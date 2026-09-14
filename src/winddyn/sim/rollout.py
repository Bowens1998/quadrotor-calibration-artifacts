"""Batched wind-coupled rollout collection on the torch reference simulator.

Each env in a batch runs one (scene, wind field, controller mode) episode:
physics at 200 Hz, control at 50 Hz, records at 20 Hz. All envs share one
domain size so per-env scenes/fields stack into fixed tensors.

The Isaac Lab backend can replace TorchQuadSim later; everything downstream
consumes only the recorded episode dict (spec §6.1: the ML layer must not
depend on simulator internals).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from winddyn.control.tracking import OUPerturbation, WaypointTracker, sample_waypoints
from winddyn.cfd.field_io import PlanarField
from winddyn.cfd.interpolate import PlanarGridWind, UniformWind, WindSource, ZeroWind
from winddyn.geometry.procedural import SceneSpec, MAX_BOXES
from winddyn.sim.controller import GeometricController
from winddyn.sim.depth import ToFDepthSensor
from winddyn.sim.torch_sim import TorchQuadSim
from winddyn.utils.config import VehicleParams, load_yaml


PHYS_HZ = 200
CTRL_EVERY = 4      # 50 Hz
REC_EVERY = 10      # 20 Hz


@dataclass
class EnvTask:
    scene: SceneSpec
    field: PlanarField | None        # None -> zero wind; used for PlanarGridWind
    wind_id: str                     # "none", "uniform_...", or field wind id
    mode: str                        # "tracking" | "perturbation"
    uniform_uv: tuple[float, float] | None = None


def _box_tensors(scenes: list[SceneSpec], device):
    c = torch.tensor(np.stack([s.box_centers for s in scenes]), dtype=torch.float32, device=device)
    h = torch.tensor(np.stack([s.box_sizes for s in scenes]) / 2.0, dtype=torch.float32, device=device)
    m = torch.tensor(
        np.stack([[i < s.n_boxes for i in range(MAX_BOXES)] for s in scenes]),
        dtype=torch.bool, device=device,
    )
    return c, h, m


def _sdf(pos, centers, half, mask):
    """(N,3) world -> (N,) 3-D signed distance to nearest box (inf if none)."""
    q = (pos.unsqueeze(1) - centers).abs() - half
    outside = q.clamp_min(0.0).norm(dim=-1)
    inside = q.max(dim=-1).values.clamp_max(0.0)
    dist = outside + inside
    return torch.where(mask, dist, torch.full_like(dist, float("inf"))).min(dim=-1).values


def _randomize(sim: TorchQuadSim, vp: VehicleParams, gen: torch.Generator):
    """Per-env domain randomization (spec §8.1) using the robot YAML ranges."""
    dr = vp.domain_randomization
    if not dr.get("enabled", True):
        return {}
    d = sim.device
    N = sim.num_envs
    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(N, generator=gen, device=d)
    scales = {
        "mass": u(*dr.get("mass", [0.9, 1.1])),
        "inertia": u(*dr.get("inertia", [0.8, 1.25])),
        "cda": u(*dr.get("body_drag_cda", [0.5, 2.0])),
        "rotor_drag": u(*dr.get("rotor_drag_coeff", [0.3, 2.5])),
        "thrust": u(*dr.get("thrust_constant", [0.9, 1.1])),
        "tau_m": u(*dr.get("motor_time_constant", [0.7, 1.4])),
    }
    sim.mass = sim.mass * scales["mass"]
    sim.inertia = sim.inertia * scales["inertia"].unsqueeze(-1)
    sim.drag.cda = sim.drag.cda * scales["cda"].unsqueeze(-1)
    sim.drag.k_rot = sim.drag.k_rot * scales["rotor_drag"].unsqueeze(-1)
    sim.rotors.thrust_scale = scales["thrust"]
    sim.rotors.time_constant = sim.rotors.time_constant * scales["tau_m"]
    return {k: v.cpu().numpy() for k, v in scales.items()}


@torch.no_grad()
def collect_batch(
    tasks: list[EnvTask],
    vp: VehicleParams,
    duration_s: float = 12.0,
    seed: int = 0,
    device: str = "cpu",
    depth_hw: tuple[int, int] = (64, 64),
    patch_size: int = 16,
    patch_spacing: float = 0.5,
    controller_cfg_path: str = "configs/sim/controller.yaml",
    commands: torch.Tensor | None = None,   # (n_ctrl_steps, N, 4) open-loop replay
    randomize: bool = True,
    start_override: np.ndarray | None = None,  # (N, 3)
) -> list[dict]:
    """Run one batch; returns one episode dict per env.

    With ``commands`` given, the velocity-command sequence is replayed verbatim
    (no waypoint tracker, no OU noise) — used for Gate A divergence checks and
    the counterfactual wind-intervention experiment (spec §17, §25).
    """
    N = len(tasks)
    d = torch.device(device)
    gen = torch.Generator(device=d).manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    scenes = [t.scene for t in tasks]
    z_ref = scenes[0].z_ref
    centers, half, mask = _box_tensors(scenes, d)

    # wind: stack the planar fields that exist; uniform/zero handled per env
    field_envs = [i for i, t in enumerate(tasks) if t.field is not None]
    wind: WindSource
    if field_envs:
        fields = [tasks[i].field for i in field_envs]
        grid_wind = PlanarGridWind(fields, N, device=d)
        idx = torch.zeros(N, dtype=torch.long)
        for j, i in enumerate(field_envs):
            idx[i] = j
        grid_wind.assign(idx)
        wind = grid_wind
    else:
        grid_wind = None
        wind = ZeroWind()
    uniform_mask = torch.tensor(
        [t.uniform_uv is not None for t in tasks], device=d
    )
    uniform_vec = torch.zeros(N, 3, device=d)
    for i, t in enumerate(tasks):
        if t.uniform_uv is not None:
            uniform_vec[i, 0], uniform_vec[i, 1] = t.uniform_uv
    zero_mask = torch.tensor(
        [t.field is None and t.uniform_uv is None for t in tasks], device=d
    )

    class MixedWind(WindSource):
        name = "mixed"
        def step(self, dt):
            pass
        def sample(self, positions, sim_time, env_ids=None):
            w = (grid_wind.sample(positions, sim_time) if grid_wind is not None
                 else torch.zeros_like(positions))
            w = torch.where(uniform_mask.unsqueeze(-1), uniform_vec, w)
            return torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(w), w)
        def reset(self, env_ids, generator):
            pass

    sim = TorchQuadSim(vp, N, MixedWind(), device=d, physics_dt=1.0 / PHYS_HZ)
    dr_scales = _randomize(sim, vp, gen) if randomize else {}

    ctrl_cfg = load_yaml(controller_cfg_path)["controller"]
    ctrl = GeometricController(vp, N, d, ctrl_cfg)
    # perturbation envs: no integral wind rejection (spec §9.3)
    pert = torch.tensor([t.mode == "perturbation" for t in tasks], device=d)
    ctrl.ki_v = ctrl.ki_v.unsqueeze(0) * (~pert).unsqueeze(-1).float()

    # waypoints + start positions in free space
    wps = []
    starts = []
    for i, t in enumerate(tasks):
        rng_i = np.random.default_rng(seed * 10007 + i)
        w8 = sample_waypoints(t.scene, rng_i)
        wps.append(w8)
        starts.append(np.array([w8[0][0], w8[0][1], z_ref]))
    tracker = WaypointTracker(
        torch.tensor(np.stack(wps), dtype=torch.float32), z_ref, d
    )
    ou = OUPerturbation(N, d, generator=gen)

    tof = vp.tof or {}
    depth_cam = ToFDepthSensor(
        centers, half, mask,
        width=depth_hw[1], height=depth_hw[0],
        hfov_deg=float(tof.get("horizontal_fov_deg", 106.0)),
        min_range=float(tof.get("min_range", 0.2)),
        max_range=float(tof.get("max_range", 6.0)),
        offset_body=tuple(tof.get("position", (-0.018, -0.078, 0.003))),
        device=d,
    )

    if start_override is not None:
        starts = list(start_override)
    sim.reset(pos=torch.tensor(np.stack(starts), dtype=torch.float32, device=d),
              generator=gen)
    ctrl.reset(torch.arange(N, device=d), sim.state.quat)

    n_steps = int(duration_s * PHYS_HZ)
    n_rec = n_steps // REC_EVERY
    H, Wd = depth_hw
    rec = {
        "t": np.zeros(n_rec, np.float32),
        "position_world": np.zeros((n_rec, N, 3), np.float32),
        "velocity_world": np.zeros((n_rec, N, 3), np.float32),
        "quaternion_world_body": np.zeros((n_rec, N, 4), np.float32),
        "angular_velocity_body": np.zeros((n_rec, N, 3), np.float32),
        "action": np.zeros((n_rec, N, 4), np.float32),
        "rotor_thrust_cmd": np.zeros((n_rec, N, 4), np.float32),
        "wind_local_world": np.zeros((n_rec, N, 3), np.float32),
        "depth": np.zeros((n_rec, N, H, Wd), np.float16),
        "wind_patch": np.zeros((n_rec, N, 2, patch_size, patch_size), np.float16),
        "sdf": np.zeros((n_rec, N), np.float32),
        "collision": np.zeros((n_rec, N), bool),
        "goal_relative": np.zeros((n_rec, N, 3), np.float32),
    }

    cmd = torch.zeros(N, 4, device=d)
    thrust_cmd = torch.zeros(N, 4, device=d)
    k = 0
    for step in range(n_steps):
        if step % CTRL_EVERY == 0:
            s = sim.state
            if commands is not None:
                ci = min(step // CTRL_EVERY, commands.shape[0] - 1)
                cmd = commands[ci].to(d)
            else:
                cmd = tracker.command(s.pos, ctrl.nose_yaw(s.quat))
                noise = ou.step(CTRL_EVERY / PHYS_HZ).clone()
                noise[:, 2] *= 0.3   # keep altitude excursions small (planar-wind MVP)
                cmd = cmd.clone()
                cmd[:, :3] += torch.where(
                    pert.unsqueeze(-1), noise, torch.zeros_like(noise)
                )
            thrust_cmd = ctrl.compute(
                s.pos, s.quat, s.lin_vel, s.ang_vel, cmd, CTRL_EVERY / PHYS_HZ
            )
        if step % REC_EVERY == 0:
            s = sim.state
            rec["t"][k] = sim.time
            rec["position_world"][k] = s.pos.cpu().numpy()
            rec["velocity_world"][k] = s.lin_vel.cpu().numpy()
            rec["quaternion_world_body"][k] = s.quat.cpu().numpy()
            rec["angular_velocity_body"][k] = s.ang_vel.cpu().numpy()
            rec["action"][k] = cmd.cpu().numpy()
            rec["rotor_thrust_cmd"][k] = thrust_cmd.cpu().numpy()
            rec["wind_local_world"][k] = sim.wind.sample(s.pos, sim.time).cpu().numpy()
            rec["depth"][k] = depth_cam.render(s.pos, s.quat).cpu().numpy().astype(np.float16)
            if grid_wind is not None:
                rec["wind_patch"][k] = (
                    grid_wind.patch(s.pos, patch_size, patch_spacing)
                    .cpu().numpy().astype(np.float16)
                )
            sd = _sdf(s.pos, centers, half, mask)
            rec["sdf"][k] = sd.cpu().numpy()
            rec["collision"][k] = (sd < 0.25).cpu().numpy() | (s.pos[:, 2] < 0.1).cpu().numpy()
            tgt = tracker.wp[torch.arange(N, device=d), tracker.idx]
            gr = torch.cat([tgt - s.pos[:, :2],
                            (z_ref - s.pos[:, 2]).unsqueeze(-1)], dim=-1)
            rec["goal_relative"][k] = gr.cpu().numpy()
            k += 1
        sim.step(thrust_cmd)

    episodes = []
    for i, t in enumerate(tasks):
        ep = {key: (val[:, i] if val.ndim > 1 else val) for key, val in rec.items()}
        ep["meta"] = {
            "scene_id": t.scene.scene_id,
            "family": t.scene.family,
            "scene_seed": int(t.scene.seed),
            "wind_id": t.wind_id,
            "mode": t.mode,
            "z_ref": float(z_ref),
            "seed": int(seed),
            "env_index": int(i),
            "record_hz": PHYS_HZ // REC_EVERY,
            "patch_convention": "world_axis_aligned_centered",
            "patch_spacing_m": float(patch_spacing),
            "field_meta": (t.field.meta if t.field is not None else
                           {"field_type": "uniform" if t.uniform_uv else "none"}),
            "domain_randomization": {kk: float(vv[i]) for kk, vv in dr_scales.items()},
        }
        episodes.append(ep)
    return episodes
