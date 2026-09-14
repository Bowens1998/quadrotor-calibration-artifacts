"""Closed-loop downstream utility of the learned wind representation (P0-C).

The waypoint-tracking controller is augmented with a wind-drag feed-forward
computed from a *current local wind estimate* using the NOMINAL vehicle model
(same nominal-model handicap as the disturbance observer):

    a_ff = drag_acc_nominal(v) - drag_acc_nominal(v - w_hat)

so a_ff -> cancels the wind-induced part of the drag when w_hat is accurate
and is exactly zero when w_hat = 0 (the uncompensated baseline).

Wind sources (paired: identical scene, wind field, waypoints, domain
randomization, controller seed per task — the ONLY difference is w_hat):

    none      w_hat = 0                       (baseline autopilot, PI only)
    true      w_hat = true local wind          (oracle upper reference)
    observer  nominal-model drag-inversion from state history (classical)
    m6_probe  PA-JEPA frozen-latent linear probe (fit on train windows)
    m8_head   direct-regression retained head  decode(P_w(z))
    m7_int    self-conditioned model's internal estimate (optional)

Learned estimators run at the 20 Hz record rate on the same (state, action,
depth) history windows they were trained on; the model never sees the feed-
forward (it acts below the velocity-command interface).

Per-episode metrics (after a warm-up of H records): mean/median distance to
the zero-wind reference trajectory of the same task, cross-track error to the
active waypoint segment, altitude error, waypoints reached, collision fraction,
control energy sum(T^2)dt. Output: outputs/metrics/closed_loop.json +
per-episode outputs/metrics/closed_loop_episodes.json.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from _common import ROOT, device_or_fallback

from train import build_splits
from self_wind_conditioning import fit_linear_probe
from winddyn.cfd.field_io import load_field
from winddyn.control.tracking import WaypointTracker, sample_waypoints
from winddyn.cfd.interpolate import PlanarGridWind
from winddyn.geometry.procedural import load_manifest as load_scenes, MAX_BOXES
from winddyn.sim.controller import GeometricController
from winddyn.sim.depth import ToFDepthSensor
from winddyn.sim.rollout import PHYS_HZ, CTRL_EVERY, REC_EVERY, _box_tensors, _sdf, _randomize
from winddyn.sim.torch_sim import TorchQuadSim
from winddyn.train.trainer import load_model
from winddyn.utils.config import GRAVITY, load_vehicle, load_yaml
from winddyn.utils.torch_math import quat_rotate_inverse

H, K = 12, 30
DT_REC = REC_EVERY / PHYS_HZ          # 0.05 s
DT_CTRL = CTRL_EVERY / PHYS_HZ        # 0.02 s


def state_features_torch(pos, quat, vel, omega):
    """Batched (N,12) egocentric features, mirrors dataset.state_features."""
    v_body = quat_rotate_inverse(quat, vel)
    g_world = torch.tensor([0.0, 0.0, -1.0], device=pos.device).expand_as(vel)
    g_body = quat_rotate_inverse(quat, g_world)
    w_, x_, y_, z_ = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    nx = -2.0 * (x_ * y_ - w_ * z_)
    ny = -(1.0 - 2.0 * (x_ * x_ + z_ * z_))
    yaw = torch.atan2(ny, nx)
    return torch.cat([v_body, omega, g_body, pos[:, 2:3],
                      torch.sin(yaw)[:, None], torch.cos(yaw)[:, None]], dim=-1)


class HistoryBuffer:
    """Rolling buffers of the last >=H records for model input assembly."""

    def __init__(self, n_envs, device, keep=H + 4):
        self.keep = keep
        self.state, self.action, self.depth = [], [], []
        self.vel, self.bz, self.thrust = [], [], []   # observer inputs
        self.n = n_envs
        self.device = device

    def push(self, state_feat, action, depth, vel, bz, thrust):
        for buf, v in ((self.state, state_feat), (self.action, action),
                       (self.depth, depth), (self.vel, vel), (self.bz, bz),
                       (self.thrust, thrust)):
            buf.append(v)
            if len(buf) > self.keep:
                buf.pop(0)

    def ready(self):
        return len(self.state) >= H

    def batch(self, need_depth=True):
        b = {"state_hist": torch.stack(self.state[-H:], dim=1),
             "action_hist": torch.stack(self.action[-H:], dim=1)}
        if need_depth:
            d0 = self.depth[-4] if len(self.depth) >= 4 else self.depth[0]
            b["depth_hist"] = torch.stack([d0, self.depth[-1]], dim=1)
        return b


class Estimators:
    """All wind sources, evaluated on the same history buffer."""

    def __init__(self, device, seed, vp):
        self.device = device
        self.vp = vp
        splits = build_splits()
        self.m6 = load_model(
            ROOT / f"outputs/checkpoints/m6_vec_teacher_seed{seed}/best.pt", device)
        self.Wp = fit_linear_probe(self.m6, splits["train"][:160], device)
        m8_ck = ROOT / f"outputs/checkpoints/m8_direct_reg_seed{seed}/best.pt"
        self.m8 = load_model(m8_ck, device) if m8_ck.exists() else None
        m7_ck = ROOT / f"outputs/checkpoints/m7_self_cond_seed{seed}/best.pt"
        self.m7 = load_model(m7_ck, device) if m7_ck.exists() else None

    @torch.no_grad()
    def estimate(self, name, buf: HistoryBuffer, true_wind_xy):
        if name == "none":
            return torch.zeros_like(true_wind_xy)
        if name == "true":
            return true_wind_xy
        if not buf.ready():
            return torch.zeros_like(true_wind_xy)
        if name == "observer":
            return self._observer(buf)
        if name == "m6_probe":
            z = self.m6.encode_context(buf.batch()).cpu().numpy()
            Zb = np.concatenate([z, np.ones((len(z), 1))], 1)
            return torch.tensor(Zb @ self.Wp, dtype=torch.float32,
                                device=self.device)
        if name == "m8_head":
            return self.m8.estimate_wind(self.m8.encode_context(buf.batch()))
        if name == "m7_int":
            b = buf.batch()
            hh = b["state_hist"].shape[1]
            b["wind_hist"] = torch.zeros(b["state_hist"].shape[0], hh, 2,
                                         device=self.device)
            b["wind_in_hist"] = b["wind_hist"]
            return self.m7.estimate_wind(self.m7.encode_context(b))
        raise ValueError(name)

    def _observer(self, buf):
        """Thrust-aware nominal-model drag inversion, mirroring
        probe_extras.observer_wind_errors (the paper's observer reference)."""
        vp = self.vp
        v3 = torch.stack(buf.vel[-H:], dim=1).cpu().numpy()      # (N, H, 3)
        bz = torch.stack(buf.bz[-H:], dim=1).cpu().numpy()       # (N, H, 3)
        T = torch.stack(buf.thrust[-H:], dim=1).cpu().numpy()    # (N, H)
        acc = np.gradient(v3, DT_REC, axis=1)
        a_res = acc - (T[..., None] * bz / vp.mass) \
            + np.array([0.0, 0.0, GRAVITY])
        v = v3[..., :2]
        cda = vp.body_drag_cda[:2].mean()
        krot = vp.rotor_drag_coeff[:2].mean()
        weight = vp.mass * GRAVITY
        tf = np.clip(T / weight, 0.0, 4.0)[..., None]
        w = np.zeros_like(v)
        for _ in range(25):
            vr = v - w
            sp = np.linalg.norm(vr, axis=-1, keepdims=True)
            coef = (0.5 * vp.air_density * cda * sp + krot * tf) / vp.mass
            w_new = v + a_res[..., :2] / np.clip(coef, 1e-3, None)
            w = 0.5 * w + 0.5 * w_new
        return torch.tensor(w[:, -6:].mean(1), dtype=torch.float32,
                            device=self.device)


def drag_accel_nominal(v_rel_xy, thrust_factor, vp):
    """(N,2) horizontal drag acceleration from the NOMINAL model."""
    cda = float(vp.body_drag_cda[:2].mean())
    krot = float(vp.rotor_drag_coeff[:2].mean())
    s = v_rel_xy.norm(dim=-1, keepdim=True)
    f = -(0.5 * vp.air_density * cda * s * v_rel_xy
          + krot * v_rel_xy * thrust_factor.unsqueeze(-1))
    return f / vp.mass


@torch.no_grad()
def run_tier(tasks, vp, estimators, condition, seed, device,
             ref_traj=None, duration_s=12.0):
    """One paired rollout batch under `condition`; returns per-env metrics
    and the recorded position trajectories."""
    N = len(tasks)
    d = torch.device(device)
    gen = torch.Generator(device=d).manual_seed(seed)

    scenes = [t["scene"] for t in tasks]
    z_ref = scenes[0].z_ref
    centers, half, mask = _box_tensors(scenes, d)

    zero_wind = condition == "ref"
    fields = [t["field"] for t in tasks]
    grid = PlanarGridWind(fields, N, device=d)
    grid.assign(torch.arange(N))

    class Wind:
        name = "grid"
        def step(self, dt): pass
        def sample(self, positions, sim_time, env_ids=None):
            if zero_wind:
                return torch.zeros_like(positions)
            return grid.sample(positions, sim_time)
        def reset(self, env_ids, generator): pass

    sim = TorchQuadSim(vp, N, Wind(), device=d, physics_dt=1.0 / PHYS_HZ)
    _randomize(sim, vp, gen)

    ctrl_cfg = load_yaml("configs/sim/controller.yaml")["controller"]
    ctrl = GeometricController(vp, N, d, ctrl_cfg)

    wps, starts = [], []
    for i, t in enumerate(tasks):
        rng_i = np.random.default_rng(t["wp_seed"])
        w8 = sample_waypoints(t["scene"], rng_i)
        wps.append(w8)
        starts.append(np.array([w8[0][0], w8[0][1], z_ref]))
    tracker = WaypointTracker(
        torch.tensor(np.stack(wps), dtype=torch.float32), z_ref, d)
    wp_t = tracker.wp

    tof = vp.tof or {}
    depth_cam = ToFDepthSensor(
        centers, half, mask, width=64, height=64,
        hfov_deg=float(tof.get("horizontal_fov_deg", 106.0)),
        min_range=float(tof.get("min_range", 0.2)),
        max_range=float(tof.get("max_range", 6.0)),
        offset_body=tuple(tof.get("position", (-0.018, -0.078, 0.003))),
        device=d)

    sim.reset(pos=torch.tensor(np.stack(starts), dtype=torch.float32, device=d),
              generator=gen)
    ctrl.reset(torch.arange(N, device=d), sim.state.quat)

    buf = HistoryBuffer(N, d)
    n_steps = int(duration_s * PHYS_HZ)
    n_rec = n_steps // REC_EVERY
    traj = np.zeros((n_rec, N, 3), np.float32)
    cross_track = np.zeros((n_rec, N), np.float32)
    alt_err = np.zeros((n_rec, N), np.float32)
    collide = np.zeros((n_rec, N), bool)
    w_err = np.full((n_rec, N), np.nan, np.float32)
    energy = np.zeros(N, np.float64)
    wp_count = np.zeros(N, np.int64)
    prev_idx = tracker.idx.clone()

    cmd = torch.zeros(N, 4, device=d)
    thrust_cmd = torch.zeros(N, 4, device=d)
    w_hat = torch.zeros(N, 2, device=d)
    k = 0
    for step in range(n_steps):
        if step % REC_EVERY == 0:
            s = sim.state
            sf = state_features_torch(s.pos, s.quat, s.lin_vel, s.ang_vel)
            depth = depth_cam.render(s.pos, s.quat)
            q = s.quat
            bz = torch.stack([2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2]),
                              2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1]),
                              1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)], dim=-1)
            buf.push(sf, cmd.clone(), depth, s.lin_vel.clone(), bz,
                     thrust_cmd.sum(-1).clone())
            true_w = sim.wind.sample(s.pos, sim.time)[:, :2] if not zero_wind \
                else grid.sample(s.pos, sim.time)[:, :2]
            if condition not in ("ref",):
                w_hat = estimators.estimate(condition, buf, true_w)
                w_hat = w_hat.clamp(-15.0, 15.0)
                w_err[k] = (w_hat - true_w).norm(dim=-1).cpu().numpy()
            traj[k] = s.pos.cpu().numpy()
            sd = _sdf(s.pos, centers, half, mask)
            collide[k] = (sd < 0.25).cpu().numpy() | (s.pos[:, 2] < 0.1).cpu().numpy()
            alt_err[k] = (s.pos[:, 2] - z_ref).abs().cpu().numpy()
            # cross-track: distance to segment prev_wp -> current_wp (xy)
            idx = tracker.idx
            cur = wp_t[torch.arange(N, device=d), idx]
            prev = wp_t[torch.arange(N, device=d), (idx - 1) % tracker.n_wp]
            seg = cur - prev
            L2 = (seg ** 2).sum(-1).clamp_min(1e-6)
            tproj = (((s.pos[:, :2] - prev) * seg).sum(-1) / L2).clamp(0, 1)
            closest = prev + tproj.unsqueeze(-1) * seg
            cross_track[k] = (s.pos[:, :2] - closest).norm(dim=-1).cpu().numpy()
            k += 1
        if step % CTRL_EVERY == 0:
            s = sim.state
            cmd = tracker.command(s.pos, ctrl.nose_yaw(s.quat))
            wp_count += (tracker.idx != prev_idx).cpu().numpy().astype(np.int64)
            prev_idx = tracker.idx.clone()
            a_ff = None
            if condition not in ("ref", "none"):
                v_xy = s.lin_vel[:, :2]
                tfac = (thrust_cmd.sum(-1) / (vp.mass * GRAVITY)).clamp(0.0, 4.0)
                a_ff_xy = (drag_accel_nominal(v_xy, tfac, vp)
                           - drag_accel_nominal(v_xy - w_hat, tfac, vp))
                a_ff = torch.cat([a_ff_xy,
                                  torch.zeros(N, 1, device=d)], dim=-1)
            thrust_cmd = ctrl.compute(s.pos, s.quat, s.lin_vel, s.ang_vel,
                                      cmd, DT_CTRL, accel_ff=a_ff)
            energy += (thrust_cmd ** 2).sum(-1).cpu().numpy() * DT_CTRL
        sim.step(thrust_cmd)

    warm = H  # skip warm-up records
    out = {
        "cross_track_mean": cross_track[warm:].mean(0),
        "alt_err_mean": alt_err[warm:].mean(0),
        "collision_frac": collide[warm:].mean(0),
        "waypoints_reached": wp_count.astype(np.float64),
        "energy": energy,
        "wind_est_rmse": (np.sqrt(np.nanmean(w_err[warm:] ** 2, axis=0))
                          if not np.isnan(w_err[warm:]).all()
                          else np.full(N, np.nan)),
    }
    if ref_traj is not None:
        dref = np.linalg.norm(traj[warm:] - ref_traj[warm:], axis=-1)
        out["ref_dev_mean"] = dref.mean(0)
        out["ref_dev_max"] = dref.max(0)
    return out, traj


def build_tasks(tier, n_tasks, rng):
    scfg = load_yaml("configs/sim/scenes_mvp.yaml")
    scenes = {s.scene_id: s for s in load_scenes(ROOT / scfg["manifest"])}
    n_ood = scfg["geo_ood_seeds"]
    ood_scene = {sid for sid, s in scenes.items()
                 if s.seed >= scfg["seeds_per_family"] - n_ood}
    with open(ROOT / "data/manifests/wind_fields.json") as f:
        fields = json.load(f)["fields"]
    if tier == "id":
        cand = [e for e in fields if e["tag"] == "train"
                and e["scene_id"] not in ood_scene]
    elif tier == "wind_extrap":
        cand = [e for e in fields if e["tag"] == "ood_extrap"
                and e["scene_id"] not in ood_scene]
    elif tier == "joint_extrap":
        cand = [e for e in fields if e["tag"] == "ood_extrap"
                and e["scene_id"] in ood_scene]
    else:
        raise ValueError(tier)
    sel = rng.choice(len(cand), size=min(n_tasks, len(cand)), replace=False)
    tasks = []
    for j, i in enumerate(sel):
        e = cand[int(i)]
        tasks.append({"scene": scenes[e["scene_id"]],
                      "field": load_field(e["path"]),
                      "wind_id": e["wind_id"],
                      "scene_id": e["scene_id"],
                      "wp_seed": 90000 + 131 * int(i)})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+",
                    default=["id", "wind_extrap", "joint_extrap"])
    ap.add_argument("--conditions", nargs="+",
                    default=["none", "true", "observer", "m6_probe",
                             "m8_head", "m7_int"])
    ap.add_argument("--n-tasks", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="model seeds for learned estimators")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = device_or_fallback(args.device)
    vp = load_vehicle()
    rng = np.random.default_rng(4242)

    episodes = []          # per-episode records for paired stats
    agg = defaultdict(dict)
    tier_off = {"id": 0, "wind_extrap": 100, "joint_extrap": 200}
    est_cache = {s: Estimators(device, s, vp) for s in args.seeds}
    for tier in args.tiers:
        tasks = build_tasks(tier, args.n_tasks, rng)
        for b0 in range(0, len(tasks), args.batch):
            chunk = tasks[b0:b0 + args.batch]
            batch_seed = 777_000 + tier_off[tier] + b0
            # zero-wind reference (model-free, shared across conditions)
            _, ref_traj = run_tier(chunk, vp, None, "ref", batch_seed, device)
            for seed in args.seeds:
                est = est_cache[seed]
                for cond in args.conditions:
                    if cond in ("none", "true", "observer") and seed != args.seeds[0]:
                        continue  # model-free: run once
                    if cond == "m8_head" and est.m8 is None:
                        continue
                    if cond == "m7_int" and est.m7 is None:
                        continue
                    m, _ = run_tier(chunk, vp, est, cond, batch_seed, device,
                                    ref_traj=ref_traj)
                    for i, t in enumerate(chunk):
                        episodes.append({
                            "tier": tier, "condition": cond,
                            "model_seed": seed if cond not in
                            ("none", "true", "observer") else -1,
                            "scene_id": t["scene_id"], "wind_id": t["wind_id"],
                            "task_key": f"{tier}/{t['scene_id']}/{t['wind_id']}",
                            **{kk: float(vv[i]) for kk, vv in m.items()},
                        })
            print(f"{tier} batch {b0 // args.batch}: done", flush=True)

    # aggregate: mean over episodes (and model seeds) per (tier, condition)
    for tier in args.tiers:
        for cond in args.conditions:
            rows = [e for e in episodes if e["tier"] == tier
                    and e["condition"] == cond]
            if not rows:
                continue
            agg[tier][cond] = {
                kk: {"mean": float(np.mean([r[kk] for r in rows])),
                     "std": float(np.std([r[kk] for r in rows]))}
                for kk in ("ref_dev_mean", "cross_track_mean", "alt_err_mean",
                           "collision_frac", "waypoints_reached", "energy",
                           "wind_est_rmse")}
    dest = ROOT / "outputs/metrics/closed_loop.json"
    with open(dest, "w") as f:
        json.dump(agg, f, indent=1)
    with open(ROOT / "outputs/metrics/closed_loop_episodes.json", "w") as f:
        json.dump(episodes, f)
    for tier, d in agg.items():
        print(f"== {tier} ==")
        for cond, m in d.items():
            print(f"  {cond:9s} ref_dev {m['ref_dev_mean']['mean']:.3f} "
                  f"xtrack {m['cross_track_mean']['mean']:.3f} "
                  f"wrmse {m['wind_est_rmse']['mean']:.3f}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
