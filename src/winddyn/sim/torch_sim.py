"""Reference batched 6-DOF rigid-body simulator in pure torch.

Purpose:
  * unit/integration tests without booting Isaac Sim (fast, CI-friendly)
  * controller tuning and world-model dataset *debugging*

It integrates the SAME force models (RotorModel, DragModel, WindField) that the
Isaac Lab backend applies, with semi-implicit Euler at the physics rate.  It is
NOT a substitute for PhysX — contacts/collisions are not simulated here (obstacle
proximity is checked analytically by the scenario SDF instead).

All state tensors carry the env dimension N.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from winddyn.sim.aero import DragModel
from winddyn.sim.rotors import RotorModel
from winddyn.utils.config import GRAVITY, VehicleParams
from winddyn.utils.torch_math import quat_integrate, quat_rotate, tilt_angle
from winddyn.cfd.interpolate import WindSource as WindField


@dataclass
class SimState:
    pos: torch.Tensor       # (N, 3) world
    quat: torch.Tensor      # (N, 4) world-from-body (w, x, y, z)
    lin_vel: torch.Tensor   # (N, 3) world
    ang_vel: torch.Tensor   # (N, 3) body


class TorchQuadSim:
    def __init__(
        self,
        vp: VehicleParams,
        num_envs: int,
        wind: WindField | None,
        device: torch.device | str = "cpu",
        physics_dt: float = 1.0 / 200.0,
    ):
        self.vp = vp
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = physics_dt
        d = self.device

        self.rotors = RotorModel(vp, num_envs, d)
        self.drag = DragModel(vp, num_envs, d)
        self.wind = wind

        self.mass = torch.full((num_envs,), float(vp.mass), device=d)
        self.inertia = (
            torch.tensor(vp.inertia_diag, dtype=torch.float32, device=d)
            .expand(num_envs, 3)
            .clone()
        )

        self.state = SimState(
            pos=torch.zeros(num_envs, 3, device=d),
            quat=torch.zeros(num_envs, 4, device=d),
            lin_vel=torch.zeros(num_envs, 3, device=d),
            ang_vel=torch.zeros(num_envs, 3, device=d),
        )
        self.state.quat[:, 0] = 1.0
        self.time = 0.0
        self.last_wind = torch.zeros(num_envs, 3, device=d)

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        s = self.state
        s.pos[env_ids] = 0.0 if pos is None else pos
        s.quat[env_ids] = 0.0
        s.quat[env_ids, 0] = 1.0
        s.lin_vel[env_ids] = 0.0
        s.ang_vel[env_ids] = 0.0
        self.rotors.reset(env_ids)
        if self.wind is not None and generator is not None:
            self.wind.reset(env_ids, generator)

    def step(self, thrust_cmd: torch.Tensor) -> SimState:
        """Advance one physics step with per-rotor thrust commands (N, 4)."""
        s, dt, d = self.state, self.dt, self.device

        thrust = self.rotors.step(thrust_cmd, dt)
        f_rot_b, tau_rot_b = self.rotors.body_wrench(thrust)

        if self.wind is not None:
            self.wind.step(dt)
            wind = self.wind.sample(s.pos, self.time)
        else:
            wind = torch.zeros_like(s.pos)
        self.last_wind = wind

        f_aero_b, tau_aero_b = self.drag.forces(
            s.quat, s.lin_vel, s.ang_vel, wind, f_rot_b[:, 2]
        )

        f_body = f_rot_b + f_aero_b
        f_world = quat_rotate(s.quat, f_body)
        f_world[:, 2] -= self.mass * GRAVITY

        # semi-implicit Euler
        acc = f_world / self.mass.unsqueeze(-1)
        s.lin_vel = s.lin_vel + acc * dt
        s.pos = s.pos + s.lin_vel * dt

        tau = tau_rot_b + tau_aero_b
        # Euler's equation with diagonal inertia
        w = s.ang_vel
        ang_acc = (tau - torch.cross(w, self.inertia * w, dim=-1)) / self.inertia
        s.ang_vel = w + ang_acc * dt
        s.quat = quat_integrate(s.quat, s.ang_vel, dt)

        self.time += dt
        return s

    # convenience diagnostics ------------------------------------------------
    def tilt_deg(self) -> torch.Tensor:
        return torch.rad2deg(tilt_angle(self.state.quat))

    def is_finite(self) -> bool:
        s = self.state
        return bool(
            torch.isfinite(s.pos).all()
            and torch.isfinite(s.quat).all()
            and torch.isfinite(s.lin_vel).all()
            and torch.isfinite(s.ang_vel).all()
        )
