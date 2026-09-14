"""GPU-vectorized SE(3) geometric controller (plan §7).

Input  per env: state (p, q, v, omega) + [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
Output per env: four rotor thrust commands (N, 4), newtons.

Structure (all steps batched, no per-env loops):
  1. velocity PI(D) -> desired acceleration (with integral wind rejection --
     the controller cannot see the wind field, like a real autopilot)
  2. accel + gravity -> desired thrust vector -> desired body Z
  3. commanded yaw (integrated from yaw_rate) -> desired attitude R_d,
     using the CAD nose axis from the robot YAML (this asset: nose = -Y)
  4. SO(3) attitude error -> body torque
  5. allocation matrix -> per-rotor thrusts, saturation-aware
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from winddyn.sim.rotors import allocate, allocation_matrix
from winddyn.utils.config import GRAVITY, VehicleParams
from winddyn.utils.torch_math import (
    matrix_to_quat,
    quat_rotate,
    quat_to_matrix,
    vee,
    yaw_of_axis,
)


class GeometricController:
    def __init__(
        self,
        vp: VehicleParams,
        num_envs: int,
        device: torch.device | str,
        cfg: dict[str, Any],
    ):
        self.vp = vp
        self.num_envs = num_envs
        self.device = torch.device(device)
        d = self.device
        g = cfg["gains"]

        t3 = lambda x: torch.tensor(x, dtype=torch.float32, device=d)
        self.kp_v = t3(g["velocity_p"])
        self.ki_v = t3(g["velocity_i"])
        self.kR = t3(g["attitude_p"])
        self.kW = t3(g["rate_p"])
        self.int_limit = float(g.get("velocity_i_limit", 3.0))

        lim = cfg["limits"]
        self.max_accel = float(lim["max_accel"])
        self.max_tilt = float(np.deg2rad(lim["max_tilt_deg"]))
        self.max_vel_cmd = float(lim["max_velocity_cmd"])
        self.max_yaw_rate = float(lim["max_yaw_rate"])

        self.mass = float(vp.mass)
        self.inertia = t3(vp.inertia_diag)
        self.nose_body = t3(vp.nose_axis_body)

        A = allocation_matrix(vp)
        self.A_inv = torch.tensor(np.linalg.inv(A), dtype=torch.float32, device=d)
        self.max_rotor_thrust = float(vp.max_rotor_thrust)

        # controller state
        self.vel_int = torch.zeros(num_envs, 3, device=d)
        self.yaw_sp = torch.zeros(num_envs, device=d)

    # ------------------------------------------------------------------ #
    def reset(self, env_ids: torch.Tensor, quat: torch.Tensor | None = None) -> None:
        """Reset integrators; if ``quat`` given, initialise yaw setpoint from it."""
        self.vel_int[env_ids] = 0.0
        if quat is not None:
            nose_w = quat_rotate(quat, self.nose_body.expand(len(env_ids), 3))
            self.yaw_sp[env_ids] = yaw_of_axis(nose_w)
        else:
            self.yaw_sp[env_ids] = 0.0

    def nose_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """(N,) current heading of the CAD nose axis, rad."""
        nose_w = quat_rotate(quat, self.nose_body.expand(quat.shape[0], 3))
        return yaw_of_axis(nose_w)

    # ------------------------------------------------------------------ #
    def compute(
        self,
        pos: torch.Tensor,        # (N, 3) world — unused, kept for interface parity
        quat: torch.Tensor,       # (N, 4) world-from-body (w, x, y, z)
        lin_vel: torch.Tensor,    # (N, 3) world
        ang_vel: torch.Tensor,    # (N, 3) body
        command: torch.Tensor,    # (N, 4) [vx, vy, vz, yaw_rate] world
        dt: float,
        accel_ff: torch.Tensor | None = None,  # (N, 3) world feed-forward
    ) -> torch.Tensor:
        """One 50 Hz control step -> (N, 4) rotor thrust commands [N]."""
        d = self.device
        N = quat.shape[0]

        vel_cmd = command[:, :3].clamp(-self.max_vel_cmd, self.max_vel_cmd)
        yaw_rate_cmd = command[:, 3].clamp(-self.max_yaw_rate, self.max_yaw_rate)

        # ---- 1. velocity loop -> desired acceleration --------------------- #
        e_v = vel_cmd - lin_vel
        self.vel_int = (self.vel_int + e_v * dt).clamp(-self.int_limit, self.int_limit)
        a_des = self.kp_v * e_v + self.ki_v * self.vel_int
        if accel_ff is not None:
            # e.g. wind-drag compensation (closed-loop utility experiment)
            a_des = a_des + accel_ff
        a_norm = a_des.norm(dim=-1, keepdim=True)
        a_des = a_des * (self.max_accel / a_norm.clamp_min(self.max_accel))

        # ---- 2. thrust vector -> desired body Z --------------------------- #
        f_des = self.mass * (a_des + torch.tensor([0.0, 0.0, GRAVITY], device=d))
        f_norm = f_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b3 = f_des / f_norm

        # tilt limit: clamp the angle of b3 from vertical
        cos_tilt = b3[:, 2].clamp(-1.0, 1.0)
        tilt = torch.acos(cos_tilt)
        over = tilt > self.max_tilt
        if over.any():
            # rotate b3 toward +Z so the tilt equals max_tilt
            z = torch.tensor([0.0, 0.0, 1.0], device=d).expand(N, 3)
            horiz = b3 - b3[:, 2:3] * z
            h_norm = horiz.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            b3_lim = np.sin(self.max_tilt) * horiz / h_norm + np.cos(self.max_tilt) * z
            b3 = torch.where(over.unsqueeze(-1), b3_lim, b3)

        # ---- 3. desired attitude from yaw setpoint ------------------------ #
        self.yaw_sp = self.yaw_sp + yaw_rate_cmd * dt
        # wrap to [-pi, pi] to keep angles well-conditioned
        self.yaw_sp = torch.remainder(self.yaw_sp + torch.pi, 2 * torch.pi) - torch.pi
        heading = torch.stack(
            [torch.cos(self.yaw_sp), torch.sin(self.yaw_sp), torch.zeros(N, device=d)],
            dim=-1,
        )
        # We want the NOSE axis (body `nose_body`) to point along `heading`.
        # Build desired axes: nose_d = normalize(heading - (heading.b3) b3)
        nose_d = heading - (heading * b3).sum(-1, keepdim=True) * b3
        nose_d = nose_d / nose_d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # Complete a right-handed frame consistent with the body layout:
        # this asset: nose_body = (0,-1,0) -> body X = nose x ... generic:
        # side_d = b3 x nose_d ; then map (nose_body, side_body, z) -> world.
        side_d = torch.cross(b3, nose_d, dim=-1)
        # body-frame anchors: nose_b, side_b = z_b x nose_b, z_b
        nose_b = self.nose_body.expand(N, 3)
        z_b = torch.tensor([0.0, 0.0, 1.0], device=d).expand(N, 3)
        side_b = torch.cross(z_b, nose_b, dim=-1)
        # R_d maps body -> world: R_d @ nose_b = nose_d etc. With B = [nose_b
        # side_b z_b] (orthonormal), W = [nose_d side_d b3]: R_d = W @ B^T.
        W = torch.stack([nose_d, side_d, b3], dim=-1)  # columns
        B = torch.stack([nose_b, side_b, z_b], dim=-1)
        R_d = W @ B.transpose(-1, -2)

        # ---- 4. SO(3) attitude error -> torque ---------------------------- #
        R = quat_to_matrix(quat)
        M = R_d.transpose(-1, -2) @ R
        e_R = 0.5 * vee(M - M.transpose(-1, -2))
        # desired body rate: only the yaw-rate feed-forward, mapped to body
        omega_d_world = torch.zeros(N, 3, device=d)
        omega_d_world[:, 2] = yaw_rate_cmd
        omega_d_body = torch.einsum("nij,nj->ni", R.transpose(-1, -2), omega_d_world)
        e_W = ang_vel - omega_d_body

        tau = (
            -self.kR * e_R
            - self.kW * e_W
            + torch.cross(ang_vel, self.inertia * ang_vel, dim=-1)
        )

        # ---- 5. collective thrust + allocation ---------------------------- #
        b3_actual = R[..., :, 2]  # third column: body Z in world
        thrust = (f_des * b3_actual).sum(-1).clamp_min(0.0)

        return allocate(self.A_inv, thrust, tau, self.max_rotor_thrust)

    def desired_quat(self) -> torch.Tensor:  # debugging helper
        raise NotImplementedError
