"""Batched rotor thrust, first-order motor response and control allocation.

Everything carries an env/batch dimension N; there are no per-env Python loops.

Actuation mode (plan §5.1): ``direct_rotor_thrust`` -- commands are per-rotor
thrusts in newtons.  Thrust constants exist (derived from the declared hover
operating point, UNIDENTIFIED_PROPULSION) so rotor speeds can be reported, but
the control variable is thrust, which keeps the demo independent of the unknown
motor curve.
"""

from __future__ import annotations

import numpy as np
import torch

from winddyn.utils.config import VehicleParams

ACTUATION_MODE = "direct_rotor_thrust"


class RotorModel:
    """First-order thrust dynamics + wrench computation for N envs.

    State: ``thrust`` (N, 4) in newtons.
    """

    def __init__(self, vp: VehicleParams, num_envs: int, device: torch.device | str):
        self.vp = vp
        self.num_envs = num_envs
        self.device = torch.device(device)
        d = self.device

        self.positions = torch.tensor(
            vp.rotor_positions, dtype=torch.float32, device=d
        )  # (4, 3)
        self.spin_dirs = torch.tensor(
            vp.spin_dirs, dtype=torch.float32, device=d
        )  # (4,)
        self.max_thrust = float(vp.max_rotor_thrust)
        self.torque_to_thrust = float(vp.torque_to_thrust)

        # Per-env randomisable multipliers (set by domain randomisation).
        self.thrust_scale = torch.ones(num_envs, device=d)
        self.torque_scale = torch.ones(num_envs, device=d)
        self.time_constant = torch.full(
            (num_envs,), float(vp.motor_time_constant), device=d
        )

        self.thrust = torch.zeros(num_envs, 4, device=d)

    # ------------------------------------------------------------------ #
    def reset(self, env_ids: torch.Tensor, hover_init: bool = True) -> None:
        if hover_init:
            self.thrust[env_ids] = self.vp.hover_thrust_per_rotor
        else:
            self.thrust[env_ids] = 0.0

    def step(self, thrust_cmd: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the first-order motor lag; returns actual thrusts (N, 4)."""
        cmd = thrust_cmd.clamp(0.0, self.max_thrust)
        alpha = 1.0 - torch.exp(-dt / self.time_constant.clamp_min(1e-4))
        self.thrust = self.thrust + alpha.unsqueeze(-1) * (cmd - self.thrust)
        return self.thrust

    def body_wrench(self, thrust: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Total (force, torque) in the body frame about the COM.

        force  (N, 3): sum of rotor thrusts along body +Z, scaled per env.
        torque (N, 3): moment of the thrusts about the COM plus reaction yaw.
        """
        f = (self.thrust if thrust is None else thrust) * self.thrust_scale.unsqueeze(-1)
        force = torch.zeros(f.shape[0], 3, device=f.device, dtype=f.dtype)
        force[:, 2] = f.sum(dim=-1)

        # torque from thrust offsets: sum_i r_i x (f_i * z)  = f_i * (r_y, -r_x, 0)
        r = self.positions  # (4, 3)
        tau_x = (f * r[:, 1]).sum(dim=-1)
        tau_y = (-f * r[:, 0]).sum(dim=-1)
        # reaction torque: each rotor drags the body opposite its spin
        tau_z = (
            -(f * self.spin_dirs).sum(dim=-1)
            * self.torque_to_thrust
            * self.torque_scale
        )
        torque = torch.stack([tau_x, tau_y, tau_z], dim=-1)
        return force, torque


def allocation_matrix(vp: VehicleParams) -> np.ndarray:
    """(4, 4) map from per-rotor thrusts to [T, tau_x, tau_y, tau_z].

    Row 0: total thrust.  Rows 1-3: torques about the COM (see body_wrench).
    Uses the REAL extracted rotor positions, so it is correct for the CAD body
    frame regardless of which way the nose points.
    """
    r = vp.rotor_positions
    s = vp.spin_dirs
    A = np.zeros((4, 4))
    A[0, :] = 1.0
    A[1, :] = r[:, 1]
    A[2, :] = -r[:, 0]
    A[3, :] = -s * vp.torque_to_thrust
    return A


def allocate(
    A_inv: torch.Tensor,
    total_thrust: torch.Tensor,
    torque: torch.Tensor,
    max_rotor_thrust: float,
) -> torch.Tensor:
    """Solve for per-rotor thrusts and saturate.

    Args:
        A_inv: (4, 4) inverse allocation matrix.
        total_thrust: (N,) desired collective thrust, N.
        torque: (N, 3) desired body torque, N m.
    Returns:
        (N, 4) rotor thrust commands in [0, max_rotor_thrust].

    Saturation policy: if any rotor leaves its limits, the TORQUE demand is
    scaled down (binary search-free, closed-form per env) before thrust is
    sacrificed -- attitude authority degrades gracefully instead of the
    collective collapsing.
    """
    wrench = torch.cat([total_thrust.unsqueeze(-1), torque], dim=-1)  # (N, 4)
    f = wrench @ A_inv.T  # (N, 4)

    lo, hi = f.min(dim=-1).values, f.max(dim=-1).values
    over = (hi > max_rotor_thrust) | (lo < 0.0)
    if over.any():
        # thrust-only component per rotor (torque = 0)
        f_coll = (
            torch.cat(
                [wrench[:, :1], torch.zeros_like(wrench[:, 1:])], dim=-1
            )
            @ A_inv.T
        )
        f_tau = f - f_coll
        # largest gamma in [0, 1] with  0 <= f_coll + gamma * f_tau <= max
        eps = 1e-9
        room_hi = (max_rotor_thrust - f_coll) / f_tau.clamp_min(eps)
        room_lo = (0.0 - f_coll) / f_tau.clamp_max(-eps)
        gamma = torch.minimum(
            torch.where(f_tau > eps, room_hi, torch.ones_like(room_hi)),
            torch.where(f_tau < -eps, room_lo, torch.ones_like(room_lo)),
        ).min(dim=-1).values.clamp(0.0, 1.0)
        f = torch.where(
            over.unsqueeze(-1), f_coll + gamma.unsqueeze(-1) * f_tau, f
        )
    return f.clamp(0.0, max_rotor_thrust)
