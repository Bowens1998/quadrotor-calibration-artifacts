"""Low-order aerodynamic drag model — UNIDENTIFIED_AERODYNAMICS.

None of these coefficients have been measured for the Starling 2 Max.  The
model exists so that wind produces physically-plausible, well-conditioned
forces for the world model to learn; it is NOT a calibrated aerodynamic model.
All coefficients are per-env tensors so domain randomisation can scale them.

Forces (plan §5.2), all batched over N envs:

    v_rel_world = v_drone - v_wind
    v_rel_body  = R^T v_rel_world
    F_body      = -0.5 rho CdA_i |v_rel_body| v_rel_body,i          (quadratic)
                  - k_rot,i * v_rel_body,i * thrust_factor          (rotor drag)
    tau_body    = -k_ang,i |omega_i| omega_i                        (quadratic)
"""

from __future__ import annotations

import torch

from winddyn.utils.config import GRAVITY, VehicleParams
from winddyn.utils.torch_math import quat_rotate, quat_rotate_inverse


class DragModel:
    def __init__(self, vp: VehicleParams, num_envs: int, device: torch.device | str):
        self.vp = vp
        d = torch.device(device)
        self.rho = float(vp.air_density)
        self.weight = float(vp.mass * GRAVITY)
        self.scale_with_thrust = bool(vp.rotor_drag_scale_with_thrust)

        base = lambda arr: torch.tensor(arr, dtype=torch.float32, device=d).expand(
            num_envs, 3
        ).clone()
        self.cda = base(vp.body_drag_cda)                # (N, 3)
        self.k_rot = base(vp.rotor_drag_coeff)           # (N, 3)
        self.k_ang = base(vp.angular_drag_coeff)         # (N, 3)

    def forces(
        self,
        quat: torch.Tensor,          # (N, 4) world-from-body
        lin_vel_world: torch.Tensor, # (N, 3)
        ang_vel_body: torch.Tensor,  # (N, 3)
        wind_world: torch.Tensor,    # (N, 3)
        total_thrust: torch.Tensor,  # (N,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (force_body (N,3), torque_body (N,3))."""
        v_rel_w = lin_vel_world - wind_world
        v_rel_b = quat_rotate_inverse(quat, v_rel_w)

        f_quad = -0.5 * self.rho * self.cda * v_rel_w.norm(dim=-1, keepdim=True) * v_rel_b

        thrust_factor = (
            (total_thrust / self.weight).clamp(0.0, 4.0).unsqueeze(-1)
            if self.scale_with_thrust
            else 1.0
        )
        f_rotor = -self.k_rot * v_rel_b * thrust_factor

        tau = -self.k_ang * ang_vel_body.abs() * ang_vel_body
        return f_quad + f_rotor, tau

    def forces_world(self, quat, lin_vel_world, ang_vel_body, wind_world, total_thrust):
        fb, tb = self.forces(quat, lin_vel_world, ang_vel_body, wind_world, total_thrust)
        return quat_rotate(quat, fb), tb
