"""Analytic ToF depth rendering for the torch simulator backend.

Raycasts the scene's axis-aligned boxes + ground plane from the Starling's ToF
frame (pose EXTRACTED from the USD audit; resolution/FOV/range ASSUMED —
UNIDENTIFIED_SENSOR_INTRINSICS). Interfaces match what an Isaac Lab depth
camera would produce: (N, H, W) float32 range images clipped to
[min_range, max_range], max_range where no return.

Body frame convention (from the asset): nose/optical axis = body -Y, up = +Z,
image-right = body -X, image-down = body -Z.
"""

from __future__ import annotations

import math

import torch

from winddyn.utils.torch_math import quat_rotate


class ToFDepthSensor:
    def __init__(
        self,
        box_centers: torch.Tensor,   # (N, B, 3) world
        box_half: torch.Tensor,      # (N, B, 3)
        box_mask: torch.Tensor,      # (N, B) bool
        width: int = 64,
        height: int = 64,
        hfov_deg: float = 106.0,
        min_range: float = 0.2,
        max_range: float = 6.0,
        offset_body: tuple[float, float, float] = (-0.018, -0.078, 0.003),
        device: str | torch.device = "cpu",
    ):
        d = torch.device(device)
        self.centers, self.half, self.mask = (
            box_centers.to(d), box_half.to(d), box_mask.to(d)
        )
        self.min_range, self.max_range = float(min_range), float(max_range)
        self.offset = torch.tensor(offset_body, dtype=torch.float32, device=d)
        self.H, self.W = height, width
        self.device = d

        # pinhole ray grid in the body frame
        tan_h = math.tan(math.radians(hfov_deg) / 2.0)
        tan_v = tan_h * height / width
        xs = torch.linspace(-tan_h, tan_h, width, device=d)
        ys = torch.linspace(-tan_v, tan_v, height, device=d)
        ty, tx = torch.meshgrid(ys, xs, indexing="ij")       # (H, W)
        fwd = torch.tensor([0.0, -1.0, 0.0], device=d)
        right = torch.tensor([-1.0, 0.0, 0.0], device=d)
        down = torch.tensor([0.0, 0.0, -1.0], device=d)
        rays = (fwd.view(1, 1, 3) + tx.unsqueeze(-1) * right
                + ty.unsqueeze(-1) * down)
        self.rays_body = (rays / rays.norm(dim=-1, keepdim=True)).reshape(-1, 3)  # (R, 3)

    @torch.no_grad()
    def render(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """pos (N, 3), quat (N, 4) -> depth (N, H, W) float32."""
        N = pos.shape[0]
        R = self.rays_body.shape[0]
        d = self.device

        origin = pos + quat_rotate(quat, self.offset.expand(N, 3))     # (N, 3)
        dirs = quat_rotate(
            quat.unsqueeze(1).expand(N, R, 4).reshape(-1, 4),
            self.rays_body.unsqueeze(0).expand(N, R, 3).reshape(-1, 3),
        ).reshape(N, R, 3)

        o = origin.unsqueeze(1)                                        # (N, 1, 3)
        inv = 1.0 / torch.where(dirs.abs() < 1e-9, torch.full_like(dirs, 1e-9), dirs)

        # slab test against every box: (N, R, B)
        lo = (self.centers - self.half).unsqueeze(1)                   # (N, 1, B, 3)
        hi = (self.centers + self.half).unsqueeze(1)
        t1 = (lo - o.unsqueeze(2)) * inv.unsqueeze(2)
        t2 = (hi - o.unsqueeze(2)) * inv.unsqueeze(2)
        t_near = torch.minimum(t1, t2).amax(dim=-1)
        t_far = torch.maximum(t1, t2).amin(dim=-1)
        hit = (t_far >= t_near) & (t_far > 0.0) & self.mask.unsqueeze(1)
        t_box = torch.where(
            hit, torch.where(t_near > 0.0, t_near, t_far),
            torch.full_like(t_near, float("inf")),
        ).amin(dim=-1)                                                 # (N, R)

        # ground plane z = 0
        dz = dirs[..., 2]
        t_gnd = torch.where(
            dz < -1e-6, -o[..., 2] / dz, torch.full_like(dz, float("inf"))
        )

        t = torch.minimum(t_box, t_gnd)
        depth = t.clamp(self.min_range, self.max_range)
        return depth.reshape(N, self.H, self.W)
