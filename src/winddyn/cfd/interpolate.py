"""Wind sources for the simulator: per-env planar-grid lookup (spec §8).

`WindSource` is the interface `TorchQuadSim` consumes:

    reset(env_ids, generator) / step(dt) / sample(pos_world, t) -> (N, 3)

`PlanarGridWind` holds a stack of planar fields (one per env, assignable) and
bilinearly interpolates [u, v] at the drone (x, y); w = 0 for the MVP.
It also extracts the privileged local wind patch for the teacher branch:
world-axis-aligned P×P patch centered on the drone (convention recorded in the
dataset metadata; spec §28.7).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from winddyn.cfd.field_io import PlanarField


class WindSource:
    """Base interface (mirrors WindJEPA's WindField)."""

    name = "base"

    def reset(self, env_ids, generator) -> None:  # pragma: no cover
        pass

    def step(self, dt: float) -> None:  # pragma: no cover
        pass

    def sample(self, positions: torch.Tensor, sim_time: float,
               env_ids: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def describe(self, env_id: int) -> dict[str, Any]:
        raise NotImplementedError


class ZeroWind(WindSource):
    name = "none"

    def sample(self, positions, sim_time, env_ids=None):
        return torch.zeros_like(positions)

    def describe(self, env_id):
        return {"wind_type": "none"}


class UniformWind(WindSource):
    """Fixed uniform horizontal wind, same for all envs (control condition)."""

    name = "uniform"

    def __init__(self, u: float, v: float, device="cpu"):
        self.vec = torch.tensor([u, v, 0.0], device=torch.device(device))

    def sample(self, positions, sim_time, env_ids=None):
        return self.vec.expand(positions.shape[0], 3)

    def describe(self, env_id):
        return {"wind_type": "uniform", "base_velocity": self.vec.tolist()}


class PlanarGridWind(WindSource):
    """Per-env planar field stack with bilinear lookup.

    fields: list of PlanarField (all same grid shape/origin/spacing).
    env_field_idx: (N,) long — which field each env uses.
    """

    name = "planar_grid"

    def __init__(self, fields: list[PlanarField], num_envs: int, device="cpu"):
        assert len(fields) > 0
        self.fields_meta = [f.meta for f in fields]
        d = torch.device(device)
        self.device = d
        shp = fields[0].shape
        for f in fields:
            assert f.shape == shp, "all fields in a stack must share the grid"
        uv = np.stack([np.stack([f.u, f.v]) for f in fields])  # (F, 2, Ny, Nx)
        self.grid = torch.tensor(uv, dtype=torch.float32, device=d)
        self.origin = torch.tensor(fields[0].origin_xy, dtype=torch.float32, device=d)
        self.spacing = torch.tensor(fields[0].spacing_xy, dtype=torch.float32, device=d)
        self.ny, self.nx = shp
        self.num_envs = num_envs
        self.env_field_idx = torch.zeros(num_envs, dtype=torch.long, device=d)

    def assign(self, env_field_idx: torch.Tensor) -> None:
        self.env_field_idx = env_field_idx.to(self.device)

    def _bilinear(self, field_idx: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """field_idx (P,), xy (P, 2) world -> (P, 2) [u, v]."""
        g = (xy - self.origin) / self.spacing
        gx = g[:, 0].clamp(0.0, self.nx - 1.0)
        gy = g[:, 1].clamp(0.0, self.ny - 1.0)
        x0 = gx.floor().long(); y0 = gy.floor().long()
        x1 = (x0 + 1).clamp(max=self.nx - 1); y1 = (y0 + 1).clamp(max=self.ny - 1)
        fx = (gx - x0.float()).unsqueeze(-1)
        fy = (gy - y0.float()).unsqueeze(-1)
        G = self.grid[field_idx]                       # (P, 2, Ny, Nx)
        idx = torch.arange(len(xy), device=self.device)
        c00 = G[idx, :, y0, x0]; c10 = G[idx, :, y0, x1]
        c01 = G[idx, :, y1, x0]; c11 = G[idx, :, y1, x1]
        return (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy)
                + c01 * (1 - fx) * fy + c11 * fx * fy)

    def sample(self, positions, sim_time, env_ids=None):
        idx = self.env_field_idx if env_ids is None else self.env_field_idx[env_ids]
        uv = self._bilinear(idx, positions[:, :2])
        return torch.cat([uv, torch.zeros_like(uv[:, :1])], dim=-1)

    def patch(self, positions: torch.Tensor, P: int, patch_spacing: float) -> torch.Tensor:
        """(N, 3) drone pos -> (N, 2, P, P) world-aligned local wind patch."""
        N = positions.shape[0]
        half = (P - 1) / 2.0
        offs = (torch.arange(P, device=self.device).float() - half) * patch_spacing
        ox, oy = torch.meshgrid(offs, offs, indexing="xy")     # (P, P)
        pts = positions[:, None, None, :2] + torch.stack([ox, oy], dim=-1)  # (N,P,P,2)
        flat = pts.reshape(-1, 2)
        fidx = self.env_field_idx.repeat_interleave(P * P)
        uv = self._bilinear(fidx, flat)                        # (N*P*P, 2)
        return uv.reshape(N, P, P, 2).permute(0, 3, 1, 2)      # (N, 2, P, P)

    def describe(self, env_id):
        m = dict(self.fields_meta[int(self.env_field_idx[env_id])])
        m["wind_type"] = self.name
        return m
