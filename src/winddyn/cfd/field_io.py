"""Canonical planar wind field storage (spec §7.1).

One `.npz` per (scene, wind condition):

    origin_xy    float32 (2,)   world coords of cell-center (0, 0)
    spacing_xy   float32 (2,)
    reference_altitude_m float32 ()
    u, v         float32 (Ny, Nx)
    meta_json    str  — solver, field_type, inlet condition, scene_id, wind_id

`field_type` must be one of: cfd | cfd_derived | learned_surrogate | synthetic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIELD_TYPES = ("cfd", "cfd_derived", "learned_surrogate", "synthetic")


@dataclass
class PlanarField:
    origin_xy: np.ndarray       # (2,)
    spacing_xy: np.ndarray      # (2,)
    z_ref: float
    u: np.ndarray               # (Ny, Nx)
    v: np.ndarray               # (Ny, Nx)
    meta: dict

    @property
    def shape(self) -> tuple[int, int]:
        return self.u.shape

    def sample_bilinear(self, xy: np.ndarray) -> np.ndarray:
        """(P, 2) world points -> (P, 2) [u, v], border-clamped bilinear."""
        ny, nx = self.u.shape
        g = (xy - self.origin_xy) / self.spacing_xy   # fractional cell coords
        gx = np.clip(g[:, 0], 0.0, nx - 1.0)
        gy = np.clip(g[:, 1], 0.0, ny - 1.0)
        x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int)
        x1 = np.minimum(x0 + 1, nx - 1); y1 = np.minimum(y0 + 1, ny - 1)
        fx = gx - x0; fy = gy - y0
        out = np.empty((len(xy), 2), dtype=np.float32)
        for k, ch in enumerate((self.u, self.v)):
            c00 = ch[y0, x0]; c10 = ch[y0, x1]
            c01 = ch[y1, x0]; c11 = ch[y1, x1]
            out[:, k] = (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy)
                         + c01 * (1 - fx) * fy + c11 * fx * fy)
        return out


def save_field(path: str | Path, field: PlanarField) -> None:
    assert field.meta.get("field_type") in FIELD_TYPES, field.meta.get("field_type")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        origin_xy=field.origin_xy.astype(np.float32),
        spacing_xy=field.spacing_xy.astype(np.float32),
        reference_altitude_m=np.float32(field.z_ref),
        u=field.u.astype(np.float32),
        v=field.v.astype(np.float32),
        meta_json=np.array(json.dumps(field.meta)),
    )


def load_field(path: str | Path) -> PlanarField:
    z = np.load(path, allow_pickle=False)
    return PlanarField(
        origin_xy=z["origin_xy"],
        spacing_xy=z["spacing_xy"],
        z_ref=float(z["reference_altitude_m"]),
        u=z["u"],
        v=z["v"],
        meta=json.loads(str(z["meta_json"])),
    )


def wind_id(direction_deg: float, speed_mps: float) -> str:
    return f"d{int(round(direction_deg)):03d}_s{speed_mps:.1f}"
