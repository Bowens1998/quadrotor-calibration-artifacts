"""Procedural simple-obstacle scenes for the fixed-altitude MVP (spec §6.2).

Five families, each parameterized by a seed:

    single_pillar, two_pillars, narrow_corridor, offset_pillars, l_shape

A scene is a set of axis-aligned 3-D boxes on a square ground plane. The wind
field is solved on the horizontal cross-section at ``z_ref``; obstacle heights
comfortably exceed ``z_ref`` so the planar cut is meaningful. All scenes share
one bounding domain so wind grids, depth rendering, and trajectory sampling
use identical extents.

Coordinates: world frame, scene centered on (0, 0). Units: meters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

SCENE_FAMILIES = (
    "single_pillar",
    "two_pillars",
    "narrow_corridor",
    "offset_pillars",
    "l_shape",
)

MAX_BOXES = 8  # fixed tensor budget; l_shape uses 2, corridor 2, offset 3


@dataclass
class SceneSpec:
    scene_id: str
    family: str
    seed: int
    bounds_xy: tuple[float, float]      # full domain size, centered on origin
    z_ref: float
    box_centers: np.ndarray             # (MAX_BOXES, 3)
    box_sizes: np.ndarray               # (MAX_BOXES, 3) full extents; 0 = unused
    n_boxes: int
    extras: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["box_centers"] = self.box_centers.tolist()
        d["box_sizes"] = self.box_sizes.tolist()
        return d

    @staticmethod
    def from_json(d: dict) -> "SceneSpec":
        d = dict(d)
        d["box_centers"] = np.asarray(d["box_centers"], dtype=np.float64)
        d["box_sizes"] = np.asarray(d["box_sizes"], dtype=np.float64)
        d["bounds_xy"] = tuple(d["bounds_xy"])
        return SceneSpec(**d)


def _pad(centers: list, sizes: list) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(centers)
    assert n <= MAX_BOXES, f"{n} boxes > MAX_BOXES={MAX_BOXES}"
    c = np.zeros((MAX_BOXES, 3))
    s = np.zeros((MAX_BOXES, 3))
    c[:n] = np.asarray(centers)
    s[:n] = np.asarray(sizes)
    return c, s, n


def _box(cx, cy, sx, sy, h):
    """Ground-mounted box: full extents (sx, sy, h), center z = h/2."""
    return [cx, cy, h / 2.0], [sx, sy, h]


# ------------------------------------------------------------------ #
# family generators: every random draw goes through rng
# ------------------------------------------------------------------ #
def single_pillar(rng, cfg, z_ref):
    w = rng.uniform(1.5, 3.5)
    d = rng.uniform(1.5, 3.5)
    cx = rng.uniform(-2.0, 2.0)
    cy = rng.uniform(-2.0, 2.0)
    h = cfg.get("obstacle_height", 8.0)
    c, s = _box(cx, cy, w, d, h)
    return _pad([c], [s]), {"pillar_w": w, "pillar_d": d}


def two_pillars(rng, cfg, z_ref):
    sep = rng.uniform(3.0, 7.0)
    w = rng.uniform(1.5, 3.0)
    d = rng.uniform(2.0, 4.0)
    ang = rng.uniform(0.0, np.pi)  # axis orientation of the pair
    h = cfg.get("obstacle_height", 8.0)
    dx, dy = 0.5 * sep * np.cos(ang), 0.5 * sep * np.sin(ang)
    c1, s1 = _box(-dx, -dy, w, d, h)
    c2, s2 = _box(dx, dy, w, d, h)
    return _pad([c1, c2], [s1, s2]), {"separation": sep, "pair_angle": ang}


def narrow_corridor(rng, cfg, z_ref):
    width = rng.uniform(2.5, 5.0)          # free gap between the walls
    length = rng.uniform(10.0, 16.0)
    thick = rng.uniform(1.0, 2.0)
    ax = rng.integers(0, 2)                # 0: corridor along x, 1: along y
    h = cfg.get("obstacle_height", 8.0)
    off = width / 2 + thick / 2
    if ax == 0:
        c1, s1 = _box(0.0, off, length, thick, h)
        c2, s2 = _box(0.0, -off, length, thick, h)
    else:
        c1, s1 = _box(off, 0.0, thick, length, h)
        c2, s2 = _box(-off, 0.0, thick, length, h)
    return _pad([c1, c2], [s1, s2]), {"gap": width, "length": length, "axis": int(ax)}


def offset_pillars(rng, cfg, z_ref):
    n = 3
    h = cfg.get("obstacle_height", 8.0)
    centers, sizes = [], []
    xs = np.linspace(-5.0, 5.0, n) + rng.uniform(-1.0, 1.0, n)
    ys = rng.uniform(-4.0, 4.0, n)
    for x, y in zip(xs, ys):
        w = rng.uniform(1.5, 3.0)
        d = rng.uniform(1.5, 3.0)
        c, s = _box(x, y, w, d, h)
        centers.append(c)
        sizes.append(s)
    return _pad(centers, sizes), {"n_pillars": n}


def l_shape(rng, cfg, z_ref):
    arm1 = rng.uniform(6.0, 10.0)
    arm2 = rng.uniform(6.0, 10.0)
    thick = rng.uniform(1.5, 2.5)
    h = cfg.get("obstacle_height", 8.0)
    rot = rng.integers(0, 4)  # which quadrant the L opens toward
    # L with corner near origin: arm A along +x, arm B along +y
    ca, sa = _box(arm1 / 2 - thick / 2, 0.0, arm1, thick, h)
    cb, sb = _box(0.0, arm2 / 2 - thick / 2, thick, arm2, h)
    # rotate by rot * 90 degrees about origin
    R = {0: (1, 0, 0, 1), 1: (0, -1, 1, 0), 2: (-1, 0, 0, -1), 3: (0, 1, -1, 0)}[int(rot)]
    def rc(c, s):
        x, y = c[0], c[1]
        cx, cy = R[0] * x + R[1] * y, R[2] * x + R[3] * y
        sx, sy = (s[0], s[1]) if rot % 2 == 0 else (s[1], s[0])
        return [cx, cy, c[2]], [sx, sy, s[2]]
    ca, sa = rc(ca, sa)
    cb, sb = rc(cb, sb)
    return _pad([ca, cb], [sa, sb]), {"arm1": arm1, "arm2": arm2, "rot90": int(rot)}


GENERATORS = {
    "single_pillar": single_pillar,
    "two_pillars": two_pillars,
    "narrow_corridor": narrow_corridor,
    "offset_pillars": offset_pillars,
    "l_shape": l_shape,
}


def make_scene(family: str, seed: int, cfg: dict | None = None) -> SceneSpec:
    cfg = cfg or {}
    rng = np.random.default_rng(seed)
    (c, s, n), extras = GENERATORS[family](rng, cfg, cfg.get("z_ref", 3.0))
    return SceneSpec(
        scene_id=f"{family}_{seed:03d}",
        family=family,
        seed=seed,
        bounds_xy=tuple(cfg.get("bounds_xy", (30.0, 30.0))),
        z_ref=float(cfg.get("z_ref", 3.0)),
        box_centers=c,
        box_sizes=s,
        n_boxes=n,
        extras=extras,
    )


def occupancy_at_zref(spec: SceneSpec, nx: int, ny: int) -> np.ndarray:
    """(ny, nx) bool occupancy of the z_ref cross-section, row y, col x."""
    Lx, Ly = spec.bounds_xy
    xs = (np.arange(nx) + 0.5) / nx * Lx - Lx / 2
    ys = (np.arange(ny) + 0.5) / ny * Ly - Ly / 2
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)
    occ = np.zeros((ny, nx), dtype=bool)
    for i in range(spec.n_boxes):
        cx, cy, cz = spec.box_centers[i]
        sx, sy, sz = spec.box_sizes[i]
        if cz - sz / 2 <= spec.z_ref <= cz + sz / 2:
            occ |= (np.abs(X - cx) <= sx / 2) & (np.abs(Y - cy) <= sy / 2)
    return occ


def free_space_sdf_xy(spec: SceneSpec, pts_xy: np.ndarray) -> np.ndarray:
    """(P, 2) points -> (P,) planar signed distance to nearest box footprint."""
    d_min = np.full(len(pts_xy), np.inf)
    for i in range(spec.n_boxes):
        c = spec.box_centers[i, :2]
        h = spec.box_sizes[i, :2] / 2
        q = np.abs(pts_xy - c) - h
        outside = np.linalg.norm(np.clip(q, 0, None), axis=-1)
        inside = np.clip(q.max(axis=-1), None, 0.0)
        d_min = np.minimum(d_min, outside + inside)
    return d_min


def save_manifest(specs: list[SceneSpec], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"version": 1, "scenes": [s.to_json() for s in specs]}, f, indent=1)


def load_manifest(path: str | Path) -> list[SceneSpec]:
    with open(path) as f:
        d = json.load(f)
    return [SceneSpec.from_json(s) for s in d["scenes"]]
