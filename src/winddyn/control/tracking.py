"""Scripted fixed-altitude waypoint tracking + perturbation data generator.

Produces the two behavior distributions of spec §9:

  * tracking:     waypoint spline following with altitude hold at z_ref
  * perturbation: same, plus temporally-correlated (OU) velocity-command
                  offsets and weakened integral wind rejection, so the causal
                  effect of wind is visible in the data instead of being
                  cancelled by the controller

Waypoints are sampled in free space (planar SDF clearance), with a bias toward
near-obstacle passages so wakes and corner regions are traversed.
"""

from __future__ import annotations

import numpy as np
import torch

from winddyn.geometry.procedural import SceneSpec, free_space_sdf_xy


def _segment_clear(spec: SceneSpec, a: np.ndarray, b: np.ndarray,
                   clearance: float) -> bool:
    """Line-of-sight check: min planar SDF along segment a->b > clearance."""
    n = max(int(np.linalg.norm(b - a) / 0.25), 2)
    pts = a[None] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None]
    return bool(free_space_sdf_xy(spec, pts).min() > clearance)


def sample_waypoints(
    spec: SceneSpec,
    rng: np.random.Generator,
    n_wp: int = 6,
    clearance: float = 1.2,
    near_frac: float = 0.5,
    margin: float = 3.0,
    segment_clearance: float = 0.6,
) -> np.ndarray:
    """(n_wp, 2) planar waypoints; ~near_frac of them hug obstacles
    (SDF 1.2-3.5 m). Consecutive waypoints keep line-of-sight clearance so the
    straight-line tracker does not fly through obstacles (the torch backend has
    no contacts; collisions would otherwise silently corrupt the data)."""
    Lx, Ly = spec.bounds_xy
    lo = np.array([-Lx / 2 + margin, -Ly / 2 + margin])
    hi = np.array([Lx / 2 - margin, Ly / 2 - margin])
    wps: list[np.ndarray] = []
    tries = 0
    while len(wps) < n_wp:
        cand = rng.uniform(lo, hi, size=(128, 2))
        sdf = free_space_sdf_xy(spec, cand)
        want_near = rng.random() < near_frac and spec.n_boxes > 0 and tries < 40
        ok = (sdf > clearance) & (sdf < 3.5) if want_near else (sdf > clearance)
        good = cand[ok]
        if not len(good):
            tries += 1
            continue
        # prefer candidates with line-of-sight from the previous waypoint
        rng.shuffle(good)
        placed = False
        for c in good[:32]:
            if not wps or _segment_clear(spec, wps[-1], c, segment_clearance):
                wps.append(c)
                placed = True
                break
        tries = tries + 1 if not placed else 0
        if tries > 200:  # degenerate scene: accept nearest-clearance candidate
            wps.append(good[0])
            tries = 0
    return np.asarray(wps)


class WaypointTracker:
    """Batched waypoint -> [vx, vy, vz, yaw_rate] command generator."""

    def __init__(
        self,
        waypoints: torch.Tensor,       # (N, n_wp, 2)
        z_ref: float,
        device,
        max_speed: float = 2.0,
        kp_pos: float = 0.8,
        kz: float = 1.2,
        k_yaw: float = 1.5,
        reach_radius: float = 0.8,
    ):
        self.wp = waypoints.to(device)
        self.z_ref = float(z_ref)
        self.idx = torch.zeros(waypoints.shape[0], dtype=torch.long, device=device)
        self.max_speed = max_speed
        self.kp = kp_pos
        self.kz = kz
        self.k_yaw = k_yaw
        self.reach = reach_radius
        self.n_wp = waypoints.shape[1]
        self.device = device

    def command(self, pos: torch.Tensor, nose_yaw: torch.Tensor) -> torch.Tensor:
        """pos (N, 3), nose_yaw (N,) -> command (N, 4)."""
        N = pos.shape[0]
        tgt = self.wp[torch.arange(N, device=self.device), self.idx]     # (N, 2)
        d = tgt - pos[:, :2]
        dist = d.norm(dim=-1)
        # advance waypoint index (loop) when reached
        reached = dist < self.reach
        self.idx = torch.where(reached, (self.idx + 1) % self.n_wp, self.idx)

        v_xy = self.kp * d
        speed = v_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_xy = v_xy * (speed.clamp(max=self.max_speed) / speed)
        vz = self.kz * (self.z_ref - pos[:, 2])

        des_yaw = torch.atan2(d[:, 1], d[:, 0])
        yaw_err = torch.remainder(des_yaw - nose_yaw + torch.pi, 2 * torch.pi) - torch.pi
        yaw_rate = (self.k_yaw * yaw_err).clamp(-1.5, 1.5)
        return torch.cat([v_xy, vz.unsqueeze(-1), yaw_rate.unsqueeze(-1)], dim=-1)


class OUPerturbation:
    """Temporally-correlated additive velocity-command noise (spec §9.3)."""

    def __init__(self, num_envs, device, sigma=0.8, tau=1.5,
                 generator: torch.Generator | None = None):
        self.x = torch.zeros(num_envs, 3, device=device)
        self.sigma, self.tau = sigma, tau
        self.gen = generator
        self.device = device

    def step(self, dt: float) -> torch.Tensor:
        n = torch.randn(self.x.shape, generator=self.gen, device=self.device)
        self.x = self.x + (-self.x / self.tau) * dt \
            + self.sigma * np.sqrt(2.0 * dt / self.tau) * n
        return self.x
