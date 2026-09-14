"""Planar D2Q9 lattice-Boltzmann (BGK) solver for geometry-conditioned wind.

This produces the fixed-altitude 2-D wind fields of the MVP (spec §7). It is a
real (if deliberately modest) CFD solve, labeled honestly:

  * D2Q9, single-relaxation-time BGK collision
  * full-way bounce-back on the obstacle footprint at z_ref
  * far-field boundary: equilibrium at (rho=1, u_inf) on all four domain edges
  * run to a quasi-steady state, then time-averaged over the final window
    (wakes may shed at these Reynolds numbers; the dataset field is the
    time-averaged mean flow — "quasi-steady planar wind", recorded in metadata)

The lattice inlet speed scales with the physical speed (U_LAT_PER_MPS), so
different physical speeds solve at different effective Reynolds numbers rather
than being scaled copies of one solution. All conditions for one scene are
solved in a single batched tensor pass.

Do NOT present this as LES-grade urban aerodynamics. It is a moderate-Reynolds
planar mean-flow model whose purpose is a *reproducible, geometry-conditioned,
spatially structured* wind field.
"""

from __future__ import annotations

import numpy as np
import torch

# D2Q9 lattice: (0) rest, (1..4) axis, (5..8) diagonal
E = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=torch.long,
)
W = torch.tensor(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
)
OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]

U_LAT_PER_MPS = 0.08 / 6.0   # lattice speed per physical m/s (0.08 at 6 m/s)
TAU = 0.56                   # nu_lat = (tau - 0.5)/3 = 0.02


def _feq(rho: torch.Tensor, u: torch.Tensor, e: torch.Tensor, w: torch.Tensor):
    """rho (B,H,W), u (B,2,H,W) -> (B,9,H,W)."""
    eu = torch.einsum("qc,bchw->bqhw", e.float(), u)
    uu = (u * u).sum(dim=1, keepdim=True)
    return w.view(1, -1, 1, 1) * rho.unsqueeze(1) * (
        1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu
    )


@torch.no_grad()
def solve_planar_lbm_batch(
    occupancy: np.ndarray,
    conditions: list[tuple[float, float]],   # (speed_mps, direction_deg)
    n_steps: int = 8000,
    avg_window: int = 2000,
    device: str | torch.device = "cpu",
    tau: float = TAU,
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """Solve all wind conditions for one occupancy grid in a single batch.

    Returns per condition: (u, v) physical m/s (ny, nx) and metadata.
    """
    d = torch.device(device)
    e, w = E.to(d), W.to(d)
    solid = torch.from_numpy(np.ascontiguousarray(occupancy)).to(d)
    ny, nx = solid.shape
    B = len(conditions)

    u_inf = torch.zeros(B, 2, ny, nx, device=d)
    for b, (speed, deg) in enumerate(conditions):
        th = np.deg2rad(deg)
        ul = U_LAT_PER_MPS * speed
        u_inf[b, 0] = ul * np.cos(th)
        u_inf[b, 1] = ul * np.sin(th)

    rho = torch.ones(B, ny, nx, device=d)
    u = u_inf.clone()
    u[:, :, solid] = 0.0
    f = _feq(rho, u, e, w)

    edge = torch.zeros(ny, nx, dtype=torch.bool, device=d)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    f_ff = _feq(torch.ones_like(rho), u_inf, e, w)

    shifts = [(int(e[q, 1]), int(e[q, 0])) for q in range(9)]
    opp = OPP
    u_acc = torch.zeros(B, 2, ny, nx, device=d)
    n_acc = 0

    for step in range(n_steps):
        rho = f.sum(dim=1)
        u = torch.einsum("qc,bqhw->bchw", e.float(), f) / rho.clamp_min(1e-9).unsqueeze(1)
        f = f + (_feq(rho, u, e, w) - f) / tau
        f = torch.where(edge.view(1, 1, ny, nx), f_ff, f)
        f = torch.stack(
            [torch.roll(f[:, q], shifts=shifts[q], dims=(1, 2)) for q in range(9)],
            dim=1,
        )
        f_s = f[:, :, solid]
        f[:, :, solid] = f_s[:, opp]
        if step >= n_steps - avg_window:
            rho_s = f.sum(dim=1)
            u_s = torch.einsum("qc,bqhw->bchw", e.float(), f) / rho_s.clamp_min(1e-9).unsqueeze(1)
            u_s[:, :, solid] = 0.0
            u_acc += u_s
            n_acc += 1

    u_mean = (u_acc / max(n_acc, 1)).cpu().numpy()
    nu_lat = (tau - 0.5) / 3.0
    L_lat = float(np.sqrt(max(occupancy.sum(), 1)))
    out = []
    for b, (speed, deg) in enumerate(conditions):
        ul = U_LAT_PER_MPS * speed
        scale = speed / ul
        meta = {
            "solver_or_model": "lbm_d2q9_bgk",
            "field_type": "cfd",
            "tau": float(tau),
            "u_lattice": float(ul),
            "n_steps": int(n_steps),
            "avg_window": int(avg_window),
            "reynolds_lattice": float(ul * L_lat / nu_lat),
            "note": (
                "time-averaged quasi-steady planar mean flow; moderate-Re LBM, "
                "equilibrium far-field boundaries; not LES-grade aerodynamics"
            ),
        }
        out.append((
            (u_mean[b, 0] * scale).astype(np.float32),
            (u_mean[b, 1] * scale).astype(np.float32),
            meta,
        ))
    return out


def solve_planar_lbm(occupancy, inlet_speed_mps, inlet_direction_deg,
                     n_steps=8000, avg_window=2000, device="cpu", tau=TAU):
    """Single-condition wrapper."""
    (u, v, meta), = solve_planar_lbm_batch(
        occupancy, [(inlet_speed_mps, inlet_direction_deg)],
        n_steps=n_steps, avg_window=avg_window, device=device, tau=tau,
    )
    return u, v, meta
