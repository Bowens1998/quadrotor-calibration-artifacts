"""Windowed torch dataset over recorded episodes.

Each sample is an (H history, K future) window with egocentric features
(spec §28.4: no global x/y position, no scene ID input):

    state_hist   (H, 12)  [v_body(3), omega_body(3), gravity_body(3), z(1),
                           yaw sin/cos(2)]
    action_hist  (H, 4)   [vx, vy, vz, yaw_rate] commands (world frame)
    action_fut   (K, 4)
    depth_hist   (2, 64, 64)  two most recent depth frames at t
    wind_hist    (H, 2)   world [u, v] at the vehicle (explicit-wind models)
    wind_fut     (K, 2)   oracle future wind along the true trajectory
    wind_patch   (2, P, P) world-aligned local patch at t (privileged teacher)
    target       (K, 12)  future [dpos_yaw(3), v_yaw(3), gravity_body(3),
                           omega_body(3)] — dpos/v in the yaw-aligned frame at t
    target_depth (2, 64, 64) depth frames at t+K (JEPA future-target encoder)

Windows lie entirely inside one episode.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

STATE_DIM = 12
ACTION_DIM = 4
TARGET_DIM = 12


def _quat_rotate_inv_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(T,4) wxyz, (T,3) world -> (T,3) body."""
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    qv = np.concatenate([-x, -y, -z], axis=1)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def nose_yaw_np(q: np.ndarray) -> np.ndarray:
    """(T,4) -> (T,) heading of the body nose axis (0,-1,0)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # world nose = R(q) @ (0,-1,0)
    nx = -2.0 * (x * y - w * z)
    ny = -(1.0 - 2.0 * (x * x + z * z))
    return np.arctan2(ny, nx)


def state_features(ep: dict) -> np.ndarray:
    """(T, 12) egocentric state features from raw episode arrays."""
    q = ep["quaternion_world_body"].astype(np.float32)
    v_body = _quat_rotate_inv_np(q, ep["velocity_world"].astype(np.float32))
    g_body = _quat_rotate_inv_np(
        q, np.repeat(np.array([[0.0, 0.0, -1.0]], np.float32), len(q), axis=0)
    )
    yaw = nose_yaw_np(q)
    return np.concatenate(
        [
            v_body,
            ep["angular_velocity_body"].astype(np.float32),
            g_body,
            ep["position_world"][:, 2:3].astype(np.float32),
            np.sin(yaw)[:, None].astype(np.float32),
            np.cos(yaw)[:, None].astype(np.float32),
        ],
        axis=1,
    )


def targets_from(ep: dict, t0: int, K: int) -> np.ndarray:
    """(K, 12) future targets in the yaw-aligned frame at t0."""
    pos = ep["position_world"].astype(np.float32)
    vel = ep["velocity_world"].astype(np.float32)
    q = ep["quaternion_world_body"].astype(np.float32)
    yaw0 = nose_yaw_np(q[t0 : t0 + 1])[0]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    R = np.array([[c, -s], [s, c]], np.float32)

    fut = slice(t0 + 1, t0 + 1 + K)
    dpos = pos[fut] - pos[t0]
    dpos_yaw = np.concatenate([dpos[:, :2] @ R.T, dpos[:, 2:3]], axis=1)
    v_yaw = np.concatenate([vel[fut][:, :2] @ R.T, vel[fut][:, 2:3]], axis=1)
    g_body = _quat_rotate_inv_np(
        q[fut], np.repeat(np.array([[0.0, 0.0, -1.0]], np.float32), K, axis=0)
    )
    omega = ep["angular_velocity_body"][fut].astype(np.float32)
    return np.concatenate([dpos_yaw, v_yaw, g_body, omega], axis=1)


def _vel_yaw_at(ep: dict, t0: int) -> np.ndarray:
    q = ep["quaternion_world_body"][t0 : t0 + 1].astype(np.float32)
    yaw0 = nose_yaw_np(q)[0]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    v = ep["velocity_world"][t0].astype(np.float32)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], np.float32)


class WindowDataset(Dataset):
    def __init__(
        self,
        episodes: list[dict],           # manifest entries with "path"
        H: int = 12,
        K: int = 30,
        stride: int = 5,
        with_depth: bool = True,
        with_patch: bool = True,
    ):
        self.H, self.K = H, K
        self.with_depth = with_depth
        self.with_patch = with_patch
        self._patch_always = with_patch
        self.entries = episodes
        self._cache: dict[str, dict] = {}
        self.index: list[tuple[int, int]] = []
        for ei, e in enumerate(episodes):
            T = e["n_steps"]
            col = None
            if e.get("collided"):
                z = np.load(e["path"])
                col = z["collision"]
            for t0 in range(H - 1, T - K - 1, stride):
                # exclude windows touching a collision step (analytic sim has
                # no contacts; penetration data would be unphysical)
                if col is not None and col[t0 - H + 1 : t0 + K + 1].any():
                    continue
                self.index.append((ei, t0))

    def _ep(self, ei: int) -> dict:
        path = self.entries[ei]["path"]
        if path not in self._cache:
            z = np.load(path, allow_pickle=False)
            skip = {"meta_json"}
            if not self.with_depth:
                skip.add("depth")
            if not self.with_patch and not self._patch_always:
                skip.add("wind_patch")
            ep = {k: z[k] for k in z.files if k not in skip}
            ep["_state"] = state_features(ep)
            if len(self._cache) > 1200:
                self._cache.clear()
            self._cache[path] = ep
        return self._cache[path]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ei, t0 = self.index[i]
        ep = self._ep(ei)
        H, K = self.H, self.K
        hist = slice(t0 - H + 1, t0 + 1)
        fut = slice(t0 + 1, t0 + 1 + K)
        out = {
            "state_hist": torch.from_numpy(ep["_state"][hist].copy()),
            "action_hist": torch.from_numpy(ep["action"][hist].copy()),
            "action_fut": torch.from_numpy(ep["action"][fut].copy()),
            "wind_hist": torch.from_numpy(ep["wind_local_world"][hist, :2].copy()),
            "wind_fut": torch.from_numpy(ep["wind_local_world"][fut, :2].copy()),
            "target": torch.from_numpy(targets_from(ep, t0, K)),
            "sdf_t": torch.tensor(float(ep["sdf"][t0])),
            # velocity at t0 in the yaw-aligned frame (M0 persistence baseline)
            "vel_yaw_t": torch.from_numpy(_vel_yaw_at(ep, t0)),
        }
        if self.with_depth:
            d0 = ep["depth"][max(t0 - 3, 0)].astype(np.float32)
            d1 = ep["depth"][t0].astype(np.float32)
            out["depth_hist"] = torch.from_numpy(np.stack([d0, d1]))
            tK = t0 + K
            dk0 = ep["depth"][max(tK - 3, 0)].astype(np.float32)
            dk1 = ep["depth"][tK].astype(np.float32)
            out["target_depth"] = torch.from_numpy(np.stack([dk0, dk1]))
            out["state_tgt"] = torch.from_numpy(ep["_state"][tK : tK + 1].copy()).squeeze(0)
        if self.with_patch:
            out["wind_patch"] = torch.from_numpy(
                ep["wind_patch"][t0].astype(np.float32)
            )
        return out


def build_window(ep: dict, t0: int, H: int, K: int,
                 with_depth=True, with_patch=True) -> dict[str, torch.Tensor]:
    """Build one sample dict directly from in-memory episode arrays
    (used by the counterfactual evaluation on replayed episodes)."""
    if "_state" not in ep:
        ep["_state"] = state_features(ep)
    hist = slice(t0 - H + 1, t0 + 1)
    fut = slice(t0 + 1, t0 + 1 + K)
    out = {
        "state_hist": torch.from_numpy(ep["_state"][hist].copy()),
        "action_hist": torch.from_numpy(ep["action"][hist].copy()),
        "action_fut": torch.from_numpy(ep["action"][fut].copy()),
        "wind_hist": torch.from_numpy(ep["wind_local_world"][hist, :2].copy()),
        "wind_fut": torch.from_numpy(ep["wind_local_world"][fut, :2].copy()),
        "target": torch.from_numpy(targets_from(ep, t0, K)),
        "vel_yaw_t": torch.from_numpy(_vel_yaw_at(ep, t0)),
    }
    if with_depth:
        d0 = ep["depth"][max(t0 - 3, 0)].astype(np.float32)
        d1 = ep["depth"][t0].astype(np.float32)
        out["depth_hist"] = torch.from_numpy(np.stack([d0, d1]))
    if with_patch:
        out["wind_patch"] = torch.from_numpy(ep["wind_patch"][t0].astype(np.float32))
    return out


def compute_norm_stats(ds: WindowDataset, n: int = 2000, seed: int = 0) -> dict:
    """Mean/std for state, action, target, wind over a subsample of windows."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    acc: dict[str, list[np.ndarray]] = {}
    for i in idx:
        s = ds[int(i)]
        for k in ("state_hist", "action_hist", "target", "wind_hist"):
            acc.setdefault(k, []).append(s[k].numpy().reshape(-1, s[k].shape[-1]))
    stats = {}
    for k, v in acc.items():
        arr = np.concatenate(v)
        stats[k] = {
            "mean": arr.mean(0).astype(np.float32),
            "std": (arr.std(0) + 1e-5).astype(np.float32),
        }
    return stats
