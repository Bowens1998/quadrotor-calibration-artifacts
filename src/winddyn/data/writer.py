"""Episode storage: one compressed npz per episode + a JSON manifest.

Schema v1 (spec §10). Arrays are (T, ...) at the record rate (20 Hz):

    t, position_world, velocity_world, quaternion_world_body,
    angular_velocity_body, action, rotor_thrust_cmd, wind_local_world,
    depth (fp16), wind_patch (fp16, world-axis-aligned), sdf, collision,
    goal_relative

Metadata lives in the manifest entry (and inside the npz as meta_json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def save_episode(out_dir: str | Path, episode_id: str, ep: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(ep["meta"])
    meta["episode_id"] = episode_id
    meta["schema_version"] = SCHEMA_VERSION
    meta["n_steps"] = int(len(ep["t"]))
    meta["collided"] = bool(ep["collision"].any())
    arrays = {k: v for k, v in ep.items() if k != "meta"}
    np.savez_compressed(out / f"{episode_id}.npz",
                        meta_json=np.array(json.dumps(meta)), **arrays)
    meta["path"] = str(out / f"{episode_id}.npz")
    return meta


def append_manifest(manifest_path: str | Path, entries: list[dict]) -> None:
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if p.exists():
        with open(p) as f:
            existing = json.load(f)["episodes"]
    seen = {e["episode_id"] for e in existing}
    for e in entries:
        if e["episode_id"] not in seen:
            existing.append(e)
    with open(p, "w") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "episodes": existing}, f, indent=1)


def load_manifest(manifest_path: str | Path) -> list[dict]:
    with open(manifest_path) as f:
        return json.load(f)["episodes"]
