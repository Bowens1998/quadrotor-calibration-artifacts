"""ID / Wind-OOD / Geometry-OOD / Joint-OOD episode splits (spec §11).

Split keys are (scene geometry seed, wind condition, episode replicate):

    train      seen scenes x seen winds, replicate < train_reps
    id_eval    seen scenes x seen winds, replicate >= train_reps (new trajectories)
    wind_ood   seen scenes x unseen wind conditions
    geo_ood    unseen scene seeds (same families) x seen winds
    joint_ood  unseen scenes x unseen winds

Windows are never split within an episode: assignment is per-episode.
"""

from __future__ import annotations


def wind_is_ood(wind_id: str, ood_wind_ids: set[str]) -> bool:
    return wind_id in ood_wind_ids


def split_episodes(
    manifest: list[dict],
    ood_scene_ids: set[str],
    ood_wind_ids: set[str],
    train_reps: int,
    extrap_wind_ids: set[str] | None = None,
) -> dict[str, list[dict]]:
    """`ood_wind_ids`: interpolative Wind-OOD (unseen directions, in-range
    speed). `extrap_wind_ids`: extrapolative Wind-OOD (speed above the training
    range) — kept as separate categories wind_extrap / joint_extrap."""
    extrap_wind_ids = extrap_wind_ids or set()
    out = {k: [] for k in ("train", "id_eval", "wind_ood", "geo_ood",
                           "joint_ood", "wind_extrap", "joint_extrap")}
    for e in manifest:
        s_ood = e["scene_id"] in ood_scene_ids
        w_ext = e["wind_id"] in extrap_wind_ids
        w_ood = e["wind_id"] in ood_wind_ids
        if w_ext:
            out["joint_extrap" if s_ood else "wind_extrap"].append(e)
        elif s_ood and w_ood:
            out["joint_ood"].append(e)
        elif s_ood:
            out["geo_ood"].append(e)
        elif w_ood:
            out["wind_ood"].append(e)
        elif e.get("replicate", 0) < train_reps:
            out["train"].append(e)
        else:
            out["id_eval"].append(e)
    return out
