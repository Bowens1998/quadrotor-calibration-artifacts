"""Collect the trajectory dataset over scenes x wind conditions (spec §9-§10).

Per (scene, wind condition): `episodes_per_pair` training replicates
(alternating tracking / perturbation modes) + 1 held-out evaluation replicate.
OOD blocks (unseen winds / unseen scenes) get evaluation replicates only.
"""
import argparse
import json
import time

from _common import ROOT, device_or_fallback

from winddyn.cfd.field_io import load_field
from winddyn.data.writer import append_manifest, save_episode
from winddyn.geometry.procedural import load_manifest
from winddyn.sim.rollout import EnvTask, collect_batch
from winddyn.utils.config import load_vehicle, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/rollouts_mvp.yaml")
    ap.add_argument("--scenes-config", default="configs/sim/scenes_mvp.yaml")
    ap.add_argument("--fields-manifest", default="data/manifests/wind_fields.json")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap task count")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    scfg = load_yaml(args.scenes_config)
    device = device_or_fallback(cfg.get("device", "cpu"))
    vp = load_vehicle()

    scenes = {s.scene_id: s for s in load_manifest(ROOT / scfg["manifest"])}
    n_ood = scfg["geo_ood_seeds"]
    ood_scene = {sid for sid, s in scenes.items()
                 if s.seed >= scfg["seeds_per_family"] - n_ood}
    with open(ROOT / args.fields_manifest) as f:
        fields = json.load(f)["fields"]

    reps = int(cfg["episodes_per_pair"])
    # matched-excitation datasets (spec P1-A): force every TRAIN replicate to
    # one behavior mode, keeping count/scene/wind coverage identical to the
    # mixed set; train_only skips the eval/OOD blocks (evaluation always uses
    # the standard mixed-manifest splits).
    force_mode = cfg.get("force_mode")
    train_only = bool(cfg.get("train_only", False))
    jobs = []  # (scene_id, field_path, wind_id, replicate, mode)
    for e in fields:
        s_ood = e["scene_id"] in ood_scene
        w_ood = e["tag"] != "train"
        if s_ood or w_ood:
            if train_only:
                continue
            n_eval = 1 if (s_ood and w_ood) else 2
            for r in range(n_eval):
                jobs.append((e["scene_id"], e["path"], e["wind_id"], reps + r,
                             "tracking" if r % 2 == 0 else "perturbation"))
        else:
            for r in range(reps):
                mode = force_mode or ("tracking" if r % 2 == 0 else "perturbation")
                jobs.append((e["scene_id"], e["path"], e["wind_id"], r, mode))
            if not train_only:
                jobs.append((e["scene_id"], e["path"], e["wind_id"], reps,
                             "tracking"))
                jobs.append((e["scene_id"], e["path"], e["wind_id"], reps + 1,
                             "perturbation"))
    if args.limit:
        jobs = jobs[: args.limit]
    # skip episodes already on disk (incremental collection)
    out_dir_pre = ROOT / cfg["out_dir"]
    jobs = [j for j in jobs
            if not (out_dir_pre / f"{j[0]}__{j[2]}__r{j[3]}_{j[4][:4]}.npz").exists()]
    print(f"{len(jobs)} episodes to collect (existing ones skipped)")

    B = int(cfg["batch_envs"])
    out_dir = ROOT / cfg["out_dir"]
    manifest_path = ROOT / cfg["manifest"]
    field_cache: dict[str, object] = {}
    t0 = time.time()
    done = 0
    for b0 in range(0, len(jobs), B):
        chunk = jobs[b0 : b0 + B]
        tasks = []
        for scene_id, fpath, wid, rep, mode in chunk:
            if fpath not in field_cache:
                if len(field_cache) > 64:
                    field_cache.clear()
                field_cache[fpath] = load_field(fpath)
            tasks.append(EnvTask(scene=scenes[scene_id], field=field_cache[fpath],
                                 wind_id=wid, mode=mode))
        eps = collect_batch(
            tasks, vp, duration_s=float(cfg["duration_s"]),
            seed=int(cfg["base_seed"]) + b0, device=device,
            depth_hw=tuple(cfg["depth_hw"]), patch_size=int(cfg["patch_size"]),
            patch_spacing=float(cfg["patch_spacing"]),
        )
        entries = []
        for (scene_id, fpath, wid, rep, mode), ep in zip(chunk, eps):
            ep["meta"]["replicate"] = rep
            eid = f"{scene_id}__{wid}__r{rep}_{mode[:4]}"
            entries.append(save_episode(out_dir, eid, ep))
        append_manifest(manifest_path, entries)
        done += len(chunk)
        print(f"{done}/{len(jobs)} episodes ({time.time()-t0:.0f}s)", flush=True)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
