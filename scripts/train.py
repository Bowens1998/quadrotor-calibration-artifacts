"""Train one model of the ablation ladder (spec §12, §15)."""
import argparse
import json

from _common import ROOT, device_or_fallback

from winddyn.data.splits import split_episodes
from winddyn.data.writer import load_manifest
from winddyn.train.trainer import train_model
from winddyn.utils.config import load_yaml


def build_splits(scenes_cfg_path="configs/sim/scenes_mvp.yaml",
                 episodes_manifest="data/manifests/episodes.json",
                 fields_manifest="data/manifests/wind_fields.json",
                 train_reps=2):
    scfg = load_yaml(scenes_cfg_path)
    from winddyn.geometry.procedural import load_manifest as load_scenes
    scenes = load_scenes(ROOT / scfg["manifest"])
    n_ood = scfg["geo_ood_seeds"]
    ood_scene = {s.scene_id for s in scenes
                 if s.seed >= scfg["seeds_per_family"] - n_ood}
    with open(ROOT / fields_manifest) as f:
        fields = json.load(f)["fields"]
    ood_wind = {e["wind_id"] for e in fields if e["tag"] == "ood"}
    extrap_wind = {e["wind_id"] for e in fields if e["tag"] == "ood_extrap"}
    manifest = load_manifest(ROOT / episodes_manifest)
    return split_episodes(manifest, ood_scene, ood_wind, train_reps,
                          extrap_wind_ids=extrap_wind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-manifest", default=None,
                    help="substitute TRAIN episodes from this manifest "
                         "(matched-excitation study); eval splits unchanged")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    splits = build_splits()
    if args.train_manifest:
        alt = build_splits(episodes_manifest=args.train_manifest)
        splits = dict(splits)
        splits["train"] = alt["train"]
        print(f"[data] train episodes from {args.train_manifest}: "
              f"{len(splits['train'])}")
    print({k: len(v) for k, v in splits.items()})
    out = ROOT / "outputs/checkpoints" / f"{cfg['name']}_seed{args.seed}"
    res = train_model(cfg, splits, out, device=device_or_fallback(args.device),
                      seed=args.seed)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
