"""Generate the controlled simple scenes + manifest (spec §6.2)."""
import argparse

from _common import ROOT  # noqa: F401  (sys.path side effect)

from winddyn.geometry.procedural import make_scene, save_manifest
from winddyn.utils.config import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim/scenes_mvp.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    specs = []
    for fam in cfg["families"]:
        for seed in range(cfg["seeds_per_family"]):
            specs.append(make_scene(fam, seed, cfg))
    save_manifest(specs, ROOT / cfg["manifest"])
    n_ood = cfg["geo_ood_seeds"]
    ood = [s.scene_id for s in specs if s.seed >= cfg["seeds_per_family"] - n_ood]
    print(f"generated {len(specs)} scenes -> {cfg['manifest']}")
    print(f"geometry-OOD held-out scenes ({len(ood)}): {ood}")


if __name__ == "__main__":
    main()
