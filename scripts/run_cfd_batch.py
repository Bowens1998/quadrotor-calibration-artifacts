"""Solve one planar LBM field per (scene, wind condition) (spec §7)."""
import argparse
import json
import time

import numpy as np

from _common import ROOT, device_or_fallback

from winddyn.cfd.field_io import PlanarField, save_field, wind_id
from winddyn.cfd.lbm2d import solve_planar_lbm_batch
from winddyn.geometry.procedural import load_manifest, occupancy_at_zref
from winddyn.utils.config import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cfd/planar_mvp.yaml")
    ap.add_argument("--scenes", default="configs/sim/scenes_mvp.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    scfg = load_yaml(args.scenes)
    device = device_or_fallback(cfg.get("device", "cpu"))
    scenes = load_manifest(ROOT / scfg["manifest"])

    conds = []
    blocks = [(cfg["train_conditions"], "train"), (cfg["ood_conditions"], "ood")]
    if "ood_extrap_conditions" in cfg:
        blocks.append((cfg["ood_extrap_conditions"], "ood_extrap"))
    for block, tag in blocks:
        for d in block["directions_deg"]:
            for s in block["speeds_mps"]:
                conds.append((float(d), float(s), tag))

    res = float(cfg["resolution_m"])
    out_dir = ROOT / cfg["out_dir"]
    entries = []
    t0 = time.time()
    for si, spec in enumerate(scenes):
        Lx, Ly = spec.bounds_xy
        nx, ny = int(round(Lx / res)), int(round(Ly / res))
        occ = occupancy_at_zref(spec, nx, ny)
        origin = np.array([-Lx / 2 + res / 2, -Ly / 2 + res / 2], np.float32)
        todo = []
        for d_deg, s_mps, tag in conds:
            wid = wind_id(d_deg, s_mps)
            path = out_dir / f"{spec.scene_id}__{wid}.npz"
            entries.append({"scene_id": spec.scene_id, "wind_id": wid,
                            "tag": tag, "path": str(path)})
            if not path.exists():
                todo.append((d_deg, s_mps, tag, wid, path))
        if todo:
            kw = {"tau": float(cfg["tau"])} if "tau" in cfg else {}
            results = solve_planar_lbm_batch(
                occ, [(s, dd) for dd, s, _, _, _ in todo],
                n_steps=int(cfg["n_steps"]), avg_window=int(cfg["avg_window"]),
                device=device, **kw,
            )
            for (d_deg, s_mps, tag, wid, path), (u, v, meta) in zip(todo, results):
                meta.update(scene_id=spec.scene_id, wind_id=wid,
                            inlet_speed_mps=s_mps, inlet_direction_deg=d_deg,
                            condition_tag=tag, grid_shape_yx=[ny, nx],
                            cell_size_xy_m=[res, res],
                            reference_altitude_m=spec.z_ref)
                save_field(path, PlanarField(origin,
                                             np.array([res, res], np.float32),
                                             spec.z_ref, u, v, meta))
        print(f"[{si+1}/{len(scenes)}] {spec.scene_id} done "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    man = ROOT / cfg["manifest"]
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w") as f:
        json.dump({"fields": entries}, f, indent=1)
    print(f"{len(entries)} fields -> {man}")


if __name__ == "__main__":
    main()
