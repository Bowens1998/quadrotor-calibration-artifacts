"""Prepare held-out geometries and unchanged planar CFD for model selection.

No predictor is imported or evaluated here. Validation/test preparation requires
an existing protocol lock; pilot geometry never enters either formal split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from _common import ROOT
from winddyn.cfd.field_io import PlanarField, load_field, save_field, wind_id
from winddyn.cfd.lbm2d import TAU, solve_planar_lbm_batch
from winddyn.geometry.procedural import (
    SCENE_FAMILIES, SceneSpec, load_manifest, make_scene, occupancy_at_zref,
)
from winddyn.utils.config import load_yaml


BASE = ROOT / "runs/icra_selection_validation_20260912"
SPLITS = ("pilot", "validation", "test")
SPEEDS = (3.0, 6.0, 8.0)


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def split_plan(split: str) -> list[dict]:
    """Deterministic metadata only: no geometry generation or CFD execution."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    seeds = {"pilot": (900,), "validation": (1000, 1001, 1002),
             "test": (2000, 2001, 2002, 2003)}[split]
    rows = []
    for fi, family in enumerate(SCENE_FAMILIES):
        for ii, seed in enumerate(seeds):
            direction = 0 if split == "pilot" else 90 * ((fi + ii) % 4)
            rows.append(dict(split=split, family=family, seed=seed,
                             scene_id=f"{family}_{seed:03d}", instance_index=ii,
                             panel=ii if split == "validation" else None,
                             direction_deg=direction, speeds_mps=list(SPEEDS)))
    return rows


def geometry_hash(spec: SceneSpec) -> str:
    """Hash actual boxes/domain; ignore seed, identifier, padding and box order."""
    boxes = [list(map(float, np.r_[c, s])) for c, s in
             zip(spec.box_centers[:spec.n_boxes], spec.box_sizes[:spec.n_boxes])]
    value = dict(bounds_xy=list(map(float, spec.bounds_xy)), z_ref=float(spec.z_ref),
                 boxes=sorted(boxes))
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def occupancy_hash(occupancy: np.ndarray) -> str:
    occupancy = np.asarray(occupancy, dtype=bool)
    return hashlib.sha256(canonical_bytes(list(occupancy.shape)) +
                          np.packbits(occupancy.ravel()).tobytes()).hexdigest()


def scene_record(spec: SceneSpec, resolution: float) -> dict:
    Lx, Ly = spec.bounds_xy
    nx, ny = int(round(Lx / resolution)), int(round(Ly / resolution))
    if not np.allclose([nx * resolution, ny * resolution], [Lx, Ly]):
        raise ValueError("Grid does not exactly tile the scene domain")
    boxes = np.r_[spec.box_centers[:spec.n_boxes].ravel(),
                  spec.box_sizes[:spec.n_boxes].ravel()]
    if not np.isfinite(boxes).all() or not (spec.box_sizes[:spec.n_boxes] > 0).all():
        raise ValueError(f"Invalid box geometry: {spec.scene_id}")
    occ = occupancy_at_zref(spec, nx, ny)
    if not occ.any() or occ.all():
        raise ValueError(f"Empty or fully blocked scene: {spec.scene_id}")
    return dict(scene_id=spec.scene_id, family=spec.family, seed=int(spec.seed),
                geometry_sha256=geometry_hash(spec), occupancy_sha256=occupancy_hash(occ),
                grid_shape_yx=[ny, nx], solid_cells=int(occ.sum()))


def assert_disjoint(records: list[dict], previous: list[dict]) -> dict:
    """Reject name, true geometry, or rasterized-occupancy reuse across splits."""
    indices = {key: {} for key in ("scene_id", "geometry_sha256", "occupancy_sha256")}
    for record in previous:
        for key, index in indices.items():
            index.setdefault(record[key], record["scene_id"])
    for record in records:
        for key, index in indices.items():
            if record[key] in index:
                raise ValueError(f"Duplicate {key}: {record['scene_id']} matches "
                                 f"{index[record[key]]}")
            index[record[key]] = record["scene_id"]
    return dict(new_scenes=len(records), compared_previous_scenes=len(previous),
                id_disjoint=True, geometry_disjoint=True, occupancy_disjoint=True)


def validate_solver_config(cfg: dict) -> dict:
    actual = dict(resolution_m=float(cfg["resolution_m"]), n_steps=int(cfg["n_steps"]),
                  avg_window=int(cfg["avg_window"]), tau=float(cfg.get("tau", TAU)))
    expected = dict(resolution_m=0.25, n_steps=8000, avg_window=2000, tau=0.56)
    if actual != expected:
        raise ValueError(f"CFD settings differ from frozen baseline: {actual}")
    return actual


def resolve_device(requested: str, allow_cpu_fallback: bool) -> tuple[str, dict]:
    torch.device(requested)  # Validate spelling before creating any artifacts.
    actual = requested
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu_fallback:
            raise RuntimeError("CUDA unavailable in this process; no implicit CPU fallback. "
                               "Check GPU access or explicitly pass --allow-cpu-fallback.")
        actual = "cpu"
    return actual, dict(requested_device=requested, actual_device=actual,
                        cpu_fallback_used=actual != requested)


def field_qc(u: np.ndarray, v: np.ndarray, occupancy: np.ndarray, speed: float) -> dict:
    """Use the old field validator's limits; failure never changes CFD settings."""
    shape_ok = u.shape == v.shape == occupancy.shape
    if not shape_ok or not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise ValueError("CFD field has wrong shape or non-finite values")
    magnitude = np.hypot(u, v)
    fluid = magnitude[~occupancy]
    solid_max = float(magnitude[occupancy].max(initial=0.0))
    median, std = float(np.median(fluid)), float(fluid.std())
    checks = dict(finite=True, solid_zero=solid_max < 1e-3,
                  far_field_sane=0.5 * speed < median < 1.6 * speed,
                  spatially_structured=std > 0.03 * speed)
    if not all(checks.values()):
        raise ValueError(f"CFD quality failure: {checks}; median={median}, std={std}")
    return dict(**checks, solid_max_mps=solid_max, fluid_median_mps=median,
                fluid_std_mps=std, max_speed_mps=float(magnitude.max()))


def protocol_lock_record(split: str, path: Path | None) -> dict | None:
    if path is None:
        if split != "pilot":
            raise ValueError("Validation/test environment requires --protocol-lock")
        return None
    if not path.is_file() or not path.read_bytes().strip():
        raise ValueError("Protocol lock must be an existing nonempty file")
    lock = json.loads(path.read_text())
    protocol = ROOT / lock["protocol_path"]
    if sha256_file(protocol) != lock["protocol_sha256"]:
        raise ValueError("Locked protocol hash mismatch")
    sources = lock["source_hashes"]
    script_key = str(Path(__file__).resolve().relative_to(ROOT))
    if script_key not in sources:
        raise ValueError("Environment generator is absent from protocol source lock")
    for source, expected in sources.items():
        if sha256_file(ROOT / source) != expected:
            raise ValueError(f"Locked source hash mismatch: {source}")
    return dict(path=portable_path(path), sha256=sha256_file(path),
                protocol_sha256=lock["protocol_sha256"])


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare(args: argparse.Namespace) -> dict:
    lock = protocol_lock_record(args.split, args.protocol_lock)
    cfg, scfg = load_yaml(args.config), load_yaml(args.scenes_config)
    solver = validate_solver_config(cfg)
    if tuple(scfg["families"]) != SCENE_FAMILIES:
        raise ValueError("Scene family list differs from frozen baseline")
    if (scfg["bounds_xy"] != [30.0, 30.0] or float(scfg["z_ref"]) != 3.0 or
            float(scfg["obstacle_height"]) != 8.0):
        raise ValueError("Scene domain/altitude/height differs from frozen baseline")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive (conditions per scene)")
    device, execution = resolve_device(args.device, args.allow_cpu_fallback)
    destination = args.base.resolve() / "environment" / args.split
    if destination.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("New environment must not be written to original data directory")
    scene_path = destination / "scenes.json"
    field_manifest = destination / "wind_fields.json"
    audit_path = destination / "environment_audit.json"
    plan = split_plan(args.split)
    specs = [make_scene(row["family"], row["seed"], scfg) for row in plan]
    for spec, row in zip(specs, plan):
        spec.extras["selection_environment"] = row
    records = [dict(**scene_record(spec, solver["resolution_m"]), **{
        "split": args.split, "panel": row["panel"], "instance_index": row["instance_index"]})
        for spec, row in zip(specs, plan)]
    previous_paths = [args.old_manifest]
    for other in SPLITS:
        path = args.base.resolve() / "environment" / other / "scenes.json"
        if other != args.split and path.exists():
            previous_paths.append(path)
    previous = [scene_record(spec, solver["resolution_m"])
                for path in previous_paths for spec in load_manifest(path)]
    disjoint = assert_disjoint(records, previous)
    scene_value = dict(version=1, scenes=[spec.to_json() for spec in specs])
    if (scene_path.exists() and
            canonical_bytes(json.loads(scene_path.read_text())) != canonical_bytes(scene_value)):
        raise ValueError("Existing scene manifest differs; refusing to overwrite")
    signature = dict(split=args.split, solver=solver, protocol_lock=lock,
                     scene_config_sha256=sha256_file(args.scenes_config),
                     cfd_config_sha256=sha256_file(args.config),
                     geometry_source_sha256=sha256_file(ROOT / "src/winddyn/geometry/procedural.py"),
                     solver_source_sha256=sha256_file(ROOT / "src/winddyn/cfd/lbm2d.py"))
    previous_field_hashes = {}
    if audit_path.exists():
        old_audit = json.loads(audit_path.read_text())
        if old_audit["signature"] != signature:
            raise ValueError("Existing environment signature differs; refusing mixed provenance")
        previous_field_hashes = {portable_path(resolve_path(name)): digest for name, digest in
            old_audit.get("committed_field_hashes", {
                entry["path"]: entry["sha256"] for entry in old_audit.get("fields", [])}).items()}
    atomic_json(scene_path, scene_value)
    audit = dict(schema_version=1, signature=signature, execution=execution,
                 disjointness=disjoint, compared_manifests=[dict(path=portable_path(p),
                 sha256=sha256_file(p)) for p in previous_paths], scenes=records,
                 status="running", solver_fallback_used=False, fields=[],
                 committed_field_hashes=previous_field_hashes)
    atomic_json(audit_path, audit)
    entries, started = [], time.monotonic()
    try:
        for spec, row, record in zip(specs, plan, records):
            ny, nx = record["grid_shape_yx"]
            occ = occupancy_at_zref(spec, nx, ny)
            resolution = solver["resolution_m"]
            origin = np.asarray(spec.bounds_xy, np.float32) * -0.5 + resolution * 0.5
            todo = []
            for speed in SPEEDS:
                wid = wind_id(row["direction_deg"], speed)
                path = destination / "wind_fields" / f"{spec.scene_id}__{wid}.npz"
                todo.append((speed, wid, path))
            for start in range(0, len(todo), args.batch_size):
                batch = todo[start:start + args.batch_size]
                missing = [(speed, wid, path) for speed, wid, path in batch if not path.exists()]
                if missing:
                    results = solve_planar_lbm_batch(
                        occ, [(speed, row["direction_deg"]) for speed, _, _ in missing],
                        n_steps=solver["n_steps"], avg_window=solver["avg_window"],
                        device=device, tau=solver["tau"])
                    if len(results) != len(missing):
                        raise ValueError("Solver returned incorrect number of fields")
                    for (speed, wid, path), (u, v, metadata) in zip(missing, results):
                        field_qc(u, v, occ, speed)
                        if any(metadata.get(k) != solver[k] for k in ("tau", "n_steps", "avg_window")):
                            raise ValueError("Solver metadata differs from requested settings")
                        metadata.update(scene_id=spec.scene_id, wind_id=wid,
                            inlet_speed_mps=speed, inlet_direction_deg=row["direction_deg"],
                            condition_tag="ood_extrap" if speed == 8 else "train",
                            grid_shape_yx=[ny, nx], cell_size_xy_m=[resolution, resolution],
                            reference_altitude_m=spec.z_ref, selection_split=args.split,
                            selection_panel=row["panel"], geometry_sha256=record["geometry_sha256"],
                            occupancy_sha256=record["occupancy_sha256"],
                            environment_signature_sha256=hashlib.sha256(canonical_bytes(signature)).hexdigest(),
                            execution=execution, solver_fallback_used=False)
                        # Field writer creates directories, but never touches a previous field.
                        temp = path.with_name(path.stem + ".tmp.npz")
                        save_field(temp, PlanarField(origin, np.full(2, resolution, np.float32),
                                                    spec.z_ref, u, v, metadata))
                        temp.replace(path)
                for speed, wid, path in batch:
                    path_key = portable_path(path)
                    if (path_key in previous_field_hashes and
                            sha256_file(path) != previous_field_hashes[path_key]):
                        raise ValueError(f"Previously audited field hash mismatch: {path}")
                    field = load_field(path)
                    expected_meta = dict(scene_id=spec.scene_id, wind_id=wid,
                        geometry_sha256=record["geometry_sha256"], occupancy_sha256=record["occupancy_sha256"],
                        environment_signature_sha256=hashlib.sha256(canonical_bytes(signature)).hexdigest(),
                        tau=solver["tau"], n_steps=solver["n_steps"], avg_window=solver["avg_window"],
                        inlet_speed_mps=speed, inlet_direction_deg=row["direction_deg"],
                        field_type="cfd", solver_or_model="lbm_d2q9_bgk")
                    if any(field.meta.get(k) != value for k, value in expected_meta.items()):
                        raise ValueError(f"Existing field metadata mismatch: {path}")
                    if (field.z_ref != spec.z_ref or not np.allclose(field.origin_xy, origin) or
                            not np.allclose(field.spacing_xy, [resolution, resolution])):
                        raise ValueError(f"Existing field grid placement mismatch: {path}")
                    qc = field_qc(field.u, field.v, occ, speed)
                    entry = dict(scene_id=spec.scene_id, wind_id=wid, family=spec.family,
                        seed=spec.seed, split=args.split, panel=row["panel"],
                        instance_index=row["instance_index"], inlet_speed_mps=speed,
                        inlet_direction_deg=row["direction_deg"],
                        speed_mps=speed, direction_deg=row["direction_deg"],
                        tag="ood_extrap" if speed == 8 else "train", path=path_key,
                        sha256=sha256_file(path), qc=qc, execution=field.meta["execution"])
                    entries.append(entry)
                    previous_field_hashes[path_key] = entry["sha256"]
                    audit["fields"] = entries
                    atomic_json(audit_path, audit)
            audit["fields"] = entries
            atomic_json(audit_path, audit)
            print(f"[{args.split}] {spec.scene_id}: {len(entries)}/{len(plan) * 3} fields", flush=True)
        atomic_json(field_manifest, dict(fields=entries))
        audit.update(status="complete", elapsed_s=time.monotonic() - started,
                     scene_manifest_sha256=sha256_file(scene_path),
                     field_manifest_sha256=sha256_file(field_manifest))
    except Exception as exc:
        audit.update(status="failed", elapsed_s=time.monotonic() - started,
                     error=f"{type(exc).__name__}: {exc}")
        atomic_json(audit_path, audit)
        raise
    atomic_json(audit_path, audit)
    return dict(split=args.split, scenes=len(specs), fields=len(entries),
                scenes_manifest=str(scene_path), wind_manifest=str(field_manifest),
                audit=str(audit_path), **execution, status="complete")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/cfd/planar_mvp.yaml")
    ap.add_argument("--scenes-config", type=Path, default=ROOT / "configs/sim/scenes_mvp.yaml")
    ap.add_argument("--old-manifest", type=Path, default=ROOT / "data/manifests/scenes.json")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--batch-size", type=int, default=3, help="CFD conditions per scene batch")
    ap.add_argument("--protocol-lock", type=Path)
    ap.add_argument("--plan-only", action="store_true", help="Print seed/condition metadata; no geometry or files")
    return ap


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(split_plan(args.split) if args.plan_only else prepare(args), indent=2))


if __name__ == "__main__":
    main()
