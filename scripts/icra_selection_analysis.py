"""Locked, parent-weighted model selection and independent scene-cluster evaluation.

This script never fits a predictor or reads test data during ``select``. The
three validation panels are fixed selection replicates, not bootstrap units.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/icra_selection_validation_20260912"
REGIMES = ("mass_1p4", "lag_3")
SELECTORS = ("R", "D", "C")
PANELS = (0, 1, 2)
MODEL_COUNT = 8
TEST_MODEL_COUNT = 10
METRICS = ("capped_loss_4", "capped_loss_1", "capped_loss_9", "contact", "success", "flight_exposure_s")
BOOTSTRAP_SEED = 2026091207
BOOTSTRAP_DRAWS = 20000
TIE_TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def relative(path, base):
    return str(Path(path).resolve().relative_to(Path(base).resolve()))


def checked_read(path, protocol_sha256):
    data = read(path)
    assert data["protocol_sha256"] == protocol_sha256, f"Protocol mismatch: {path}"
    return data


def source_receipt(path, base, sources, expected=None):
    path = Path(path).resolve()
    name = relative(path, base)  # Reject sources outside this experiment.
    digest = sha(path)
    assert expected is None or digest == expected, f"Raw receipt mismatch: {path}"
    assert name not in sources or sources[name] == digest
    sources[name] = digest
    return path


def raw_receipt(receipt, base, sources):
    path = Path(receipt["path"])
    if not path.is_absolute():
        path = ROOT / path
    return source_receipt(path, base, sources, receipt["sha256"])


def formal_lock(base, protocol_hash):
    path = Path(base) / "LOCK.json"
    lock = checked_read(path, protocol_hash)
    assert lock["source_hashes"], "Formal source lock cannot be empty"
    for name, digest in lock["source_hashes"].items():
        source = (ROOT / name).resolve()
        source.relative_to(ROOT)
        assert sha(source) == digest, f"Formal locked source changed: {name}"
    return path


def check_episode(episode):
    for key in ("task_id", "scene", "family", "wind", "prefix_sha256"):
        assert isinstance(episode[key], str) and episode[key]
    assert isinstance(episode["collided"], bool) and isinstance(episode["success"], bool)
    for key in ("actual_mass_kg", "actual_motor_tau_s", "terminal_error", "first_contact_s"):
        assert episode[key] is not None and np.isfinite(episode[key])
    assert episode["actual_mass_kg"] > 0 and episode["actual_motor_tau_s"] > 0
    assert episode["terminal_error"] >= 0 and 0 <= episode["first_contact_s"] <= 12
    assert episode["collided"] or episode["first_contact_s"] == 12
    assert episode["success"] == (not episode["collided"] and episode["terminal_error"] < .5)
    assert isinstance(episode["valid_records"], int) and 0 <= episode["valid_records"] <= 200
    if episode["valid_records"] == 0:
        assert episode["tracking_rmse"] is None
    else:
        assert episode["tracking_rmse"] is not None and np.isfinite(episode["tracking_rmse"])
        assert episode["tracking_rmse"] >= 0
    assert float(episode["speed_mps"]) in (3., 6., 8.)
    for cap in (1, 4, 9):
        value = episode[f"capped_loss_{cap}"]
        assert value is not None and np.isfinite(value) and 0 <= value <= cap + 1e-12


def tie_argmin(values, tolerance=TIE_TOL):
    """Lowest index within the locked absolute tolerance of the minimum."""
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) > 0 and np.isfinite(values).all()
    return int(np.flatnonzero(values <= values.min() + tolerance)[0])


def capped_tracking_loss(error, valid, cap=4.0, warmup_records=40):
    """Fixed-grid mean: alive min(error^2, cap), invalid cap.

    The accepted 12 s harness has 240 records at 0,...,11.95 s; the 200
    post-warm-up records are 2,...,11.95 s. Dead-state NaNs cannot contaminate
    this declared loss, but nonfinite live observations are an invalid run.
    """
    error = np.asarray(error, dtype=float)
    valid = np.asarray(valid)
    assert error.shape == valid.shape and valid.dtype == bool
    assert error.ndim in (1, 2) and cap > 0 and np.isfinite(cap)
    assert 0 <= warmup_records < len(error)
    assert np.isfinite(error[valid]).all(), "Nonfinite live tracking error"
    assert (error[valid] >= 0).all(), "Tracking error is a norm"
    e, v = error[warmup_records:], valid[warmup_records:]
    safe_error = np.where(v, e, 0.0)
    # Clip before squaring, so very large finite errors do not overflow.
    loss = np.minimum(safe_error, np.sqrt(cap)) ** 2
    loss = np.where(v, loss, cap)
    return loss.mean(axis=0)


def parent_mean(rows, field):
    groups = {}
    for row in rows:
        if row["common_valid"]:
            value = row[field]
            assert value is not None and np.isfinite(value)
            groups.setdefault(row["task_id"], []).append(float(value))
    if not groups:
        return None
    return float(np.mean([np.mean(values) for values in groups.values()]))


def selection_for_panel(rows, flight_losses):
    """Return all model scores and prespecified support/fallback decisions."""
    by_model = {}
    for row in rows:
        index = row["model_index"]
        assert type(index) is int and index in range(MODEL_COUNT)
        assert isinstance(row["initial_alive"], bool) and isinstance(row["common_valid"], bool)
        for mode in ("held", "continuation"):
            value = row[f"selected_contact_{mode}"]
            assert isinstance(value, bool) if row["initial_alive"] else value is None
        by_model.setdefault(index, []).append(row)
    assert set(by_model) == set(range(MODEL_COUNT))
    key = lambda r: (r["task_id"], int(r["step"]))
    reference = sorted(by_model[0], key=key)
    assert len(reference) == 45 and len({key(r) for r in reference}) == 45
    assert len({r["task_id"] for r in reference}) == 15
    for task_id in {r["task_id"] for r in reference}:
        assert {r["step"] for r in reference if r["task_id"] == task_id} == {600, 1200, 1800}
    signature = lambda r: (key(r), r["scene"], r["family"], r["wind"],
                           bool(r["initial_alive"]), bool(r["common_valid"]))
    for index, values in by_model.items():
        assert [signature(r) for r in sorted(values, key=key)] == [signature(r) for r in reference]
        for row in values:
            if row["common_valid"]:
                assert row["initial_alive"]
                for field in ("held_mse", "continuation_regret", "held_regret"):
                    assert row[field] is not None and np.isfinite(row[field]) and row[field] >= -1e-10
                assert row["held_mse"] >= 0
            else:
                assert all(row[field] is None for field in ("held_mse", "continuation_regret", "held_regret"))

    common = [r for r in reference if r["common_valid"]]
    support = dict(total_snapshots=len(reference), initially_alive=sum(bool(r["initial_alive"]) for r in reference),
                   common_valid_snapshots=len(common), supported_parents=len({r["task_id"] for r in common}),
                   supported_families=sorted({r["family"] for r in common}),
                   snapshots_by_family={f: sum(r["family"] == f for r in common)
                                        for f in sorted({r["family"] for r in reference})})
    supported = len(common) >= 15 and len(support["supported_families"]) >= 3
    scores = []
    for index in range(MODEL_COUNT):
        values = by_model[index]
        mse = parent_mean(values, "held_mse")
        contact = {}
        for mode in ("held", "continuation"):
            alive = [r for r in values if r["initial_alive"]]
            for r in alive:
                assert r[f"selected_contact_{mode}"] is not None
            # Contact is descriptive on all alive snapshots, also parent-weighted.
            grouped = {}
            for r in alive:
                grouped.setdefault(r["task_id"], []).append(bool(r[f"selected_contact_{mode}"]))
            contact[mode] = float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None
        closed = np.asarray(flight_losses[index], float)
        assert closed.shape == (15,) and np.isfinite(closed).all()
        assert ((closed >= 0) & (closed <= 4 + 1e-12)).all()
        scores.append(dict(index=index, held_mse=mse, held_rmse=np.sqrt(mse).item() if mse is not None else None,
                           continuation_regret=parent_mean(values, "continuation_regret"),
                           held_regret=parent_mean(values, "held_regret"), selected_contact=contact,
                           selected_contact_counts={mode: sum(bool(r[f"selected_contact_{mode}"]) for r in values if r["initial_alive"])
                                                    for mode in ("held", "continuation")},
                           closed_loop_capped_loss_4=float(closed.mean())))
    selected = {}
    for selector, field in (("R", "held_rmse"), ("D", "continuation_regret"), ("C", "closed_loop_capped_loss_4")):
        enabled = selector == "C" or supported
        index = tie_argmin([r[field] for r in scores]) if enabled else 0
        selected[selector] = dict(index=index, supported=enabled,
                                  fallback=None if enabled else "scalar: insufficient shared diagnostic support",
                                  score=scores[index][field], score_name=field)
    return dict(selected=selected, support=support, scores=scores)


def paired_identity(episodes):
    fields = ("task_id", "scene", "family", "wind", "speed_mps", "panel", "actual_mass_kg", "actual_motor_tau_s", "prefix_sha256")
    return [tuple(e[field] for field in fields) for e in sorted(episodes, key=lambda e: e["task_id"])]


def select(base=BASE, protocol=None):
    base = Path(base)
    protocol = Path(protocol) if protocol else base / "PROTOCOL.md"
    assert not (base / "selection.json").exists(), "Selection is immutable"
    assert not (base / "flights/test").exists(), "Selection must precede every test flight"
    protocol_hash = sha(protocol)
    selected, details, sources, provenance = {}, {}, {}, {}
    source_receipt(formal_lock(base, protocol_hash), base, sources)
    validation_scenes, validation_tasks = set(), set()
    for regime in REGIMES:
        selected[regime], details[regime], provenance[regime] = {}, {}, {}
        for panel in PANELS:
            bp = base / "branches/validation" / regime / f"panel{panel}/scores.json"
            branches = checked_read(bp, protocol_hash)
            assert (branches["split"], branches["regime"], branches["panel"]) == ("validation", regime, panel)
            sources[relative(bp, base)] = sha(bp)
            source_receipt(bp.with_name("parent.npz"), base, sources)
            source_receipt(bp.with_name("parent.json"), base, sources)
            assert len(branches["snapshots"]) == 3 and {r["step"] for r in branches["snapshots"]} == {600, 1200, 1800}
            for receipt in branches["snapshots"]:
                raw_receipt(receipt, base, sources)
            losses, identity = {}, None
            for index in range(MODEL_COUNT):
                path = base / "flights/validation" / regime / f"panel{panel}/model{index:02d}/results.json"
                data = checked_read(path, protocol_hash)
                assert (data["split"], data["regime"], data["panel"], data["model_index"]) == ("validation", regime, panel, index)
                episodes = data["episodes"]
                assert len(episodes) == 15 and len({e["task_id"] for e in episodes}) == 15
                for episode in episodes:
                    check_episode(episode)
                    assert episode["panel"] == panel
                assert len({e["scene"] for e in episodes}) == 5
                assert len({e["family"] for e in episodes}) == 5
                for scene in {e["scene"] for e in episodes}:
                    group = [e for e in episodes if e["scene"] == scene]
                    assert len(group) == 3 and len({e["family"] for e in group}) == 1
                    assert {float(e["speed_mps"]) for e in group} == {3., 6., 8.}
                if index == 0:
                    validation_scenes.update(e["scene"] for e in episodes)
                    validation_tasks.update(e["task_id"] for e in episodes)
                current = paired_identity(episodes)
                if identity is None:
                    identity = current
                assert identity == current, "Validation controllers are not paired"
                expected = {(r["task_id"], r["scene"], r["family"], r["wind"]) for r in branches["rows"]}
                assert expected == {(e["task_id"], e["scene"], e["family"], e["wind"]) for e in episodes}
                losses[index] = [e["capped_loss_4"] for e in episodes]
                key = str(index)
                if key in provenance[regime]:
                    assert provenance[regime][key] == data["predictor_provenance"]
                assert data["predictor_provenance"] and data["predictor_provenance"] == branches["predictor_provenance"][key]
                provenance[regime][key] = data["predictor_provenance"]
                sources[relative(path, base)] = sha(path)
                assert data["arrays"]
                for receipt in data["arrays"]:
                    raw_receipt(receipt, base, sources)
            outcome = selection_for_panel(branches["rows"], losses)
            selected[regime][str(panel)] = outcome.pop("selected")
            details[regime][str(panel)] = outcome
    assert len(validation_scenes) == 15 and len(validation_tasks) == 45, "Validation panels overlap or task manifest changed"
    result = dict(protocol_sha256=protocol_hash, created_utc=datetime.now(timezone.utc).isoformat(),
                  analysis_source_sha256=sha(__file__),
                  selected=selected, validation=details, validation_source_hashes=sources,
                  validation_scenes=sorted(validation_scenes), validation_tasks=sorted(validation_tasks),
                  predictor_provenance=provenance, candidate_indices=list(range(MODEL_COUNT)),
                  tie_tolerance=TIE_TOL, fallback_index=0,
                  weighting="Within-parent mean of common snapshots, then equal supported-parent mean; all 3 fixed panels retained",
                  test_data_read=False, status="selection_locked_before_test_evaluation")
    write_new(base / "selection.json", result)
    return result


def stratified_cluster_indices(families, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """One shared scene index matrix; wind tasks/panels always travel together."""
    families = np.asarray(families)
    assert families.ndim == 1 and len(families) > 0
    rng = np.random.default_rng(seed)
    blocks = []
    for family in sorted(set(families.tolist())):
        indices = np.flatnonzero(families == family)
        blocks.append(rng.choice(indices, size=(draws, len(indices)), replace=True))
    return np.concatenate(blocks, axis=1)


def bootstrap_contrast(left, right, indices, interval_level=.975):
    """Inputs are fixed panel × scene × wind-task arrays."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    assert left.shape == right.shape and left.ndim == 3
    assert np.isfinite(left).all() and np.isfinite(right).all()
    assert indices.shape[1] == left.shape[1]
    scene_difference = (left - right).mean(axis=(0, 2))
    draws = scene_difference[indices].mean(axis=1)
    alpha = (1 - interval_level) / 2
    return dict(difference=float(scene_difference.mean()),
                interval_level=interval_level,
                interval=np.quantile(draws, [alpha, 1-alpha]).tolist(),
                draws=len(draws)), draws


def episode_metric(episode, metric):
    key = {"contact": "collided", "flight_exposure_s": "first_contact_s"}.get(metric, metric)
    value = episode[key]
    assert value is not None and np.isfinite(value)
    value = float(value)
    if metric in ("contact", "success"):
        assert value in (0.0, 1.0)
    elif metric == "flight_exposure_s":
        assert 0 <= value <= 12
    else:
        cap = float(metric.rsplit("_", 1)[1])
        assert 0 <= value <= cap + 1e-12
    return value


def flight_arrays(data, episodes, base, sources):
    """Read only hashed, task-labelled test observations for paired RMSE."""
    observed = {}
    for receipt in data["arrays"]:
        path = raw_receipt(receipt, base, sources)
        tasks = receipt["tasks"]
        assert len(tasks) == len(set(tasks))
        with np.load(path, allow_pickle=False) as arrays:
            errors, valid = arrays["tracking_error"], arrays["valid"]
            assert errors.shape == valid.shape == (240, len(tasks)) and valid.dtype == bool
            assert np.isfinite(errors[valid]).all() and (errors[valid] >= 0).all()
            assert not ((~valid[:-1]) & valid[1:]).any()
            for column, task_id in enumerate(tasks):
                assert task_id not in observed, "Duplicated task in test array blocks"
                observed[task_id] = (errors[:, column].copy(), valid[:, column].copy())
    assert set(observed) == {e["task_id"] for e in episodes}
    return (np.stack([observed[e["task_id"]][0] for e in episodes], axis=1)[40:],
            np.stack([observed[e["task_id"]][1] for e in episodes], axis=1)[40:])


def common_time_rmse(left_error, left_valid, right_error, right_valid):
    """Paired per-task RMSE on the intersection; missing support stays missing."""
    assert left_error.shape == left_valid.shape == right_error.shape == right_valid.shape
    mask = left_valid & right_valid
    count = mask.sum(axis=0)
    def rmse(error):
        safe = np.where(mask, error, 0.0).astype(float)
        mean_square = np.divide(np.square(safe).sum(axis=0), count,
                                out=np.full(count.shape, np.nan), where=count > 0)
        assert np.isfinite(mean_square[count > 0]).all(), "Nonfinite uncapped common-time error"
        return np.sqrt(mean_square)
    return rmse(left_error), rmse(right_error), count


def nullable_mean(values):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else None


def common_time_contrast(left, right, indices):
    """Equal fixed-panel means, each over its pair-specific eligible tasks."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    assert left.shape == right.shape and left.ndim == 3
    mask = np.isfinite(left)
    assert np.array_equal(mask, np.isfinite(right))
    counts = mask.sum(axis=2)
    sums = np.where(mask, left - right, 0.0).sum(axis=2)
    denominators = counts[:, indices].sum(axis=2)
    numerators = sums[:, indices].sum(axis=2)
    usable = (denominators > 0).all(axis=0)
    draws = np.full(len(indices), np.nan)
    draws[usable] = (numerators[:, usable] / denominators[:, usable]).mean(axis=0)
    point_counts = counts.sum(axis=1)
    supported = bool((point_counts > 0).all())
    point = float((sums.sum(axis=1) / point_counts).mean()) if supported else None
    # Report no interval for an entirely unsupported fixed panel.
    interval = np.quantile(draws[usable], [.025, .975]).tolist() if supported and usable.any() else None
    return dict(difference=point, interval_level=.95, interval=interval,
                eligible_tasks_by_panel=point_counts.tolist(), requested_draws=len(indices),
                valid_draws=int(usable.sum()), zero_support_draws=int((~usable).sum()),
                support="Pair-specific common flight time; an empty fixed panel yields no pooled estimate"), draws


def summarize(base=BASE, protocol=None):
    base = Path(base)
    protocol = Path(protocol) if protocol else base / "PROTOCOL.md"
    protocol_hash = sha(protocol)
    lock_path = base / "selection.json"
    lock = checked_read(lock_path, protocol_hash)
    assert lock["test_data_read"] is False
    assert lock["analysis_source_sha256"] == sha(__file__), "Analysis code changed after selection"
    formal_lock(base, protocol_hash)
    for name, digest in lock["validation_source_hashes"].items():
        path = (base / name).resolve()
        assert relative(path, base) == name, "Invalid validation receipt path"
        assert sha(path) == digest, f"Validation changed after selection: {name}"
    outdir = base / "analysis"
    assert not outdir.exists(), "Analysis output must be a new directory"
    sources = {"selection.json": sha(lock_path)}
    primary, secondary, panel_rows, means, matrices, bootstrap_arrays = [], [], [], [], {}, {}
    common_time_panels, common_time_references = [], []
    test_geometry = {}
    for regime in REGIMES:
        models, traces, identity = {}, {}, None
        for index in range(TEST_MODEL_COUNT):
            path = base / "flights/test" / regime / f"model{index:02d}/results.json"
            data = checked_read(path, protocol_hash)
            assert (data["split"], data["regime"], data["model_index"]) == ("test", regime, index)
            episodes = sorted(data["episodes"], key=lambda e: e["task_id"])
            assert len(episodes) == 60 and len({e["task_id"] for e in episodes}) == 60
            for episode in episodes:
                check_episode(episode)
            current = paired_identity(episodes)
            if identity is None:
                identity = current
            assert identity == current, "Test controllers are not paired"
            if index < MODEL_COUNT:
                assert data["predictor_provenance"] == lock["predictor_provenance"][regime][str(index)]
            models[index] = episodes
            sources[relative(path, base)] = sha(path)
            traces[index] = flight_arrays(data, episodes, base, sources)
        scenes = sorted({e["scene"] for e in models[0]})
        assert len(scenes) == 20
        assert not set(scenes) & set(lock["validation_scenes"])
        assert not {e["task_id"] for e in models[0]} & set(lock["validation_tasks"])
        families, order = [], []
        for scene in scenes:
            episode_ids = [i for i, e in enumerate(models[0]) if e["scene"] == scene]
            assert len(episode_ids) == 3
            assert len({models[0][i]["family"] for i in episode_ids}) == 1
            assert {float(models[0][i]["speed_mps"]) for i in episode_ids} == {3.0, 6.0, 8.0}
            families.append(models[0][episode_ids[0]]["family"])
            order.append(sorted(episode_ids, key=lambda i: models[0][i]["speed_mps"]))
        assert len(set(families)) == 5 and all(families.count(f) == 4 for f in set(families))
        order = np.asarray(order)
        indices = stratified_cluster_indices(families)
        bootstrap_arrays[f"{regime}__scene_indices"] = indices
        test_geometry[regime] = dict(scenes=scenes, families=families, task_ids=[[models[0][i]["task_id"] for i in group] for group in order])
        cubes = {metric: np.stack([np.asarray([episode_metric(e, metric) for e in models[i]])[order]
                                  for i in range(TEST_MODEL_COUNT)]) for metric in METRICS}
        matrices[regime] = {}
        for index in range(TEST_MODEL_COUNT):
            valid_rmse = [e["tracking_rmse"] for e in models[index] if e["tracking_rmse"] is not None]
            matrices[regime][str(index)] = dict(index=index, tasks=60,
                **{metric: float(cubes[metric][index].mean()) for metric in METRICS},
                tracking_rmse=float(np.mean(valid_rmse)) if valid_rmse else None,
                tracking_rmse_available=len(valid_rmse),
                terminal_error=float(np.mean([e["terminal_error"] for e in models[index]])))
        selected_values = {}
        for selector in SELECTORS:
            choices = [lock["selected"][regime][str(panel)][selector]["index"] for panel in PANELS]
            assert all(index in range(MODEL_COUNT) for index in choices)
            selected_values[selector] = {metric: cubes[metric][choices] for metric in METRICS}
            for panel, index in zip(PANELS, choices):
                decision = lock["selected"][regime][str(panel)][selector]
                panel_rows.append(dict(regime=regime, panel=panel, selector=selector,
                    index=index, supported=decision["supported"], fallback=decision["fallback"],
                    **{metric: float(cubes[metric][index].mean()) for metric in METRICS}))
            means.append(dict(regime=regime, selector=selector, selected_indices=choices,
                supported_panels=sum(lock["selected"][regime][str(panel)][selector]["supported"] for panel in PANELS),
                **{metric: float(values.mean()) for metric, values in selected_values[selector].items()}))
        for metric in METRICS:
            for left, right in (("D", "R"), ("D", "C"), ("R", "C")):
                is_primary = metric == "capped_loss_4" and (left, right) == ("D", "R")
                result, draws = bootstrap_contrast(selected_values[left][metric], selected_values[right][metric],
                                                   indices, .975 if is_primary else .95)
                item = dict(regime=regime, comparison=f"{left} minus {right}", metric=metric,
                            status="primary" if is_primary else "secondary_exploratory",
                            adjustment="Bonferroni: two regime contrasts" if is_primary else "none",
                            **result)
                (primary if is_primary else secondary).append(item)
                bootstrap_arrays[f"{regime}__{metric}__{left}_minus_{right}"] = draws
        for left, right in (("D", "R"), ("D", "C"), ("R", "C")):
            left_values, right_values = [], []
            for panel in PANELS:
                li = lock["selected"][regime][str(panel)][left]["index"]
                ri = lock["selected"][regime][str(panel)][right]["index"]
                lv, rv, count = common_time_rmse(*traces[li], *traces[ri])
                left_values.append(lv[order]); right_values.append(rv[order])
                common_time_panels.append(dict(regime=regime, panel=panel, comparison=f"{left} minus {right}",
                    left_index=li, right_index=ri, eligible_tasks=int((count > 0).sum()),
                    common_observations=int(count.sum()), left_rmse=nullable_mean(lv), right_rmse=nullable_mean(rv),
                    difference=nullable_mean(lv-rv)))
            result, draws = common_time_contrast(left_values, right_values, indices)
            secondary.append(dict(regime=regime, comparison=f"{left} minus {right}",
                metric="common_time_tracking_rmse", status="secondary_exploratory", adjustment="none", **result))
            bootstrap_arrays[f"{regime}__common_time_tracking_rmse__{left}_minus_{right}"] = draws
        for index in range(MODEL_COUNT):
            for reference in (8, 9):
                lv, rv, count = common_time_rmse(*traces[index], *traces[reference])
                common_time_references.append(dict(regime=regime, model_index=index, reference_index=reference,
                    eligible_tasks=int((count > 0).sum()), common_observations=int(count.sum()),
                    model_rmse=nullable_mean(lv), reference_rmse=nullable_mean(rv), difference=nullable_mean(lv-rv),
                    status="descriptive only; pair-specific common flight time"))
        for metric in METRICS:
            bootstrap_arrays[f"{regime}__model_{metric}"] = cubes[metric]
    assert len(primary) == 2 and len(panel_rows) == 18
    outdir.mkdir()
    np.savez_compressed(outdir / "bootstrap.npz", **bootstrap_arrays)
    result = dict(protocol_sha256=protocol_hash, selection_sha256=sha(lock_path),
                  test_result_source_hashes=sources, primary=primary, secondary=secondary,
                  selector_means=means, panel_results=panel_rows, all_test_models=matrices,
                  common_time_panel_comparisons=common_time_panels, common_time_reference_comparisons=common_time_references,
                  test_geometry=test_geometry, bootstrap_seed=BOOTSTRAP_SEED,
                  bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_file_sha256=sha(outdir / "bootstrap.npz"),
                  statistical_scope="Paired family-stratified test-scene bootstrap, conditional on the fixed candidate pool and three fixed validation panels; no panel/model-seed resampling",
                  primary_metric="Mean over the 200 records at 2,...,11.95 s: alive min(error squared,4), otherwise 4 (m^2)",
                  sensitivity_caps_m2=[1, 9], failure_warning="Capped tracking loss is a declared utility, not a safety measure; contact and success are reported separately",
                  status="complete_pending_independent_acceptance")
    write_new(outdir / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "summarize"))
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    result = (select if args.command == "select" else summarize)(args.base, args.protocol)
    print(json.dumps(dict(command=args.command, status=result["status"]), indent=2))


if __name__ == "__main__":
    main()
