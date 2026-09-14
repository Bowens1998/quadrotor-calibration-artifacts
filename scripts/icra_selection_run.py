"""Versioned, frozen-model selection experiment; no old source or results mutated."""
import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from _common import ROOT
import icra_snapshot_rollout as harness
from icra_paired_mpc import Predictor
from icra_snapshot_branches import branch as held_branch, tensor_state
from icra_snapshot_prediction import model_input, cost
from icra_mpc_policy import candidates, reference_at, physics, features, predict
from icra_vertical_authority import command_with_cap
from winddyn.geometry.procedural import SceneSpec
from winddyn.cfd.field_io import load_field

BASE = ROOT / 'runs/icra_selection_validation_20260912'
MODELS = [('scalar', 0), ('physics_features', 0), ('raw', 0)] + [
    ('supervised', k) for k in range(5)] + [('feedback', 0), ('observer', 0)]
REGIMES = ['mass_1p4', 'lag_3']
CAPS = [1, 4, 9]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def protocol_sha():
    return sha(BASE / 'PROTOCOL.md')


def check_lock(split):
    if split == 'pilot':
        return
    lock = json.loads((BASE / 'LOCK.json').read_text())
    assert lock['protocol_sha256'] == protocol_sha()
    for name, expected in lock['source_hashes'].items():
        assert sha(ROOT / name) == expected, name
    if split == 'test':
        selection = json.loads((BASE / 'selection.json').read_text())
        assert selection['protocol_sha256'] == protocol_sha()
        for name, expected in selection['validation_source_hashes'].items():
            assert sha(BASE / name) == expected, name


def configure(device):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable; do not silently change execution device')


def tasks_for(split, panel=None):
    folder = BASE / 'environment' / split
    raw = json.loads((folder / 'scenes.json').read_text())
    scenes = {s['scene_id']: SceneSpec.from_json(s) for s in
              (raw['scenes'] if isinstance(raw, dict) else raw)}
    entries = json.loads((folder / 'wind_fields.json').read_text())['fields']
    tasks = []
    for entry in entries:
        scene = scenes[entry['scene_id']]
        meta = scene.extras.get('selection_environment', {})
        p = entry.get('panel', meta.get('panel'))
        if split == 'validation' and p != panel:
            continue
        wind_id = entry['wind_id']
        ident = f'{split}/{scene.scene_id}/{wind_id}'
        wp_seed = int(hashlib.sha256(('selection-waypoint/' + ident).encode()).hexdigest()[:8], 16)
        fieldpath = Path(entry['path'])
        if not fieldpath.is_absolute():
            fieldpath = ROOT / fieldpath
        assert sha(fieldpath) == entry['sha256']
        tasks.append(dict(task_id=ident, scene=scene, scene_id=scene.scene_id,
                          field=load_field(fieldpath), wind_id=wind_id, wp_seed=wp_seed,
                          family=scene.family, panel=p, speed_mps=entry['speed_mps'],
                          tier='fresh_geometry_' + ('wind8' if entry['speed_mps'] == 8 else 'wind3or6')))
    tasks.sort(key=lambda t: (t['family'], t['scene_id'], t['speed_mps']))
    assert len(tasks) == (60 if split == 'test' else 15), (split, panel, len(tasks))
    # Check the path before any simulation. Do not resample based on flight outcomes.
    for task in tasks:
        wps = harness.sample_waypoints(task['scene'], np.random.default_rng(task['wp_seed']))
        assert all(harness._segment_clear(task['scene'], a, b, .6)
                   for a, b in zip(wps[:-1], wps[1:])), task['task_id']
    return tasks


def blocks_for(tasks, split):
    if split != 'test':
        return [tasks]
    return [[t for t in tasks if t['family'] == family] for family in sorted({t['family'] for t in tasks})]


def simulation_seed(split, panel, block):
    text = f'selection-plant/{split}/{panel}/{block}'
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def run_harness(tasks, regime, method, seed, device, sim_seed, hook=None):
    calls = 0
    def nominal(pos, ref, velocity):
        nonlocal calls
        calls += 1
        return command_with_cap(pos, ref, velocity, .5 if calls <= 100 else 1.5)
    old_predictor, old_command = harness.MatrixPredictor, harness.nominal_command
    harness.MatrixPredictor, harness.nominal_command = Predictor, nominal
    try:
        result = harness.rollout(tasks, regime, method, device, seed, None, sim_seed,
                                 duration_s=12., snapshot_hook=hook)
        assert calls == 600
        return result
    finally:
        harness.MatrixPredictor, harness.nominal_command = old_predictor, old_command


@torch.no_grad()
def nominal_continuation(sim, ctrl, thrust, active, command, alive,
                         centers, half, mask, waypoints, zref, step, no_intervention=False):
    """N-independent equivalent of the accepted nominal continuation branch."""
    ss, cc = copy.deepcopy((sim, ctrl))
    force, valid = thrust.clone(), alive.clone()
    current = active.clone()
    positions, validity, controls = [], [], [current.cpu().numpy().copy()]
    for k in range(300):
        if k >= 4 and k % 4 == 0:
            if no_intervention or k >= 24:
                ref, velocity = reference_at(waypoints, [(step + k) / 200], zref)
                u = command_with_cap(ss.state.pos.cpu().numpy(), ref[:, 0], velocity[:, 0], 1.5)
                current = torch.as_tensor(u, dtype=torch.float32, device=ss.device)
            else:
                current = command.clone()
            force = cc.compute(ss.state.pos, ss.state.quat, ss.state.lin_vel,
                               ss.state.ang_vel, current, .02)
            controls.append(current.cpu().numpy().copy())
        previous = [getattr(ss.state, key).clone() for key in ['pos', 'quat', 'lin_vel', 'ang_vel']]
        ss.step(force)
        for key, value in zip(['pos', 'quat', 'lin_vel', 'ang_vel'], previous):
            getattr(ss.state, key)[~valid] = value[~valid]
        valid &= ~((harness._sdf(ss.state.pos, centers, half, mask) < .25) | (ss.state.pos[:, 2] < .1))
        if (k + 1) % 10 == 0:
            positions.append(ss.state.pos.cpu().numpy().copy())
            validity.append(valid.cpu().numpy().copy())
    return np.stack(positions), np.stack(validity), np.stack(controls)


@torch.no_grad()
def forecast(adapter, snapshot):
    arr = model_input(snapshot)
    n = len(snapshot['initial_alive'])
    nominal = physics(arr, 'x_', adapter.tau if adapter.method == 'scalar' else .3)
    if adapter.method == 'scalar':
        pred = nominal
    else:
        residual = predict(features(arr, 'x_', adapter.recipe, adapter.model, adapter.device), adapter.fit)
        pred = nominal + residual.reshape(n * 9, 30, 3)
    pred = pred.reshape(n, 9, 30, 3)
    sf = snapshot['state_hist'][:, -1]
    c, s = sf[:, 11], sf[:, 10]
    world = pred.copy()
    world[..., 0] = c[:, None, None] * pred[..., 0] - s[:, None, None] * pred[..., 1]
    world[..., 1] = s[:, None, None] * pred[..., 0] + c[:, None, None] * pred[..., 1]
    return world + snapshot['initial_position'][:, None, None]


def capped_loss(errors, valid, cap):
    e = np.asarray(errors)[40:]
    v = np.asarray(valid)[40:]
    assert e.shape == v.shape and len(e) == 200
    assert np.isfinite(e[v]).all()
    loss = np.full(e.shape, float(cap))
    loss[v] = np.minimum(np.abs(e[v]), np.sqrt(cap)) ** 2
    return loss.mean(0)


def augment_episodes(out, arrays, tasks):
    assert arrays['position'].shape == (240, len(tasks), 3)
    assert np.isfinite(arrays['position']).all()
    assert np.isfinite(arrays['action']).all()
    assert not ((~arrays['valid'][:-1]) & arrays['valid'][1:]).any()
    values = {cap: capped_loss(arrays['tracking_error'], arrays['valid'], cap) for cap in CAPS}
    for j, (row, task) in enumerate(zip(out['episodes'], tasks)):
        row.update({key: task[key] for key in ['task_id', 'family', 'panel', 'speed_mps']})
        for cap in CAPS:
            row[f'capped_loss_{cap}'] = float(values[cap][j])
        m = arrays['valid'][40:, j]
        expected = float(np.sqrt(np.mean(arrays['tracking_error'][40:, j][m] ** 2))) if m.any() else None
        assert (expected is None and row['tracking_rmse'] is None) or np.isclose(expected, row['tracking_rmse'])


def runtime(device, started):
    return dict(device=str(device), gpu=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None,
                torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                wall_s=time.monotonic() - started, tf32=False)


@torch.no_grad()
def collect_branches(split, regime, panel, device):
    assert split in ['pilot', 'validation']
    check_lock(split)
    started = time.monotonic()
    tasks = tasks_for(split, panel)
    dest = BASE / 'branches' / split / regime / f'panel{panel}'
    dest.mkdir(parents=True, exist_ok=False)
    adapters = [Predictor(method, regime, device, seed, None) for method, seed in MODELS[:8]]
    all_rows, receipts, no_interventions = [], [], {}
    def hook(sim, ctrl, buf, thrust, active, alive, centers, half, mask, wps, zref, step):
        before, clock = tensor_state(sim, ctrl), sim.time
        ref, velocity = reference_at(wps, [step / 200], zref)
        nom = command_with_cap(sim.state.pos.cpu().numpy(), ref[:, 0], velocity[:, 0], 1.5)
        commands = candidates(nom)
        hp, hv, cp, cv = [], [], [], []
        for action in range(9):
            command = torch.as_tensor(commands[:, action], dtype=torch.float32, device=device)
            p, v = held_branch(sim, ctrl, thrust, command, alive, centers, half, mask)
            hp.append(p); hv.append(v)
            p, v, control = nominal_continuation(sim, ctrl, thrust, active, command, alive,
                                               centers, half, mask, wps, zref, step)
            np.testing.assert_array_equal(control[1:6], np.broadcast_to(commands[:, action].astype(np.float32), control[1:6].shape))
            cp.append(p); cv.append(v)
        if split == 'pilot':
            center = torch.as_tensor(commands[:, 4], dtype=torch.float32, device=device)
            p, v = held_branch(sim, ctrl, thrust, center, alive, centers, half, mask)
            np.testing.assert_array_equal(p, hp[4]); np.testing.assert_array_equal(v, hv[4])
            p, v, _ = nominal_continuation(sim, ctrl, thrust, active, center, alive,
                                         centers, half, mask, wps, zref, step)
            np.testing.assert_array_equal(p, cp[4]); np.testing.assert_array_equal(v, cv[4])
            no_interventions[step] = nominal_continuation(sim, ctrl, thrust, active, center, alive,
                                                        centers, half, mask, wps, zref, step, True)[:2]
        for key, value in tensor_state(sim, ctrl).items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert sim.time == clock
        a = {key: value.cpu().numpy() for key, value in buf.batch().items()}
        a.update(position_world=np.stack(hp, 1), valid=np.stack(hv, 1),
                 continuation_position_world=np.stack(cp, 1), continuation_valid=np.stack(cv, 1),
                 candidate_command=commands, initial_position=sim.state.pos.cpu().numpy().copy(),
                 initial_velocity=sim.state.lin_vel.cpu().numpy().copy(), initial_alive=alive.cpu().numpy().copy(),
                 reference_world=reference_at(wps, step / 200 + .05 * np.arange(1, 31), zref)[0],
                 previous_command=active.cpu().numpy().copy())
        held = a['position_world'].transpose(2, 1, 0, 3)
        cont = a['continuation_position_world'].transpose(2, 1, 0, 3)
        held_valid = a['valid'].transpose(2, 1, 0)
        cont_valid = a['continuation_valid'].transpose(2, 1, 0)
        support = a['initial_alive'] & held_valid.all((1, 2)) & cont_valid.all((1, 2))
        held_cost, cont_cost = cost(held, a), cost(cont, a)
        predictions = []
        for index, adapter in enumerate(adapters):
            world = forecast(adapter, a)
            assert np.isfinite(world).all()
            pc = cost(world, a)
            batch = {key: torch.as_tensor(a[key], device=device) for key in ['state_hist', 'action_hist', 'depth_hist']}
            fake_buf = SimpleNamespace(batch=lambda: batch, vel=[torch.as_tensor(a['initial_velocity'], device=device)])
            _, actual = adapter.plan(fake_buf, a['initial_position'], a['reference_world'], nom, a['previous_command'])
            np.testing.assert_allclose(pc, actual['cost'], atol=1e-5, rtol=1e-6)
            choice = pc.argmin(1)
            predictions.append(world)
            for j, task in enumerate(tasks):
                initial = bool(a['initial_alive'][j]); valid = bool(support[j]); selected = int(choice[j])
                all_rows.append(dict(model_index=index, task_id=task['task_id'], scene=task['scene_id'],
                    family=task['family'], wind=task['wind_id'], step=step, initial_alive=initial,
                    common_valid=valid, choice=selected,
                    held_mse=float(((world[j] - held[j]) ** 2).sum(-1).mean()) if valid else None,
                    continuation_regret=float(cont_cost[j, selected] - cont_cost[j].min()) if valid else None,
                    held_regret=float(held_cost[j, selected] - held_cost[j].min()) if valid else None,
                    selected_contact_held=bool(not held_valid[j, selected, -1]) if initial else None,
                    selected_contact_continuation=bool(not cont_valid[j, selected, -1]) if initial else None))
        a['forecast_world'] = np.stack(predictions)
        path = dest / f'step{step}.npz'
        np.savez_compressed(path, **a)
        receipts.append(dict(step=step, activation_step=step + 4, continuation_step=step + 24,
                             parents=len(tasks), common_valid=int(support.sum()),
                             path=str(path.relative_to(ROOT)), sha256=sha(path), parent_unmodified=True))
        print('SNAPSHOT_ACCEPTED', split, regime, panel, step, int(support.sum()), flush=True)
    out, arrays = run_harness(tasks, regime, 'feedback', 0, device,
                              simulation_seed(split, panel, 0), hook)
    augment_episodes(out, arrays, tasks)
    if split == 'pilot':
        replay, repeated = run_harness(tasks, regime, 'feedback', 0, device, simulation_seed(split, panel, 0))
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], repeated[key])
        for step, (position, valid) in no_interventions.items():
            lo = step // 10 + 1
            np.testing.assert_allclose(position, arrays['position'][lo:lo + 30], rtol=0, atol=1e-5)
            np.testing.assert_array_equal(valid, arrays['valid'][lo:lo + 30])
    np.savez_compressed(dest / 'parent.npz', **arrays)
    write(dest / 'parent.json', out)
    write(dest / 'scores.json', dict(protocol_sha256=protocol_sha(), split=split, regime=regime, panel=panel,
        candidate_count=8, rows=all_rows, snapshots=receipts,
        parent_receipts=[dict(path=str((dest / name).relative_to(ROOT)), sha256=sha(dest / name))
                         for name in ['parent.npz', 'parent.json']],
        predictor_provenance={str(i): p.hashes for i, p in enumerate(adapters)},
        validation_simulation_seconds=93 * len(tasks),
        qa_simulation_seconds=(12 + 3 * 3 * 1.5) * len(tasks) if split == 'pilot' else 0,
        runtime=runtime(device, started)))
    print('BRANCHES_COMPLETE', split, regime, panel, flush=True)


@torch.no_grad()
def flight(split, regime, panel, index, device):
    check_lock(split)
    if split == 'validation':
        assert index < 8
    started = time.monotonic()
    tasks = tasks_for(split, panel)
    dest = BASE / 'flights' / split / regime
    if split != 'test':
        dest = dest / f'panel{panel}'
    dest = dest / f'model{index:02d}'
    dest.mkdir(parents=True, exist_ok=False)
    episodes, decisions, provenance, files = [], [], {}, []
    method, model_seed = MODELS[index]
    for bi, block in enumerate(blocks_for(tasks, split)):
        out, arrays = run_harness(block, regime, method, model_seed, device, simulation_seed(split, panel, bi))
        augment_episodes(out, arrays, block)
        for decision in out['decisions']:
            assert decision['applied_step'] == decision['observation_step'] + 4
            decision['block'] = bi
        episodes.extend(out['episodes']); decisions.extend(out['decisions']); provenance.update(out['predictor_provenance'])
        path = dest / f'block{bi}.npz'
        np.savez_compressed(path, **arrays)
        files.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), tasks=[t['task_id'] for t in block]))
    write(dest / 'results.json', dict(protocol_sha256=protocol_sha(), split=split, regime=regime, panel=panel,
          model_index=index, method=method, model_seed=model_seed, stage='prospective frozen-model selection validation',
          episodes=episodes, decisions=decisions, predictor_provenance=provenance, arrays=files,
          simulation_seconds=12 * len(tasks), runtime=runtime(device, started)))
    print('FLIGHT_COMPLETE', split, regime, panel, index, flush=True)


def preflight():
    hashes = {}
    for regime in REGIMES:
        for method, seed in MODELS[:8]:
            adapter = Predictor(method, regime, 'cpu', seed, None)
            hashes.update(adapter.hashes)
    write(BASE / 'candidate_provenance.json', hashes)
    print('CANDIDATES_VERIFIED', len(hashes), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['preflight', 'branches', 'flight'])
    parser.add_argument('--split', choices=['pilot', 'validation', 'test'], default='pilot')
    parser.add_argument('--regime', choices=REGIMES, default='lag_3')
    parser.add_argument('--panel', type=int, default=0)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    configure('cpu' if args.operation == 'preflight' else args.device)
    if args.operation == 'preflight':
        preflight()
    elif args.operation == 'branches':
        collect_branches(args.split, args.regime, args.panel, args.device)
    else:
        flight(args.split, args.regime, args.panel, args.index, args.device)


if __name__ == '__main__':
    main()
