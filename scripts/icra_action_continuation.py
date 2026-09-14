"""Locked one-interval action intervention with two causal shared continuations."""
import argparse
import copy
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import icra_snapshot_rollout as parent
from icra_snapshot_branches import branch as held_branch, tensor_state
from icra_mpc_policy import candidates, reference_at
from icra_vertical_authority import command_with_cap

ROOT = parent.ROOT
BASE = ROOT/'runs/icra_action_continuation_20260909'
OLD = ROOT/'runs/icra_snapshot_pilot_20260908'
CHOICES = ROOT/'runs/icra_campaign_snapshot_20260909/attempt01'
REGIMES = ['mass_1p4', 'lag_3']
TIERS = ['id', 'wind_extrap', 'joint_extrap']
MODES = ['nominal', 'observer']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def compare(a, b, exact=False):
    assert a.shape == b.shape
    if exact or a.dtype == bool:
        np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
    return float(np.max(np.abs(a.astype(float)-b.astype(float))))


@torch.no_grad()
def continuation(sim, ctrl, buf, thrust, active, command, alive,
                 centers, half, mask, wps, zref, step, mode):
    """Clones are independent; control precedes observation at coincident ticks."""
    assert mode in MODES+['no_intervention']
    ss, cc = copy.deepcopy((sim, ctrl))
    force, valid = thrust.clone(), alive.clone()
    current = active.clone()
    history = SimpleNamespace(state=[v.clone() for v in buf.state], ready=lambda: True)
    initial_history = torch.stack(history.state, 1).cpu().numpy().copy()
    vp = parent.load_vehicle()
    observer = parent.Estimators('observer', 0, ss.device, vp) if mode == 'observer' else None
    wind = torch.zeros(len(alive), 2, device=ss.device)
    zero = torch.zeros(len(alive), 3, device=ss.device)
    trace = {k: [] for k in ['position_world', 'valid', 'observation_state',
             'observer_wind', 'control_command', 'commanded_thrust',
             'control_position', 'control_velocity', 'control_wind', 'accel_ff']}

    def record_observation():
        nonlocal wind
        if observer is not None:
            # The third argument is deliberately unavailable; the causal observer ignores it.
            wind = observer.estimate('observer', history,
                    torch.full_like(wind, float('nan'))).clamp(-15, 15)
        trace['observation_state'].append(history.state[-1].cpu().numpy().copy())
        trace['observer_wind'].append(wind.cpu().numpy().copy())

    def record_control(ff):
        for key, value in [('control_command', current), ('commanded_thrust', force),
                           ('control_position', ss.state.pos),
                           ('control_velocity', ss.state.lin_vel),
                           ('control_wind', wind), ('accel_ff', ff)]:
            trace[key].append(value.cpu().numpy().copy())

    # k=0 control was already executed by the parent before the snapshot observation.
    record_control(zero)
    record_observation()
    for k in range(300):
        if k >= 4 and k % 4 == 0:
            follow = mode == 'no_intervention' or k >= 24
            if follow:
                ref, velocity = reference_at(wps, [(step+k)/200], zref)
                u = command_with_cap(ss.state.pos.cpu().numpy(), ref[:, 0], velocity[:, 0], 1.5)
                current = torch.tensor(u, dtype=torch.float32, device=ss.device)
            else:
                current = command.clone()
            ff = zero
            if mode == 'observer' and k >= 24:
                factor = (force.sum(-1)/(vp.mass*parent.GRAVITY)).clamp(0, 4)
                horizontal = (parent.drag_accel_nominal(ss.state.lin_vel[:, :2], factor, vp)
                              - parent.drag_accel_nominal(ss.state.lin_vel[:, :2]-wind, factor, vp))
                ff = torch.cat([horizontal, zero[:, 2:]], -1)
            force = cc.compute(ss.state.pos, ss.state.quat, ss.state.lin_vel,
                               ss.state.ang_vel, current, .02,
                               accel_ff=ff if mode == 'observer' and k >= 24 else None)
            record_control(ff)
        if k > 0 and k % 10 == 0:
            history.state.append(parent.state_features_torch(ss.state.pos, ss.state.quat,
                                 ss.state.lin_vel, ss.state.ang_vel))
            history.state = history.state[-16:]
            record_observation()
        previous = [getattr(ss.state, key).clone() for key in ['pos', 'quat', 'lin_vel', 'ang_vel']]
        ss.step(force)
        for key, value in zip(['pos', 'quat', 'lin_vel', 'ang_vel'], previous):
            getattr(ss.state, key)[~valid] = value[~valid]
        hit = (parent._sdf(ss.state.pos, centers, half, mask) < .25) | (ss.state.pos[:, 2] < .1)
        valid &= ~hit
        if (k+1) % 10 == 0:
            trace['position_world'].append(ss.state.pos.cpu().numpy().copy())
            trace['valid'].append(valid.cpu().numpy().copy())
    history.state.append(parent.state_features_torch(ss.state.pos, ss.state.quat,
                         ss.state.lin_vel, ss.state.ang_vel))
    history.state = history.state[-16:]
    record_observation()  # final observation is saved but cannot affect a control tick.
    result = {k: np.stack(v) for k, v in trace.items()}
    result.update(initial_state_history=initial_history,
                  observation_steps=np.arange(0, 301, 10), control_steps=np.arange(0, 300, 4))
    assert result['position_world'].shape == (30, 12, 3)
    assert result['observation_state'].shape == (31, 12, 12)
    return result


def source_lock(dest):
    files = set((ROOT/'scripts').glob('*.py')) | set((ROOT/'src').rglob('*.py'))
    files |= set((ROOT/'configs').rglob('*.yaml'))
    files |= {BASE/'PROTOCOL.md', ROOT/'data/manifests/scenes.json', ROOT/'data/manifests/wind_fields.json'}
    files |= set(CHOICES.glob('config*.json'))
    files |= {CHOICES/'source_lock.json', CHOICES/'output_hashes.json', CHOICES/'summary.json'}
    old_receipt = json.loads((CHOICES/'output_hashes.json').read_text())
    for p in CHOICES.glob('config*.json'):
        assert sha(p) == old_receipt[str(p.relative_to(ROOT))]
    fields = json.loads((ROOT/'data/manifests/wind_fields.json').read_text())['fields']
    selected = []
    for ti, tier in enumerate(TIERS):
        tasks = parent.build_tasks(tier, 12, selection_seed=55112007+ti, waypoint_base=56112007)
        for t in tasks:
            selected.append(dict(tier=tier, scene=t['scene_id'], wind=t['wind_id'], wp_seed=t['wp_seed']))
            entry = next(f for f in fields if f['scene_id'] == t['scene_id'] and f['wind_id'] == t['wind_id'])
            files.add(Path(parent.portable_path(entry['path'], ROOT)))
    for reg in REGIMES:
        files |= set((OLD/reg).glob('*.npz'))
        pp = ROOT/'runs/icra_vertical_authority_20260908'/('config001' if reg == 'mass_1p4' else 'config005')
        files |= set(pp.glob('*.npz')) | {pp/'results.json'}
    write(dest/'source_lock.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)})
    protected = list((ROOT/'paper/icra_2027').rglob('*'))
    protected += [ROOT/'output/pdf/icra_campaign_robustness_20260909.pdf',
                  ROOT/'output/pdf/icra_campaign_supplement_20260909.pdf',
                  ROOT/'output/data/icra_campaign_evidence_bundle_20260909.zip']
    write(dest/'protected_files.json', {str(p.relative_to(ROOT)): sha(p) for p in protected if p.is_file()})
    write(dest/'preflight.json', dict(tasks=selected, source_files=len(files), choices=190,
          torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
          gpu=torch.cuda.get_device_name(0), tf32=False, physics_hz=200,
          control_hz=50, observation_hz=20, activation_step=4, continuation_step=24))


@torch.no_grad()
def run(attempt):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dest = BASE/attempt
    dest.mkdir(exist_ok=False)
    started = time.time()
    try:
        source_lock(dest)
        for reg in REGIMES:
            rd = dest/reg
            rd.mkdir()
            original = ROOT/'runs/icra_vertical_authority_20260908'/('config001' if reg == 'mass_1p4' else 'config005')
            for ti, tier in enumerate(TIERS):
                tasks = parent.build_tasks(tier, 12, selection_seed=55112007+ti, waypoint_base=56112007)
                for task in tasks:
                    task['tier'] = tier
                old_parent = load(original/f'{tier}.npz')
                calls = 0
                checks = []

                def command(pos, ref, velocity):
                    nonlocal calls
                    calls += 1
                    return command_with_cap(pos, ref, velocity, .5 if calls <= 100 else 1.5)

                parent.nominal_command = command

                def hook(sim, ctrl, buf, thrust, active, alive, centers, half, mask, wps, zref, step):
                    name = f'{tier}_step{step}'
                    old = load(OLD/reg/f'{name}.npz')
                    before, clock = tensor_state(sim, ctrl), sim.time
                    ref, vel = reference_at(wps, [step/200], zref)
                    u = candidates(command_with_cap(sim.state.pos.cpu().numpy(), ref[:, 0], vel[:, 0], 1.5))
                    observed = {k: v.cpu().numpy() for k, v in buf.batch().items()}
                    observed.update(candidate_command=u, initial_position=sim.state.pos.cpu().numpy(),
                                    initial_velocity=sim.state.lin_vel.cpu().numpy(), initial_alive=alive.cpu().numpy(),
                                    previous_command=active.cpu().numpy(),
                                    reference_world=reference_at(wps, step/200+.05*np.arange(1, 31), zref)[0])
                    input_errors = {k: compare(v, old[k]) for k, v in observed.items()}
                    cmds = [torch.tensor(u[:, c], dtype=torch.float32, device=sim.device) for c in range(9)]
                    hp, hv = held_branch(sim, ctrl, thrust, cmds[4], alive, centers, half, mask)
                    held_error = compare(hp, old['position_world'][:, 4])
                    compare(hv, old['valid'][:, 4], True)
                    common_args = (sim, ctrl, buf, thrust, active)
                    tail = (alive, centers, half, mask, wps, zref, step)
                    ni = continuation(*common_args, cmds[4], *tail, 'no_intervention')
                    sl = slice(step//10+1, step//10+31)
                    no_intervention_error = compare(ni['position_world'], old_parent['position'][sl])
                    compare(ni['valid'], old_parent['valid'][sl], True)
                    np.savez_compressed(rd/f'{name}_mechanics.npz', **observed,
                                        held_center_position=hp, held_center_valid=hv,
                                        no_intervention_position=ni['position_world'], no_intervention_valid=ni['valid'])
                    for mode in MODES:
                        runs = [continuation(*common_args, cmd, *tail, mode) for cmd in cmds]
                        dup = continuation(*common_args, cmds[4], *tail, mode)
                        for key in dup:
                            compare(dup[key], runs[4][key], True)
                        for c, result in enumerate(runs):
                            compare(result['position_world'][:2], old['position_world'][:2, c])
                            compare(result['valid'][:2], old['valid'][:2, c], True)
                        # Candidate is axis 1 for all time traces; history/time arrays are shared.
                        shared = ['initial_state_history', 'control_steps', 'observation_steps']
                        output = {k: runs[0][k] if k in shared else np.stack([v[k] for v in runs], 1) for k in runs[0]}
                        np.savez_compressed(rd/f'{name}_{mode}.npz', **output)
                    after = tensor_state(sim, ctrl)
                    assert before.keys() == after.keys() and sim.time == clock
                    for key in before:
                        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
                    checks.append(dict(snapshot=name, input_max_abs=input_errors,
                         old_center_max_abs=held_error, no_intervention_max_abs=no_intervention_error,
                         duplicate_centers_exact=True, pre_switch_all_nine_match=True, parent_unmodified=True))
                    write(rd/f'{name}_checks.json', checks[-1])
                    print('snapshot verified', reg, name, round(time.time()-started, 1), 's', flush=True)

                result, arrays = parent.rollout(tasks, reg, 'feedback', 'cuda', 0, None,
                                               57112007+1000*ti, snapshot_hook=hook)
                errors = {k: compare(arrays[k], v) for k, v in old_parent.items()}
                assert len(checks) == 3
                np.savez_compressed(rd/f'{tier}_parent.npz', **arrays)
                write(rd/f'{tier}_parent.json', dict(**result, replay_max_abs=errors))
        receipt = {str(p.relative_to(ROOT)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
        write(dest/'output_hashes.json', receipt)
        write(dest/'completion.json', dict(status='simulation_complete_pending_analysis_acceptance',
              seconds=time.time()-started, snapshots=18, modes=2, unique_states=216))
    except Exception:
        write(dest/'failure.json', dict(error=traceback.format_exc(), seconds=time.time()-started))
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--attempt', required=True)
    run(ap.parse_args().attempt)
