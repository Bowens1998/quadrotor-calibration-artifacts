"""Oracle projection audit of shared versus state-dependent affine action responses."""
import argparse
import json
from pathlib import Path

import numpy as np

from _common import ROOT
from icra_adaptation_fit import physics, sha
from icra_decision_evaluation import model_input
from icra_decision_fit import CONFIGS
from icra_readout_structure import centered, world_vector

BASE = ROOT / 'runs/icra_response_capacity_20260908'
DATA = ROOT / 'runs/icra_paired_calibration_20260908'
FIT = ROOT / 'runs/icra_decision_fit_20260908'


def inverse_heading(y, state):
    c, s = state[:, -1, 11], state[:, -1, 10]
    q = c.astype(float)**2 + s.astype(float)**2
    assert np.all(np.abs(q - 1) < 1e-5)
    out = y.copy()
    out[..., 0] = (c[:, None, None] * y[..., 0] + s[:, None, None] * y[..., 1]) / q[:, None, None]
    out[..., 1] = (-s[:, None, None] * y[..., 0] + c[:, None, None] * y[..., 1]) / q[:, None, None]
    return out, q


def shared_projection(design, target, state):
    """Exact least-squares projection with loss measured in world coordinates."""
    yaw, q = inverse_heading(target, state)
    coefficients = []
    ranks = []
    singular_values = []
    for axis in range(3):
        scale = np.sqrt(q) if axis < 2 else np.ones(len(q))
        x = (design * scale[:, None, None]).reshape(-1, 4)
        y = (yaw[..., axis] * scale[:, None, None]).reshape(-1, 30)
        coef, _, rank, sv = np.linalg.lstsq(x, y, rcond=1e-10)
        coefficients.append(coef)
        ranks.append(int(rank))
        singular_values.append(sv.tolist())
    coef = np.stack(coefficients, -1)
    return world_vector(np.einsum('ncd,dtj->nctj', design, coef), state), ranks, singular_values


def local_projection(u, target):
    predictions, ranks, conditions = [], [], []
    for x, y in zip(u, target):
        coef, _, rank, sv = np.linalg.lstsq(x, y.reshape(9, 90), rcond=1e-10)
        assert rank == 2
        predictions.append((x @ coef).reshape(9, 30, 3))
        ranks.append(int(rank))
        conditions.append(float(sv[0] / sv[-1]))
    return np.array(predictions), ranks, conditions


def state_mse(x):
    return (x**2).sum(-1).mean((1, 2))


def synthetic_check():
    rng = np.random.default_rng(20260908)
    u = centered(rng.normal(size=(5, 9, 2)))
    state = np.zeros((5, 12, 12))
    angle = np.linspace(-1, 1, 5)
    state[:, -1, 11], state[:, -1, 10] = np.cos(angle), np.sin(angle)
    c, s = state[:, -1, 11], state[:, -1, 10]
    uy = np.stack([c[:, None]*u[..., 0]+s[:, None]*u[..., 1],
                   -s[:, None]*u[..., 0]+c[:, None]*u[..., 1]], -1)
    design = np.concatenate([u, uy], -1)
    shared = world_vector(np.einsum('ncd,dtj->nctj', design, rng.normal(size=(4, 30, 3))), state)
    recovered, _, _ = shared_projection(design, shared, state)
    np.testing.assert_allclose(recovered, shared, atol=1e-11)
    target = shared + centered(rng.normal(size=shared.shape))
    ps, _, _ = shared_projection(design, target, state)
    pl, _, _ = local_projection(u, target)
    np.testing.assert_allclose(state_mse(target-ps).mean(),
        state_mse(target-pl).mean()+state_mse(pl-ps).mean(), atol=1e-11)
    return 'passed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=BASE/'attempt01')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    check = synthetic_check()
    sources = {}

    def record(path):
        digest = sha(path)
        key = str(path.relative_to(ROOT))
        if key in sources:
            assert sources[key] == digest
        sources[key] = digest
        return digest

    def read(path):
        record(path)
        return json.loads(path.read_text())

    for name in ['scripts/icra_response_capacity.py', 'scripts/icra_readout_structure.py',
                 'scripts/icra_adaptation_fit.py', 'scripts/icra_decision_evaluation.py',
                 'scripts/icra_decision_fit.py', 'scripts/icra_paired_calibration.py',
                 'scripts/icra_mpc_policy.py', 'src/winddyn/sim/aero.py',
                 'src/winddyn/sim/rotors.py', 'src/winddyn/sim/controller.py',
                 'runs/icra_response_capacity_20260908/PROTOCOL.md']:
        record(ROOT/name)
    all_rows, aggregates, containment = [], [], []
    for regime in ['mass_1p4', 'lag_3']:
        manifest = read(DATA/regime/'manifest.json')
        assert len(manifest) == 32
        assert [(m['parent_id'], m['split']) for m in manifest] == [
            (i, 'fit' if i < 24 else 'validation') for i in range(32)]
        states, tails_u, tails_p, designs, truths, rows = [], [], [], [], [], []
        for batch in range(4):
            for step in [600, 1200, 1800]:
                path = DATA/regime/f'batch{batch}_step{step}.npz'
                record(path)
                with np.load(path) as z:
                    a = {k: z[k] for k in z.files}
                assert a['state_hist'].shape == (8, 12, 12)
                arr = model_input(a)
                p = physics(arr, 'x_').reshape(8, 9, 90)
                ut = arr['x_action_fut'].astype(float).reshape(8, 9, 120)
                u = a['candidate_command'].astype(np.float32).astype(float)
                assert np.max(np.ptp(u[..., 2:], axis=1)) == 0
                ux = centered(u[..., :2])
                sf = a['state_hist'][:, -1]
                c, s = sf[:, 11], sf[:, 10]
                uy = np.stack([c[:, None]*ux[..., 0]+s[:, None]*ux[..., 1],
                               -s[:, None]*ux[..., 0]+c[:, None]*ux[..., 1]], -1)
                design = np.concatenate([ux, uy], -1)
                truth = centered(a['position_world'].transpose(2, 1, 0, 3).astype(float))
                assert truth.shape == (8, 9, 30, 3)
                for j in range(8):
                    rows.append(dict(regime=regime, parent_id=batch*8+j, snapshot=path.stem,
                        split='fit' if batch < 3 else 'validation',
                        prefix_alive=bool(a['initial_alive'][j]),
                        all_candidates_valid=bool(a['valid'][:, :, j].all()),
                        valid_candidate_records=int(a['valid'][:, :, j].sum())))
                for dest, value in [(states, a['state_hist']), (tails_u, ut), (tails_p, p),
                                    (designs, design), (truths, truth)]:
                    dest.append(value)
        state, ut, p, design, truth = [np.concatenate(v) for v in [states, tails_u, tails_p, designs, truths]]
        assert len(rows) == 96
        # Containment uses model predictions, never branch labels or encoders.
        for index, (method, seed) in enumerate(CONFIGS):
            folder = FIT/regime/f'config{index:02d}'
            fit_records = read(folder/'results.json')
            for objective in ['trajectory', 'decision']:
                selected = next(r['selected'] for r in fit_records['records'] if r['objective'] == objective)
                path = folder/selected['weight_file']
                assert record(path) == selected['weight_sha256']
                with np.load(path) as z:
                    fit = {k: z[k] for k in z.files}
                n_context = {'physics_features': 0, 'raw': 192, 'supervised': 128}[method]
                tail = p if method == 'physics_features' else np.concatenate([ut, p], -1)
                yaw = centered(p) + (centered(tail)/fit['std'][n_context:]) @ fit['coef'][n_context:-1]
                target = world_vector(yaw.reshape(96, 9, 30, 3), state)
                projected, ranks, _ = shared_projection(design, target, state)
                error = float(np.max(np.abs(projected-target)))
                assert error < 1e-7, (regime, method, seed, objective, error)
                containment.append(dict(regime=regime, method=method, seed=seed, objective=objective,
                    weight=str(path.relative_to(ROOT)), max_absolute_error=error, ranks=ranks))
        for split in ['fit', 'validation']:
            eligible = np.array([r['split'] == split and r['all_candidates_valid'] for r in rows])
            idx = np.flatnonzero(eligible)
            y, x, st = truth[eligible], design[eligible], state[eligible]
            shared, rank, sv = shared_projection(x, y, st)
            local, local_ranks, conditions = local_projection(x[..., :2], y)
            metrics = {'centered_truth_energy': state_mse(y),
                       'shared_affine_residual_mse': state_mse(y-shared),
                       'per_state_affine_residual_mse': state_mse(y-local),
                       'extra_state_dependent_affine_mse': state_mse(local-shared)}
            values = {k: float(v.mean()) for k, v in metrics.items()}
            identity_error = abs(values['shared_affine_residual_mse'] -
                values['per_state_affine_residual_mse'] - values['extra_state_dependent_affine_mse'])
            np.testing.assert_allclose(values['shared_affine_residual_mse'],
                values['per_state_affine_residual_mse']+values['extra_state_dependent_affine_mse'],
                atol=1e-10, rtol=1e-7)
            for position, original_index in enumerate(idx):
                rows[original_index].update({k: float(v[position]) for k, v in metrics.items()})
                rows[original_index].update(command_design_rank=local_ranks[position],
                                           command_design_condition=conditions[position])
            own = [r for r in rows if r['split'] == split]
            aggregates.append(dict(regime=regime, split=split, interpretation='truth-informed in-split projection',
                total_states=len(own), parents=len({r['parent_id'] for r in own}),
                prefix_alive=sum(r['prefix_alive'] for r in own), eligible_states=len(idx),
                global_ranks=rank, global_singular_values=sv, max_command_condition=max(conditions),
                identity_absolute_error=identity_error,
                extra_state_dependent_fraction=values['extra_state_dependent_affine_mse']/values['shared_affine_residual_mse'],
                means=values))
        all_rows.extend(rows)
    assert len(all_rows) == 192 and len(containment) == 28 and len(aggregates) == 4
    assert [a['eligible_states'] for a in aggregates] == [62, 24, 69, 24]
    result = dict(status='accepted', stage='posthoc_oracle_capacity_audit',
        synthetic_check=check, states=192, eligible_states=179, selected_heads_contained=28,
        max_containment_error=max(r['max_absolute_error'] for r in containment),
        encoder_checkpoints_loaded=False, new_simulator_rollouts=0, new_deployable_predictors=0,
        aggregates=aggregates, containment=containment)
    for name, value in [('summary.json', result), ('rows.json', all_rows), ('sources.json', sources)]:
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'containment'}, indent=2))


if __name__ == '__main__':
    main()
