"""Unfitted audit of candidate contrasts in the deployed affine readout."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import ROOT
from icra_adaptation_fit import physics, sha
from icra_adaptation_ridge import predict
from icra_decision_evaluation import CONFIGS as EVAL_CONFIGS, model_input
from icra_decision_fit import CONFIGS as FIT_CONFIGS

BASE = ROOT / 'runs/icra_readout_structure_20260908'
FIT = ROOT / 'runs/icra_decision_fit_20260908'
EVAL = ROOT / 'runs/icra_decision_evaluation_20260908'
SNAPSHOTS = ROOT / 'runs/icra_snapshot_pilot_20260908'


def centered(x):
    return x - x.mean(axis=1, keepdims=True)


def world_vector(x, state):
    c, s = state[:, -1, 11], state[:, -1, 10]
    out = x.copy()
    out[..., 0] = c[:, None, None] * x[..., 0] - s[:, None, None] * x[..., 1]
    out[..., 1] = s[:, None, None] * x[..., 0] + c[:, None, None] * x[..., 1]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE / 'attempt01')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def record(path):
        key = str(path.relative_to(ROOT))
        digest = sha(path)
        if key in sources:
            assert sources[key] == digest
        sources[key] = digest
        return digest

    def read(path):
        record(path)
        return json.loads(path.read_text())

    for name in [
        'scripts/icra_readout_structure.py', 'scripts/icra_adaptation_fit.py',
        'scripts/icra_adaptation_ridge.py', 'scripts/icra_decision_evaluation.py',
        'scripts/icra_decision_fit.py', 'scripts/icra_mpc_matrix_policy.py',
        'scripts/icra_mpc_policy.py', 'src/winddyn/models/wm.py',
        'runs/icra_readout_structure_20260908/PROTOCOL.md',
    ]:
        record(ROOT / name)
    assert read(FIT / 'accepted.json')['accepted_fits'] == 84
    heads = []
    per_state = []
    for regime in ['mass_1p4', 'lag_3']:
        paths = sorted((SNAPSHOTS / regime).glob('*.npz'))
        assert len(paths) == 9
        inputs = {}
        for path in paths:
            record(path)
            with np.load(path) as z:
                a = {k: z[k] for k in z.files}
            assert a['state_hist'].shape == (12, 12, 12)
            assert a['candidate_command'].shape == (12, 9, 4)
            arr = model_input(a)
            p = physics(arr, 'x_').reshape(12, 9, 90)
            u = arr['x_action_fut'].astype(np.float64).reshape(12, 9, 120)
            truth = a['position_world'].transpose(2, 1, 0, 3)
            assert p.shape == (12, 9, 90) and truth.shape == (12, 9, 30, 3)
            inputs[path.stem] = (a, p, u, centered(truth))
        for fit_index, (method, seed) in enumerate(FIT_CONFIGS):
            folder = FIT / regime / f'config{fit_index:02d}'
            selected_records = read(folder / 'results.json')
            assert (selected_records['regime'], selected_records['method'],
                    selected_records['seed']) == (regime, method, seed)
            for objective in ['trajectory', 'decision']:
                eval_index = EVAL_CONFIGS.index((method, seed, objective))
                evaluation = read(EVAL / regime / f'config{eval_index:03d}.json')
                assert (evaluation['regime'], evaluation['method'],
                        evaluation['model_seed'], evaluation['objective']) == (
                            regime, method, seed, objective)
                selected = next(r['selected'] for r in selected_records['records']
                                if r['objective'] == objective)
                weight = folder / selected['weight_file']
                digest = record(weight)
                assert digest == selected['weight_sha256']
                assert digest == evaluation['predictor_provenance'][str(weight.relative_to(ROOT))]
                with np.load(weight) as z:
                    fit = {k: z[k] for k in z.files}
                n_context = {'physics_features': 0, 'raw': 192, 'supervised': 128}[method]
                n_action = 90 if method == 'physics_features' else 210
                n_features = n_context + n_action
                assert fit['coef'].shape == (n_features + 1, 90)
                assert fit['mean'].shape == fit['std'].shape == (n_features,)
                assert all(np.isfinite(fit[k]).all() for k in ['mean', 'std', 'coef'])
                assert (fit['std'] > 0).all()
                rows = {(r['snapshot'], r['task_index']): r for r in evaluation['rows']}
                assert len(rows) == len(evaluation['rows']) == 108
                assert set(rows) == {(name, j) for name in inputs for j in range(12)}
                head_rows = []
                affine_errors = []
                perturbation_errors = []
                for name, (a, p, u, truth_c) in inputs.items():
                    tail = p if method == 'physics_features' else np.concatenate([u, p], -1)
                    tail_c = centered(tail)
                    w = fit['coef'][n_context:-1]
                    pred_c = centered(p) + (tail_c / fit['std'][n_context:]) @ w
                    assert np.isfinite(pred_c).all()

                    # Functional check with full affine code and actual weights.
                    # Synthetic shared values are not encoded observations.
                    def synthetic_prediction(offset):
                        x = np.empty((12, 9, n_features), dtype=np.float64)
                        x[..., n_context:] = tail
                        if n_context:
                            values = np.linspace(-.1, .1, n_context)[None, :] + np.arange(12)[:, None] / 100
                            x[..., :n_context] = (fit['mean'][:n_context] +
                                fit['std'][:n_context] * (values + offset))[:, None, :]
                        return predict(x, fit) + p

                    full = synthetic_prediction(0)
                    shifted = synthetic_prediction(.25)
                    numerical_error = float(np.max(np.abs(centered(full) - pred_c)))
                    perturbation_error = float(np.max(np.abs(centered(shifted) - centered(full))))
                    np.testing.assert_allclose(centered(full), pred_c, atol=1e-10, rtol=1e-8)
                    np.testing.assert_allclose(centered(shifted), centered(full), atol=1e-10, rtol=1e-8)
                    affine_errors.append(numerical_error)
                    perturbation_errors.append(perturbation_error)
                    wc = world_vector(pred_c.reshape(12, 9, 30, 3), a['state_hist'])
                    contrast_error = wc - truth_c
                    contrast_mse = (contrast_error ** 2).sum(-1).mean((1, 2))
                    valid = a['valid'].transpose(2, 1, 0)
                    for j in range(12):
                        saved = rows[(name, j)]
                        full_valid = bool(valid[j].all())
                        prefix = bool(a['initial_alive'][j])
                        assert saved['all_candidates_valid'] == full_valid
                        assert saved['prefix_alive'] == prefix
                        assert saved['valid_candidate_records'] == int(valid[j].sum())
                        row = dict(regime=regime, method=method, model_seed=seed,
                                   objective=objective, snapshot=name, task_index=j,
                                   prefix_alive=prefix, all_candidates_valid=full_valid,
                                   valid_candidate_records=int(valid[j].sum()))
                        if full_valid:
                            value = float(contrast_mse[j])
                            np.testing.assert_allclose(value, saved['contrast_mse'], atol=1e-8, rtol=1e-6)
                            row.update(reconstructed_contrast_mse=value,
                                       saved_contrast_mse=saved['contrast_mse'],
                                       absolute_error=abs(value - saved['contrast_mse']),
                                       saved_common_mse=saved['common_mse'],
                                       saved_total_mse=saved['total_mse'],
                                       saved_regret=saved['regret'])
                        head_rows.append(row)
                eligible = [r for r in head_rows if r['all_candidates_valid']]
                assert len(eligible) == {'mass_1p4': 98, 'lag_3': 106}[regime]
                heads.append(dict(regime=regime, method=method, model_seed=seed,
                    objective=objective, weight=str(weight.relative_to(ROOT)), weight_sha256=digest,
                    feature_count=n_features, shared_history_feature_count=n_context,
                    candidate_varying_feature_count=n_action, states=len(head_rows),
                    prefix_alive=sum(r['prefix_alive'] for r in head_rows),
                    all_candidates_valid=len(eligible),
                    max_saved_metric_error=max(r['absolute_error'] for r in eligible),
                    max_full_affine_error=max(affine_errors),
                    max_synthetic_context_shift_error=max(perturbation_errors),
                    means={k: float(np.mean([r[k] for r in eligible])) for k in [
                        'reconstructed_contrast_mse', 'saved_common_mse',
                        'saved_total_mse', 'saved_regret']}))
                per_state.extend(head_rows)
    assert len(heads) == 28 and len(per_state) == 3024
    assert sum(r['all_candidates_valid'] for r in per_state) == 2856
    groups = defaultdict(list)
    for h in heads:
        groups[(h['regime'], h['method'], h['objective'])].append(h)
    aggregates = []
    for (regime, method, objective), values in groups.items():
        aggregates.append(dict(regime=regime, method=method, objective=objective,
            heads=len(values), unique_states_per_head=108,
            all_candidates_valid_per_head=values[0]['all_candidates_valid'],
            means={k: float(np.mean([h['means'][k] for h in values])) for k in values[0]['means']}))
    assert len(aggregates) == 12
    result = dict(status='accepted', stage='posthoc_unfitted_development_audit',
        selected_heads=len(heads), unique_states=216, state_head_records=len(per_state),
        all_valid_state_head_records=2856,
        all_valid_unique_states=204,
        max_saved_metric_error=max(h['max_saved_metric_error'] for h in heads),
        max_full_affine_error=max(h['max_full_affine_error'] for h in heads),
        max_synthetic_context_shift_error=max(h['max_synthetic_context_shift_error'] for h in heads),
        encoder_checkpoints_loaded=False, new_fits=0, new_rollouts=0,
        heads=heads, aggregates=aggregates)
    for filename, value in [('summary.json', result), ('rows.json', per_state), ('sources.json', sources)]:
        (args.output / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['heads', 'aggregates']}, indent=2))


if __name__ == '__main__':
    main()
