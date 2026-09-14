"""FIT-group-selected conditional response probes; no new controller or rollouts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from _common import ROOT
from icra_adaptation_fit import physics, sha
from icra_readout_structure import centered, world_vector
from icra_response_capacity import inverse_heading, state_mse
from revision_closedloop import Estimators
from winddyn.utils.config import load_vehicle

BASE = ROOT/'runs/icra_history_response_20260908'
DATA = ROOT/'runs/icra_paired_calibration_20260908'
METHODS = ['nominal_physics', 'shared_affine', 'current', 'current_quadratic8',
           'current_observer', 'current_history8']
LAMBDAS = [1e-4, 1e-2, 1.]
INPUT_KEYS = ['state_hist', 'action_hist', 'candidate_command', 'previous_command', 'initial_velocity']


def response_inputs(a, observer):
    # No truth, validity, reference, field/plant identity or depth enters features.
    a = {k: a[k] for k in INPUT_KEYS}
    st = a['state_hist'].astype(float)
    n = len(st)
    cmd = a['candidate_command'].astype(np.float32).astype(float)
    prev = a['previous_command'].astype(float)
    np.testing.assert_allclose(a['action_hist'][:, -1], prev, atol=1e-7)
    assert np.max(np.ptp(cmd[..., 2:], axis=1)) == 0
    cur = np.concatenate([st[:, -1], prev, cmd[:, 4, :3], cmd[..., :2].mean(1)], -1)
    hist = np.concatenate([(st[:, :-1]-st[:, -1:]).reshape(n, -1),
                           (a['action_hist'].astype(float)-prev[:, None]).reshape(n, -1)], -1)
    assert cur.shape == (n, 21) and hist.shape == (n, 180)
    c, s = st[:, -1, 11], st[:, -1, 10]
    ux = centered(cmd[..., :2])
    uy = np.stack([c[:, None]*ux[..., 0]+s[:, None]*ux[..., 1],
                  -s[:, None]*ux[..., 0]+c[:, None]*ux[..., 1]], -1)
    v = a['initial_velocity'].astype(float).copy()
    vx, vy = v[:, 0].copy(), v[:, 1].copy()
    v[:, 0], v[:, 1] = c*vx+s*vy, -s*vx+c*vy
    arr = {'x_state_hist': np.repeat(a['state_hist'], 9, 0),
           'x_action_fut': np.repeat(cmd.reshape(n*9, 4)[:, None], 30, 1),
           'x_vel_yaw_t': np.repeat(v, 9, 0)}
    p = centered(physics(arr, 'x_').reshape(n, 9, 30, 3))
    buf = SimpleNamespace(state=[torch.tensor(a['state_hist'][:, i]) for i in range(12)], ready=lambda: True)
    obs = observer.estimate('observer', buf, torch.zeros(n, 2)).clamp(-15, 15).numpy().astype(float)
    out = dict(current=cur, history=hist, observer=obs, state=st,
               design=np.concatenate([ux, uy], -1), nominal_yaw=p)
    assert all(np.isfinite(v).all() for v in out.values())
    return out


def parent_weights(parents):
    unique, counts = np.unique(parents, return_counts=True)
    lookup = dict(zip(unique, counts))
    w = np.array([1/lookup[p] for p in parents], dtype=float)
    return w/w.sum()


def weighted_stats(x, weights):
    mean = np.einsum('n,nd->d', weights, x)
    std = np.sqrt(np.einsum('n,nd->d', weights, (x-mean)**2))
    return mean, np.maximum(std, 1e-5)


def extra_source(data, current_normalized, method):
    if method == 'current_history8':
        return data['history']
    if method == 'current_observer':
        return data['observer']
    if method == 'current_quadratic8':
        return np.stack([current_normalized[:, i]*current_normalized[:, j]
                         for i in range(21) for j in range(i, 21)], -1)
    raise ValueError(method)


def fit_transform(data, idx, weights, method):
    params = {'method': np.array(method)}
    if method != 'shared_affine':
        params['current_mean'], params['current_std'] = weighted_stats(data['current'][idx], weights)
        current = (data['current']-params['current_mean'])/params['current_std']
        if method != 'current':
            source = extra_source(data, current, method)
            params['extra_mean'], params['extra_std'] = weighted_stats(source[idx], weights)
            norm = (source-params['extra_mean'])/params['extra_std']
            if method.endswith('8'):
                _, singular, vectors = np.linalg.svd(norm[idx]*np.sqrt(weights)[:, None], full_matrices=False)
                params['projection'] = vectors[:8].T
                params['pca_singular_values'] = singular
                score = norm @ params['projection']
                params['score_mean'], params['score_std'] = weighted_stats(score[idx], weights)
    return params


def design_matrix(data, params):
    method = str(params['method'])
    parts = [data['design']]
    if method != 'shared_affine':
        current = (data['current']-params['current_mean'])/params['current_std']
        context = [current]
        if method != 'current':
            source = extra_source(data, current, method)
            extra = (source-params['extra_mean'])/params['extra_std']
            if method.endswith('8'):
                extra = (extra @ params['projection']-params['score_mean'])/params['score_std']
            context.append(extra)
        context = np.concatenate(context, -1)
        for axis in range(2):
            parts.append(data['design'][..., axis, None]*context[:, None])
    return np.concatenate(parts, -1)


def fit_model(data, idx, method, lam):
    weights = parent_weights(data['parent'][idx])
    params = fit_transform(data, idx, weights, method)
    x = design_matrix(data, params)[idx]
    scale = np.sqrt(np.einsum('n,ncd->d', weights, x**2)/9)
    params['design_scale'] = np.maximum(scale, 1e-5)
    x = x/params['design_scale']
    target = (data['truth_yaw']-data['nominal_yaw'])[idx]
    sf = data['state'][idx, -1]
    q = sf[:, 11]**2+sf[:, 10]**2
    coefficients = []
    for axis in range(3):
        row_weights = np.repeat(weights*(q if axis < 2 else 1)/9, 9)
        xx = x.reshape(-1, x.shape[-1])
        y = target[..., axis].reshape(-1, 30)
        gram = xx.T @ (row_weights[:, None]*xx)
        rhs = xx.T @ (row_weights[:, None]*y)
        coefficients.append(np.linalg.solve(gram+lam*np.eye(len(gram)), rhs))
    params['coef'] = np.stack(coefficients, -1)
    params['lambda'] = np.array(lam)
    assert all(np.isfinite(v).all() for k,v in params.items() if k != 'method')
    return params


def predict_model(data, params=None):
    yaw = data['nominal_yaw'].copy()
    if params is not None:
        x = design_matrix(data, params)/params['design_scale']
        yaw += np.einsum('ncd,dtj->nctj', x, params['coef'])
    out = world_vector(yaw, data['state'])
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.mean(1), 0, atol=1e-8)
    return out


def summarize(errors, parents):
    values = [dict(parent_id=int(p), states=int(np.sum(parents == p)),
                   mse=float(errors[parents == p].mean())) for p in np.unique(parents)]
    return dict(parent_macro_mse=float(np.mean([v['mse'] for v in values])),
                state_micro_mse=float(errors.mean()), states=len(errors),
                parents=len(values), per_parent=values)


def synthetic_checks():
    rng = np.random.default_rng(20260908)
    n = 48
    u = centered(rng.normal(size=(n, 9, 2)))
    state = np.zeros((n, 12, 12)); state[:, -1, 11] = 1
    hidden = rng.normal(size=(n, 8))
    data = dict(current=rng.normal(size=(n, 21)), history=hidden @ rng.normal(size=(8, 180)),
                observer=rng.normal(size=(n, 2)), state=state,
                design=np.concatenate([u, u], -1), nominal_yaw=np.zeros((n, 9, 30, 3)),
                parent=np.arange(n))
    true = u[..., 0, None, None]*hidden[:, None, 0, None, None]*rng.normal(size=(1, 1, 30, 3))*.01
    data['truth_yaw'] = true
    train = np.arange(36); val = np.arange(36, 48)
    hp = fit_model(data, train, 'current_history8', 1e-8)
    cp = fit_model(data, train, 'current', 1e-8)
    he = state_mse(predict_model(data, hp)[val]-true[val]).mean()
    ce = state_mse(predict_model(data, cp)[val]-true[val]).mean()
    assert he < 1e-10 and ce > 100*max(he, 1e-12), (he, ce)
    other = dict(data); other['truth_yaw'] = data['truth_yaw'].copy(); other['truth_yaw'][val] = np.nan
    hp2 = fit_model(other, train, 'current_history8', 1e-8)
    for key in hp:
        np.testing.assert_array_equal(hp[key], hp2[key])
    return dict(history_interaction_recovery_mse=float(he), current_only_mse=float(ce),
                validation_label_perturbation_fit_invariant=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE/'attempt01')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    checks = synthetic_checks()
    sources = {}
    def record(path):
        rel = str(path.relative_to(ROOT)); value = sha(path)
        assert rel not in sources or sources[rel] == value
        sources[rel] = value
        return value
    def read(path):
        record(path); return json.loads(path.read_text())
    previous = read(ROOT/'runs/icra_response_capacity_20260908/attempt01/sources.json')
    for rel, expected in previous.items():
        assert record(ROOT/rel) == expected
    for rel in ['scripts/icra_history_response.py', 'runs/icra_history_response_20260908/PROTOCOL.md',
                'scripts/revision_closedloop.py', 'scripts/closed_loop_eval.py',
                'src/winddyn/revision/protocol.py', 'src/winddyn/utils/config.py',
                'configs/robot/starling_2_max.yaml']:
        record(ROOT/rel)
    observer = Estimators('observer', 0, 'cpu', load_vehicle())
    datasets, selections, fitted, all_rows = {}, [], {}, []
    for regime in ['mass_1p4', 'lag_3']:
        manifest = read(DATA/regime/'manifest.json')
        assert [(m['parent_id'], m['split']) for m in manifest] == [
            (i, 'fit' if i < 24 else 'validation') for i in range(32)]
        batches = defaultdict(list); records = []
        for batch in range(4):
            for step in [600, 1200, 1800]:
                path = DATA/regime/f'batch{batch}_step{step}.npz'; record(path)
                with np.load(path) as z:
                    a = {k:z[k] for k in z.files}
                inputs = response_inputs(a, observer)
                forbidden = dict(a)
                for key in set(a)-set(INPUT_KEYS):
                    forbidden[key] = np.full(a[key].shape, np.nan)
                inputs2 = response_inputs(forbidden, observer)
                for key in inputs:
                    np.testing.assert_array_equal(inputs[key], inputs2[key])
                truth = centered(a['position_world'].transpose(2,1,0,3).astype(float))
                yaw, _ = inverse_heading(truth, inputs['state'])
                for key, value in dict(inputs, truth_world=truth, truth_yaw=yaw).items():
                    batches[key].append(value)
                for j in range(8):
                    records.append(dict(regime=regime, snapshot=path.stem, parent_id=batch*8+j,
                        split='fit' if batch < 3 else 'validation',
                        prefix_alive=bool(a['initial_alive'][j]),
                        all_candidates_valid=bool(a['valid'][:,:,j].all()),
                        valid_candidate_records=int(a['valid'][:,:,j].sum())))
        data = {k:np.concatenate(v) for k,v in batches.items()}
        data['parent'] = np.array([r['parent_id'] for r in records])
        data['valid'] = np.array([r['all_candidates_valid'] for r in records])
        fit_idx = np.flatnonzero((data['parent'] < 24)&data['valid'])
        val_idx = np.flatnonzero((data['parent'] >= 24)&data['valid'])
        assert len(fit_idx) == {'mass_1p4':62, 'lag_3':69}[regime] and len(val_idx) == 24
        data['records'] = records; datasets[regime] = data
        for method in METHODS[1:]:
            candidates = []
            for lam in LAMBDAS:
                oof = np.full(96, np.nan); folds = []
                for fold in range(6):
                    tr = fit_idx[data['parent'][fit_idx]%6 != fold]
                    va = fit_idx[data['parent'][fit_idx]%6 == fold]
                    trp, vap = set(data['parent'][tr]), set(data['parent'][va])
                    assert trp.isdisjoint(vap) and max(trp|vap) < 24
                    model = fit_model(data, tr, method, lam)
                    pred = predict_model(data, model)
                    e = state_mse(pred[va]-data['truth_world'][va]); oof[va] = e
                    folds.append(dict(fold=fold, train_parents=sorted(map(int,trp)),
                        held_parents=sorted(map(int,vap)), score=summarize(e,data['parent'][va])))
                assert np.isfinite(oof[fit_idx]).all() and np.isnan(oof[val_idx]).all()
                candidates.append(dict(lambda_=lam, score=summarize(oof[fit_idx],data['parent'][fit_idx]), folds=folds))
            selected = min(candidates,key=lambda c:(c['score']['parent_macro_mse'],-c['lambda_']))
            model = fit_model(data, fit_idx, method, selected['lambda_'])
            path = args.output/f'{regime}_{method}.npz'; np.savez_compressed(path, **model)
            with np.load(path) as z:
                reloaded = {k:z[k] for k in z.files}
            np.testing.assert_allclose(predict_model(data,model),predict_model(data,reloaded),rtol=0,atol=0)
            fitted[regime,method] = reloaded
            selections.append(dict(regime=regime,method=method,candidates=candidates,
                selected_lambda=selected['lambda_'], selected_cv_mse=selected['score']['parent_macro_mse'],
                weight_file=path.name,weight_sha256=sha(path),design_columns=int(model['coef'].shape[0]),
                fit_parents=sorted(map(int,set(data['parent'][fit_idx])))))
    # Commit all choices before validation labels are used for scoring any model.
    selection_path = args.output/'selection.json'
    selection_path.write_text(json.dumps(selections,indent=2,allow_nan=False)+'\n')
    selection_hash = sha(selection_path)
    summaries = []
    for regime, data in datasets.items():
        for method in METHODS:
            model = None if method == 'nominal_physics' else fitted[regime,method]
            pred = predict_model(data,model)
            error = state_mse(pred-data['truth_world'])
            for i, r in enumerate(data['records']):
                row = dict(r,method=method)
                if r['all_candidates_valid']:
                    row['contrast_mse'] = float(error[i])
                all_rows.append(row)
            for split in ['fit','validation']:
                mask = data['valid'] & ((data['parent']<24) if split=='fit' else (data['parent']>=24))
                summaries.append(dict(regime=regime,method=method,split=split,
                    score=summarize(error[mask],data['parent'][mask])))
    def validation(method):
        return next(s['score'] for s in summaries if s['regime']=='lag_3' and s['method']==method and s['split']=='validation')
    h = validation('current_history8'); contrasts = []
    for method in ['shared_affine','current','current_quadratic8','current_observer']:
        comp = validation(method)
        hm = {r['parent_id']:r['mse'] for r in h['per_parent']}
        cm = {r['parent_id']:r['mse'] for r in comp['per_parent']}
        assert set(hm)==set(cm)==set(range(24,32))
        reduction = 1-h['parent_macro_mse']/comp['parent_macro_mse']
        wins = sum(hm[p] < cm[p] for p in hm)
        contrasts.append(dict(comparator=method, relative_mse_reduction=reduction,
            strictly_better_parents=wins,total_parents=8,passes=reduction>=.10 and wins>=6))
    history_support = all(c['passes'] for c in contrasts[:3])
    checks.update(input_whitelist_invariant=True, source_hashes_verified=len(sources),
                  saved_weight_reload_exact=True, fit_parent_groups_disjoint=True,
                  selection_saved_before_validation_scoring=True)
    assert len(summaries)==24 and len(all_rows)==1152
    assert sum(r['all_candidates_valid'] for r in all_rows)==1074
    assert sha(selection_path)==selection_hash
    result = dict(status='accepted',stage='posthoc_development_response_probe',checks=checks,
        configurations=12,selected_fits=10,cv_fits=180,state_method_records=1152,
        eligible_state_method_records=1074,unique_states=192,eligible_unique_states=179,
        selection_sha256=selection_hash, summaries=summaries,
        gate=dict(primary_regime='lag_3', contrasts=contrasts,
                  motion_history_support=history_support,
                  beyond_observer_support=history_support and contrasts[3]['passes'],
                  statistical_confirmation=False,control_benefit_evaluated=False))
    for name,value in [('summary.json',result),('rows.json',all_rows),('sources.json',sources)]:
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['gate'],indent=2))
    for s in summaries:
        print(s['regime'],s['method'],s['split'],s['score']['parent_macro_mse'])


if __name__=='__main__':
    main()
