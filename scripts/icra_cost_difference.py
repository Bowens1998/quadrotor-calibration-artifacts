"""Exact cost-difference accounting from previously accepted frozen forecasts."""
import json
from pathlib import Path
import numpy as np
from _common import ROOT
from icra_adaptation_fit import sha

BASE = ROOT/'runs/icra_cost_difference_20260908'
INPUT = ROOT/'runs/icra_response_choice_20260908/attempt01'
SNAP = ROOT/'runs/icra_snapshot_pilot_20260908'
COMMONS = ['physics_features', 'raw']
MODES = ['native', 'shared_affine', 'current']
PAIRS = np.array([(i, j) for i in range(9) for j in range(i+1, 9)])


def full_cost(world, a):
    u = a['candidate_command'][:, :, :3]
    tracking = np.mean(np.sum((world-a['reference_world'][:, None])**2, axis=-1), axis=-1)
    return tracking+.02*np.sum(u**2, axis=-1)+.05*np.sum((u-a['previous_command'][:, None, :3])**2, axis=-1)


def terms(pred, truth, reference):
    mu, mu_hat = truth.mean(1, keepdims=True), pred.mean(1, keepdims=True)
    d, q = truth-mu, pred-mu_hat
    e0, eps = mu_hat-mu, q-d
    dot = lambda a, b: np.mean(np.sum(a*b, axis=-1), axis=-1)
    return np.stack([2*dot(e0, d), 2*dot(mu_hat-reference[:, None], eps), dot(q, q)-dot(d, d)], axis=-1)


def main():
    out = BASE/'attempt01'; out.mkdir(parents=True, exist_ok=False)
    sources = {}
    def record(path):
        h = sha(path); rel = str(path.relative_to(ROOT))
        assert rel not in sources or sources[rel] == h
        sources[rel] = h
        return h
    def read(path):
        record(path)
        return json.loads(path.read_text())
    accepted = read(INPUT/'accepted.json')
    assert accepted['status'] == 'accepted'
    for rel, h in read(INPUT/'sources.json').items():
        assert record(ROOT/rel) == h
    for rel, h in accepted['output_hashes'].items():
        assert record(INPUT/rel) == h
    record(Path(__file__).resolve()); record(BASE/'PROTOCOL.md')
    original = read(INPUT/'summary.json')
    old_rows = read(INPUT/'rows.json')
    key = lambda r: (r['regime'], r['common_source'], r['response'], r['snapshot'], r['task_index'])
    old_rows = {key(r): r for r in old_rows}
    rows, files, moments = [], [], {}
    max_pair_error = 0.; max_selected_error = 0.; max_choice_metric_error = 0.
    for file in original['component_files']:
        regime, name = file['regime'], file['snapshot']
        p = INPUT/file['file']; assert record(p) == file['sha256']
        with np.load(p) as z:
            native, response = z['native_world'], z['response_world']
        p = SNAP/regime/(name+'.npz'); record(p)
        with np.load(p) as z:
            a = dict(z)
        truth = a['position_world'].transpose(2, 1, 0, 3).astype(float)
        valid = a['valid'].transpose(2, 1, 0)
        full = valid.all(axis=(1, 2))
        tc = full_cost(truth, a)
        best = tc.argmin(1)
        all_terms = np.empty((2, 3, 12, 36, 3))
        all_costs = np.empty((2, 3, 12, 9))
        for i, common in enumerate(COMMONS):
            mean = native[i].mean(1, keepdims=True)
            for k, mode in enumerate(MODES):
                pred = native[i] if mode == 'native' else mean+response[k-1]
                pc = full_cost(pred, a); chosen = pc.argmin(1)
                action_terms = terms(pred, truth, a['reference_world'])
                pair_terms = action_terms[:, PAIRS[:, 0]]-action_terms[:, PAIRS[:, 1]]
                direct = (pc-tc)[:, PAIRS[:, 0]]-(pc-tc)[:, PAIRS[:, 1]]
                np.testing.assert_allclose(pair_terms.sum(-1), direct, rtol=1e-9, atol=1e-11)
                max_pair_error = max(max_pair_error, float(np.max(np.abs(pair_terms.sum(-1)-direct))))
                all_terms[i, k], all_costs[i, k] = pair_terms, pc
                group = moments.setdefault((regime, common, mode), [])
                group.append(pair_terms[full].reshape(-1, 3))
                for j in range(12):
                    old = old_rows[regime, common, mode, name, j]
                    assert int(chosen[j]) == old['choice']
                    alive = bool(a['initial_alive'][j])
                    contact = bool(not valid[j, chosen[j], -1]) if alive else None
                    assert contact == old['selected_contact'] and alive == old['prefix_alive']
                    assert bool(full[j]) == old['all_candidates_valid']
                    row = dict(regime=regime, common_source=common, response=mode, snapshot=name,
                        task_index=j, all_candidates_valid=bool(full[j]), prefix_alive=alive,
                        choice=int(chosen[j]), selected_contact=contact)
                    if full[j]:
                        margin = float(pc[j, chosen[j]]-pc[j, best[j]])
                        regret = float(tc[j, chosen[j]]-tc[j, best[j]])
                        delta = action_terms[j, chosen[j]]-action_terms[j, best[j]]
                        correct = bool(chosen[j] == best[j])
                        assert correct == old['correct'] and margin <= 1e-12 and regret >= -1e-12
                        np.testing.assert_allclose(regret, old['regret'], atol=1e-11, rtol=1e-8)
                        np.testing.assert_allclose(delta.sum(), margin-regret, atol=1e-11, rtol=1e-9)
                        max_choice_metric_error = max(max_choice_metric_error, abs(regret-old['regret']))
                        max_selected_error = max(max_selected_error, abs(float(delta.sum())-margin+regret))
                        row.update(true_best=int(best[j]), correct=correct, regret=regret,
                            selected_predicted_cost_gap=margin,
                            selected_pair_terms=delta.tolist(),
                            pair_second_moment=(pair_terms[j].T@pair_terms[j]/36).tolist())
                    rows.append(row)
        p = out/f'{regime}_{name}_pairs.npz'
        np.savez_compressed(p, pair_indices=PAIRS, terms=all_terms, predicted_cost=all_costs,
            true_cost=tc, all_candidates_valid=full, initial_alive=a['initial_alive'])
        files.append(dict(regime=regime, snapshot=name, file=p.name, sha256=sha(p)))
    means = []
    for group, arrays in moments.items():
        t = np.concatenate(arrays); second = t.T@t/len(t)
        total = float(np.mean(t.sum(-1)**2))
        np.testing.assert_allclose(second.sum(), total, atol=1e-12, rtol=1e-9)
        rr = [r for r in rows if r['all_candidates_valid'] and (r['regime'], r['common_source'], r['response']) == group]
        n = {'mass_1p4': 98, 'lag_3': 106}[group[0]]
        assert len(rr) == n and len(t) == n*36
        means.append(dict(regime=group[0], common_source=group[1], response=group[2],
            eligible_states=n, pair_records=len(t), accuracy=float(np.mean([r['correct'] for r in rr])),
            regret=float(np.mean([r['regret'] for r in rr])),
            pair_cost_error_rms=float(np.sqrt(total)), term_rms=np.sqrt(np.diag(second)).tolist(),
            signed_second_moment=second.tolist()))
    assert len(rows) == 1296 and len(means) == 12
    assert sum(r['pair_records'] for r in means) == 44064
    result = dict(status='verified_by_runner', stage='posthoc_cost_difference_accounting',
        unique_states=216, eligible_unique_states=204, state_configuration_records=1296,
        eligible_state_configuration_records=1224, eligible_pair_configuration_records=44064,
        component_terms=['common_error_interaction', 'response_alignment', 'response_energy'],
        max_pair_identity_error=max_pair_error, max_selected_identity_error=max_selected_error,
        max_original_regret_difference=max_choice_metric_error,
        no_new_fits_or_rollouts=True, means=means, pair_files=files)
    for name, value in [('summary.json', result), ('rows.json', rows), ('sources.json', sources)]:
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['means', 'pair_files']}, indent=2))
    for r in means:
        print(r['regime'], r['common_source'], r['response'], r['pair_cost_error_rms'], r['term_rms'])


if __name__ == '__main__':
    main()
