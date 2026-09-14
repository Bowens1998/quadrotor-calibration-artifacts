"""Fixed choices and fixed 16-contrast family for continuation sensitivity."""
import argparse
import json
from pathlib import Path
import numpy as np
from icra_action_continuation import ROOT, BASE, OLD, CHOICES, REGIMES, TIERS, sha, write, load

MODES = ['held', 'nominal', 'observer']
METHODS = ['frozen', 'scalar', 'physics_features', 'raw', 'raw_depth', 'adapt_random', 'adapt_pretrained']
METRICS = ['regret', 'accuracy', 'selected_contact', 'own_regret', 'own_accuracy']


def mean(values):
    good = [x for x in values if x is not None]
    return float(np.mean(good)) if good else None


def snapshot_truth(run, reg, name):
    a = load(OLD/reg/f'{name}.npz')
    penalty = .02*np.square(a['candidate_command'][..., :3].astype(float)).sum(-1)
    penalty += .05*np.square(a['candidate_command'][..., :3].astype(float)-a['previous_command'][:, None, :3]).sum(-1)
    truths = {}
    for mode in MODES:
        d = a if mode == 'held' else load(run/reg/f'{name}_{mode}.npz')
        p = d['position_world'].astype(float).transpose(2, 1, 0, 3)
        valid = d['valid'].transpose(2, 1, 0)
        cost = np.square(p-a['reference_world'][:, None]).sum(-1).mean(-1)+penalty
        truths[mode] = dict(cost=cost, all_valid=valid.all((1, 2)),
                            selected_contact=~valid[:, :, -1], oracle=cost.argmin(1))
    common = np.logical_and.reduce([x['all_valid'] for x in truths.values()])
    return dict(modes=truths, common=common, alive=a['initial_alive'])


def summarize(rows):
    return dict(states=len(rows), common_valid=sum(r['common_valid'] for r in rows),
                own_valid=sum(r['own_valid'] for r in rows), initially_alive=sum(r['prefix_alive'] for r in rows),
                **{k: mean([r[k] for r in rows]) for k in METRICS})


def interval(diff, counts, geometry):
    rng = np.random.default_rng(30919008)
    reps = 20000
    sw = np.zeros((reps, len(counts)), int)
    for g in np.unique(geometry):
        ix = np.flatnonzero(geometry == g)
        sw[:, ix] = rng.multinomial(len(ix), np.ones(len(ix))/len(ix), size=reps)
    cw = rng.multinomial(5, np.ones(5)/5, size=reps)
    mw = rng.multinomial(5, np.ones(5)/5, size=reps)
    den = sw@counts
    ok = den > 0
    values = np.einsum('msc,rc->rms', diff, sw[ok])/den[ok, None, None]
    draws = np.einsum('rms,rm,rs->r', values, mw[ok], cw[ok])/25
    return dict(difference=float((diff.sum(-1)/counts.sum()).mean()),
                interval_95=np.quantile(draws, [.025, .975]).tolist(), requested_draws=reps,
                valid_draws=int(ok.sum()), zero_denominator_draws=int((~ok).sum()),
                seed=30919008, family_size=16, adjustment='none; exploratory')


def aggregate(configs):
    tiers = TIERS+['pooled']
    ct = []
    for d in configs:
        for mode in MODES:
            for tier in tiers:
                rr = [r for r in d['rows'] if r['mode'] == mode and (tier == 'pooled' or r['tier'] == tier)]
                ct.append({**{k: d[k] for k in ['index', 'regime', 'campaign', 'method', 'seed']},
                           'mode': mode, 'tier': tier, **summarize(rr)})
    campaigns, means = [], []
    for reg in REGIMES:
        for method in METHODS:
            for mode in MODES:
                for tier in tiers:
                    cc = []
                    for campaign in range(5):
                        vals = [x for x in ct if (x['regime'], x['method'], x['mode'], x['tier'], x['campaign']) == (reg, method, mode, tier, campaign)]
                        assert len(vals) == (5 if method in ['frozen', 'adapt_random', 'adapt_pretrained'] else 1)
                        counts = {k: vals[0][k] for k in ['states', 'common_valid', 'own_valid', 'initially_alive']}
                        assert all(all(x[k] == v for k, v in counts.items()) for x in vals)
                        row = dict(regime=reg, method=method, mode=mode, tier=tier, campaign=campaign,
                                   **counts, **{k: mean([x[k] for x in vals]) for k in METRICS})
                        campaigns.append(row)
                        cc.append(row)
                    means.append(dict(regime=reg, method=method, mode=mode, tier=tier, **counts,
                                      **{k: mean([x[k] for x in cc]) for k in METRICS}))
    contrasts = []
    for reg in REGIMES:
        ds = [d for d in configs if d['regime'] == reg]
        first = [r for r in ds[0]['rows'] if r['mode'] == 'held']
        scenes = sorted({r['scene'] for r in first})
        assert len(scenes) == 21
        si = {s: i for i, s in enumerate(scenes)}
        geo = np.array([int(any(r['scene'] == s and r['tier'] == 'joint_extrap' for r in first)) for s in scenes])
        for metric in ['regret', 'accuracy', 'selected_contact']:
            validkey = 'prefix_alive' if metric == 'selected_contact' else 'common_valid'
            counts = np.zeros(len(scenes))
            for row in first:
                counts[si[row['scene']]] += row[validkey]
            cubes = {mode: {m: np.zeros((5 if m == 'frozen' else 1, 5, len(scenes))) for m in ['frozen', 'scalar']} for mode in MODES}
            for d in ds:
                if d['method'] not in ['frozen', 'scalar']:
                    continue
                for r in d['rows']:
                    if r[validkey]:
                        cubes[r['mode']][d['method']][d['seed'], d['campaign'], si[r['scene']]] += r[metric]
            gaps = {m: cubes[m]['frozen']-cubes[m]['scalar'] for m in MODES}
            for mode in MODES[1:]:
                contrasts.append(dict(regime=reg, mode=mode, metric=metric,
                     comparison='frozen minus scalar', eligible_states=int(counts.sum()),
                     scene_clusters=len(scenes), **interval(gaps[mode], counts, geo)))
                if metric == 'regret':
                    contrasts.append(dict(regime=reg, mode=mode, metric='regret_gap_change',
                         comparison='continuation minus held: frozen-minus-scalar gap',
                         eligible_states=int(counts.sum()), scene_clusters=len(scenes),
                         **interval(gaps[mode]-gaps['held'], counts, geo)))
    assert (len(ct), len(campaigns), len(means), len(contrasts)) == (2280, 840, 168, 16)
    return dict(configuration_tier=ct, campaign_means=campaigns, method_means=means, comparisons=contrasts)


def analyze(run):
    run = run.resolve()
    assert json.loads((run/'completion.json').read_text())['status'] == 'simulation_complete_pending_analysis_acceptance'
    dest = run/'analysis'
    dest.mkdir(exist_ok=False)
    source = {str(p.relative_to(ROOT)): sha(p) for p in list(run.rglob('*.npz'))+list(CHOICES.glob('config*.json'))
              +[Path(__file__), BASE/'PROTOCOL.md', run/'output_hashes.json']}
    write(dest/'source_lock.json', source)
    cache = {(reg, p.stem): snapshot_truth(run, reg, p.stem) for reg in REGIMES for p in sorted((OLD/reg).glob('*.npz'))}
    np.savez_compressed(dest/'truth_costs.npz', **{
        f'{reg}__{name}__{mode}': q['cost']
        for (reg, name), d in cache.items() for mode, q in d['modes'].items()})
    configs = []
    for path in sorted(CHOICES.glob('config*.json')):
        old = json.loads(path.read_text())
        rows = []
        for r in old['rows']:
            q = cache[old['regime'], r['snapshot']]
            j, choice = r['task_index'], r['choice']
            for mode, truth in q['modes'].items():
                costs = truth['cost'][j]
                regret = float(costs[choice]-costs.min())
                accuracy = bool(choice == truth['oracle'][j])
                common, own, alive = bool(q['common'][j]), bool(truth['all_valid'][j]), bool(q['alive'][j])
                row = {k: r[k] for k in ['snapshot', 'task_index', 'tier', 'scene', 'choice']}
                row.update(mode=mode, common_valid=common, own_valid=own, prefix_alive=alive,
                           oracle=int(truth['oracle'][j]), regret=regret if common else None,
                           accuracy=accuracy if common else None, own_regret=regret if own else None,
                           own_accuracy=accuracy if own else None,
                           selected_contact=bool(truth['selected_contact'][j, choice]) if alive else None)
                if mode == 'held':
                    assert own == r['all_candidates_valid'] and alive == r['prefix_alive']
                    for k, oldkey in [('own_regret', 'regret'), ('own_accuracy', 'correct'), ('selected_contact', 'selected_contact')]:
                        assert (row[k] is None) == (r[oldkey] is None)
                        if row[k] is not None:
                            np.testing.assert_allclose(row[k], r[oldkey], atol=1e-8, rtol=1e-9)
                rows.append(row)
        d = {k: old[k] for k in ['index', 'regime', 'campaign', 'method', 'seed']}
        d['rows'] = rows
        write(dest/path.name, d)
        configs.append(d)
    assert len(configs) == 190 and sum(len(d['rows']) for d in configs) == 61560
    summaries = aggregate(configs)
    ordering, support = [], []
    pairs = np.triu_indices(9, 1)
    for reg in REGIMES:
        for tier in TIERS+['pooled']:
            items = [q for (r, name), q in cache.items() if r == reg and (tier == 'pooled' or name.split('_step')[0] == tier)]
            common = np.concatenate([q['common'] for q in items])
            support.append(dict(regime=reg, tier=tier, states=len(common), common_valid=int(common.sum()),
                initially_alive=int(np.concatenate([q['alive'] for q in items]).sum()),
                own_valid={m: int(sum(q['modes'][m]['all_valid'].sum() for q in items)) for m in MODES}))
            held = np.concatenate([q['modes']['held']['cost'] for q in items])[common]
            for mode in MODES[1:]:
                new = np.concatenate([q['modes'][mode]['cost'] for q in items])[common]
                a = held[:, pairs[0]]-held[:, pairs[1]]
                b = new[:, pairs[0]]-new[:, pairs[1]]
                ordering.append(dict(regime=reg, tier=tier, mode=mode, common_valid=int(common.sum()),
                    pairs=int(common.sum()*36), pair_order_agreement=float(np.mean(np.sign(a) == np.sign(b))),
                    argmin_agreement=float(np.mean(held.argmin(1) == new.argmin(1))),
                    held_cost_range=float(np.mean(np.ptp(held, axis=1))),
                    continuation_cost_range=float(np.mean(np.ptp(new, axis=1))),
                    exact_pair_ties_held=int((a == 0).sum()), exact_pair_ties_continuation=int((b == 0).sum())))
    summaries.update(stage='Development evaluation-target sensitivity; fixed prior predictor choices.',
                     support=support, candidate_ordering=ordering, configurations=190, state_configuration_mode_rows=61560)
    write(dest/'summary.json', summaries)
    write(dest/'output_hashes.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted(dest.iterdir()) if p.is_file()})
    print(json.dumps(dict(support=support, comparisons=summaries['comparisons']), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    analyze(ap.parse_args().run)
