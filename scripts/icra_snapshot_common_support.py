"""Post-hoc common-support aggregation of saved same-state prediction outputs."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/icra_snapshot_common_support_20260909'
OLD = ROOT / 'runs/icra_snapshot_prediction_20260908'
REGIMES = ['mass_1p4', 'lag_3']
METHODS = ['scalar', 'axis', 'physics_features', 'raw', 'supervised']
TIERS = ['id', 'wind_extrap', 'joint_extrap', 'pooled']


def main():
    dest = BASE / 'attempt01'
    dest.mkdir(exist_ok=False)
    sources = [Path(__file__), BASE / 'PROTOCOL.md', OLD / 'descriptive_summary.json', OLD / 'accepted_intervals.json']
    original = json.loads((OLD / 'descriptive_summary.json').read_text())['per_config']
    oldmap = {(r['regime'], r['method'], r['seed'], r['subset']): r for r in original}
    records, support = [], []
    for regime in REGIMES:
        masks = {}
        for p in sorted((ROOT / 'runs/icra_snapshot_pilot_20260908' / regime).glob('*.npz')):
            sources.append(p)
            with np.load(p) as z:
                assert z['valid'].shape == (30, 9, 12)
                for j in range(12):
                    masks[p.stem, j] = (bool(z['initial_alive'][j]), bool(z['valid'][:, :, j].all()), int(z['valid'][:, :, j].sum()))
        assert len(masks) == 108
        files = sorted((OLD / regime).glob('config*.json'))
        assert len(files) == 45
        for p in files:
            sources.append(p)
            d = json.loads(p.read_text())
            assert len(d['rows']) == 108
            assert {(r['snapshot'], r['task_index']) for r in d['rows']} == set(masks)
            for r in d['rows']:
                assert (r['prefix_alive'], r['all_candidates_valid'], r['valid_candidate_records']) == masks[r['snapshot'], r['task_index']]
                if r['all_candidates_valid']:
                    assert math.isclose(r['total_mse'], r['position_squared_error'] / 270, rel_tol=1e-11, abs_tol=1e-11)
                    assert math.isclose(r['total_mse'], r['common_mse'] + r['contrast_mse'], rel_tol=1e-11, abs_tol=1e-11)
            for tier in TIERS:
                rows = [r for r in d['rows'] if tier == 'pooled' or r['snapshot'].rsplit('_step', 1)[0] == tier]
                shared = [r for r in rows if r['all_candidates_valid']]
                assert rows and shared
                nrecords = sum(r['valid_candidate_records'] for r in rows)
                row = dict(regime=regime, method=d['method'], seed=d['model_seed'], subset=d['subset'], tier=tier,
                           snapshots=len(rows), initially_alive=sum(r['prefix_alive'] for r in rows),
                           all_valid=len(shared), original_position_records=nrecords, common_position_records=270 * len(shared),
                           original_rmse=math.sqrt(sum(r['position_squared_error'] for r in rows) / nrecords),
                           common_rmse=math.sqrt(sum(r['total_mse'] for r in shared) / len(shared)),
                           accuracy=sum(r['correct'] for r in shared) / len(shared),
                           regret=sum(r['regret'] for r in shared) / len(shared))
                if tier == 'pooled':
                    old = oldmap[regime, d['method'], d['model_seed'], d['subset']]
                    for newkey, oldkey in [('original_rmse', 'rmse'), ('accuracy', 'accuracy'), ('regret', 'regret')]:
                        assert math.isclose(row[newkey], old[oldkey], rel_tol=1e-11, abs_tol=1e-11)
                records.append(row)
        for tier in TIERS:
            selected = [v for (name, _), v in masks.items() if tier == 'pooled' or name.rsplit('_step', 1)[0] == tier]
            support.append(dict(regime=regime, tier=tier, snapshots=len(selected), initially_alive=sum(v[0] for v in selected), all_valid=sum(v[1] for v in selected)))
    means = []
    for reg in REGIMES:
        for tier in TIERS:
            for method in METHODS:
                rows = [r for r in records if (r['regime'], r['tier'], r['method']) == (reg, tier, method)]
                assert len(rows) == (25 if method == 'supervised' else 5)
                assert len({r['all_valid'] for r in rows}) == 1
                means.append(dict(regime=reg, tier=tier, method=method, configs=len(rows), unique_states=rows[0]['all_valid'],
                                  **{k:sum(r[k] for r in rows) / len(rows) for k in ['original_rmse', 'common_rmse', 'accuracy', 'regret']}))
    differences = []
    for reg in REGIMES:
        for tier in TIERS:
            pair = {r['method']:r for r in means if r['regime'] == reg and r['tier'] == tier}
            differences.append(dict(regime=reg, tier=tier,
                                    **{k:pair['supervised'][k] - pair['scalar'][k] for k in ['original_rmse', 'common_rmse', 'accuracy', 'regret']}))
    result = dict(stage='post-hoc development denominator audit; no new inference or confidence intervals',
                  protocol_commit='f912c08', per_config=records, means=means, support=support, frozen_minus_scalar=differences)
    (dest / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (dest / 'sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2) + '\n')
    print(json.dumps(differences, indent=2))


if __name__ == '__main__':
    main()
