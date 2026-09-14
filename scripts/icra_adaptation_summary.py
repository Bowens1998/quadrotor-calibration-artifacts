"""Complete-matrix descriptive development summary; no post-selection claims."""
import argparse,json
from collections import defaultdict
import numpy as np
from _common import ROOT
from icra_adaptation_fit import BASE,CONFIGS


def aggregate(out,index,mask):
    n=out['n_positions'][index,mask].sum();sq=out['position_sq'][index,mask].sum()
    complete=out['all_branches_valid'][index,mask]
    return dict(position_rmse=float(np.sqrt(sq/n)) if n else None,n_positions=int(n),
        prefix_tasks=int(mask.sum()),all_valid_tasks=int(complete.sum()),
        selected_contact=float(out['selected_collision'][index,mask].mean()),
        accuracy=float(out['correct'][index,mask][complete].mean()) if complete.any() else None,
        regret=float(out['regret'][index,mask][complete].mean()) if complete.any() else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',default='formal_v1');a=ap.parse_args()
    missing=[str(BASE/reg/a.attempt/f'{r}_seed{s}'/'results.json') for reg in ['mass_1p4','lag_3'] for r,s in CONFIGS if not (BASE/reg/a.attempt/f'{r}_seed{s}'/'results.json').exists()]
    if missing:raise SystemExit(f'Incomplete matrix: {len(missing)} of 66 configurations missing; no aggregate written')
    rows=[];eligibility={}
    for reg in ['mass_1p4','lag_3']:
        reference=None
        for recipe,seed in CONFIGS:
            dest=BASE/reg/a.attempt/f'{recipe}_seed{seed}';m=json.loads((dest/'results.json').read_text())
            assert len(m['records'])==40 and m['recipe']==recipe and m['model_seed']==seed
            if reference is None:reference=m['branch_tasks']
            assert reference==m['branch_tasks']
            tasks=sorted([t for t in reference if 'sample_index' in t],key=lambda t:t['sample_index'])
            eligibility[reg]=dict(total_tasks=len(reference),prefix_valid=len(tasks))
            tiers=['pooled']+sorted({t['tier'] for t in tasks})
            for filename,key,physical in [('outcomes.npz','records',False),('physical_outcomes.npz','physical_records',True)]:
                if not m[key]:continue
                with np.load(dest/filename) as z:out={k:z[k] for k in z.files}
                for rec in m[key]:
                    if rec.get('status')=='unavailable_empty_split':
                        rows.append(dict(regime=reg,recipe=recipe,model_seed=seed,**rec));continue
                    for tier in tiers:
                        mask=np.array([tier=='pooled' or t['tier']==tier for t in tasks])
                        row=dict(regime=reg,recipe=rec['name'] if physical else recipe,model_seed=seed,subset_seed=rec['subset_seed'],budget=rec['budget'],center='physical_control' if physical else rec['center'],tier=tier,status='complete')
                        row.update(aggregate(out,rec['outcome_index'],mask));rows.append(row)
    groups=defaultdict(list)
    for row in rows:
        if row['status']=='complete':groups[tuple(row[k] for k in ['regime','recipe','budget','center','tier'])].append(row)
    summary=[]
    for keys,values in groups.items():
        row=dict(zip(['regime','recipe','budget','center','tier'],keys));row['fits']=len(values)
        for metric in ['position_rmse','selected_contact','accuracy','regret']:
            vs=[v[metric] for v in values if v[metric] is not None];row[metric]=float(np.mean(vs)) if vs else None
        # Descriptive marginal spreads only: these are not independent-run standard errors or CIs.
        for axis in ['model_seed','subset_seed']:
            means=[np.mean([v['position_rmse'] for v in values if v[axis]==s and v['position_rmse'] is not None]) for s in sorted({v[axis] for v in values})]
            row[axis+'_marginal_rmse_sd']=float(np.std(means,ddof=1)) if len(means)>1 else None
        summary.append(row)
    dest=BASE/f'{a.attempt}_summary';dest.mkdir(exist_ok=False)
    (dest/'summary.json').write_text(json.dumps(dict(stage='development descriptive; no confirmatory inference',eligibility=eligibility,summary=summary,per_fit=rows),indent=2)+'\n')
    lines=['# New-regime calibration: full descriptive development matrix','','All budgets include validation. Mean RMSE averages model/subset fits; these fits are dependent, not independent experiments. Contact exclusions are retained. No confidence or superiority claim is made by this script.','','| Regime | Input | Budget | Center | RMSE m | Selected contact | Accuracy |','|---|---|---:|---|---:|---:|---:|']
    for r in summary:
        if r['tier']=='pooled':lines.append(f"| {r['regime']} | {r['recipe']} | {r['budget']} | {r['center']} | {r['position_rmse']:.4f} | {r['selected_contact']:.2%} | {r['accuracy']:.2%} |")
    (dest/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
