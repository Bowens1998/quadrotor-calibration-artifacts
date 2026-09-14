"""All prespecified lag nonlinear-readout contrasts; no fitting or selection."""
import argparse
from pathlib import Path
import numpy as np
from icra_nonlinear_readout_20260914 import (
    ROOT, PREPARED, DEFAULT_RUN, read, checked, sha, key, load_arrays, write_new,
    verify_prepared)

FAMILIES = ['scalar','affine','additive','joint']
METRICS = ['forecast_rmse','centered_mse','held_regret','nominal_regret','observer_regret']
PAIRS = [('joint','affine'),('additive','affine'),('joint','additive'),('joint','scalar')]


def array_metric(values, metric):
    if metric == 'forecast_rmse':return values['total_mse']
    if metric == 'centered_mse':return values['centered_mse']
    return values['regret'][['held_regret','nominal_regret','observer_regret'].index(metric)]


def cluster_bootstrap(cube, counts, sw, cw, ew, sqrt=False):
    """cube[campaign,encoder,head,scene]; head seeds are fixed, averaged metrics."""
    denominator = sw@counts
    assert (denominator > 0).all()
    state_mean = np.einsum('cehs,rs->rceh',cube,sw,optimize=True)/denominator[:,None,None,None]
    if sqrt:state_mean=np.sqrt(state_mean)
    cell = state_mean.mean(-1)
    encoder_weights = ew if cube.shape[1] == 5 else np.ones((len(ew),1))
    draws = np.einsum('rce,rc,re->r',cell,cw,encoder_weights,optimize=True)/(5*cube.shape[1])
    point = cube.sum(-1)/counts.sum()
    if sqrt:point=np.sqrt(point)
    return float(point.mean()),draws


def summarize(prepared=PREPARED, run=DEFAULT_RUN):
    verify_prepared(prepared)
    run_lock=read(run/'source_lock.json')
    for rel,digest in run_lock.items():checked(ROOT/rel,digest)
    checked(prepared/'evaluation.npz',run_lock[key(prepared/'evaluation.npz')])
    dest = run/'summary'
    if dest.exists():raise FileExistsError(dest)
    meta = read(run/'evaluation/configurations.json')
    assert meta['complete'] and len(meta['configurations'])==180
    source={key(run/'evaluation/configurations.json'):sha(run/'evaluation/configurations.json'),
            key(Path(__file__)):sha(Path(__file__)),
            key(prepared/'evaluation.npz'):sha(prepared/'evaluation.npz')}
    data = load_arrays(prepared/'evaluation.npz')
    mask = data['common_valid'].reshape(108)
    scene = data['scene'].reshape(108)
    tier = data['tier'].reshape(108)
    scenes = sorted(np.unique(scene))
    assert len(scenes)==21 and mask.sum()==106
    counts = np.array([np.sum(mask & (scene==s)) for s in scenes])
    strata = np.array([int(np.any((scene==s)&(tier=='joint_extrap'))) for s in scenes])
    cubes = {metric:{family:np.zeros((5,1 if family=='scalar' else 5,
                                    3 if family in ['joint','additive'] else 1,len(scenes)))
                     for family in FAMILIES} for metric in METRICS}
    configurations=[]
    seen=set()
    expected={(family,campaign,encoder,head) for family in FAMILIES
              for campaign in range(5)
              for encoder in ([0] if family=='scalar' else range(5))
              for head in (range(3) if family in ['additive','joint'] else [-1])}
    assert {(c['family'],c['campaign'],c['encoder_seed'],c['head_seed'])
            for c in meta['configurations']}==expected
    for c in meta['configurations']:
        checked(ROOT/c['path'],c['sha256'],source)
        values=load_arrays(ROOT/c['path'])
        cell=(c['family'],c['campaign'],c['encoder_seed'],c['head_seed'])
        assert cell not in seen
        seen.add(cell)
        for metric in METRICS:
            v=array_metric(values,metric)
            h=max(0,c['head_seed'])
            cubes[metric][c['family']][c['campaign'],c['encoder_seed'],h]=[
                v[mask & (scene==s)].sum() for s in scenes]
        for group in ['pooled','id','wind_extrap','joint_extrap']:
            supported=mask & (np.ones(108,bool) if group=='pooled' else tier==group)
            alive=data['initial_alive'].reshape(108) & (np.ones(108,bool) if group=='pooled' else tier==group)
            row={k:c[k] for k in ['family','campaign','encoder_seed','head_seed']}
            row.update(tier=group,states=int(supported.sum()))
            for metric in METRICS:
                v=array_metric(values,metric)[supported].mean()
                row[metric]=float(np.sqrt(v) if metric=='forecast_rmse' else v)
            row['accuracy']=values['accuracy'][:,supported].mean(1).tolist()
            row['selected_contact']=values['selected_contact'][:,alive].mean(1).tolist()
            configurations.append(row)
    rng=np.random.default_rng(2026091401)
    repetitions=20000
    sw=np.zeros((repetitions,len(scenes)),int)
    for g in np.unique(strata):
        ix=np.flatnonzero(strata==g)
        sw[:,ix]=rng.multinomial(len(ix),np.ones(len(ix))/len(ix),size=repetitions)
    cw=rng.multinomial(5,np.ones(5)/5,size=repetitions)
    ew=rng.multinomial(5,np.ones(5)/5,size=repetitions)
    points={};draws={};contrasts=[]
    for metric in METRICS:
        for family in FAMILIES:
            point,draw=cluster_bootstrap(cubes[metric][family],counts,sw,cw,ew,metric=='forecast_rmse')
            points[metric,family]=point;draws[metric,family]=draw
            check=[x[metric] for x in configurations if x['family']==family and x['tier']=='pooled']
            np.testing.assert_allclose(point,np.mean(check),atol=1e-12)
        for left,right in PAIRS:
            difference=draws[metric,left]-draws[metric,right]
            contrasts.append(dict(metric=metric,left=left,right=right,
                difference=points[metric,left]-points[metric,right],
                interval_95=np.quantile(difference,[.025,.975]).tolist(),
                bootstrap_draws=repetitions,bootstrap_seed=2026091401,
                adjustment='none; exploratory 20-contrast family on reused development states',
                head_seed_handling='metrics averaged within campaign/encoder; three fixed head seeds',
                scene_clusters=len(scenes),states=int(mask.sum())))
    assert len(contrasts)==20
    campaign_means=[];means=[]
    for group in ['pooled','id','wind_extrap','joint_extrap']:
        for family in FAMILIES:
            rows=[r for r in configurations if r['tier']==group and r['family']==family]
            for campaign in range(5):
                rr=[r for r in rows if r['campaign']==campaign]
                campaign_means.append(dict(tier=group,family=family,campaign=campaign,
                    **{metric:float(np.mean([r[metric] for r in rr])) for metric in METRICS}))
            means.append(dict(tier=group,family=family,
                    **{metric:float(np.mean([r[metric] for r in rows])) for metric in METRICS}))
    doc=dict(stage='Development sensitivity to a specified nonlinear readout; no full-flight test',
             configurations=configurations,campaign_means=campaign_means,means=means,contrasts=contrasts,
             unique_states=108,common_valid=106,scene_clusters=21,
             families=FAMILIES,metrics=METRICS,all_models_retained=180)
    dest.mkdir()
    write_new(dest/'summary.json',doc)
    write_new(dest/'sources.json',source)
    lines=['# Frozen-context nonlinear readout: all lag development results','',
           'Five calibration campaigns × five encoder seeds; three head seeds for each nonlinear family.',
           '106 shared valid states, 21 scene clusters. Heads are averaged as metrics, never ensembled.',
           'All 20 intervals are exploratory, unadjusted and conditional on the existing states and three head seeds.','',
           '| Readout | Forecast RMSE (m) | Centered MSE (m²) | Held regret | Nominal regret | Observer regret |',
           '|---|---:|---:|---:|---:|---:|']
    for r in means:
        if r['tier']=='pooled':lines.append('| '+r['family']+' | '+' | '.join(f'{r[m]:.7f}' for m in METRICS)+' |')
    lines += ['', '## Every contrast','', '| Metric | Contrast | Difference | 95% exploratory interval |','|---|---|---:|---|']
    for r in contrasts:
        lines.append(f'| {r["metric"]} | {r["left"]} − {r["right"]} | {r["difference"]:.7f} | [{r["interval_95"][0]:.7f}, {r["interval_95"][1]:.7f}] |')
    lines+=['','## Every campaign, pooled states','', '| Campaign | Readout | RMSE | Centered MSE | Held | Nominal | Observer |','|---|---|---:|---:|---:|---:|---:|']
    for r in campaign_means:
        if r['tier']=='pooled':lines.append(f'| {r["campaign"]} | {r["family"]} | '+' | '.join(f'{r[m]:.7f}' for m in METRICS)+' |')
    lines+=['','The JSON retains all seed-specific results and distribution tiers. These results do not evaluate recurrent latent models, establish a sufficient belief state, or demonstrate closed-loop adoption.','']
    (dest/'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:13]))
    return doc


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=PREPARED)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args()
    summarize(args.prepared,args.run)
