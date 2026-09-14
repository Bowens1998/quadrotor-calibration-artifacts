"""Accept the complete fresh matrix and report all eight locked comparisons."""
import json
from collections import defaultdict
import numpy as np
import torch
from _common import ROOT
from icra_confirm_common import BASE,FROZEN_JOBS,ADAPT_JOBS,REGIMES
from icra_adaptation_fit import CONFIGS,sha
from icra_adaptation_summary import aggregate
from icra_confirm_statistics import PRIMARY,interval

def summarize(out,i,mask):
    if not mask.any():
        return dict(position_rmse=None,n_positions=0,prefix_tasks=0,all_valid_tasks=0,selected_contact=None,accuracy=None,regret=None)
    return aggregate(out,i,mask)

def main():
    frozen=[(r,c,*CONFIGS[k]) for r,c,k in FROZEN_JOBS]
    paths=[BASE/f'campaign{c}'/r/'frozen_v1'/f'{recipe}_seed{s}' for r,c,recipe,s in frozen]
    paths += [BASE/f'campaign{c}'/arm/r/'scratch_v1'/f'budget{b}_subset901_seed{s}' for arm,r,b,c,s in ADAPT_JOBS]
    missing=[str(p) for p in paths if not (p/'results.json').exists()]
    if missing:raise SystemExit(f'Incomplete confirmation: {len(missing)}/530 configurations missing; no results written')
    lock=json.loads((BASE/'LOCK.json').read_text());assert lock['formal_collection_authorized']
    for line in (BASE/'source.sha256').read_text().splitlines():
        h,p=line.split('  ',1);assert sha(ROOT/p)==h
    meta={};truth={};scene_info={};eligibility={};cached_hashes={}
    def verified(path,expected):
        path=path.resolve()
        if path not in cached_hashes:cached_hashes[path]=sha(path)
        assert cached_hashes[path]==expected
    for r in REGIMES:
        reference=None
        for c in range(5):
            path=BASE/f'campaign{c}'/r/'cache';m=json.loads((path/'metadata.json').read_text())
            assert not m['fixture'] and m['campaign']==c and len(m['calibration_episodes'])==64
            if reference is None:reference=m['branch_tasks']
            assert m['branch_tasks']==reference and len(reference)==736
            with np.load(path/'arrays.npz') as z:
                valid=z['branch_valid'].copy();cost=z['branch_cost_actual'].copy();which=z['cal_episode_index'].copy()
            if r in truth:
                np.testing.assert_array_equal(truth[r][0],valid);np.testing.assert_array_equal(truth[r][1],cost)
            truth[r]=valid,cost
            m['window_episode_ids']=np.array([e['episode_id'] for e in m['calibration_episodes']])[which];meta[r,c]=m
        tasks=sorted([t for t in reference if 'sample_index' in t],key=lambda t:t['sample_index']);scenes=sorted({t['scene'] for t in reference});index={s:i for i,s in enumerate(scenes)}
        geometry=np.array([int(any('heldout_geometry' in t['tier'] for t in reference if t['scene']==s)) for s in scenes])
        scene_info[r]=tasks,index,geometry
        eligibility[r]=dict(total_tasks=736,prefix_valid=len(tasks),all_four_valid=int(truth[r][0].all(1).sum()),scene_clusters=len(scenes))
    rows=[];cubes={};denominators={};unavailable=[];accepted=[];normalization={}
    def add(r,c,method,s,b,head,out,i):
        valid,cost=truth[r];tasks,index,geometry=scene_info[r]
        assert np.isfinite(out['position_sq'][i]).all() and np.isfinite(out['predicted_costs'][i]).all()
        np.testing.assert_array_equal(out['n_positions'][i],valid.sum(1)*30)
        np.testing.assert_array_equal(out['all_branches_valid'][i],valid.all(1))
        np.testing.assert_array_equal(out['oracle'][i],cost.argmin(1))
        np.testing.assert_array_equal(out['selected_collision'][i],~valid[np.arange(len(valid)),out['choice'][i]])
        for key in ['regret','correct']:np.testing.assert_array_equal(np.isnan(out[key][i]),~valid.all(1))
        for tier in ['pooled']+sorted({t['tier'] for t in tasks}):
            mask=np.array([tier=='pooled' or t['tier']==tier for t in tasks])
            rows.append(dict(regime=r,campaign=c,method=method,model_seed=s,budget=b,head=head,tier=tier,**summarize(out,i,mask)))
        if b!=32 or head not in ('physical','physical_control') or method not in ('frozen_supervised','adapt_random','adapt_pretrained','physics_only','calibrated'):return
        nm=1 if method in ('physics_only','calibrated') else 5
        for metric,num,n in [('position',out['position_sq'][i],out['n_positions'][i]),('accuracy',np.nan_to_num(out['correct'][i]),out['all_branches_valid'][i].astype(float)),('contact',out['selected_collision'][i].astype(float),np.ones(len(tasks)))]:
            sums=np.zeros(len(index));counts=np.zeros(len(index))
            for j,t in enumerate(tasks):sums[index[t['scene']]]+=num[j];counts[index[t['scene']]]+=n[j]
            key=r,method,metric
            if key not in cubes:cubes[key]=np.zeros((nm,5,len(index)))
            cubes[key][s,c]=sums
            if (r,metric) in denominators:np.testing.assert_array_equal(denominators[r,metric],counts)
            denominators[r,metric]=counts
    def checks(dest,m,r,c):
        assert m['stage']=='confirmation' and not m['fixture'] and m['campaign']==c
        if m.get('status')=='unavailable_empty_split':unavailable.append(str(dest));return False
        assert m['branch_tasks']==meta[r,c]['branch_tasks']
        for p,h in json.loads((dest/'provenance.json').read_text()).items():verified(ROOT/p,h)
        return True
    def check_split(m,r,c,b):
        expected=meta[r,c]['subsets']['901'][str(b)]
        for part in ['fit','validation']:
            assert m[part+'_episodes']==expected[part]
            assert m[part+'_windows']==int(np.isin(meta[r,c]['window_episode_ids'],expected[part]).sum())
    for r,c,recipe,s in frozen:
        dest=BASE/f'campaign{c}'/r/'frozen_v1'/f'{recipe}_seed{s}';m=json.loads((dest/'results.json').read_text())
        assert m['recipe']==recipe and m['model_seed']==s and len(m['records'])==4
        assert checks(dest,m,r,c)
        complete=[rec for rec in m['records'] if rec['status']=='complete']
        for rec in m['records']:
            check_split(rec,r,c,rec['budget'])
            if rec['status']!='complete':unavailable.append(str(dest)+f"/{rec['budget']}/{rec['center']}")
        if complete:
            with np.load(dest/'outcomes.npz') as z:out={k:z[k] for k in z.files}
            for rec in complete:
                verified(dest/rec['weight_file'],rec['weight_sha256']);assert rec['validation_rmse']==min(v['validation_rmse'] for v in rec['grid'])
                add(r,c,'physics_only' if recipe=='physics_only' else 'frozen_'+recipe,s,rec['budget'],rec['center'],out,rec['outcome_index'])
        if m['physical_records']:
            with np.load(dest/'physical_outcomes.npz') as z:out={k:z[k] for k in z.files}
            for rec in m['physical_records']:add(r,c,rec['name'],0,rec['budget'],'physical_control',out,rec['outcome_index'])
        accepted.append(str(dest.relative_to(BASE)))
    for arm,r,b,c,s in ADAPT_JOBS:
        dest=BASE/f'campaign{c}'/arm/r/'scratch_v1'/f'budget{b}_subset901_seed{s}';m=json.loads((dest/'results.json').read_text())
        if not checks(dest,m,r,c):accepted.append(str(dest.relative_to(BASE)));continue
        assert m['initialization']==arm and m['budget']==b and m['model_seed']==s
        check_split(m,r,c,b);assert len(m['curves'])==202 and len(m['records'])==2
        assert m['selection']['validation_rmse']==min(v['validation_rmse'] for v in m['curves'])
        assert abs(m['retained_validation_rmse_reproduced']-m['selection']['validation_rmse'])<1e-4
        verified(dest/'encoder.pt',m['encoder_sha256']);ck=torch.load(dest/'encoder.pt',map_location='cpu',weights_only=False)
        assert all(torch.isfinite(v).all() for v in ck['model'].values())
        current=[np.asarray(ck['stats'][k][v]) for k in ['state_hist','action_hist'] for v in ['mean','std']]+[ck['extra_mean'],ck['extra_std']]
        nk=r,b,c,s
        if nk in normalization:
            for x,y in zip(current,normalization[nk]):np.testing.assert_array_equal(x,y)
        else:normalization[nk]=current
        with np.load(dest/'outcomes.npz') as z:out={k:z[k] for k in z.files}
        for rec in m['records']:
            verified(dest/(rec['center']+'_ridge.npz'),rec['weight_sha256']);assert rec['validation_rmse']==min(v['validation_rmse'] for v in rec['grid'])
            add(r,c,'adapt_'+arm,s,b,rec['center'],out,rec['outcome_index'])
        with np.load(dest/'retained_outcomes.npz') as z:out={k:z[k][None] for k in z.files}
        add(r,c,'adapt_'+arm,s,b,'retained',out,0);accepted.append(str(dest.relative_to(BASE)))
    for r,left,right,metric in PRIMARY:
        if (r,metric) not in denominators or denominators[r,metric].sum()==0:
            unavailable.append(f'{r}/{metric}: no eligible primary observations')
    dest=BASE/'summary';dest.mkdir(exist_ok=False)
    if unavailable:
        (dest/'unavailable.json').write_text(json.dumps(dict(accepted=accepted,unavailable=unavailable,decision='Do not drop incomplete campaigns or compute a reduced primary family.'),indent=2)+'\n')
        raise SystemExit('Empty calibration splits: explicit unavailable report written; primary analysis not reduced')
    primary=[]
    for r,left,right,metric in PRIMARY:
        primary.append(dict(regime=r,budget=32,left=left,right=right,metric=metric,**interval(cubes[r,left,metric],cubes[r,right,metric],denominators[r,metric],scene_info[r][2],metric)))
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in ['regime','method','budget','head','tier'])].append(row)
    means=[]
    for keys,vs in groups.items():
        item=dict(zip(['regime','method','budget','head','tier'],keys));item['fits']=len(vs)
        for key in ['position_rmse','selected_contact','accuracy','regret']:
            vals=[v[key] for v in vs if v[key] is not None];item[key]=float(np.mean(vals)) if vals else None
        means.append(item)
    (dest/'summary.json').write_text(json.dumps(dict(stage='fresh algorithm confirmation',accepted=accepted,eligibility=eligibility,primary=primary,secondary_means=means,per_fit=rows),indent=2)+'\n')
    lines=['# Fresh calibration confirmation — all eight primary comparisons','','Budget32 includesvalidation; five independent calibration campaigns. Two branch realizations stay inside scene clusters. Differences are left minus right. Contact and accuracy are fractions. Family-adjusted99.375% and descriptive95% intervals are distinct.','','| Regime | Left | Right | Metric | Difference | Adjusted99.375% interval |','|---|---|---|---|---:|---|']
    for p in primary:
        lo,hi=p['adjusted_99_375_interval'];lines.append(f"| {p['regime']} | {p['left']} | {p['right']} | {p['metric']} | {p['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    (dest/'SUMMARY.md').write_text('\n'.join(lines)+'\n');print('ACCEPTED all530 confirmation configurations')
if __name__=='__main__':main()
