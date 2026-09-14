"""Exploratory paired controller differences and common-flight-time sensitivity."""
import json
from collections import defaultdict
import numpy as np
from _common import ROOT
from icra_paired_mpc import CONFIGS
BASE=ROOT/'runs/icra_paired_mpc_20260908'

def paired(a,b,counts,geometry,reps=10000):
    rng=np.random.default_rng(58112007);w=np.zeros((reps,len(counts)))
    for g in np.unique(geometry):
        ix=np.flatnonzero(geometry==g);w[:,ix]=rng.multinomial(len(ix),np.ones(len(ix))/len(ix),size=reps)
    seeds=rng.multinomial(5,np.ones(5)/5,size=reps);den=w@counts
    assert (den>0).all()
    # Fixed mean over the five old shared-pool subsets, paired seed resampling.
    values=np.einsum('msc,rc->rms',a-b,w)/den[:,None,None]
    diff=np.einsum('rm,rm->r',values.mean(2),seeds)/5
    return dict(difference=float((a-b).sum(-1).mean()/counts.sum()),exploratory_95_interval=np.quantile(diff,[.025,.975]).tolist(),draws=reps)

def main():
    assert len(json.loads((BASE/'summary/summary.json').read_text())['accepted'])==30
    configs={};logs={}
    for i,key in enumerate(CONFIGS):
        d=BASE/'configs'/f'config{i:03d}';m=json.loads((d/'results.json').read_text());configs[key]=m
        err=[];valid=[]
        for tier in ['id','wind_extrap','joint_extrap']:
            with np.load(d/f'{tier}.npz') as z:err.append(z['tracking_error'][40:]);valid.append(z['valid'][40:])
        logs[key]=np.concatenate(err,1),np.concatenate(valid,1)
    rows=[]
    for reg in ['mass_1p4','lag_3']:
        tasks=configs[reg,'feedback',0,None]['episodes'];scenes=sorted({e['scene'] for e in tasks});ix={s:i for i,s in enumerate(scenes)};counts=np.zeros(len(scenes));geo=np.array([int(any(t['tier']=='joint_extrap' for t in tasks if t['scene']==s)) for s in scenes])
        for e in tasks:counts[ix[e['scene']]]+=1
        for other in ['feedback','observer','scalar','scalar_old','physics_features','raw']:
            cubes={metric:np.zeros((2,5,5,len(scenes))) for metric in ['tracking_rmse','success','contact','common_time_tracking_rmse']}
            for seed in range(5):
                for c,subset in enumerate(range(901,906)):
                    ka=(reg,'supervised',seed,None);kb=(reg,other,0,subset if other=='scalar_old' else None)
                    ma,mb=configs[ka],configs[kb];ea,va=logs[ka];eb,vb=logs[kb];mask=va&vb;assert (mask.sum(0)>0).all()
                    common=[np.sqrt(np.where(mask,e**2,0).sum(0)/mask.sum(0)) for e in [ea,eb]]
                    for side,m in enumerate([ma,mb]):
                        for j,e in enumerate(m['episodes']):
                            assert e['tracking_rmse'] is not None
                            for metric in cubes:
                                val=common[side][j] if metric=='common_time_tracking_rmse' else e[metric] if metric!='contact' else e['collided']
                                cubes[metric][side,seed,c,ix[e['scene']]]+=val
            for metric,cube in cubes.items():rows.append(dict(regime=reg,comparison='supervised minus '+other,metric=metric,scene_clusters=len(scenes),**paired(cube[0],cube[1],counts,geo)))
    with (BASE/'summary/exploratory_intervals.json').open('x') as f:json.dump(dict(stage='48 unadjusted exploratory paired scene/seed comparisons; conditional on one new calibration campaign; old scalar subsets fixed-averaged, other entries algebraically repeated',comparisons=rows),f,indent=2)
    for r in rows:
        if r['comparison'] in ['supervised minus observer','supervised minus scalar']:print(r)
if __name__=='__main__':main()
