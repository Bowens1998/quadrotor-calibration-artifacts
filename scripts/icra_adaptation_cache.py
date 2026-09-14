"""Cache measured-history inputs separately from calibration/evaluation labels."""
import argparse,json
from collections import defaultdict
import numpy as np
from _common import ROOT
from winddyn.revision.protocol import portable_path,RevisionDataset,dump_new
from winddyn.data.dataset import build_window,nose_yaw_np
from icra_adaptation_subsets import calibration_subsets,SUBSET_SEEDS

BASE=ROOT/'runs/icra_adaptation_20260907'
INPUT_KEYS=('state_hist','action_hist','action_fut','depth_hist','vel_yaw_t')


def input_arrays(sample):
    # Future positions, wind targets and actual plant parameters are never
    # placed in the model-input dictionary.
    return {k:sample[k].detach().cpu().numpy() for k in INPUT_KEYS}


def main():
    p=argparse.ArgumentParser();p.add_argument('--regime',required=True,choices=['mass_1p4','lag_3']);a=p.parse_args()
    base=BASE/a.regime;dest=base/'cache';dest.mkdir(exist_ok=False)
    manifest=json.loads((base/'calibration_manifest.json').read_text());entries=manifest['episodes'];assert len(entries)==64 and not manifest['pilot']
    for e in entries:e['path']=str(portable_path(e['path'],ROOT))
    splits={seed:{budget:{kind:[e['episode_id'] for e in rows] for kind,rows in parts.items()} for budget,parts in calibration_subsets(entries,seed).items()} for seed in SUBSET_SEEDS}
    ds=RevisionDataset(entries,H=12,K=30,stride=7,with_depth=True)
    cal=defaultdict(list);y=[];which=[]
    for i,(ei,anchor) in enumerate(ds.index):
        sample=ds[i]
        for k,v in input_arrays(sample).items():cal[k].append(v)
        y.append(sample['target'].numpy()[...,:3]);which.append(ei)
    assert len(y)>0
    arrays={'cal_'+k:np.stack(v) for k,v in cal.items()};arrays.update(cal_target_position=np.stack(y),cal_episode_index=np.array(which))
    branch=ROOT/f'runs/icra_adaptation_branches_20260907_{a.regime}'
    bm=json.loads((branch/'branch_manifest.json').read_text());assert len(bm['episodes'])==1472
    assert (bm['anchor'],bm['H'],bm['K'])==(79,12,30)
    groups=defaultdict(list)
    for e in bm['episodes']:groups[e['task']].append(e)
    assert len(groups)==368
    inp=defaultdict(list);truth=[];valid=[];cost=[];metadata=[]
    for task,es in groups.items():
        es=sorted(es,key=lambda e:e['branch']);assert [e['branch'] for e in es]==[0,1,2,3]
        eps=[]
        for e in es:
            with np.load(portable_path(e['path'],ROOT)) as z:eps.append({k:z[k] for k in z.files if k not in ['meta_json','wind_patch']})
        row=dict(task=task,scene=es[0]['scene_id'],tier=es[0]['tier'],mode=es[0]['mode'])
        if any(ep['collision'][:80].any() for ep in eps):row['excluded_prefix_collision']=True;metadata.append(row);continue
        row['sample_index']=len(truth);metadata.append(row)
        samples=[build_window(ep,79,12,30,with_depth=True,with_patch=False) for ep in eps]
        for sample in samples:
            for k,v in input_arrays(sample).items():inp[k].append(v)
        yy=np.stack([sample['target'].numpy()[...,:3] for sample in samples]);truth.append(yy)
        valid.append([not ep['collision'][:110].any() for ep in eps])
        yaw=nose_yaw_np(eps[0]['quaternion_world_body'][79:80])[0];c,s=np.cos(-yaw),np.sin(-yaw)
        goal=np.array([[c,-s],[s,c]])@np.array(es[0]['desired_delta']);cost.append(((yy[:,-1,:2]-goal)**2).sum(-1))
        row['goal_yaw']=goal.tolist()
    arrays.update({'branch_'+k:np.stack(v) for k,v in inp.items()})
    arrays.update(branch_target_position=np.stack(truth),branch_valid=np.array(valid,bool),branch_cost_actual=np.stack(cost))
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(dest/'arrays.npz',**arrays)
    dump_new(dest/'metadata.json',dict(regime=a.regime,input_keys=INPUT_KEYS,calibration_episodes=entries,calibration_windows=len(y),
        episode_valid_records=[e['n_steps'] for e in ds.entries],subsets=splits,branch_tasks=metadata,branch_prefix_valid=len(truth),branch_all_valid=int(np.all(valid,axis=1).sum()),stage='development, no fit performed'))
    print(a.regime,'cache complete',len(y),'calibration windows',len(truth),'valid branch prefixes',flush=True)
if __name__=='__main__':main()
