"""Cache independent calibration pools and correctly namespaced branch batches."""
import argparse,copy,json
from collections import defaultdict
import numpy as np
from _common import ROOT
from icra_confirm_common import BASE,FIXTURE,CALIBRATION_JOBS,BUDGETS,verify_plant
from icra_adaptation_cache import input_arrays,INPUT_KEYS
from icra_adaptation_subsets import calibration_subsets
from winddyn.revision.protocol import portable_path,RevisionDataset,dump_new
from winddyn.data.dataset import build_window,nose_yaw_np


def merge_branches(parts):
    arrays={k:np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    tasks=[];offset=0
    for batch,(a,rows) in enumerate(parts):
        assert set(a)==set(arrays)
        for original in rows:
            row=copy.deepcopy(original);row.update(original_task=row['task'],branch_batch=batch,task=f'batch{batch}/{row["task"]}')
            if 'sample_index' in row:row['sample_index']+=offset
            tasks.append(row)
        offset+=len(a['branch_target_position'])
    assert len({t['task'] for t in tasks})==len(tasks)
    assert sorted(t['sample_index'] for t in tasks if 'sample_index' in t)==list(range(offset))
    for k in INPUT_KEYS:assert len(arrays['branch_'+k])==4*offset
    return arrays,tasks


def branch_cache(path,regime):
    bm=json.loads((path/'branch_manifest.json').read_text());assert len(bm['episodes'])==1472 and (bm['anchor'],bm['H'],bm['K'])==(79,12,30)
    verify_plant(bm['episodes'],regime)
    groups=defaultdict(list)
    for e in bm['episodes']:groups[e['task']].append(e)
    inp=defaultdict(list);truth=[];valid=[];cost=[];rows=[]
    assert len(groups)==368
    for task,es in groups.items():
        es=sorted(es,key=lambda e:e['branch']);assert [e['branch'] for e in es]==[0,1,2,3];eps=[]
        for e in es:
            with np.load(portable_path(e['path'],ROOT)) as z:eps.append({k:z[k] for k in z.files if k not in ['meta_json','wind_patch']})
        row=dict(task=task,scene=es[0]['scene_id'],tier=es[0]['tier'],mode=es[0]['mode'])
        # Recheck the complete shared history, not just the input windows.
        for ep in eps[1:]:
            for k in ['position_world','velocity_world','quaternion_world_body','collision']:
                np.testing.assert_array_equal(ep[k][:80],eps[0][k][:80])
        if any(ep['collision'][:80].any() for ep in eps):row['excluded_prefix_collision']=True;rows.append(row);continue
        row['sample_index']=len(truth);rows.append(row)
        samples=[build_window(ep,79,12,30,with_depth=True,with_patch=False) for ep in eps]
        for sample in samples:
            for k,v in input_arrays(sample).items():inp[k].append(v)
        yy=np.stack([s['target'].numpy()[...,:3] for s in samples]);truth.append(yy)
        valid.append([not ep['collision'][:110].any() for ep in eps])
        yaw=nose_yaw_np(eps[0]['quaternion_world_body'][79:80])[0];c,s=np.cos(-yaw),np.sin(-yaw)
        goal=np.array([[c,-s],[s,c]])@np.array(es[0]['desired_delta']);row['goal_yaw']=goal.tolist();cost.append(((yy[:,-1,:2]-goal)**2).sum(-1))
    arrays={'branch_'+k:np.stack(v) for k,v in inp.items()}
    arrays.update(branch_target_position=np.stack(truth),branch_valid=np.array(valid,bool),branch_cost_actual=np.stack(cost))
    return arrays,rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--fixture',action='store_true');a=ap.parse_args()
    regime,campaign=CALIBRATION_JOBS[a.index]
    if a.fixture:
        assert campaign==0
        old=ROOT/'runs/icra_adaptation_20260907'/regime/'cache'
        with np.load(old/'arrays.npz') as z:oldarrays={k:z[k] for k in z.files}
        meta=json.loads((old/'metadata.json').read_text());arrays={k:v for k,v in oldarrays.items() if k.startswith('cal_')}
        part=({k:v for k,v in oldarrays.items() if k.startswith('branch_')},meta['branch_tasks'])
        branch,tasks=merge_branches([part,part]);root=FIXTURE
    else:
        assert json.loads((BASE/'LOCK.json').read_text())['formal_collection_authorized'];root=BASE
        manifest=json.loads((BASE/f'campaign{campaign}'/regime/'calibration_manifest.json').read_text());assert not manifest['pilot'] and manifest['campaign']==campaign
        entries=copy.deepcopy(manifest['episodes']);assert len(entries)==64
        verify_plant(entries,regime)
        for e in entries:e['path']=str(portable_path(e['path'],ROOT))
        ds=RevisionDataset(entries,H=12,K=30,stride=7,with_depth=True);cal=defaultdict(list);y=[];which=[]
        for i,(ei,anchor) in enumerate(ds.index):
            sample=ds[i]
            for k,v in input_arrays(sample).items():cal[k].append(v)
            y.append(sample['target'].numpy()[...,:3]);which.append(ei)
        arrays={'cal_'+k:np.stack(v) for k,v in cal.items()};arrays.update(cal_target_position=np.stack(y),cal_episode_index=np.array(which))
        meta=dict(calibration_episodes=entries,calibration_windows=len(y),episode_valid_records=[e['n_steps'] for e in ds.entries])
        branch,tasks=merge_branches([branch_cache(BASE/f'icra_confirm_branches_20260908_{regime}_{batch}',regime) for batch in range(2)])
    arrays.update(branch)
    meta.update(regime=regime,campaign=campaign,fixture=a.fixture,stage='excluded operational fixture' if a.fixture else 'fresh calibration confirmation',input_keys=INPUT_KEYS,branch_tasks=tasks,
        branch_prefix_valid=len(arrays['branch_valid']),branch_all_valid=int(arrays['branch_valid'].all(1).sum()))
    splits=calibration_subsets(meta['calibration_episodes'],901)
    meta['subsets']={'901':{str(b):{k:[e['episode_id'] for e in es] for k,es in splits[b].items()} for b in BUDGETS}}
    assert len(tasks)==736 and all(np.isfinite(v).all() for v in arrays.values())
    dest=root/f'campaign{campaign}'/regime/'cache';dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'arrays.npz',**arrays);dump_new(dest/'metadata.json',meta)
    for arm in ['random','pretrained']:
        alias=root/f'campaign{campaign}'/arm/regime/'cache';alias.parent.mkdir(parents=True,exist_ok=True);alias.symlink_to(dest,target_is_directory=True)
    print('CACHE COMPLETE',regime,campaign,'fixture',a.fixture,flush=True)
if __name__=='__main__':main()
