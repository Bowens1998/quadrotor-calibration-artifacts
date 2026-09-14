"""Revision action-branch test: identical observed histories, candidate commands.

No model/metric access during collection. Four branches share initial state,
vehicle randomization, controller state and the full prefix within each task.
This tests action-conditioned prediction under a new behavior, not a latent
wind intervention. Each task is a bootstrap cluster.
"""
import argparse,json,time
from collections import defaultdict
import numpy as np
import torch
from _common import ROOT
from revision_v2 import BASE as TRAIN_BASE, RECIPES, batch_device
BASE=TRAIN_BASE
from winddyn.revision.protocol import VERSION,portable_path,dump_new
from winddyn.cfd.field_io import load_field
from winddyn.geometry.procedural import load_manifest
from winddyn.sim.rollout import collect_batch,EnvTask
from winddyn.data.writer import save_episode
from winddyn.data.dataset import build_window,nose_yaw_np
from winddyn.train.trainer import load_model
from winddyn.utils.config import load_vehicle

ANCHOR=79  # t=3.95 seconds; command branches start at 4.0 seconds
K=30;H=12


def select_branch_tasks(scenes,fields,rng,geometry_strata=False):
    # 48 distinct fields from ID and wind-speed extrapolation, balanced modes.
    tasks=[]
    for heldout in ([False,True] if geometry_strata else [False]):
        for tag in ['train','ood_extrap']:
            cand=sorted([e for e in fields if e['tag']==tag and (scenes[e['scene_id']].seed>=4)==heldout],
                        key=lambda e:(e['scene_id'],e['wind_id']))
            for i in rng.choice(len(cand),min(48,len(cand)),replace=False):
                for mode in ['tracking','perturbation']:
                    field=dict(cand[int(i)])
                    field['evaluation_tier']=tag+('_heldout_geometry' if heldout else '')
                    tasks.append((field,mode))
    return tasks


def collect(device, collection_seed=912007, goal_mode="heading", geometry_strata=False):
    scenes={s.scene_id:s for s in load_manifest(ROOT/'data/manifests/scenes.json')}
    fields=json.load(open(ROOT/'data/manifests/wind_fields.json'))['fields']
    rng=np.random.default_rng(collection_seed)
    tasks=select_branch_tasks(scenes,fields,rng,geometry_strata)
    out=ROOT/'data'/('episodes_revision_branches' if BASE==TRAIN_BASE else 'episodes_'+BASE.name);entries=[]
    for offset in range(0,len(tasks),16):
        chunk=tasks[offset:offset+16];N=len(chunk)
        envs=[EnvTask(scenes[e['scene_id']],load_field(portable_path(e['path'],ROOT)),e['wind_id'],mode)
              for e,mode in chunk]
        t=np.arange(350)/50
        cmds=np.zeros((len(t),N,4),np.float32)
        for i in range(N):
            phase=rng.uniform(0,2*np.pi)
            cmds[:,i,0]=1.2*np.sin(2*np.pi*.18*t+phase)
            cmds[:,i,1]=1.2*np.cos(2*np.pi*.15*t+phase)
        # Shared prefix plus {continue, brake, left, right}; lateral branch
        # directions use commanded prefix heading, not observed test outcomes.
        prefix_last=cmds[199,:,:2].copy()
        directions=prefix_last/np.maximum(np.linalg.norm(prefix_last,axis=1,keepdims=True),1e-6)
        side=np.stack([-directions[:,1],directions[:,0]],-1)
        suffixes=[prefix_last,np.zeros((N,2)),2*side,-2*side]
        if goal_mode=='uniform':
            # Draw goals before simulating any candidate outcome, independently
            # of prefix commands. Separate RNG avoids changing command draws.
            goal_rng=np.random.default_rng(collection_seed+100000+offset)
            angles=goal_rng.uniform(-np.pi,np.pi,N)
            goals=2*np.stack([np.cos(angles),np.sin(angles)],-1)
        else:goals=directions*2
        first=None
        for branch,velocity in enumerate(suffixes):
            cc=cmds.copy();cc[200:,:,:2]=velocity
            eps=collect_batch(envs,load_vehicle(),duration_s=7.,seed=collection_seed+7993+offset,
                              device=device,commands=torch.tensor(cc),randomize=True)
            states={key:np.stack([e[key][:ANCHOR+1] for e in eps]) for key in
                    ['position_world','velocity_world','quaternion_world_body','angular_velocity_body','action','depth']}
            if first is None:first=states
            else:
                for key in states:np.testing.assert_array_equal(first[key],states[key])
            for i,((field,mode),ep) in enumerate(zip(chunk,eps)):
                task=f"branch_{offset+i:04d}";eid=f'{task}_b{branch}'
                ep['meta'].update(protocol=VERSION,task=task,branch=branch,
                                  tier=field['evaluation_tier'],desired_delta=goals[i].tolist(),goal_mode=goal_mode,collection_seed=collection_seed)
                if (out/(eid+'.npz')).exists():raise FileExistsError(eid)
                entries.append(save_episode(out,eid,ep))
        print('branch collection',offset+N,'/',len(tasks),flush=True)
    dump_new(BASE/'branch_manifest.json',dict(protocol=VERSION,anchor=ANCHOR,H=H,K=K,goal_mode=goal_mode,collection_seed=collection_seed,geometry_strata=geometry_strata,episodes=entries))


@torch.no_grad()
def evaluate(recipe,seed,device):
    if recipe.startswith('wind_only'):raise ValueError('wind-only has no predictor')
    name=f'{recipe}_seed{seed}';out=TRAIN_BASE/'checkpoints'/name
    model=load_model(out/('readout.pt' if (out/'readout.pt').exists() else 'best.pt'),device)
    es=json.load(open(BASE/'branch_manifest.json'))['episodes'];groups=defaultdict(list)
    for e in es:groups[e['task']].append(e)
    rows=[]
    for task,entries in groups.items():
        eps=[]
        for e in sorted(entries,key=lambda e:e['branch']):
            with np.load(portable_path(e['path'],ROOT)) as z:eps.append({k:z[k] for k in z.files if k!='meta_json'})
        # Pre-prefix collisions invalidate all four branches equally.
        if any(ep['collision'][:ANCHOR+1].any() for ep in eps):
            rows.append(dict(task=task,excluded_prefix_collision=True));continue
        samples=[build_window(ep,ANCHOR,H,K,with_depth=model.use_depth,with_patch=False) for ep in eps]
        batch={k:torch.stack([b[k] for b in samples]).to(device) for k in samples[0]}
        pred=model.predict_targets(batch).cpu().numpy()[...,:3]
        truth=batch['target'].cpu().numpy()[...,:3]
        # Cost is progress toward a predeclared 2m planar displacement. This is
        # a ranking diagnostic, not a collision-aware planning benchmark.
        yaw=nose_yaw_np(eps[0]['quaternion_world_body'][ANCHOR:ANCHOR+1])[0]
        c,s=np.cos(-yaw),np.sin(-yaw);R=np.array([[c,-s],[s,c]])
        goal=R@np.array(entries[0]['desired_delta'])
        actual=((truth[:,-1,:2]-goal)**2).sum(-1)
        predicted=((pred[:,-1,:2]-goal)**2).sum(-1)
        collision=np.array([ep['collision'][ANCHOR+1:ANCHOR+K+1].any() for ep in eps])
        choice=int(predicted.argmin());oracle=int(actual.argmin())
        # Position error restricted to valid prefix before first branch contact;
        # rank/regret uses only tasks with every candidate contact-free.
        valid=np.array([not ep['collision'][:ANCHOR+K+1].any() for ep in eps])
        rows.append(dict(task=task,tier=entries[0]['tier'],mode=entries[0]['mode'],
          scene=entries[0]['scene_id'],n_valid_branches=int(valid.sum()),
          position_sq=float(((pred[valid]-truth[valid])**2).sum()),n_positions=int(valid.sum())*K,
          all_branches_valid=bool(valid.all()),selected_collision=bool(collision[choice]),
          choice=choice,oracle=oracle,regret=float(actual[choice]-actual[oracle]) if valid.all() else None,
          correct=bool(choice==oracle) if valid.all() else None,
          actual_costs=actual.tolist(),predicted_costs=predicted.tolist()))
    dump_new(BASE/'branch_metrics'/f'{name}.json',dict(protocol=VERSION,name=name,tasks=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','evaluate'])
    p.add_argument('--recipe',choices=RECIPES,default='pajepa');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda')
    p.add_argument('--run-dir',default=str(TRAIN_BASE))
    p.add_argument('--collection-seed',type=int,default=912007)
    p.add_argument('--goal-mode',choices=['heading','uniform'],default='heading')
    p.add_argument('--geometry-strata',action='store_true')
    a=p.parse_args()
    from pathlib import Path
    BASE=Path(a.run_dir).resolve();BASE.mkdir(parents=True,exist_ok=True)
    if a.command=='collect':collect(a.device,a.collection_seed,a.goal_mode,a.geometry_strata)
    else:evaluate(a.recipe,a.seed,a.device)
