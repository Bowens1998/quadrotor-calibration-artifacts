"""Frozen context residual decoding with a common fixed physical baseline."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from _common import ROOT
from revision_v2 import plan_splits,batch_device,BASE as TRAIN_BASE
from winddyn.revision.protocol import RevisionDataset,fit_ridge,predict_ridge,dump_new,portable_path
from winddyn.data.dataset import build_window,nose_yaw_np
from winddyn.train.trainer import load_model
BASE=ROOT/'runs/revision_residual_20260907'
BRANCH=ROOT/'runs/revision_20260907_goaldev'
TAU=.3


def physics(b):
    # Same observed anchor velocity and candidate command sequence for everyone.
    commands=b['action_fut'][...,:3].detach().cpu().numpy().astype(float)
    sf=b['state_hist'][:,-1].detach().cpu().numpy()
    c,s=sf[:,11],sf[:,10]
    u=commands.copy();u[...,0]=c[:,None]*commands[...,0]+s[:,None]*commands[...,1]
    u[...,1]=-s[:,None]*commands[...,0]+c[:,None]*commands[...,1]
    v=b['vel_yaw_t'].detach().cpu().numpy().astype(float)
    p=np.zeros_like(v);out=[];decay=np.exp(-.05/TAU)
    for k in range(u.shape[1]):
        p=p+.05*u[:,k]+TAU*(1-decay)*(v-u[:,k]);v=u[:,k]+(v-u[:,k])*decay
        out.append(p.copy())
    return np.stack(out,1)


@torch.no_grad()
def features(model,b):
    context=model.encode_context(b) if model is not None else b['state_hist'].flatten(1)
    return torch.cat([context,b['action_fut'].flatten(1)],-1).cpu().numpy()


@torch.no_grad()
def extract(model,ds,device):
    xx=[];yy=[];pp=[]
    for b in DataLoader(ds,batch_size=256,shuffle=False):
        b=batch_device(b,device);xx.append(features(model,b));pp.append(physics(b));yy.append(b['target'][...,:3].cpu().numpy())
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(pp)


def train_readout(model, recipe, seed, device, dest):
    splits=plan_splits();sets={k:RevisionDataset(es,H=12,K=30,stride=7,with_depth=model.use_depth if model else False) for k,es in splits.items()}
    x,y,p=extract(model,sets['train'],device);xv,yv,pv=extract(model,sets['validation'],device)
    target=(y-p).reshape(len(y),-1);choices=[]
    for lam in [1e-5,1e-4,1e-3,1e-2,.1,1.,10.]:
        fit=fit_ridge(x,target,lam);pred=pv+predict_ridge(xv,fit).reshape(yv.shape)
        choices.append((float(np.sqrt(np.mean(np.sum((pred-yv)**2,-1)))),lam,fit))
    error,lam,fit=min(choices,key=lambda z:z[0]);np.savez(dest/'readout.npz',**fit)
    dump_new(dest/'fit.json',dict(recipe=recipe,seed=seed,tau=TAU,train_windows=len(y),validation_windows=len(yv),
        selected_lambda=lam,validation_rmse=error,grid=[{'lambda':l,'validation_rmse':e} for e,l,_ in choices]))
    return fit


@torch.no_grad()
def evaluate_frozen(model, fit, name, branch, output, device):
    start=time.time()
    manifest=json.load(open(branch/'branch_manifest.json'));t=manifest['anchor'];K=manifest['K'];groups=defaultdict(list)
    for e in manifest['episodes']:groups[e['task']].append(e)
    rows=[]
    for task,es in groups.items():
        es=sorted(es,key=lambda e:e['branch']);eps=[]
        for e in es:
            with np.load(portable_path(e['path'],ROOT)) as z:eps.append({k:z[k] for k in z.files if k not in ['meta_json','wind_patch']})
        if any(ep['collision'][:t+1].any() for ep in eps):rows.append(dict(task=task,excluded_prefix_collision=True));continue
        samples=[build_window(ep,t,12,K,with_depth=model.use_depth if model else False,with_patch=False) for ep in eps]
        b=batch_device({k:torch.stack([s[k] for s in samples]) for k in samples[0]},device)
        pred=physics(b)+predict_ridge(features(model,b),fit).reshape(4,K,3);truth=b['target'][...,:3].cpu().numpy()
        yaw=nose_yaw_np(eps[0]['quaternion_world_body'][t:t+1])[0];c,s=np.cos(-yaw),np.sin(-yaw)
        goal=np.array([[c,-s],[s,c]])@np.array(es[0]['desired_delta'])
        actual=((truth[:,-1,:2]-goal)**2).sum(-1);costs=((pred[:,-1,:2]-goal)**2).sum(-1)
        valid=np.array([not ep['collision'][:t+K+1].any() for ep in eps]);choice=int(costs.argmin());oracle=int(actual.argmin())
        rows.append(dict(task=task,tier=es[0]['tier'],mode=es[0]['mode'],scene=es[0]['scene_id'],
          n_valid_branches=int(valid.sum()),position_sq=float(((pred[valid]-truth[valid])**2).sum()),n_positions=int(valid.sum())*K,
          all_branches_valid=bool(valid.all()),selected_collision=bool(not valid[choice]),choice=choice,oracle=oracle,
          regret=float(actual[choice]-actual[oracle]) if valid.all() else None,correct=bool(choice==oracle) if valid.all() else None,
          actual_costs=actual.tolist(),predicted_costs=costs.tolist()))
    dump_new(output/'branch_metrics'/f'{name}.json',dict(name=name,tau=TAU,tasks=rows,wall_s=time.time()-start))
    print(name,'done',time.time()-start,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--recipe',required=True,choices=['blind','pajepa','direct','supervised','supervised_wind','wind_only','raw'])
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--device',default='cuda')
    ap.add_argument('--eval-only',action='store_true')
    ap.add_argument('--branch-dir',type=Path,default=BRANCH)
    ap.add_argument('--output-dir',type=Path,default=BASE)
    ap.add_argument('--fit-dir',type=Path,default=BASE/'fits')
    a=ap.parse_args();torch.set_num_threads(4);name=f'{a.recipe}_seed{a.seed}'
    model=None if a.recipe=='raw' else load_model(TRAIN_BASE/'checkpoints'/name/'best.pt',a.device)
    dest=a.fit_dir/name
    if a.eval_only:
        fit=load_frozen_fit(dest)
    else:
        dest.mkdir(parents=True,exist_ok=False)
        fit=train_readout(model,a.recipe,a.seed,a.device,dest)
    evaluate_frozen(model,fit,name,a.branch_dir,a.output_dir,a.device)


def load_frozen_fit(dest):
    metadata=json.loads((dest/'fit.json').read_text())
    if metadata['tau'] != TAU:raise ValueError('Frozen readout physics mismatch')
    with np.load(dest/'readout.npz') as z:return {k:z[k].copy() for k in z.files}


if __name__=='__main__':main()
