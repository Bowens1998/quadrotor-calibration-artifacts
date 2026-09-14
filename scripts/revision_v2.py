"""Independent September revision entry point. Never writes legacy outputs.

prepare -> collect -> train -> evaluate -> summarize (separate Slurm jobs).
Old ID becomes validation. New test trajectories are generated with disjoint
seeds and balanced behavior for every scene x wind cell.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, platform, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from _common import ROOT
from train import build_splits
from winddyn.revision.protocol import (VERSION, RevisionDataset, relocate,
    portable_path, dump_new, digest, causal_observer, matched_input_observer, fit_ridge, predict_ridge)
from winddyn.data.dataset import compute_norm_stats
from winddyn.models.wm import WorldModel, compute_loss
from winddyn.train.trainer import make_loader, load_model
from winddyn.utils.config import load_yaml, load_vehicle

BASE=ROOT/'runs/revision_20260907'
RECIPES={
 'blind':dict(jepa=True,use_depth=True,privileged=False),
 'pajepa':dict(jepa=True,use_depth=True,privileged=True),
 'direct':dict(jepa=True,use_depth=True,privileged=True,wind_objective='direct'),
 'supervised':dict(jepa=False,use_depth=True,privileged=False),
 'supervised_wind':dict(jepa=False,use_depth=True,privileged=True,wind_objective='direct'),
 'wind_only':dict(jepa=False,use_depth=True,privileged=True,wind_objective='direct'),
 'pajepa_nodepth':dict(jepa=True,use_depth=False,privileged=True),
 'wind_only_nodepth':dict(jepa=False,use_depth=False,privileged=True,wind_objective='direct'),
}


def code_hash():
    h=hashlib.sha256()
    for p in sorted(list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))):
        h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def prepare(args):
    original=build_splits()
    plan={"protocol":VERSION,"created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          "train":original['train'],"validation":original['id_eval'],
          "legacy_ood_status":"development only, previously inspected",
          "test_seed_base":730000,"test_behavior":["tracking","perturbation"],
          "probe_fit":"all training episodes, uniform stride, no prefix truncation",
          "ridge_grid":[1e-5,1e-4,1e-3,1e-2,.1,1.],
          "checkpoint":"validation state RMSE; wind-only uses validation head RMSE",
          "state_readout":"JEPA: freeze trunk then refit readout on train for 20 epochs; select on validation",
          "seeds":[0,1,2,3,4],"recipes":RECIPES,
          "primary":"prediction accuracy and wind head/probe error under unseen trajectories; paired by episode",
          "statistical_unit":"episode; report seed variation separately; scene cluster sensitivity",
          "code_hash":code_hash()}
    dump_new(BASE/'protocol.json',plan)
    print('Protocol locked:',BASE/'protocol.json',flush=True)


def plan_splits():
    p=json.load(open(BASE/'protocol.json'))
    return {k:relocate(p[k],ROOT) for k in ['train','validation']}


def collect(args):
    from winddyn.cfd.field_io import load_field
    from winddyn.geometry.procedural import load_manifest
    from winddyn.sim.rollout import EnvTask,collect_batch
    from winddyn.data.writer import save_episode
    scenes={s.scene_id:s for s in load_manifest(ROOT/'data/manifests/scenes.json')}
    fields=json.load(open(ROOT/'data/manifests/wind_fields.json'))['fields']
    fields=sorted(fields,key=lambda e:(e['scene_id'],e['wind_id']))
    protocol=json.load(open(BASE/'protocol.json'))
    jobs=[(e,mode) for e in fields for mode in protocol['test_behavior']]
    if args.limit:jobs=jobs[:args.limit]
    out=ROOT/'data/episodes_revision_v2'
    manifest=BASE/('test_manifest.json' if not args.limit else 'pilot_test_manifest.json')
    if manifest.exists():raise FileExistsError(manifest)
    entries=[];vp=load_vehicle();t0=time.time()
    for start in range(0,len(jobs),16):
        chunk=jobs[start:start+16]
        tasks=[EnvTask(scenes[e['scene_id']],load_field(portable_path(e['path'],ROOT)),e['wind_id'],mode)
               for e,mode in chunk]
        # Batched generation has deterministic per-batch seeds; identity in metadata.
        eps=collect_batch(tasks,vp,seed=protocol['test_seed_base']+start,device=args.device)
        for (e,mode),ep in zip(chunk,eps):
            ep['meta']['replicate']=100
            ep['meta']['protocol']=VERSION
            eid=f"v2_{e['scene_id']}__{e['wind_id']}__{mode}"
            path=out/(eid+'.npz')
            if path.exists():
                with np.load(path) as z:meta=json.loads(str(z['meta_json']))
                meta['path']=str(path);entries.append(meta)
            else:entries.append(save_episode(out,eid,ep))
        print(f'collect {start+len(chunk)}/{len(jobs)} {time.time()-t0:.1f}s',flush=True)
    dump_new(manifest,{"protocol":VERSION,"episodes":entries,"seconds":time.time()-t0})


def datasets(recipe, H=12, K=30, stride=5):
    splits=plan_splits();depth=RECIPES[recipe]['use_depth']
    return {k:RevisionDataset(es,H=H,K=K,stride=stride,with_depth=depth)
            for k,es in splits.items()}


def batch_device(b,device):return {k:v.to(device,non_blocking=True) for k,v in b.items()}


@torch.no_grad()
def validation(model,loader,device,wind_only=False):
    model.eval();n=0;sq=0.
    for b in loader:
        b=batch_device(b,device)
        if wind_only:
            err=model.estimate_wind(model.encode_context(b))-b['wind_hist'][:,-1]
        else:
            err=model.predict_targets(b)[...,:3]-b['target'][...,:3]
        sq+=float(err.square().sum());n+=err.numel()/err.shape[-1]
    if not n:raise ValueError('Empty validation set')
    return float(np.sqrt(sq/n))


def train(args):
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    name=f'{args.recipe}_seed{args.seed}'+('_pilot' if args.epochs!=50 else '')
    out=BASE/'checkpoints'/name
    out.mkdir(parents=True,exist_ok=False)
    ds=datasets(args.recipe);stats=compute_norm_stats(ds['train'],n=4000,seed=0)
    cfg=load_yaml('configs/experiments/m6_vec_teacher.yaml')
    cfg['name']=name;cfg['model'].update(RECIPES[args.recipe])
    cfg['model']['detach_probe']=cfg['model']['jepa']
    cfg['model']['wind_teacher']='vector'
    cfg['train']['epochs']=args.epochs
    wind_only=args.recipe.startswith('wind_only')
    if wind_only:cfg['train']['loss_weights']['probe']=0.
    model=WorldModel(cfg['model'],stats).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    loaders={k:make_loader(d,256,k=='train',workers=2) for k,d in ds.items()}
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    t0=time.time();best=float('inf');history=[]
    for ep in range(args.epochs):
        model.train();losses=[]
        for b in loaders['train']:
            loss,_=compute_loss(model,batch_device(b,args.device),cfg['train']['loss_weights'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();model.update_ema()
            losses.append(float(loss.detach()))
        sched.step();v=validation(model,loaders['validation'],args.device,wind_only)
        history.append(dict(epoch=ep,loss=float(np.mean(losses)),validation_rmse=v))
        if v<best:
            best=v;torch.save(dict(model=model.state_dict(),cfg=cfg,stats=stats,epoch=ep),out/'best.pt')
        print(name,ep,round(v,5),flush=True)
    model=load_model(out/'best.pt',args.device)
    # A genuine frozen-trunk readout; supervised models keep their trained head.
    if model.jepa:
        for p in model.parameters():p.requires_grad_(False)
        torch.manual_seed(args.seed+10000)
        for layer in model.probe.modules():
            if hasattr(layer,'reset_parameters'):layer.reset_parameters()
        for p in model.probe.parameters():p.requires_grad_(True)
        popt=torch.optim.AdamW(model.probe.parameters(),lr=1e-3,weight_decay=1e-5)
        best_read=float('inf')
        for ep in range(20 if args.epochs==50 else 2):
            model.eval()
            for b in loaders['train']:
                b=batch_device(b,args.device)
                pred=model(b)['probe'];target=model.normalizer.norm('target',b['target'])
                loss=torch.nn.functional.smooth_l1_loss(pred,target,beta=.5)
                popt.zero_grad();loss.backward();popt.step()
            v=validation(model,loaders['validation'],args.device)
            if v<best_read:
                best_read=v;torch.save(dict(model=model.state_dict(),cfg=cfg,stats=stats,
                        readout_epoch=ep),out/'readout.pt')
        print('frozen readout validation',best_read,flush=True)
    log=dict(protocol=VERSION,recipe=args.recipe,seed=args.seed,epochs=args.epochs,
       code_hash=code_hash(),protocol_hash=digest(json.load(open(BASE/'protocol.json'))),
       train_windows=len(ds['train']),validation_windows=len(ds['validation']),
       best_validation=best,history=history,wall_s=time.time()-t0,
       torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
       slurm_job=os.environ.get('SLURM_JOB_ID'),gpu=(torch.cuda.get_device_name() if args.device=='cuda' else 'cpu'),
       peak_gpu_bytes=(torch.cuda.max_memory_allocated() if args.device=='cuda' else 0))
    dump_new(out/'train_log.json',log)


@torch.no_grad()
def encode(model,ds,device):
    zs=[];ws=[];heads=[];pos=[]
    for b in make_loader(ds,256,False,workers=2):
        b=batch_device(b,device);z=model.encode_context(b)
        zs.append(z.cpu().numpy());ws.append(b['wind_hist'][:,-1].cpu().numpy())
        if model.privileged:heads.append(model.estimate_wind(z).cpu().numpy())
        pred=model.predict_targets(b)
        pos.append(((pred[...,:3]-b['target'][...,:3])**2).sum(-1).cpu().numpy())
    if not zs:raise ValueError('Empty dataset')
    return np.concatenate(zs),np.concatenate(ws),(np.concatenate(heads) if heads else None),np.concatenate(pos)


def evaluate(args):
    torch.set_num_threads(4)
    name=f'{args.recipe}_seed{args.seed}'
    dest=BASE/'metrics'/f'{name}.json'
    if dest.exists():raise FileExistsError(dest)
    out=BASE/'checkpoints'/name
    model=load_model(out/('readout.pt' if (out/'readout.pt').exists() else 'best.pt'),args.device)
    ds=datasets(args.recipe,stride=7)
    ztr,wtr,_,_=encode(model,ds['train'],args.device)
    zv,wv,_,_=encode(model,ds['validation'],args.device)
    candidates=[]
    for lam in json.load(open(BASE/'protocol.json'))['ridge_grid']:
        fit=fit_ridge(ztr,wtr,lam);err=np.mean(np.sum((predict_ridge(zv,fit)-wv)**2,1))
        candidates.append((err,lam,fit))
    _,lam,fit=min(candidates,key=lambda x:x[0])
    test=relocate(json.load(open(BASE/'test_manifest.json'))['episodes'],ROOT)
    fields=json.load(open(ROOT/'data/manifests/wind_fields.json'))['fields']
    tags={e['wind_id']:e['tag'] for e in fields}
    groups=defaultdict(list)
    for e in test:
        geo=int(e['scene_seed'])>=4;tag=tags[e['wind_id']]
        split=('joint_extrap' if geo else 'wind_extrap') if tag=='ood_extrap' else (
              ('joint_ood' if geo else 'wind_ood') if tag=='ood' else ('geo_ood' if geo else 'id_test'))
        groups[split].append(e)
    rows=[];summ={};vp=load_vehicle()
    for sp,es in groups.items():
        ds_sp=RevisionDataset(es,stride=7,with_depth=model.use_depth)
        z,w,head,pos=encode(model,ds_sp,args.device)
        pe=((predict_ridge(z,fit)-w)**2).sum(-1)
        he=((head-w)**2).sum(-1) if head is not None else None
        ids=np.array([ei for ei,_ in ds_sp.index]);obs=[]
        for ei,t in ds_sp.index:
            obs.append((ei,t))
        observers={ei:causal_observer(ds_sp._ep(ei),vp) for ei in np.unique(ids)}
        oe=np.array([np.sum((observers[ei][t]-w[i])**2) for i,(ei,t) in enumerate(obs)])
        matched={ei:matched_input_observer(ds_sp._ep(ei),vp) for ei in np.unique(ids)}
        me=np.array([np.sum((matched[ei][t]-w[i])**2) for i,(ei,t) in enumerate(obs)])
        for ei in np.unique(ids):
            m=ids==ei;e=es[int(ei)]
            row=dict(split=sp,episode=e['episode_id'],scene=e['scene_id'],family=e['family'],
              wind=e['wind_id'],mode=e['mode'],n=int(m.sum()),probe_sq=float(pe[m].sum()),
              observer_sq=float(oe[m].sum()),matched_observer_sq=float(me[m].sum()),position_sq=float(pos[m].sum()),horizon=pos.shape[1])
            if he is not None:row['head_sq']=float(he[m].sum())
            rows.append(row)
        summ[sp]=dict(n_episodes=len(np.unique(ids)),n_windows=len(ids),probe_rmse=float(np.sqrt(pe.mean())),
          head_rmse=float(np.sqrt(he.mean())) if he is not None else None,
          position_rmse=None if args.recipe.startswith('wind_only') else float(np.sqrt(pos.mean())),
          observer_rmse=float(np.sqrt(oe.mean())),matched_observer_rmse=float(np.sqrt(me.mean())))
    dump_new(dest,dict(protocol=VERSION,code_hash=code_hash(),name=name,ridge_lambda=lam,
          probe_fit_episodes=len(ds['train'].entries),summary=summ,episodes=rows))
    # Preserve the fitted inference path for downstream use, no extra wind labels.
    np.savez(out/'wind_probe.npz',**fit)
    print(json.dumps(summ,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','collect','train','evaluate'])
    ap.add_argument('--recipe',choices=list(RECIPES),default='pajepa');ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--device',default='cuda')
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args();globals()[args.command](args)

if __name__=='__main__':main()
