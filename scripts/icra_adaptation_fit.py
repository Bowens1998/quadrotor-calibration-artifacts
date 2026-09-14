"""Development-only budgeted regime calibration from isolated input/label caches."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from _common import ROOT
from icra_adaptation_cache import INPUT_KEYS,BASE
from icra_adaptation_ridge import RidgePath,predict
from winddyn.train.trainer import load_model
from revision_v2 import BASE as TRAIN_BASE

GRID=[1e-5,1e-4,1e-3,1e-2,.1,1.,10.]
RECIPES=['blind','pajepa','direct','supervised','supervised_wind','wind_only']
CONFIGS=[(r,s) for r in RECIPES for s in range(5)]+[(r,0) for r in ['raw_state','raw_depth','physics_only']]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def physics(arr,prefix,tau=.3):
    commands=arr[prefix+'action_fut'][...,:3].astype(float)
    sf=arr[prefix+'state_hist'][:,-1];c,s=sf[:,11],sf[:,10]
    u=commands.copy();u[...,0]=c[:,None]*commands[...,0]+s[:,None]*commands[...,1]
    u[...,1]=-s[:,None]*commands[...,0]+c[:,None]*commands[...,1]
    v=arr[prefix+'vel_yaw_t'].astype(float);p=np.zeros_like(v);out=[];decay=np.exp(-.05/tau)
    for k in range(u.shape[1]):
        p=p+.05*u[:,k]+tau*(1-decay)*(v-u[:,k]);v=u[:,k]+(v-u[:,k])*decay;out.append(p.copy())
    return np.stack(out,1)


@torch.no_grad()
def features(arr,prefix,recipe,model,device):
    n=len(arr[prefix+'state_hist']);p=physics(arr,prefix).reshape(n,-1)
    if recipe=='physics_only':return p
    if model is not None:
        assert not model.use_wind
        parts=[]
        for start in range(0,n,128):
            b={k:torch.from_numpy(arr[prefix+k][start:start+128]).to(device) for k in INPUT_KEYS}
            parts.append(model.encode_context(b).cpu().numpy())
        context=np.concatenate(parts)
        return np.concatenate([context,arr[prefix+'action_fut'].reshape(n,-1),p],1)
    keys=['state_hist','action_hist','action_fut']
    if recipe=='raw_depth':keys.append('depth_hist')
    return np.concatenate([arr[prefix+k].reshape(n,-1) for k in keys]+[p],1)


def task_metrics(pred,arr,meta):
    truth=arr['branch_target_position'];pred=pred.reshape(truth.shape);valid=arr['branch_valid']
    tasks=sorted([r for r in meta['branch_tasks'] if 'sample_index' in r],key=lambda r:r['sample_index'])
    goals=np.array([r['goal_yaw'] for r in tasks]);cost=((pred[:,:,-1,:2]-goals[:,None,:])**2).sum(-1)
    actual=arr['branch_cost_actual'];choice=cost.argmin(1);oracle=actual.argmin(1);rows=np.arange(len(choice))
    all_valid=valid.all(1)
    return dict(position_sq=(((pred-truth)**2).sum((2,3))*valid).sum(1),n_positions=valid.sum(1)*truth.shape[2],
        choice=choice,oracle=oracle,selected_collision=~valid[rows,choice],all_branches_valid=all_valid,
        regret=np.where(all_valid,actual[rows,choice]-actual[rows,oracle],np.nan),
        correct=np.where(all_valid,choice==oracle,np.nan),predicted_costs=cost)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--regime',required=True,choices=['mass_1p4','lag_3'])
    ap.add_argument('--config',type=int,required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--attempt',default='formal_v1')
    a=ap.parse_args();torch.set_num_threads(4);started=time.time();recipe,seed=CONFIGS[a.config]
    cache=BASE/a.regime/'cache';meta=json.loads((cache/'metadata.json').read_text())
    dest=BASE/a.regime/a.attempt/f'{recipe}_seed{seed}';dest.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'scripts/icra_adaptation_ridge.py',ROOT/'scripts/icra_adaptation_cache.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources+[cache/'arrays.npz',cache/'metadata.json']}
    checkpoint=TRAIN_BASE/'checkpoints'/f'{recipe}_seed{seed}'/'best.pt'
    model=None
    if recipe in RECIPES:
        hashes[str(checkpoint.relative_to(ROOT))]=sha(checkpoint);model=load_model(checkpoint,a.device);model.eval()
    # Write provenance before extraction or fitting; the complete source/model lock is external.
    (dest/'provenance.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with np.load(cache/'arrays.npz') as f:arr={k:f[k] for k in f.files}
    assert all(np.isfinite(v).all() for v in arr.values())
    assert len(meta['calibration_episodes'])==64 and len(meta['branch_tasks'])==368
    x=features(arr,'cal_',recipe,model,a.device);xb=features(arr,'branch_',recipe,model,a.device)
    y=arr['cal_target_position'].reshape(len(x),-1);p=physics(arr,'cal_').reshape(y.shape);pb=physics(arr,'branch_').reshape(len(xb),-1)
    assert np.isfinite(x).all() and np.isfinite(xb).all()
    episode_ids=np.array([e['episode_id'] for e in meta['calibration_episodes']])[arr['cal_episode_index']]
    records=[];outcomes=[];physical_records=[];physical_outcomes=[]
    for subset,bybudget in meta['subsets'].items():
        for budget,parts in bybudget.items():
            fitmask=np.isin(episode_ids,parts['fit']);valmask=np.isin(episode_ids,parts['validation'])
            assert not set(parts['fit'])&set(parts['validation']) and not np.any(fitmask&valmask)
            common=dict(subset_seed=int(subset),budget=int(budget),fit_episodes=parts['fit'],validation_episodes=parts['validation'],fit_windows=int(fitmask.sum()),validation_windows=int(valmask.sum()))
            if not fitmask.any() or not valmask.any():
                records.extend([dict(common,center=c,status='unavailable_empty_split') for c in ['zero','physical']]);continue
            path=RidgePath(x[fitmask])
            for center in ['zero','physical']:
                prior=p if center=='physical' else np.zeros_like(p);branch_prior=pb if center=='physical' else np.zeros_like(pb)
                candidates=[];best=None
                for lam in GRID:
                    fit=path.fit((y-prior)[fitmask],lam)
                    delta=(predict(x[valmask],fit)+prior[valmask]-y[valmask]).reshape(-1,30,3)
                    error=float(np.sqrt(np.mean(np.sum(delta**2,-1))));candidates.append(dict(lambda_=lam,validation_rmse=error))
                    if best is None or error<best[0]:best=(error,lam,fit)
                error,lam,fit=best;pred=predict(xb,fit)+branch_prior;assert np.isfinite(pred).all()
                idx=len(outcomes);outcomes.append(task_metrics(pred,arr,meta))
                weight=dest/f'fit_{subset}_{budget}_{center}.npz';np.savez_compressed(weight,**fit)
                records.append(dict(common,center=center,status='complete',outcome_index=idx,selected_lambda=lam,validation_rmse=error,grid=candidates,weight_file=weight.name,weight_sha256=sha(weight)))
            # Physical controls only in one deterministic configuration, without duplicate seed reporting.
            if recipe=='physics_only':
                taus=[.1,.2,.3,.5,.7,1.,1.5]
                errors=[float(np.mean((physics(arr,'cal_',t).reshape(y.shape)[fitmask]-y[fitmask])**2)) for t in taus]
                selected=taus[int(np.argmin(errors))]
                for name,tau in [('fixed_0.3',.3),('fixed_1.0',1.),('calibrated',selected)]:
                    pp=physics(arr,'cal_',tau).reshape(y.shape)
                    val=float(np.sqrt(np.mean(np.sum((pp[valmask]-y[valmask]).reshape(-1,30,3)**2,-1))))
                    physical_records.append(dict(common,name=name,tau=tau,validation_rmse=val,outcome_index=len(physical_outcomes),tau_grid=taus,training_mse_grid=errors))
                    physical_outcomes.append(task_metrics(physics(arr,'branch_',tau),arr,meta))
            print(a.regime,recipe,subset,budget,'complete',flush=True)
    def save_outcomes(name,values):
        if values:np.savez_compressed(dest/name,**{k:np.stack([v[k] for v in values]) for k in values[0]})
    save_outcomes('outcomes.npz',outcomes);save_outcomes('physical_outcomes.npz',physical_outcomes)
    result=dict(regime=a.regime,recipe=recipe,model_seed=seed,stage='development',feature_dim=x.shape[1],branch_tasks=meta['branch_tasks'],records=records,physical_records=physical_records,wall_s=time.time()-started)
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',dest,len(outcomes),'fits',flush=True)
if __name__=='__main__':main()
