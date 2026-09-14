"""From-scratch history supervision using only the new-regime label budget."""
import argparse,copy,json,time
import numpy as np
import torch
from torch import nn
from _common import ROOT
from icra_adaptation_fit import BASE,INPUT_KEYS,GRID,physics,features,task_metrics,sha
from icra_adaptation_ridge import RidgePath,predict
from winddyn.models.wm import WorldModel

JOBS=[(r,b,s,i) for r in ['mass_1p4','lag_3'] for b in [8,16,32,64] for s in range(901,906) for i in range(5)]


def fit_stats(arr,mask):
    return {k:dict(mean=arr['cal_'+k][mask].mean((0,1)).astype('float32'),std=np.maximum(arr['cal_'+k][mask].std((0,1)),1e-5).astype('float32')) for k in ['state_hist','action_hist']}


def make_encoder(stats):
    return WorldModel(dict(jepa=False,use_depth=True,use_wind=False,privileged=False,hidden_dim=256,latent_dim=128,action_embed_dim=64),stats)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    regime,budget,subset,seed=JOBS[a.index];torch.set_num_threads(4);start=time.time()
    dest=BASE/regime/'scratch_v1'/f'budget{budget}_subset{subset}_seed{seed}';dest.mkdir(parents=True,exist_ok=False)
    cache=BASE/regime/'cache';meta=json.loads((cache/'metadata.json').read_text())
    source_paths=[ROOT/'scripts/icra_adaptation_scratch.py',ROOT/'scripts/icra_adaptation_fit.py',ROOT/'scripts/icra_adaptation_ridge.py',ROOT/'src/winddyn/models/wm.py',cache/'arrays.npz',cache/'metadata.json']
    (dest/'provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in source_paths},indent=2)+'\n')
    with np.load(cache/'arrays.npz') as z:arr={k:z[k] for k in z.files}
    parts=meta['subsets'][str(subset)][str(budget)]
    ids=np.array([e['episode_id'] for e in meta['calibration_episodes']])[arr['cal_episode_index']]
    fm=np.isin(ids,parts['fit']);vm=np.isin(ids,parts['validation']);assert not np.any(fm&vm)
    if not fm.any() or not vm.any():
        (dest/'results.json').write_text(json.dumps(dict(status='unavailable_empty_split',budget=budget,subset_seed=subset,model_seed=seed)));return
    stats=fit_stats(arr,fm);pc=physics(arr,'cal_').reshape(len(ids),-1);pb=physics(arr,'branch_').reshape(-1,90)
    y=arr['cal_target_position'].reshape(len(ids),-1).astype('float64')
    extra=np.concatenate([arr['cal_action_fut'].reshape(len(ids),-1),pc],1)
    mean=extra[fm].mean(0);std=np.maximum(extra[fm].std(0),1e-5)
    xt=torch.tensor((extra-mean)/std,dtype=torch.float32,device=a.device)
    target=torch.tensor(y-pc,dtype=torch.float32,device=a.device)
    observed={k:torch.from_numpy(arr['cal_'+k]).to(a.device) for k in INPUT_KEYS}
    fitids=np.flatnonzero(fm);validids=np.flatnonzero(vm);curves=[];best=None
    for decay in [1e-4,1e-2]:
        torch.manual_seed(seed);np.random.seed(seed)
        model=make_encoder(stats).to(a.device);head=nn.Linear(128+210,90).to(a.device)
        opt=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=3e-4,weight_decay=decay)
        rng=np.random.default_rng(seed)
        def run(idx):
            b={k:v[idx] for k,v in observed.items()}
            return head(torch.cat([model.encode_context(b),xt[idx]],1))
        for epoch in range(101):
            loss_sum=0.
            if epoch:
                model.train();head.train()
                for ii in np.array_split(rng.permutation(fitids),max(1,int(np.ceil(len(fitids)/128)))):
                    opt.zero_grad(set_to_none=True);output=run(ii);loss=(output-target[ii]).square().mean()
                    assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.)
                    opt.step();loss_sum+=float(loss.detach())*len(ii)
            model.eval();head.eval()
            with torch.no_grad():
                error=0.
                for ii in np.array_split(validids,max(1,int(np.ceil(len(validids)/128)))):
                    error+=float((run(ii)-target[ii]).square().sum())
            val=float(np.sqrt(error/(len(validids)*30)))
            curves.append(dict(weight_decay=decay,epoch=epoch,validation_rmse=val,training_mse=loss_sum/len(fitids) if epoch else None))
            if best is None or val<best['validation_rmse']:
                best=dict(validation_rmse=val,weight_decay=decay,epoch=epoch)
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        print(regime,budget,subset,seed,'trained decay',decay,flush=True)
    model.load_state_dict(best_state);model.eval()
    checkpoint=dest/'encoder.pt';torch.save(dict(model=best_state,training_head=best_head,stats=stats,extra_mean=mean,extra_std=std,selection=best),checkpoint)
    x=features(arr,'cal_','scratch',model,a.device);xb=features(arr,'branch_','scratch',model,a.device);path=RidgePath(x[fm]);records=[];outcomes=[]
    for center in ['zero','physical']:
        prior=pc if center=='physical' else np.zeros_like(pc);priorb=pb if center=='physical' else np.zeros_like(pb)
        grid=[];selected=None
        for lam in GRID:
            fit=path.fit((y-prior)[fm],lam);delta=(predict(x[vm],fit)+prior[vm]-y[vm]).reshape(-1,30,3)
            val=float(np.sqrt(np.mean(np.sum(delta**2,-1))));grid.append(dict(lambda_=lam,validation_rmse=val))
            if selected is None or val<selected[0]:selected=(val,lam,fit)
        val,lam,fit=selected;pred=predict(xb,fit)+priorb;assert np.isfinite(pred).all()
        outcomes.append(task_metrics(pred,arr,meta));weight=dest/f'{center}_ridge.npz';np.savez_compressed(weight,**fit)
        records.append(dict(center=center,selected_lambda=lam,validation_rmse=val,grid=grid,outcome_index=len(outcomes)-1,weight_sha256=sha(weight)))
    np.savez_compressed(dest/'outcomes.npz',**{k:np.stack([o[k] for o in outcomes]) for k in outcomes[0]})
    result=dict(status='complete',regime=regime,budget=budget,subset_seed=subset,model_seed=seed,fit_episodes=parts['fit'],validation_episodes=parts['validation'],fit_windows=int(fm.sum()),validation_windows=int(vm.sum()),selection=best,curves=curves,records=records,encoder_sha256=sha(checkpoint),branch_tasks=meta['branch_tasks'],wall_s=time.time()-start)
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',dest,flush=True)
if __name__=='__main__':main()
