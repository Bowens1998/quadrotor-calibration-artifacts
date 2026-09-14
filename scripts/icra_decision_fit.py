"""Matched-budget linear readouts: trajectory versus centered candidate-cost loss."""
import argparse,json
import numpy as np
import torch
from _common import ROOT
from icra_snapshot_prediction import model_input
from icra_mpc_matrix_policy import MatrixPredictor
from icra_adaptation_fit import features,physics,sha
BASE=ROOT/'runs/icra_decision_fit_20260908'
CONFIGS=[('physics_features',0),('raw',0)]+[('supervised',i) for i in range(5)]
LAMBDAS=[1e-4,1e-2,1.]

def costs(pred,ref,penalty):return ((pred-ref[:,None])**2).sum(-1).mean(-1)+penalty

def objective(pred,target,ref,penalty,kind):
    if kind=='trajectory':return ((pred-target)**2).sum(-1).mean()
    e=costs(pred,ref,penalty)-costs(target,ref,penalty)
    return ((e-e.mean(-1,keepdim=True))**2).mean()

def load_split(regime,recipe,model,split):
    xs=[];ps=[];ys=[];refs=[];pen=[];count=0
    for batch in (range(3) if split=='fit' else [3]):
        for step in [600,1200,1800]:
            with np.load(ROOT/'runs/icra_paired_calibration_20260908'/regime/f'batch{batch}_step{step}.npz') as z:a={k:z[k] for k in z.files}
            arr=model_input(a);n=len(a['initial_alive']);x=features(arr,'x_',recipe,model,'cuda').reshape(n,9,-1);p=physics(arr,'x_').reshape(n,9,30,3)
            sf=a['state_hist'][:,-1];c,s=sf[:,11],sf[:,10]
            def yaw(w):
                v=w-a['initial_position'].reshape((n,)+(1,)*(w.ndim-2)+(3,));out=v.copy();cc=c.reshape((n,)+(1,)*(w.ndim-2));ss=s.reshape((n,)+(1,)*(w.ndim-2));out[...,0]=cc*v[...,0]+ss*v[...,1];out[...,1]=-ss*v[...,0]+cc*v[...,1];return out
            y=yaw(a['position_world'].transpose(2,1,0,3));ref=yaw(a['reference_world']);u=a['candidate_command'];penalty=.02*(u[:,:,:3]**2).sum(-1)+.05*((u[:,:,:3]-a['previous_command'][:,None,:3])**2).sum(-1)
            valid=a['valid'].all((0,1));count+=n
            for dst,value in zip([xs,ps,ys,refs,pen],[x,p,y,ref,penalty]):dst.append(value[valid])
    return [np.concatenate(v) for v in [xs,ps,ys,refs,pen]],count

@torch.no_grad()
def validation(w,x,p,y,r,pen):
    pred=p+(x@w).reshape(p.shape);pc=costs(pred,r,pen);tc=costs(y,r,pen);idx=pc.argmin(-1);rows=torch.arange(len(idx),device=idx.device)
    return dict(regret=float((tc[rows,idx]-tc.min(-1).values).mean()),accuracy=float((idx==tc.argmin(-1)).double().mean()),rmse=float(((pred-y)**2).sum(-1).mean().sqrt()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--regime',required=True);ap.add_argument('--index',type=int,required=True);a=ap.parse_args();torch.set_num_threads(4)
    method,seed=CONFIGS[a.index];dest=BASE/a.regime/f'config{a.index:02d}';dest.mkdir(parents=True,exist_ok=False)
    adapter=MatrixPredictor(method,a.regime,'cuda',seed,901);recipe={'physics_features':'physics_only','raw':'raw_state','supervised':'supervised'}[method]
    train,nt=load_split(a.regime,recipe,adapter.model,'fit');val,nv=load_split(a.regime,recipe,adapter.model,'validation')
    mean=train[0].mean((0,1));std=np.maximum(train[0].std((0,1)),1e-5)
    def tensors(v):
        x=(v[0]-mean)/std;x=np.concatenate([x,np.ones((*x.shape[:2],1))],-1)
        return [torch.tensor(t,dtype=torch.float64,device='cuda') for t in [x]+v[1:]]
    x,p,y,r,pen=tensors(train);vx,vp,vy,vr,vpen=tensors(val);records=[]
    for kind in ['trajectory','decision']:
        norm=float(objective(p,y,r,pen,kind));assert norm>1e-12
        candidates=[]
        for lam in LAMBDAS:
            w=torch.zeros((x.shape[-1],90),dtype=torch.float64,device='cuda',requires_grad=True);opt=torch.optim.Adam([w],lr=.003)
            for iteration in range(500):
                opt.zero_grad();pred=p+(x@w).reshape(p.shape);loss=objective(pred,y,r,pen,kind)/norm+lam*w[:-1].square().sum()/90
                assert torch.isfinite(loss);loss.backward();opt.step()
            score=validation(w,vx,vp,vy,vr,vpen);file=f'{kind}_lambda{lam}.npz';np.savez_compressed(dest/file,mean=mean,std=std,coef=w.detach().cpu().numpy())
            candidates.append(dict(lambda_=lam,validation=score,weight_file=file,weight_sha256=sha(dest/file),final_train_loss=float(loss.detach())))
        selected=min(candidates,key=lambda q:(q['validation']['regret'],-q['lambda_']))
        records.append(dict(objective=kind,normalization=norm,candidates=candidates,selected=selected))
    (dest/'results.json').write_text(json.dumps(dict(regime=a.regime,method=method,seed=seed,fit_total=nt,fit_eligible=len(x),validation_total=nv,validation_eligible=len(vx),records=records,encoder_provenance={k:v for k,v in adapter.hashes.items() if k.endswith('best.pt')}),indent=2)+'\n')
    print('DECISION FIT COMPLETE',a.regime,a.index,flush=True)
if __name__=='__main__':main()
