"""Low-dimensional response identification; old development caches only."""
import argparse,json
import numpy as np
from _common import ROOT
from icra_adaptation_fit import sha,task_metrics
BASE=ROOT/'runs/icra_sysid_development_20260908'
OLD=ROOT/'runs/icra_adaptation_20260907'
TAUS=[.1,.2,.3,.5,.7,1.,1.5]
JOBS=[(r,b,s) for r in ['mass_1p4','lag_3'] for b in [32,64] for s in range(901,906)]

def basis(u,v,tau,dt=.05):
    """Exact affine trajectory in gain and steady-velocity bias, for one axis."""
    decay=np.exp(-dt/tau);vv=v.copy();p=np.zeros_like(v);vg=np.zeros_like(v);pg=np.zeros_like(v);vb=np.zeros_like(v);pb=np.zeros_like(v)
    offsets=[];features=[]
    for k in range(u.shape[1]):
        p+=tau*(1-decay)*vv;vv*=decay
        pg+=dt*u[:,k]+tau*(1-decay)*(vg-u[:,k]);vg=decay*vg+(1-decay)*u[:,k]
        pb+=dt+tau*(1-decay)*(vb-1);vb=decay*vb+(1-decay)
        offsets.append(p.copy());features.append(np.stack([pg,pb],-1))
    return np.stack(offsets,1),np.stack(features,1)

def commands(arr,prefix):
    u=arr[prefix+'action_fut'][...,:3].astype(float).copy();state=arr[prefix+'state_hist'][:,-1];c,s=state[:,11],state[:,10]
    x,y=u[...,0].copy(),u[...,1].copy();u[...,0]=c[:,None]*x+s[:,None]*y;u[...,1]=-s[:,None]*x+c[:,None]*y
    return u,arr[prefix+'vel_yaw_t'].astype(float)

def fit_axis(u,v,y,fitmask,valmask):
    grid=[];best=None
    for tau in TAUS:
        offset,x=basis(u,v,tau)
        coef=np.linalg.lstsq(x[fitmask].reshape(-1,2),(y-offset)[fitmask].reshape(-1),rcond=None)[0]
        mse=float(np.mean((offset[valmask]+x[valmask]@coef-y[valmask])**2));grid.append(dict(tau=tau,gain=float(coef[0]),bias=float(coef[1]),validation_mse=mse))
        if best is None or mse<best['validation_mse']:best=grid[-1]
    return dict(best),grid

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);a=ap.parse_args();r,b,s=JOBS[a.index]
    cache=OLD/r/'cache';m=json.loads((cache/'metadata.json').read_text());split=m['subsets'][str(s)][str(b)]
    with np.load(cache/'arrays.npz') as z:arr={k:z[k] for k in z.files}
    ids=np.array([e['episode_id'] for e in m['calibration_episodes']])[arr['cal_episode_index']];fm=np.isin(ids,split['fit']);vm=np.isin(ids,split['validation']);assert fm.any() and vm.any() and not (fm&vm).any()
    u,v=commands(arr,'cal_');ub,vb=commands(arr,'branch_');pred=[];selected=[];grids=[]
    for axis in range(3):
        best,grid=fit_axis(u[...,axis],v[:,axis],arr['cal_target_position'][...,axis],fm,vm)
        offset,x=basis(ub[...,axis],vb[:,axis],best['tau']);pred.append(offset+x@np.array([best['gain'],best['bias']]))
        selected.append(best);grids.append(grid)
    prediction=np.stack(pred,-1);assert np.isfinite(prediction).all();out=task_metrics(prediction,arr,m)
    d=BASE/r/f'budget{b}_subset{s}';d.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(d/'outcomes.npz',**out);np.savez_compressed(d/'weights.npz',tau=[q['tau'] for q in selected],gain=[q['gain'] for q in selected],bias=[q['bias'] for q in selected])
    result=dict(stage='development',regime=r,budget=b,subset=s,split=split,selected=selected,grids=grids,branch_tasks=m['branch_tasks'],fit_windows=int(fm.sum()),validation_windows=int(vm.sum()),weights_sha256=sha(d/'weights.npz'))
    (d/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    sources=[ROOT/'scripts/icra_sysid_development.py',ROOT/'scripts/icra_adaptation_fit.py',cache/'metadata.json',cache/'arrays.npz',BASE/'PROTOCOL.md']
    (d/'provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sources},indent=2)+'\n')
    print('SYSID COMPLETE',r,b,s,flush=True)
if __name__=='__main__':main()
