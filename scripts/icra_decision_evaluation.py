"""Unfitted model comparisons on shared held-command controller-state branches."""
import argparse,json
from types import SimpleNamespace
import numpy as np
import torch
from icra_mpc_matrix_policy import MatrixPredictor
from icra_mpc_policy import physics,features,predict,commands,basis
from _common import ROOT
BASE=ROOT/'runs/icra_decision_evaluation_20260908'
from icra_decision_fit import BASE as FIT_BASE, CONFIGS as FIT_CONFIGS
CONFIGS=[('scalar',0,'calibrated')]+[(m,seed,objective) for m,seed in FIT_CONFIGS for objective in ['trajectory','decision']]

def model_input(a):
    # Deliberate whitelist: branch truth, reference and alive are never model features.
    arr={'x_'+k:np.repeat(a[k],9,axis=0) for k in ['state_hist','action_hist','depth_hist']}
    sf=a['state_hist'][:,-1];c,s=sf[:,11],sf[:,10];v=a['initial_velocity'].copy();vx,vy=v[:,0].copy(),v[:,1].copy();v[:,0]=c*vx+s*vy;v[:,1]=-s*vx+c*vy
    arr['x_vel_yaw_t']=np.repeat(v,9,axis=0)
    arr['x_action_fut']=np.repeat(a['candidate_command'].reshape(-1,4)[:,None],30,axis=1).astype(np.float32)
    return arr

def forecast(p,a):
    arr=model_input(a)
    if p.method=='scalar':pred=physics(arr,'x_',p.tau)
    elif p.method=='axis':
        cmd,v=commands(arr,'x_');parts=[]
        for axis,q in enumerate(p.selected):
            off,x=basis(cmd[...,axis],v[:,axis],q['tau']);parts.append(off+x@np.array([q['gain'],q['bias']]))
        pred=np.stack(parts,-1)
    else:pred=(predict(features(arr,'x_',p.recipe,p.model,p.device),p.fit)+physics(arr,'x_').reshape(108,-1)).reshape(108,30,3)
    pred=pred.reshape(12,9,30,3);sf=a['state_hist'][:,-1];c,s=sf[:,11],sf[:,10];world=pred.copy();world[...,0]=c[:,None,None]*pred[...,0]-s[:,None,None]*pred[...,1];world[...,1]=s[:,None,None]*pred[...,0]+c[:,None,None]*pred[...,1];return world+a['initial_position'][:,None,None]

def cost(world,a):
    u=a['candidate_command'];return ((world-a['reference_world'][:,None])**2).sum(-1).mean(-1)+.02*(u[:,:,:3]**2).sum(-1)+.05*((u[:,:,:3]-a['previous_command'][:,None,:3])**2).sum(-1)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--regime',required=True,choices=['mass_1p4','lag_3']);arg=ap.parse_args();torch.set_num_threads(4)
    dest=BASE/arg.regime;dest.mkdir(parents=True,exist_ok=False);inputs={}
    for path in sorted((ROOT/'runs/icra_snapshot_pilot_20260908'/arg.regime).glob('*.npz')):
        with np.load(path) as z:inputs[path.stem]={k:z[k] for k in z.files}
    assert len(inputs)==9
    accepted=json.loads((FIT_BASE/'accepted.json').read_text());assert accepted['accepted_fits']==84
    for index,(method,seed,objective) in enumerate(CONFIGS):
        p=MatrixPredictor(method,arg.regime,'cuda',seed,901);rows=[]
        p.hashes={k:v for k,v in p.hashes.items() if k.endswith('best.pt')}
        if method=='scalar':
            p.tau=next(r for r in accepted['scalar_new_data'] if r['regime']==arg.regime)['selected']['tau'];p.record(FIT_BASE/'accepted.json')
        else:
            folder=FIT_BASE/arg.regime/f'config{FIT_CONFIGS.index((method,seed)):02d}'
            record=next(r for r in json.loads((folder/'results.json').read_text())['records'] if r['objective']==objective)['selected']
            path=folder/record['weight_file'];p.record(path);assert p.hashes[str(path.relative_to(ROOT))]==record['weight_sha256']
            with np.load(path) as z:p.fit={k:z[k] for k in z.files}

        for name,a in inputs.items():
            world=forecast(p,a);pc=cost(world,a);assert np.isfinite(world).all()
            # Reproduce the actual controller's candidate cost, including penalties.
            b={k:torch.tensor(a[k],device='cuda') for k in ['state_hist','action_hist','depth_hist']}
            buf=SimpleNamespace(batch=lambda:b,vel=[torch.tensor(a['initial_velocity'],device='cuda')])
            _,info=p.plan(buf,a['initial_position'],a['reference_world'],a['candidate_command'][:,4],a['previous_command'])
            np.testing.assert_allclose(pc,info['cost'],atol=1e-5,rtol=1e-6)
            truth=a['position_world'].transpose(2,1,0,3);valid=a['valid'].transpose(2,1,0);tc=cost(truth,a);choice=pc.argmin(1)
            error=world-truth;common=error.mean(1,keepdims=True);contrast=error-common
            for j in range(12):
                full=bool(valid[j].all());prefix=bool(a['initial_alive'][j]);n=int(valid[j].sum())
                row=dict(snapshot=name,task_index=j,prefix_alive=prefix,all_candidates_valid=full,valid_candidate_records=n,position_squared_error=float(np.where(valid[j],(error[j]**2).sum(-1),0).sum()),choice=int(choice[j]),selected_contact=bool(not valid[j,choice[j],-1]) if prefix else None)
                if full:
                    row.update(correct=bool(choice[j]==tc[j].argmin()),regret=float(tc[j,choice[j]]-tc[j].min()),common_mse=float((common[j]**2).sum(-1).mean()),contrast_mse=float((contrast[j]**2).sum(-1).mean()),total_mse=float((error[j]**2).sum(-1).mean()))
                    np.testing.assert_allclose(row['common_mse']+row['contrast_mse'],row['total_mse'],atol=1e-8)
                rows.append(row)
        (dest/f'config{index:03d}.json').write_text(json.dumps(dict(regime=arg.regime,method=method,model_seed=seed,objective=objective,predictor_provenance=p.hashes,rows=rows),indent=2)+'\n')
    print('DECISION EVALUATION COMPLETE',arg.regime,len(CONFIGS),flush=True)
if __name__=='__main__':main()
