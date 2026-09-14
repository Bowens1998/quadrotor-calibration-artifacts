"""No-refit development diagnosis of shared versus action-specific errors."""
import argparse,json
import numpy as np
import torch
from _common import ROOT
from icra_adaptation_fit import features,physics,sha,TRAIN_BASE,task_metrics
from icra_adaptation_ridge import predict
from winddyn.train.trainer import load_model
from icra_sysid_development import basis,commands
BASE=ROOT/'runs/icra_action_decomposition_20260908'
OLD=ROOT/'runs/icra_adaptation_20260907'

def decompose(pred,truth,actual,goals,valid):
    e=pred-truth;mean=e.mean(1,keepdims=True);contrast=e-mean
    total=(e**2).sum(-1).mean((1,2));common=(mean**2).sum(-1).mean((1,2));action=(contrast**2).sum(-1).mean((1,2))
    np.testing.assert_allclose(total,common+action,rtol=1e-9,atol=1e-10)
    costs=((pred[:,:,-1,:2]-goals[:,None])**2).sum(-1)
    ce=costs-actual;pairs=np.stack([ce[:,a]-ce[:,b] for a in range(4) for b in range(a+1,4)],1)
    ordered=np.sort(actual,axis=1)
    return dict(all_valid=valid.all(1),trajectory_sq=total,common_sq=common,action_contrast_sq=action,cost_difference_sq=(pairs**2).mean(1),oracle_margin=ordered[:,1]-ordered[:,0])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);args=ap.parse_args();regime=['mass_1p4','lag_3'][args.index];torch.set_num_threads(2)
    cache=OLD/regime/'cache';meta=json.loads((cache/'metadata.json').read_text())
    with np.load(cache/'arrays.npz') as z:arr={k:z[k] for k in z.files}
    tasks=sorted([t for t in meta['branch_tasks'] if 'sample_index' in t],key=lambda t:t['sample_index']);goals=np.array([t['goal_yaw'] for t in tasks]);truth=arr['branch_target_position'];dest=BASE/regime;dest.mkdir(exist_ok=False)
    rows=[];hashes={}
    def verify(p,h=None):
        key=str(p.relative_to(ROOT))
        if key not in hashes:hashes[key]=sha(p)
        if h is not None:assert hashes[key]==h
    for p in [cache/'arrays.npz',cache/'metadata.json',ROOT/'scripts/icra_action_decomposition.py',BASE/'PROTOCOL.md']:verify(p)
    def add(pred,method,subset,seed,reference=None):
        pred=pred.reshape(truth.shape);out=task_metrics(pred,arr,meta)
        if reference is not None:
            np.testing.assert_allclose(out['position_sq'],reference['position_sq'],rtol=1e-5,atol=1e-5)
            np.testing.assert_array_equal(out['choice'],reference['choice'])
        decom=decompose(pred,truth,arr['branch_cost_actual'],goals,arr['branch_valid']);mask=decom['all_valid'];assert mask.any()
        row=dict(method=method,subset=subset,seed=seed,regime=regime,all_valid=int(mask.sum()),prefix_valid=len(mask),valid_branch_rmse=float(np.sqrt(out['position_sq'].sum()/out['n_positions'].sum())),accuracy=float(out['correct'][mask].mean()),regret=float(out['regret'][mask].mean()),selected_contact=float(out['selected_collision'].mean()))
        for k in ['trajectory_sq','common_sq','action_contrast_sq','cost_difference_sq']:row[k]=float(decom[k][mask].mean())
        row['oracle_margin_quantiles']=np.quantile(decom['oracle_margin'][mask],[0,.25,.5,.75,1]).tolist()
        np.savez_compressed(dest/f'{method}_subset{subset}_seed{seed}.npz',**decom,**{k:out[k] for k in ['correct','regret','choice','selected_collision']});rows.append(row)
    for recipe,seeds in [('physics_only',[0]),('supervised',range(5))]:
        for seed in seeds:
            old=OLD/regime/'formal_v1'/f'{recipe}_seed{seed}';m=json.loads((old/'results.json').read_text());model=None
            if recipe=='supervised':
                ck=TRAIN_BASE/'checkpoints'/f'supervised_seed{seed}/best.pt';verify(ck);model=load_model(ck,'cpu');model.eval()
            xb=features(arr,'branch_',recipe,model,'cpu');prior=physics(arr,'branch_').reshape(len(xb),-1)
            with np.load(old/'outcomes.npz') as z:ref={k:z[k] for k in z.files}
            for subset in range(901,906):
                rec=next(r for r in m['records'] if r['subset_seed']==subset and r['budget']==32 and r['center']=='physical');wp=old/rec['weight_file'];verify(wp,rec['weight_sha256'])
                with np.load(wp) as z:fit={k:z[k] for k in z.files}
                add(predict(xb,fit)+prior,'frozen_supervised' if model is not None else 'physics_features',subset,seed,{k:v[rec['outcome_index']] for k,v in ref.items()})
                if recipe=='physics_only':
                    pr=next(r for r in m['physical_records'] if r['subset_seed']==subset and r['budget']==32 and r['name']=='calibrated')
                    add(physics(arr,'branch_',pr['tau']),'scalar_response',subset,0)
    ub,vb=commands(arr,'branch_')
    for subset in range(901,906):
        d=ROOT/'runs/icra_sysid_development_20260908'/regime/f'budget32_subset{subset}';m=json.loads((d/'results.json').read_text());verify(d/'weights.npz',m['weights_sha256']);parts=[]
        for axis,q in enumerate(m['selected']):
            off,x=basis(ub[...,axis],vb[:,axis],q['tau']);parts.append(off+x@np.array([q['gain'],q['bias']]))
        with np.load(d/'outcomes.npz') as z:ref={k:z[k] for k in z.files}
        add(np.stack(parts,-1),'axis_response',subset,0,ref)
    assert len(rows)==40
    (dest/'results.json').write_text(json.dumps(dict(stage='development descriptive; no refitting',records=rows,branch_tasks=meta['branch_tasks']),indent=2)+'\n');(dest/'provenance.json').write_text(json.dumps(hashes,indent=2)+'\n');print('DECOMPOSITION COMPLETE',regime,flush=True)
if __name__=='__main__':main()
