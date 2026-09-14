"""Evaluate each scratch model's retained validation-selected neural head."""
import argparse,json
import numpy as np
import torch
from torch import nn
from _common import ROOT
from icra_adaptation_fit import BASE,INPUT_KEYS,physics,task_metrics,sha
from icra_adaptation_scratch import JOBS,make_encoder


@torch.no_grad()
def predict_retained(model,head,arr,prefix,mean,std):
    p=physics(arr,prefix).reshape(-1,90)
    extra=np.concatenate([arr[prefix+'action_fut'].reshape(len(p),-1),p],1)
    x=torch.tensor((extra-mean)/std,dtype=torch.float32);out=[]
    for start in range(0,len(p),128):
        b={k:torch.from_numpy(arr[prefix+k][start:start+128]) for k in INPUT_KEYS}
        out.append(head(torch.cat([model.encode_context(b),x[start:start+128]],1)).numpy())
    return p+np.concatenate(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',required=True,type=int);a=ap.parse_args();torch.set_num_threads(4)
    regime,budget,subset,seed=JOBS[a.index];name=f'budget{budget}_subset{subset}_seed{seed}'
    source=BASE/regime/'scratch_v1'/name;m=json.loads((source/'results.json').read_text())
    dest=BASE/regime/'retained_head_v1'/name;dest.mkdir(parents=True,exist_ok=False)
    assert sha(source/'encoder.pt')==m['encoder_sha256']
    ck=torch.load(source/'encoder.pt',map_location='cpu',weights_only=False)
    model=make_encoder(ck['stats']);model.load_state_dict(ck['model']);model.eval()
    head=nn.Linear(338,90);head.load_state_dict(ck['training_head']);head.eval()
    cache=BASE/regime/'cache';meta=json.loads((cache/'metadata.json').read_text())
    with np.load(cache/'arrays.npz') as z:arr={k:z[k] for k in z.files}
    ids=np.array([e['episode_id'] for e in meta['calibration_episodes']])[arr['cal_episode_index']]
    val=np.isin(ids,m['validation_episodes']);cal=predict_retained(model,head,arr,'cal_',ck['extra_mean'],ck['extra_std'])
    error=float(np.sqrt(np.mean(np.sum((cal[val]-arr['cal_target_position'].reshape(-1,90)[val]).reshape(-1,30,3)**2,-1))))
    assert abs(error-m['selection']['validation_rmse'])<1e-4,(error,m['selection'])
    pred=predict_retained(model,head,arr,'branch_',ck['extra_mean'],ck['extra_std']);assert np.isfinite(pred).all()
    metrics=task_metrics(pred,arr,meta);np.savez_compressed(dest/'outcomes.npz',**metrics)
    result=dict(regime=regime,budget=budget,subset_seed=subset,model_seed=seed,validation_rmse_reproduced=error,
        original_selection=m['selection'],encoder_sha256=m['encoder_sha256'],branch_tasks=meta['branch_tasks'],
        source_sha256=sha(ROOT/'scripts/icra_adaptation_retained_head.py'),scope='development retained head; no refit or test selection')
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',name,flush=True)
if __name__=='__main__':main()
