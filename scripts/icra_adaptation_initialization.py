"""Matched learned-initialization audit with shared nominal observation statistics."""
import argparse,copy,json,sys
import numpy as np
import torch
from torch import nn
from _common import ROOT
import icra_adaptation_scratch as scratch
from icra_adaptation_fit import TRAIN_BASE,sha,task_metrics
from icra_adaptation_retained_head import predict_retained

BASE=ROOT/'runs/icra_initialization_20260908'
CONTEXT=('step_embed','ctx_gru','depth_enc','ctx_proj')


def initialize(factory,stats,checkpoint,pretrained):
    model=factory(stats)
    if pretrained:
        state=model.state_dict()
        for key,value in checkpoint['model'].items():
            if key.split('.')[0] in CONTEXT:state[key]=value.clone()
        model.load_state_dict(state)
    return model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',required=True,type=int);a=ap.parse_args()
    assert 0<=a.index<400
    arm='random' if a.index<200 else 'pretrained';index=a.index%200
    regime,budget,subset,seed=scratch.JOBS[index]
    checkpoint_path=TRAIN_BASE/'checkpoints'/f'supervised_seed{seed}'/'best.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    # Random arm receives nominal observed-channel statistics but no learned weights.
    stats={k:copy.deepcopy(checkpoint['stats'][k]) for k in ['state_hist','action_hist']}
    factory=scratch.make_encoder
    scratch.make_encoder=lambda incoming:initialize(factory,incoming,checkpoint,arm=='pretrained')
    scratch.fit_stats=lambda arr,mask:copy.deepcopy(stats)
    scratch.BASE=BASE/arm
    sys.argv=[sys.argv[0],'--index',str(index)]
    scratch.main()
    dest=BASE/arm/regime/'scratch_v1'/f'budget{budget}_subset{subset}_seed{seed}'
    m=json.loads((dest/'results.json').read_text())
    m.update(initialization=arm,normalization='shared nominal observed-channel statistics',nominal_checkpoint_sha256=sha(checkpoint_path),wrapper_sha256=sha(ROOT/'scripts/icra_adaptation_initialization.py'))
    if m['status']=='complete':
        # Preserve the neural head in the same run; no separate post-hoc head selection.
        trained=torch.load(dest/'encoder.pt',map_location='cpu',weights_only=False)
        model=factory(trained['stats']);model.load_state_dict(trained['model']);model.eval()
        head=nn.Linear(338,90);head.load_state_dict(trained['training_head']);head.eval()
        cache=BASE/arm/regime/'cache'
        with np.load(cache/'arrays.npz') as z:arr={k:z[k] for k in z.files}
        meta=json.loads((cache/'metadata.json').read_text())
        ids=np.array([e['episode_id'] for e in meta['calibration_episodes']])[arr['cal_episode_index']]
        vm=np.isin(ids,m['validation_episodes'])
        cal=predict_retained(model,head,arr,'cal_',trained['extra_mean'],trained['extra_std'])
        err=float(np.sqrt(np.mean(np.sum((cal[vm]-arr['cal_target_position'].reshape(-1,90)[vm]).reshape(-1,30,3)**2,-1))))
        assert abs(err-m['selection']['validation_rmse'])<1e-4
        pred=predict_retained(model,head,arr,'branch_',trained['extra_mean'],trained['extra_std'])
        assert np.isfinite(pred).all()
        np.savez_compressed(dest/'retained_outcomes.npz',**task_metrics(pred,arr,meta))
        m['retained_validation_rmse_reproduced']=err
    (dest/'results.json').write_text(json.dumps(m,indent=2)+'\n')
    print('MATCHED COMPLETE',arm,index,flush=True)
if __name__=='__main__':main()
