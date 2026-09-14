"""Fixed matched adaptation on a new independent confirmation calibration pool."""
import argparse,copy,json,sys
import numpy as np
import torch
from torch import nn
from _common import ROOT
from icra_confirm_common import BASE,FIXTURE,ADAPT_JOBS
import icra_adaptation_scratch as scratch
from icra_adaptation_initialization import initialize
from icra_adaptation_fit import TRAIN_BASE,sha,task_metrics
from icra_adaptation_retained_head import predict_retained


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',required=True,type=int);ap.add_argument('--fixture',action='store_true');a=ap.parse_args()
    arm,regime,budget,campaign,seed=ADAPT_JOBS[a.index];root=FIXTURE if a.fixture else BASE
    if not a.fixture:assert json.loads((BASE/'LOCK.json').read_text())['formal_collection_authorized']
    meta=json.loads((root/f'campaign{campaign}'/regime/'cache/metadata.json').read_text())
    assert meta['fixture']==a.fixture and meta['campaign']==campaign
    path=TRAIN_BASE/'checkpoints'/f'supervised_seed{seed}'/'best.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    stats={k:copy.deepcopy(checkpoint['stats'][k]) for k in ['state_hist','action_hist']};factory=scratch.make_encoder
    scratch.make_encoder=lambda incoming:initialize(factory,incoming,checkpoint,arm=='pretrained')
    scratch.fit_stats=lambda arr,mask:copy.deepcopy(stats);scratch.BASE=root/f'campaign{campaign}'/arm
    scratch.JOBS=[(regime,budget,901,seed)];sys.argv=[sys.argv[0],'--index','0'];scratch.main()
    dest=root/f'campaign{campaign}'/arm/regime/'scratch_v1'/f'budget{budget}_subset901_seed{seed}'
    m=json.loads((dest/'results.json').read_text());m.update(initialization=arm,campaign=campaign,fixture=a.fixture,stage='excluded fixture' if a.fixture else 'confirmation',normalization='shared nominal observed-channel statistics',nominal_checkpoint_sha256=sha(path),wrapper_sha256=sha(ROOT/'scripts/icra_confirm_adapt.py'))
    if m['status']=='complete':
        ck=torch.load(dest/'encoder.pt',map_location='cpu',weights_only=False);model=factory(ck['stats']);model.load_state_dict(ck['model']);model.eval()
        head=nn.Linear(338,90);head.load_state_dict(ck['training_head']);head.eval()
        with np.load(root/f'campaign{campaign}'/regime/'cache/arrays.npz') as z:arr={k:z[k] for k in z.files}
        ids=np.array([e['episode_id'] for e in meta['calibration_episodes']])[arr['cal_episode_index']];vm=np.isin(ids,m['validation_episodes'])
        cal=predict_retained(model,head,arr,'cal_',ck['extra_mean'],ck['extra_std'])
        err=float(np.sqrt(np.mean(np.sum((cal[vm]-arr['cal_target_position'].reshape(-1,90)[vm]).reshape(-1,30,3)**2,-1))))
        assert abs(err-m['selection']['validation_rmse'])<1e-4
        pred=predict_retained(model,head,arr,'branch_',ck['extra_mean'],ck['extra_std']);assert np.isfinite(pred).all()
        np.savez_compressed(dest/'retained_outcomes.npz',**task_metrics(pred,arr,meta));m['retained_validation_rmse_reproduced']=err
    (dest/'results.json').write_text(json.dumps(m,indent=2)+'\n');print('CONFIRM ADAPT COMPLETE',arm,regime,campaign,budget,seed,flush=True)
if __name__=='__main__':main()
