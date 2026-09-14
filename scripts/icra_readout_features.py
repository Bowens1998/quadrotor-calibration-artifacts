"""Give both ridge arms exactly the same explicit physical prediction feature."""
import argparse,json
import numpy as np
import torch
from _common import ROOT
import revision_residual as residual
from icra_readout_factorial import zero_prior

base_features=residual.features
physical_prediction=residual.physics


def augmented_features(model,batch):
    return np.concatenate([base_features(model,batch),physical_prediction(batch).reshape(len(batch['action_fut']),-1)],axis=1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True,choices=['blind','pajepa','direct','supervised','supervised_wind','wind_only','raw']);p.add_argument('--seed',type=int,default=0);p.add_argument('--prior',choices=['zero','physical'],required=True);a=p.parse_args()
    torch.set_num_threads(4);name=f'{a.recipe}_seed{a.seed}';base=ROOT/'runs/icra_readout_features_20260907'/a.prior;dest=base/'fits'/name;dest.mkdir(parents=True,exist_ok=False)
    model=None if a.recipe=='raw' else residual.load_model(residual.TRAIN_BASE/'checkpoints'/name/'best.pt','cuda')
    residual.features=augmented_features
    if a.prior=='zero':residual.physics=zero_prior
    # TAU stays .3 for the explicit physical feature in BOTH arms.
    fit=residual.train_readout(model,a.recipe,a.seed,'cuda',dest)
    residual.evaluate_frozen(model,fit,name,residual.BRANCH,base,'cuda')
    for path in [dest/'fit.json',base/'branch_metrics'/f'{name}.json']:
        d=json.loads(path.read_text());d.update(additive_prior=a.prior,physical_feature_tau=.3,feature_spec='original_features concatenated with flattened P_0.3s');path.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
