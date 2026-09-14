"""Matched linear readout without the additive physical prediction."""
import argparse,json,time
import numpy as np
import torch
from _common import ROOT
import revision_residual as residual
from winddyn.revision.protocol import dump_new


def zero_prior(batch):
    actions=batch['action_fut']
    return np.zeros((len(actions),actions.shape[1],3),dtype=float)


def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True,choices=['blind','pajepa','direct','supervised','supervised_wind','wind_only','raw']);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(4);name=f'{a.recipe}_seed{a.seed}';base=ROOT/'runs/icra_readout_factorial_20260907/zero';dest=base/'fits'/name;dest.mkdir(parents=True,exist_ok=False)
    model=None if a.recipe=='raw' else residual.load_model(residual.TRAIN_BASE/'checkpoints'/name/'best.pt','cuda')
    # Runtime substitution is confined to this process; shared source, existing
    # fits and the already completed confirmation remain unchanged.
    residual.physics=zero_prior;residual.TAU=None
    fit=residual.train_readout(model,a.recipe,a.seed,'cuda',dest)
    residual.evaluate_frozen(model,fit,name,residual.BRANCH,base,'cuda')
    dump_new(dest/'diagnostic.json',dict(prior_mode='zero',reference='runs/revision_residual_20260907',evaluation='old goaldev development only',recipe=a.recipe,seed=a.seed))
if __name__=='__main__':main()
