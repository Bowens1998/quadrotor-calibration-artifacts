"""Paired development intervention on outer-loop vertical command authority."""
import argparse,json
import numpy as np
import torch
import icra_mpc_matrix as parent
from icra_mpc_policy import nominal_command
ROOT=parent.ROOT
BASE=ROOT/'runs/icra_vertical_authority_20260908'
CONFIGS=[(r,m,cap) for r in ['mass_1p4','lag_3'] for m in ['feedback','observer'] for cap in [.5,1.5]]

def command_with_cap(pos,ref,velocity,cap):
    out=nominal_command(pos,ref,velocity)
    out[:,2]=np.clip(velocity[:,2]+ref[:,2]-pos[:,2],-cap,cap)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);a=ap.parse_args();torch.set_num_threads(4)
    regime,method,cap=CONFIGS[a.index];dest=BASE/f'config{a.index:03d}';dest.mkdir(parents=True,exist_ok=False)
    episodes=[]
    for ti,tier in enumerate(['id','wind_extrap','joint_extrap']):
        calls=0
        def command(pos,ref,velocity):
            nonlocal calls
            calls+=1
            return command_with_cap(pos,ref,velocity,.5 if calls<=100 else cap)
        parent.nominal_command=command
        tasks=parent.build_tasks(tier,12,selection_seed=55112007+ti,waypoint_base=56112007)
        for t in tasks:t['tier']=tier
        out,arrays=parent.rollout(tasks,regime,method,'cuda',0,None,57112007+1000*ti)
        assert calls==600
        np.savez_compressed(dest/f'{tier}.npz',**arrays);episodes.extend(out['episodes'])
    (dest/'results.json').write_text(json.dumps(dict(stage='post-hoc development intervention',regime=regime,method=method,vertical_cap=cap,index=a.index,episodes=episodes),indent=2)+'\n')
    print('VERTICAL AUTHORITY COMPLETE',a.index,flush=True)
if __name__=='__main__':main()
