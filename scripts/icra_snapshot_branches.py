"""Held-command counterfactuals from identical simulator/controller snapshots."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
import icra_snapshot_rollout as parent
from icra_mpc_policy import candidates,reference_at
from icra_vertical_authority import command_with_cap
BASE=parent.ROOT/'runs/icra_snapshot_pilot_20260908'

def tensor_state(sim,ctrl):
    out={}
    for name,obj in [('sim',sim),('state',sim.state),('rotors',sim.rotors),('drag',sim.drag),('ctrl',ctrl)]:
        for key,value in vars(obj).items():
            if torch.is_tensor(value):out[name+'.'+key]=value.detach().clone()
    return out

@torch.no_grad()
def branch(sim,ctrl,thrust,command,alive,centers,half,mask):
    ss,cc=copy.deepcopy((sim,ctrl));force=thrust.clone();valid=alive.clone();positions=[];validity=[]
    for k in range(300):
        if k>=4 and k%4==0:force=cc.compute(ss.state.pos,ss.state.quat,ss.state.lin_vel,ss.state.ang_vel,command,.02)
        previous=[getattr(ss.state,key).clone() for key in ['pos','quat','lin_vel','ang_vel']]
        ss.step(force)
        for key,value in zip(['pos','quat','lin_vel','ang_vel'],previous):getattr(ss.state,key)[~valid]=value[~valid]
        hit=(parent._sdf(ss.state.pos,centers,half,mask)<.25)|(ss.state.pos[:,2]<.1);valid &= ~hit
        if (k+1)%10==0:positions.append(ss.state.pos.cpu().numpy().copy());validity.append(valid.cpu().numpy().copy())
    return np.array(positions),np.array(validity)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--regime',required=True,choices=['mass_1p4','lag_3']);a=ap.parse_args();torch.set_num_threads(4)
    dest=BASE/a.regime;dest.mkdir(parents=True,exist_ok=False);metadata=[]
    for ti,tier in enumerate(['id','wind_extrap','joint_extrap']):
        # Keep all 12 parent envs so randomization exactly matches accepted matrix; pilot writes all.
        tasks=parent.build_tasks(tier,12,selection_seed=55112007+ti,waypoint_base=56112007)
        for task in tasks:task['tier']=tier
        calls=0
        def command(pos,ref,velocity):
            nonlocal calls
            calls+=1
            return command_with_cap(pos,ref,velocity,.5 if calls<=100 else 1.5)
        parent.nominal_command=command
        def hook(sim,ctrl,buf,thrust,active,alive,centers,half,mask,wps,zref,step):
            before=tensor_state(sim,ctrl);clock=sim.time
            ref,vel=reference_at(wps,[step/200],zref);nom=command_with_cap(sim.state.pos.cpu().numpy(),ref[:,0],vel[:,0],1.5);u=candidates(nom)
            pp=[];vv=[]
            for c in range(9):
                p,v=branch(sim,ctrl,thrust,torch.as_tensor(u[:,c],dtype=torch.float32,device=sim.device),alive,centers,half,mask);pp.append(p);vv.append(v)
            p,v=branch(sim,ctrl,thrust,torch.as_tensor(u[:,4],dtype=torch.float32,device=sim.device),alive,centers,half,mask)
            np.testing.assert_array_equal(p,pp[4]);np.testing.assert_array_equal(v,vv[4])
            after=tensor_state(sim,ctrl)
            for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
            assert sim.time==clock
            arrays={k:value.cpu().numpy() for k,value in buf.batch().items()}
            arrays.update(position_world=np.stack(pp,1),valid=np.stack(vv,1),candidate_command=u,initial_position=sim.state.pos.cpu().numpy(),initial_velocity=sim.state.lin_vel.cpu().numpy(),initial_alive=alive.cpu().numpy(),reference_world=reference_at(wps,step/200+.05*np.arange(1,31),zref)[0],previous_command=active.cpu().numpy())
            np.savez_compressed(dest/f'{tier}_step{step}.npz',**arrays)
            metadata.append(dict(tier=tier,step=step,activation_step=step+4,tasks=12,prefix_alive=int(alive.sum()),duplicate_center_exact=True,parent_unmodified=True))
        out,arrays=parent.rollout(tasks,a.regime,'feedback','cuda',0,None,57112007+1000*ti,snapshot_hook=hook)
        # Instrumentation must not alter parent control trajectory, including randomization.
        original=parent.ROOT/'runs/icra_vertical_authority_20260908'/('config001' if a.regime=='mass_1p4' else 'config005')
        with np.load(original/f'{tier}.npz') as z:
            for key in z.files:np.testing.assert_allclose(arrays[key],z[key],atol=1e-5,rtol=0)
        metadata[-1]['full_parent_replay_verified']=True
    (dest/'acceptance.json').write_text(json.dumps(dict(regime=a.regime,snapshots=metadata,stage='mechanical pilot; no model comparisons'),indent=2)+'\n')
    print('SNAPSHOT PILOT COMPLETE',a.regime,flush=True)
if __name__=='__main__':main()
