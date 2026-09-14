"""Independent shifted-plant calibration episodes and development branches."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from _common import ROOT
import winddyn.sim.rollout as rollout
import revision_branching as branches
from winddyn.revision.protocol import dump_new,portable_path
from winddyn.data.writer import save_episode
from winddyn.cfd.field_io import load_field
from winddyn.geometry.procedural import load_manifest
from winddyn.utils.config import load_vehicle

BASE=ROOT/'runs/icra_adaptation_20260907'


def shifted_randomizer(original,mass_factor,lag_factor):
    def randomize(sim,vp,gen):
        scales={k:v.copy() for k,v in original(sim,vp,gen).items()}
        sim.mass=sim.mass*mass_factor
        sim.rotors.time_constant=sim.rotors.time_constant*lag_factor
        scales['adapt_mass_multiplier']=np.full(sim.num_envs,mass_factor)
        scales['adapt_lag_multiplier']=np.full(sim.num_envs,lag_factor)
        scales['realized_mass_kg']=sim.mass.cpu().numpy().copy()
        scales['realized_motor_tau_s']=sim.rotors.time_constant.cpu().numpy().reshape(-1).copy()
        return scales
    return randomize


def main():
    p=argparse.ArgumentParser();p.add_argument('--regime',choices=['mass_1p4','lag_3'],required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    mf,lf=(1.4,1.) if a.regime=='mass_1p4' else (1.,3.)
    rollout._randomize=shifted_randomizer(rollout._randomize,mf,lf)
    tag=('pilot_' if a.pilot else '')+a.regime
    dest=BASE/tag;dest.mkdir(parents=True,exist_ok=False)
    scenes={s.scene_id:s for s in load_manifest(ROOT/'data/manifests/scenes.json')}
    fields=json.loads((ROOT/'data/manifests/wind_fields.json').read_text())['fields']
    candidates=sorted([f for f in fields if f['tag']=='train' and scenes[f['scene_id']].seed<4],key=lambda f:(f['scene_id'],f['wind_id']))
    seed=6712007+(990000 if a.pilot else 0);rng=np.random.default_rng(seed)
    selected=[candidates[int(i)] for i in rng.choice(len(candidates),2 if a.pilot else 32,replace=False)]
    specs=[(f,mode) for f in selected for mode in ['tracking','perturbation']]
    entries=[];out=ROOT/'data'/f'episodes_icra_adaptation_20260907_{tag}'
    for off in range(0,len(specs),16):
        chunk=specs[off:off+16]
        tasks=[rollout.EnvTask(scenes[f['scene_id']],load_field(portable_path(f['path'],ROOT)),f['wind_id'],mode) for f,mode in chunk]
        eps=rollout.collect_batch(tasks,load_vehicle(),duration_s=3. if a.pilot else 7.,seed=seed+200000+off,device='cuda',randomize=True)
        for i,((field,mode),ep) in enumerate(zip(chunk,eps)):
            eid=f'calibration_{off+i:03d}'
            if (out/f'{eid}.npz').exists():raise FileExistsError(eid)
            ep['meta'].update(adaptation_regime=a.regime,pilot=a.pilot,calibration_pair=(off+i)//2,calibration_seed=seed)
            entries.append(save_episode(out,eid,ep))
    dump_new(dest/'calibration_manifest.json',dict(regime=a.regime,pilot=a.pilot,mass_factor=mf,lag_factor=lf,episodes=entries))
    print(a.regime,'calibration episodes',len(entries),flush=True)
    if not a.pilot:
        branches.BASE=ROOT/f'runs/icra_adaptation_branches_20260907_{a.regime}'
        branches.BASE.mkdir(exist_ok=False)
        branches.collect('cuda',8712007,'uniform',True)
    print(a.regime,'collection complete',flush=True)
if __name__=='__main__':main()
