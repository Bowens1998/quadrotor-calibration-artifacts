"""New confirmation collection; formal collection requires a protocol lock."""
import argparse,json
import numpy as np
import torch
from _common import ROOT
from icra_confirm_common import BASE,CALIBRATION_JOBS,BRANCH_JOBS,calibration_seed,branch_seed
from icra_adaptation_collect import shifted_randomizer
import winddyn.sim.rollout as rollout
import revision_branching as branches
from winddyn.revision.protocol import dump_new,portable_path
from winddyn.data.writer import save_episode
from winddyn.cfd.field_io import load_field
from winddyn.geometry.procedural import load_manifest
from winddyn.utils.config import load_vehicle


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',required=True,choices=['pilot','calibration','branch']);ap.add_argument('--index',required=True,type=int);a=ap.parse_args();torch.set_num_threads(4)
    if a.kind!='pilot':
        lock=json.loads((BASE/'LOCK.json').read_text());assert lock['formal_collection_authorized']
    if a.kind=='branch':regime,batch=BRANCH_JOBS[a.index]
    elif a.kind=='pilot':regime,campaign=CALIBRATION_JOBS[a.index];assert a.index in (0,1)
    else:regime,campaign=CALIBRATION_JOBS[a.index]
    mf,lf=(1.4,1.) if regime=='mass_1p4' else (1.,3.)
    rollout._randomize=shifted_randomizer(rollout._randomize,mf,lf)
    if a.kind=='branch':
        branches.BASE=BASE/f'icra_confirm_branches_20260908_{regime}_{batch}';branches.BASE.mkdir(exist_ok=False)
        branches.collect('cuda',branch_seed(batch),'uniform',True);return
    pilot=a.kind=='pilot';seed=20012007 if pilot else calibration_seed(campaign)
    dest=BASE/('pilot' if pilot else f'campaign{campaign}')/regime;dest.mkdir(parents=True,exist_ok=False)
    scenes={s.scene_id:s for s in load_manifest(ROOT/'data/manifests/scenes.json')}
    fields=json.loads((ROOT/'data/manifests/wind_fields.json').read_text())['fields']
    candidates=sorted([f for f in fields if f['tag']=='train' and scenes[f['scene_id']].seed<4],key=lambda f:(f['scene_id'],f['wind_id']))
    selected=[candidates[int(i)] for i in np.random.default_rng(seed).choice(len(candidates),2 if pilot else 32,replace=False)]
    specs=[(f,mode) for f in selected for mode in ('tracking','perturbation')]
    out=ROOT/'data'/f'episodes_icra_confirmation_calibration_20260908_{"pilot" if pilot else campaign}_{regime}';entries=[]
    for off in range(0,len(specs),16):
        chunk=specs[off:off+16];tasks=[rollout.EnvTask(scenes[f['scene_id']],load_field(portable_path(f['path'],ROOT)),f['wind_id'],mode) for f,mode in chunk]
        eps=rollout.collect_batch(tasks,load_vehicle(),duration_s=3. if pilot else 7.,seed=seed+200000+off,device='cuda',randomize=True)
        for i,((field,mode),ep) in enumerate(zip(chunk,eps)):
            eid=f'calibration_{off+i:03d}';assert not (out/f'{eid}.npz').exists()
            ep['meta'].update(adaptation_regime=regime,pilot=pilot,calibration_pair=(off+i)//2,calibration_seed=seed,confirmation_campaign=None if pilot else campaign)
            entries.append(save_episode(out,eid,ep))
    dump_new(dest/'calibration_manifest.json',dict(regime=regime,pilot=pilot,campaign=None if pilot else campaign,calibration_seed=seed,mass_factor=mf,lag_factor=lf,episodes=entries))
    print('COMPLETE',a.kind,regime,len(entries),flush=True)
if __name__=='__main__':main()
