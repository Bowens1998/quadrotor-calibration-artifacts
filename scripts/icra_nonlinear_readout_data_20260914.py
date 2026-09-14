"""Prepare source-verified C1 lag features for a frozen-encoder readout study.

No fitting, hyperparameter selection, model ranking, or simulation is performed.
Missing original C1 calibration arrays are an error; paired-data substitutes are
not accepted. All outputs are new and an existing output directory is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from _common import ROOT
from icra_confirm_frozen import features, physics
from icra_snapshot_prediction import model_input, cost
from winddyn.train.trainer import load_model

C1 = ROOT/'runs/icra_calibration_confirmation_20260908'
CAMPAIGN = ROOT/'runs/icra_campaign_snapshot_20260909'
RECOVER = ROOT/'data/icra_campaign_snapshot_20260909/recovered'
SNAPSHOTS = ROOT/'runs/icra_snapshot_pilot_20260908/lag_3'
CONTINUATION = ROOT/'runs/icra_action_continuation_20260909/attempt02'
DEST = ROOT/'runs/icra_nonlinear_readout_20260914/prepared'
EXPECTED_WINDOWS = [(864,316,100),(859,330,112),(880,322,112),(873,336,112),(846,326,103)]
EXPECTED_CACHE_SHA = [
    'c963380a777ff2ed882be9a242b27e873ff73c2ea4b9fb129a2b4c4015067810',
    'd8cab234e357738dd8a4747dac2079b3e11a46d68a6ed9a6758ff45d21cef671',
    '954ce75bc35d5192e2b82efb596289d876cfaad63e8dd95212dd64b6352a20fa',
    '5c2f230297f07977b9428a5bb9c14648adc08037d95ff9a957a831517dbd30bc',
    '701eda2240a0b88864d6bf7776919b92535ec60fb55d7562342c9eb1349ad438',
]
MODES = ['held','nominal','observer']
CAL_KEYS = ['cal_state_hist','cal_action_hist','cal_action_fut','cal_depth_hist',
            'cal_vel_yaw_t','cal_target_position','cal_episode_index']


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def key(path):
    p=Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def checked(path, expected=None, sources=None):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Required original source missing: {path}')
    value=sha(path)
    if expected is not None and value!=expected:raise ValueError(f'Source SHA256 mismatch: {path}')
    if sources is not None:sources[key(path)]=value
    return path


def load_arrays(path, keys=None):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in (z.files if keys is None else keys)}


def check_science_lock(sources=None):
    path=ROOT/'runs/icra_selection_validation_20260912/LOCK.json'
    doc=read(path);checked(path,sources=sources)
    for name,wanted in doc['source_hashes'].items():checked(ROOT/name,wanted)
    return dict(path=key(path),sha256=sha(path),files=len(doc['source_hashes']),all_unchanged=True)


def resolve_calibration_cache(calibration_root, campaign):
    """Root mirrors C1: <root>/campaignN/lag_3/cache/arrays.npz."""
    return Path(calibration_root)/f'campaign{campaign}/lag_3/cache/arrays.npz'


def calibration_sources(calibration_root=C1, sources=None):
    """Validate all five exact arrays before any extraction or output creation."""
    paths=[]
    for campaign in range(5):
        source=C1/f'campaign{campaign}/lag_3'
        prov_path=source/'frozen_v1/supervised_seed0/provenance.json'
        prov=read(checked(prov_path,sources=sources))
        canonical=key(source/'cache/arrays.npz')
        if prov[canonical]!=EXPECTED_CACHE_SHA[campaign]:raise ValueError('Original C1 provenance differs from audited cache hash')
        p=resolve_calibration_cache(calibration_root,campaign)
        if not p.is_file():
            raise FileNotFoundError(f'Original C1 campaign {campaign} cache is missing: {p}. Restore the exact cache (SHA {EXPECTED_CACHE_SHA[campaign]}); paired calibration is not a valid substitute.')
        checked(p,EXPECTED_CACHE_SHA[campaign],sources)
        m=source/'cache/metadata.json';checked(m,prov[key(m)],sources)
        paths.append((p,m))
    return paths


def build_masks(arr,meta):
    """Use episode-level original subset901/budget32; never split windows anew."""
    parts=meta['subsets']['901']['32']
    if len(parts['fit'])!=24 or len(parts['validation'])!=8 or set(parts['fit'])&set(parts['validation']):
        raise ValueError('Not the original 24-FIT / 8-VAL episode split')
    ids=np.asarray([e['episode_id'] for e in meta['calibration_episodes']])[arr['cal_episode_index']]
    fm=np.isin(ids,parts['fit']);vm=np.isin(ids,parts['validation'])
    if np.any(fm&vm) or not fm.any() or not vm.any():raise ValueError('Invalid calibration window masks')
    return ids,fm,vm


def load_encoders(device='cpu',sources=None):
    inv=read(checked(CAMPAIGN/'inventory.json',sources=sources));models=[]
    extra=read(checked(CAMPAIGN/'additional_nominal_recovery.json',sources=sources))['files']
    recovered={**inv['recover_files'],**extra}
    for seed in range(5):
        rel=f'runs/revision_20260907/checkpoints/supervised_seed{seed}/best.pt'
        p=checked(RECOVER/rel,recovered[rel],sources)
        model=load_model(p,device).eval().requires_grad_(False)
        if model.use_wind or model.privileged or model.jepa or model.latent!=128:
            raise ValueError('Wrong supervised nominal encoder configuration')
        models.append(model)
    return models


def world_to_local(world,a):
    n=len(a['initial_position']);world=np.asarray(world,dtype=np.float64)
    delta=world-a['initial_position'].reshape((n,)+(1,)*(world.ndim-2)+(3,))
    sf=a['state_hist'][:,-1];shape=(n,)+(1,)*(world.ndim-2)
    c=sf[:,11].astype(float).reshape(shape);s=sf[:,10].astype(float).reshape(shape)
    out=delta.copy();out[...,0]=c*delta[...,0]+s*delta[...,1];out[...,1]=-s*delta[...,0]+c*delta[...,1]
    return out


@torch.no_grad()
def prepare_calibration(campaign,cache_path,meta_path,models,device='cpu'):
    """Return arrays, metadata. Features[encoder_seed,window,338], no fits."""
    checked(cache_path,EXPECTED_CACHE_SHA[campaign])
    arr=load_arrays(cache_path,CAL_KEYS);meta=read(meta_path)
    if meta['regime']!='lag_3' or meta['campaign']!=campaign or meta['fixture']:
        raise ValueError('Wrong original calibration cache')
    for a in arr.values():
        if not np.isfinite(a).all():raise ValueError('Non-finite calibration source')
    ids,fm,vm=build_masks(arr,meta)
    if (len(ids),int(fm.sum()),int(vm.sum()))!=EXPECTED_WINDOWS[campaign]:
        raise ValueError('Calibration windows do not match original campaign records')
    feats=np.stack([features(arr,'cal_','supervised',m,device) for m in models])
    nominal=physics(arr,'cal_',.3)
    if feats.shape!=(5,len(ids),338):raise ValueError('Unexpected frozen-context feature dimensions')
    np.testing.assert_array_equal(feats[:,:,128:248],np.broadcast_to(arr['cal_action_fut'].reshape(len(ids),120),(5,len(ids),120)))
    np.testing.assert_array_equal(feats[:,:,248:],np.broadcast_to(nominal.reshape(len(ids),90),(5,len(ids),90)))
    out=dict(features=feats,target=arr['cal_target_position'],nominal=nominal,
             residual_target=arr['cal_target_position'].astype(float)-nominal,
             fit_mask=fm,validation_mask=vm,episode_index=arr['cal_episode_index'],episode_id=ids)
    # Retain measured inputs so later feature checks do not depend on hidden caches.
    out.update({k:arr['cal_'+k] for k in ['state_hist','action_hist','action_fut','depth_hist','vel_yaw_t']})
    info=dict(campaign=campaign,regime='lag_3',all_windows=len(ids),fit_windows=int(fm.sum()),
              validation_windows=int(vm.sum()),unused_windows=int((~(fm|vm)).sum()),
              subset_seed=901,budget_episodes=32,fit_episodes=meta['subsets']['901']['32']['fit'],
              validation_episodes=meta['subsets']['901']['32']['validation'],
              all_episode_metadata=meta['calibration_episodes'])
    return out,info


@torch.no_grad()
def prepare_evaluation(models=None,device='cpu',sources=None):
    """Return existing nine-snapshot features/truth, with no rollout or fitting.

    features: (5 encoder seeds,9 snapshot files,12 tasks,9 candidates,338)
    truth/cost modes: held, nominal, observer. Common validity is across every
    candidate/time/mode. All arrays retain their original file/task ordering.
    """
    if models is None:models=load_encoders(device,sources)
    snapshot_lock=read(checked(CAMPAIGN/'attempt01/source_lock.json',sources=sources))
    output_lock=read(checked(CAMPAIGN/'attempt01/output_hashes.json',sources=sources))
    continuation_lock=read(checked(CONTINUATION/'output_hashes.json',sources=sources))
    config=read(checked(CAMPAIGN/'attempt01/config095.json',output_lock[key(CAMPAIGN/'attempt01/config095.json')],sources))
    rowmap={(r['snapshot'],r['task_index']):r for r in config['rows']}
    paths=sorted(SNAPSHOTS.glob('*.npz'))
    if len(paths)!=9:raise ValueError('Expected nine original lag snapshot files')
    accepted=[]
    for seed in range(5):
        p=ROOT/f'data/icra_campaign_snapshot_20260909/attempt01/config{95+seed:03d}.npz'
        accepted.append(load_arrays(checked(p,output_lock[key(p)],sources),['context'])['context'])
    arrays={k:[] for k in ['features','nominal','held_target_local','position_world','valid','cost',
        'common_valid','initial_alive','candidate_command','previous_command','initial_position',
        'initial_velocity','reference_world','state_hist','action_hist','depth_hist','scene','tier']}
    context_max=[]
    for si,path in enumerate(paths):
        a=load_arrays(checked(path,snapshot_lock[key(path)],sources));inp=model_input(a)
        fs=np.stack([features(inp,'x_','supervised',m,device) for m in models])
        for seed in range(5):
            np.testing.assert_allclose(fs[seed,:,:128],accepted[seed][si],atol=5e-4,rtol=2e-5)
            context_max.append(float(np.max(np.abs(fs[seed,:,:128]-accepted[seed][si]))))
        truths=[];valids=[];costs=[]
        ca={k:a[k].astype(np.float64) for k in ['candidate_command','previous_command','reference_world']}
        for mode in MODES:
            if mode=='held':d=a
            else:
                p=CONTINUATION/'lag_3'/f'{path.stem}_{mode}.npz'
                d=load_arrays(checked(p,continuation_lock[key(p)],sources),['position_world','valid'])
            p=d['position_world'].transpose(2,1,0,3);v=d['valid'].transpose(2,1,0)
            truths.append(p);valids.append(v);costs.append(cost(p.astype(float),ca))
        full=np.stack(valids).all((0,2,3))
        arrays['features'].append(fs.reshape(5,12,9,338))
        arrays['nominal'].append(physics(inp,'x_',.3).reshape(12,9,30,3))
        arrays['held_target_local'].append(world_to_local(truths[0],a))
        arrays['position_world'].append(np.stack(truths))
        arrays['valid'].append(np.stack(valids));arrays['cost'].append(np.stack(costs))
        arrays['common_valid'].append(full)
        for k in ['candidate_command','previous_command','initial_position','initial_velocity','reference_world','state_hist','action_hist','depth_hist']:
            arrays[k].append(a[k])
        arrays['initial_alive'].append(a['initial_alive'])
        arrays['scene'].append(np.asarray([rowmap[path.stem,j]['scene'] for j in range(12)]))
        arrays['tier'].append(np.asarray([rowmap[path.stem,j]['tier'] for j in range(12)]))
    out={k:np.stack(v) for k,v in arrays.items()}
    # Put seed and continuation-mode axes first for a stable public interface.
    out['features']=out['features'].transpose(1,0,2,3,4)
    for k in ['position_world','valid','cost']:out[k]=np.swapaxes(out[k],0,1)
    out['branch_relative_time_s']=.05*np.arange(1,31)
    if out['features'].shape!=(5,9,12,9,338) or int(out['common_valid'].sum())!=106:
        raise ValueError('Evaluation shape/support differs from accepted lag source')
    info=dict(snapshot_names=[p.stem for p in paths],mode_order=MODES,model_seed_order=list(range(5)),
              total_states=108,common_valid_states=106,initially_alive=int(out['initial_alive'].sum()),
              contexts_max_abs_against_accepted=float(max(context_max)),
              context_tolerance=dict(atol=5e-4,rtol=2e-5),
              truth_axis_order=['mode','snapshot','task','candidate','time','xyz'],
              feature_axis_order=['encoder_seed','snapshot','task','candidate','feature'],
              cost_axis_order=['mode','snapshot','task','candidate'])
    return out,info


def save_npz(path,arrays):
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**arrays)
    with np.load(path,allow_pickle=False) as z:
        for k,v in arrays.items():np.testing.assert_array_equal(z[k],v)
    return dict(path=key(path),sha256=sha(path),shapes={k:list(v.shape) for k,v in arrays.items()})


def prepare(calibration_root=C1,output=DEST,device='cpu'):
    """Validate sources, freeze the five encoders, prepare arrays, return manifest."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(f'Will not overwrite prepared output: {output}')
    sources={};paths=calibration_sources(calibration_root,sources)
    scientific_lock=check_science_lock(sources)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this process; no silent CPU fallback')
    for p in [Path(__file__),ROOT/'scripts/icra_confirm_frozen.py',ROOT/'scripts/icra_adaptation_ridge.py',
              ROOT/'scripts/icra_snapshot_prediction.py',ROOT/'scripts/icra_adaptation_cache.py',
              ROOT/'src/winddyn/models/wm.py',ROOT/'src/winddyn/train/trainer.py']:
        checked(p,sources=sources)
    started=time.monotonic();models=load_encoders(device,sources)
    # A new output directory is created only after all five original caches pass.
    output.mkdir(parents=True,exist_ok=False)
    outputs={};cal_meta=[]
    for campaign,(cache,meta) in enumerate(paths):
        arrays,info=prepare_calibration(campaign,cache,meta,models,device)
        outputs[f'campaign{campaign}.npz']=save_npz(output/f'campaign{campaign}.npz',arrays);cal_meta.append(info)
        print(f'PREPARED C1 lag campaign {campaign}: {info["fit_windows"]} FIT / {info["validation_windows"]} VAL windows',flush=True)
    evaluation,eval_meta=prepare_evaluation(models,device,sources)
    outputs['evaluation.npz']=save_npz(output/'evaluation.npz',evaluation)
    for p,wanted in sources.items():checked(Path(p) if Path(p).is_absolute() else ROOT/p,wanted)
    check_science_lock()
    (output/'source_lock.json').write_text(json.dumps(sources,indent=2)+'\n')
    doc=dict(status='prepared_no_fitting',schema_version=1,device=device,torch_version=torch.__version__,
             nominal_tau_s=.3,feature_dim=338,feature_blocks=dict(context=[0,128],future_command=[128,248],nominal_displacement=[248,338]),
             feature_dtype='float64; encoded context and future commands originate in float32',
             target='30x3 relative displacement in current yaw frame, metres',
             normalization='No fitted readout normalization here; derive mean/std only from each campaign FIT mask, std floor 1e-5. Encoder uses original nominal checkpoint statistics.',
             calibration=cal_meta,evaluation=eval_meta,outputs=outputs,
             source_lock=dict(path=key(output/'source_lock.json'),sha256=sha(output/'source_lock.json')),
             scientific_lock=scientific_lock,elapsed_s=time.monotonic()-started,
             scope='C1 budget32 subset901 lag calibration; existing Fig3/Q3 development-state evaluation. Not paired-data calibration, not a new independent test, no fitted model results.',
             original_four_action_confirmation='Original C1 branch arrays remain in the hash-verified restored caches; this utility deliberately prepares only calibration windows and the nine-action Fig3/Q3 evaluation.')
    (output/'manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
    return doc


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--calibration-root',type=Path,default=C1,help='Root containing campaign0..4/lag_3/cache/arrays.npz; original metadata is always verified at its original repository location')
    ap.add_argument('--output',type=Path,default=DEST)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=ap.parse_args();doc=prepare(a.calibration_root,a.output,a.device)
    print(json.dumps(dict(status=doc['status'],output=str(a.output),evaluation=doc['evaluation']),indent=2))


if __name__=='__main__':main()
