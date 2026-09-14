"""Fit and evaluate the prespecified C1 lag nonlinear readout development control."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from _common import ROOT
from icra_adaptation_ridge import RidgePath, predict
from icra_nonlinear_readout_data_20260914 import (
    DEST as PREPARED, CAMPAIGN, RECOVER, check_science_lock, checked, key,
    load_arrays, read, sha)
from icra_nonlinear_readout_heads_20260914 import (
    GRID, HEAD_SEEDS, STEPS, fit_family, predict_saved, synthetic_checks)
from icra_snapshot_prediction import cost

BASE = ROOT/'runs/icra_nonlinear_readout_20260914'
DEFAULT_RUN = BASE/'attempt01'


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify_prepared(path):
    manifest = read(path/'manifest.json')
    assert manifest['status'] == 'prepared_no_fitting'
    for item in manifest['outputs'].values():
        checked(ROOT/item['path'], item['sha256'])
    checked(ROOT/manifest['source_lock']['path'], manifest['source_lock']['sha256'])
    for rel, value in read(path/'source_lock.json').items():
        checked(Path(rel) if Path(rel).is_absolute() else ROOT/rel, value)
    check_science_lock()
    return manifest


def configuration(campaign, seed, method='frozen'):
    inv = read(CAMPAIGN/'inventory.json')
    return next(c for c in inv['configurations'] if c['regime'] == 'lag_3'
                and c['campaign'] == campaign and c['seed'] == seed and c['method'] == method)


def original_fit(campaign, seed):
    c = configuration(campaign, seed)
    inv = read(CAMPAIGN/'inventory.json')
    path = checked(RECOVER/c['weight'], inv['recover_files'][c['weight']])
    old = load_arrays(path)
    source_lock = read(CAMPAIGN/'attempt01/source_lock.json')
    m = read(checked(ROOT/c['metadata'],source_lock[c['metadata']]))
    record = next(r for r in m['records'] if r['budget'] == 32
                  and r['subset_seed'] == 901 and r['center'] == 'physical')
    return c, old, record


def reproduce_ridge(data, campaign, seed):
    x = data['features'][seed]
    y = data['residual_target'].reshape(len(x), 90)
    fm, vm = data['fit_mask'], data['validation_mask']
    path = RidgePath(x[fm])
    candidates = []
    fits = []
    for lam in GRID:
        fit = path.fit(y[fm], lam)
        err = (predict(x[vm], fit)-y[vm]).reshape(-1, 30, 3)
        candidates.append(dict(lambda_=lam, validation_rmse=float(np.sqrt((err**2).sum(-1).mean()))))
        fits.append(fit)
    best = min(range(len(GRID)), key=lambda i: (candidates[i]['validation_rmse'], i))
    c, old, record = original_fit(campaign, seed)
    assert GRID[best] == record['selected_lambda']
    assert abs(candidates[best]['validation_rmse']-record['validation_rmse']) < 1e-4
    # Encoder kernels can introduce tiny float32 differences across CPU/GPU.
    # A different selected lambda or a material forecast change is a failed audit.
    np.testing.assert_allclose(fits[best]['mean'], old['mean'], atol=5e-4, rtol=2e-5)
    np.testing.assert_allclose(fits[best]['std'], old['std'], atol=5e-4, rtol=2e-5)
    coefficient_error = float(np.max(np.abs(fits[best]['coef']-old['coef'])))
    np.testing.assert_allclose(fits[best]['coef'], old['coef'], atol=1e-3, rtol=1e-3)
    forecast_error = float(np.max(np.abs(predict(x, fits[best])-predict(x, old))))
    assert forecast_error < 5e-4
    return dict(campaign=campaign, encoder_seed=seed, selected_lambda=GRID[best],
                original_index=c['index'], original_metadata=c['metadata'],
                candidates=candidates, coefficient_max_abs=coefficient_error,
                forecast_max_abs=forecast_error,
                validation_rmse_difference=candidates[best]['validation_rmse']-record['validation_rmse'],
                baseline='Original saved coefficients and forecasts remain the reported affine baseline')


def fit_all(prepared, run, device):
    manifest = verify_prepared(prepared)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent device switch')
    sources = [Path(__file__), Path(__file__).with_name('icra_nonlinear_readout_heads_20260914.py'),
               Path(__file__).with_name('icra_nonlinear_readout_data_20260914.py'),
               ROOT/'scripts/icra_adaptation_ridge.py', BASE/'PROTOCOL.md',
               prepared/'manifest.json', prepared/'source_lock.json']
    lock = {key(p): sha(p) for p in sources}
    lock.update({v['path']: v['sha256'] for v in manifest['outputs'].values()})
    if run.exists():
        assert read(run/'source_lock.json') == lock, 'Cannot resume after source/protocol changes'
        assert not (run/'fit_complete.json').exists(), 'Fitting already completed'
        environment = read(run/'environment.json')
        assert environment['device'] == device and environment['torch'] == torch.__version__
        assert environment['numpy'] == np.__version__, 'Resume must preserve numerical environment'
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_new(run/'source_lock.json', lock)
        write_new(run/'synthetic_checks.json', synthetic_checks())
        write_new(run/'environment.json', dict(torch=torch.__version__, numpy=np.__version__,
                  device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                  started_unix=time.time(), stages='fit all before evaluating any new model'))
    receipts = []
    for campaign in range(5):
        data = load_arrays(prepared/f'campaign{campaign}.npz')
        for seed in range(5):
            folder = run/f'campaign{campaign}_encoder{seed}'
            folder.mkdir(exist_ok=True)
            audit = folder/'affine_reproduction.json'
            if not audit.exists():
                write_new(audit, reproduce_ridge(data, campaign, seed))
            receipts.append(dict(path=key(audit), sha256=sha(audit)))
            for family in ['additive', 'joint']:
                weight, receipt = folder/f'{family}.pt', folder/f'{family}.json'
                if receipt.exists():
                    saved = read(receipt)
                    checked(weight, saved['sha256'])
                else:
                    if weight.exists():
                        raise FileExistsError(f'Unreceipted checkpoint requires inspection: {weight}')
                    started = time.monotonic()
                    ck = fit_family(data['features'][seed], data['residual_target'].reshape(-1, 90),
                                    data['fit_mask'], data['validation_mask'], family, device,
                                    progress=lambda fam, step, total: print(
                                      f'FIT campaign={campaign} encoder={seed} {fam} update={step}/{total}', flush=True))
                    assert ck['updates'] == 1000 and len(ck['candidates']) == 21
                    assert [r['head_seed'] for r in ck['selected']] == [0, 1, 2]
                    ck.update(campaign=campaign, encoder_seed=seed, protocol_sha256=lock[key(BASE/'PROTOCOL.md')])
                    torch.save(ck, weight)
                    saved = {k: v for k, v in ck.items() if k not in ['parameters', 'mean', 'std']}
                    saved.update(path=key(weight), sha256=sha(weight), elapsed_s=time.monotonic()-started)
                    write_new(receipt, saved)
                receipts.append(dict(path=key(receipt), sha256=sha(receipt)))
                receipts.append(dict(path=key(weight), sha256=sha(weight)))
    assert len(receipts) == 125
    for p, digest in lock.items():
        checked(ROOT/p, digest)
    check_science_lock()
    write_new(run/'fit_complete.json', dict(status='all_heads_selected_before_evaluation',
              completed_unix=time.time(), families=['affine','additive','joint'], campaigns=5,
              encoder_seeds=5, nonlinear_head_seeds=3, trained_nonlinear_grid_heads=1050,
              selected_nonlinear_heads=150, affine_heads=25, files=receipts,
              data_scope='C1 budget32 lag; existing development states'))


def world_from_local(local, data):
    local = np.asarray(local, dtype=np.float64).reshape(108, 9, 30, 3)
    state = data['state_hist'].reshape(108, 12, 12)[:, -1]
    c, s = state[:, 11].astype(float), state[:, 10].astype(float)
    world = local.copy()
    world[..., 0] = c[:,None,None]*local[...,0]-s[:,None,None]*local[...,1]
    world[..., 1] = s[:,None,None]*local[...,0]+c[:,None,None]*local[...,1]
    return world+data['initial_position'].reshape(108,1,1,3)


def score_world(world, data):
    assert world.shape == (108,9,30,3) and np.isfinite(world).all()
    a = {k: data[k].reshape((108,)+data[k].shape[2:]).astype(float)
         for k in ['candidate_command','previous_command','reference_world']}
    predicted_cost = cost(world, a)
    chosen = predicted_cost.argmin(1)
    true = data['position_world'][0].reshape(108,9,30,3).astype(float)
    error = world-true
    shared = error.mean(1,keepdims=True)
    total_mse = (error**2).sum(-1).mean((1,2))
    shared_mse = (shared**2).sum(-1).mean((1,2))
    centered_mse = ((error-shared)**2).sum(-1).mean((1,2))
    np.testing.assert_allclose(total_mse,shared_mse+centered_mse,atol=1e-10)
    true_cost = data['cost'].reshape(3,108,9).astype(float)
    oracle = true_cost.argmin(-1)
    regret = np.take_along_axis(true_cost,chosen[None,:,None].repeat(3,0),axis=2)[...,0]-true_cost.min(-1)
    contacts = ~data['valid'].reshape(3,108,9,30)[..., -1]
    contact = np.take_along_axis(contacts,chosen[None,:,None].repeat(3,0),axis=2)[...,0]
    return dict(choice=chosen, predicted_cost=predicted_cost, total_mse=total_mse,
                shared_mse=shared_mse, centered_mse=centered_mse, regret=regret,
                accuracy=oracle==chosen[None], selected_contact=contact)


def evaluate(prepared, run, device='cpu'):
    manifest = verify_prepared(prepared)
    receipt = read(run/'fit_complete.json')
    assert receipt['status'] == 'all_heads_selected_before_evaluation'
    for item in receipt['files']:
        checked(ROOT/item['path'], item['sha256'])
    for rel, digest in read(run/'source_lock.json').items():
        checked(ROOT/rel,digest)
    dest = run/'evaluation'
    dest.mkdir(exist_ok=False)
    write_new(dest/'start.json',dict(fit_receipt_sha256=sha(run/'fit_complete.json'),started_unix=time.time()))
    data = load_arrays(prepared/'evaluation.npz')
    mask = data['common_valid'].reshape(108)
    alive = data['initial_alive'].reshape(108)
    rows, sources = [], {}
    original_outputs = read(CAMPAIGN/'attempt01/output_hashes.json')
    # Check every original baseline before revealing any nonlinear outcome.
    baseline_worlds, baseline_audits = {}, []
    for campaign in range(5):
        for family in ['scalar','affine']:
            for seed in ([0] if family == 'scalar' else range(5)):
                cfg = configuration(campaign,seed,'scalar' if family == 'scalar' else 'frozen')
                p = ROOT/f'data/icra_campaign_snapshot_20260909/attempt01/config{cfg["index"]:03d}.npz'
                checked(p,original_outputs[key(p)],sources)
                old_world = load_arrays(p)['world'].reshape(108,9,30,3)
                row_path = CAMPAIGN/f'attempt01/config{cfg["index"]:03d}.json'
                original_rows = read(checked(row_path,original_outputs[key(row_path)],sources))['rows']
                expected_keys = [(name,j) for name in manifest['evaluation']['snapshot_names'] for j in range(12)]
                assert [(r['snapshot'],r['task_index']) for r in original_rows] == expected_keys
                baseline_score = score_world(old_world,data)
                np.testing.assert_array_equal(baseline_score['choice'],[r['choice'] for r in original_rows])
                original_mask = np.array([r['all_candidates_valid'] for r in original_rows])
                np.testing.assert_allclose(baseline_score['regret'][0,original_mask],
                    [r['regret'] for r in original_rows if r['all_candidates_valid']],atol=1e-10,rtol=1e-9)
                rebuild_max_abs = 0.
                if family == 'affine':
                    _, fit, _ = original_fit(campaign,seed)
                    x = data['features'][seed].reshape(-1,338)
                    rebuilt = world_from_local(predict(x,fit).reshape(108,9,30,3)+data['nominal'].reshape(108,9,30,3),data)
                    np.testing.assert_allclose(rebuilt,old_world,atol=5e-4,rtol=1e-4)
                    np.testing.assert_array_equal(score_world(rebuilt,data)['choice'],baseline_score['choice'])
                    rebuild_max_abs=float(np.max(np.abs(rebuilt-old_world)))
                baseline_worlds[campaign,family,seed]=old_world
                baseline_audits.append(dict(campaign=campaign,family=family,encoder_seed=seed,
                    original_choices_exact=True,original_regret_reproduced=True,
                    forecast_rebuild_max_abs=rebuild_max_abs))
    assert len(baseline_audits)==30
    write_new(dest/'baseline_acceptance.json',dict(status='all_original_baselines_reproduced_before_new_model_scoring',
              baselines=baseline_audits,completed_unix=time.time()))
    for campaign in range(5):
        for family in ['scalar','affine','additive','joint']:
            for seed in ([0] if family == 'scalar' else range(5)):
                if family in ['scalar','affine']:
                    worlds = [(-1,baseline_worlds[campaign,family,seed])]
                else:
                    ck = torch.load(run/f'campaign{campaign}_encoder{seed}/{family}.pt',map_location='cpu',weights_only=False)
                    x = data['features'][seed].reshape(-1,338)
                    worlds = [(s['head_seed'],world_from_local(
                        predict_saved(x,ck,s['index'],device).reshape(108,9,30,3)+data['nominal'].reshape(108,9,30,3),data))
                        for s in ck['selected']]
                for head_seed,world in worlds:
                    values = score_world(world,data)
                    stem = f'{family}_campaign{campaign}_encoder{seed}_head{head_seed}'
                    np.savez_compressed(dest/f'{stem}.npz',world=world,**values)
                    summary = dict(campaign=campaign,encoder_seed=seed,head_seed=head_seed,family=family,
                         path=key(dest/f'{stem}.npz'),sha256=sha(dest/f'{stem}.npz'),
                         forecast_rmse=float(np.sqrt(values['total_mse'][mask].mean())),
                         centered_mse=float(values['centered_mse'][mask].mean()),
                         shared_mse=float(values['shared_mse'][mask].mean()),
                         regret=values['regret'][:,mask].mean(1).tolist(),
                         accuracy=values['accuracy'][:,mask].mean(1).tolist(),
                         selected_contact=values['selected_contact'][:,alive].mean(1).tolist())
                    rows.append(summary)
    assert len(rows)==180
    for rel,digest in sources.items():checked(ROOT/rel,digest)
    write_new(dest/'baseline_sources.json',sources)
    write_new(dest/'configurations.json',dict(configurations=rows,
         common_valid=int(mask.sum()),initially_alive=int(alive.sum()),mode_order=['held','nominal','observer'],
         state_count=108,scene_clusters=len(np.unique(data['scene'])),
         calibration_and_encoder_seeds=5,head_seeds=3,heads_are_not_forecast_ensembled=True,
         complete=True,completed_unix=time.time(),scope='Exploratory development readout comparison; no complete-flight evidence'))
    print(f'EVALUATION COMPLETE: {len(rows)} configurations; {mask.sum()} common-valid states',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fit','evaluate','all','check'])
    parser.add_argument('--prepared',type=Path,default=PREPARED)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'check':
        print(json.dumps(synthetic_checks(),indent=2))
        return
    try:
        if args.stage in ['fit','all']:fit_all(args.prepared,args.run,args.device)
        if args.stage in ['evaluate','all']:evaluate(args.prepared,args.run,args.device)
    except Exception:
        if args.run.exists():
            path=args.run/f'failure_{time.time_ns()}.json'
            write_new(path,dict(stage=args.stage,error=traceback.format_exc(),time_unix=time.time()))
        raise


if __name__ == '__main__':main()
