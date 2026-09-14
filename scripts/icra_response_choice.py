"""Unfitted candidate-choice test with deployable common forecasts held fixed."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from _common import ROOT
from icra_adaptation_fit import sha
from icra_decision_evaluation import forecast, cost, CONFIGS as EVAL_CONFIGS
from icra_history_response import response_inputs, predict_model, Estimators, load_vehicle

BASE = ROOT/'runs/icra_response_choice_20260908'
SNAP = ROOT/'runs/icra_snapshot_pilot_20260908'
FULL = ROOT/'runs/icra_decision_fit_20260908'
RESPONSE = ROOT/'runs/icra_history_response_20260908/attempt01'
COMMONS = ['physics_features','raw']
MODES = ['native','shared_affine','current']


def predictions(a, full_fits, response_fits, observer):
    native = {}
    for common in COMMONS:
        p = SimpleNamespace(method=common, recipe='physics_only' if common=='physics_features' else 'raw_state',
                            fit=full_fits[common], model=None, device='cpu')
        native[common] = forecast(p,a)
    inputs = response_inputs(a,observer)
    response = {kind:predict_model(inputs,response_fits[kind]) for kind in MODES[1:]}
    outputs = {}
    max_mean_error = 0.
    for common in COMMONS:
        mu = native[common].mean(1,keepdims=True)
        np.testing.assert_allclose(mu+(native[common]-mu),native[common],atol=1e-12,rtol=0)
        outputs[common,'native'] = native[common]
        for mode in MODES[1:]:
            np.testing.assert_allclose(response[mode].mean(1),0,atol=1e-8)
            outputs[common,mode] = mu+response[mode]
            error = float(np.max(np.abs(outputs[common,mode].mean(1,keepdims=True)-mu)))
            assert error < 1e-8
            max_mean_error = max(max_mean_error,error)
    assert all(np.isfinite(v).all() for v in outputs.values())
    return outputs, np.stack([native[k] for k in COMMONS]), np.stack([response[k] for k in MODES[1:]]), max_mean_error


def evaluate(pred,a,regime,common,mode,name):
    truth=a['position_world'].transpose(2,1,0,3)
    valid=a['valid'].transpose(2,1,0)
    pc,tc=cost(pred,a),cost(truth,a)
    choice=pc.argmin(1)
    error=pred-truth; common_error=error.mean(1,keepdims=True);contrast=error-common_error
    rows=[]
    for j in range(12):
        full=bool(valid[j].all());prefix=bool(a['initial_alive'][j])
        assert not full or prefix
        row=dict(regime=regime,common_source=common,response=mode,snapshot=name,
            tier=name.rsplit('_step',1)[0],task_index=j,prefix_alive=prefix,
            all_candidates_valid=full,valid_candidate_records=int(valid[j].sum()),
            position_squared_error=float(np.where(valid[j],(error[j]**2).sum(-1),0).sum()),
            choice=int(choice[j]),selected_contact=bool(not valid[j,choice[j],-1]) if prefix else None)
        if full:
            row.update(correct=bool(choice[j]==tc[j].argmin()),
                regret=float(tc[j,choice[j]]-tc[j].min()),
                common_mse=float((common_error[j]**2).sum(-1).mean()),
                contrast_mse=float((contrast[j]**2).sum(-1).mean()),
                total_mse=float((error[j]**2).sum(-1).mean()))
            np.testing.assert_allclose(row['total_mse'],row['common_mse']+row['contrast_mse'],atol=1e-8)
        rows.append(row)
    return rows


def aggregate(rows):
    full=[r for r in rows if r['all_candidates_valid']]
    alive=[r for r in rows if r['prefix_alive']]
    positions=sum(r['valid_candidate_records'] for r in rows)
    assert full and alive and positions
    return dict(states=len(rows),initially_alive=len(alive),all_candidates_valid=len(full),
        valid_candidate_positions=positions,
        position_rmse=float(np.sqrt(sum(r['position_squared_error'] for r in rows)/positions)),
        selected_contacts=sum(r['selected_contact'] for r in alive),
        selected_contact_rate=float(np.mean([r['selected_contact'] for r in alive])),
        accuracy=float(np.mean([r['correct'] for r in full])),
        regret=float(np.mean([r['regret'] for r in full])),
        common_mse=float(np.mean([r['common_mse'] for r in full])),
        contrast_mse=float(np.mean([r['contrast_mse'] for r in full])),
        total_mse=float(np.mean([r['total_mse'] for r in full])))


def comparisons(rows, means):
    contrasts=[];gates=[]
    for regime in ['mass_1p4','lag_3']:
        for common in COMMONS:
            mr={r['response']:r['metrics'] for r in means if r['regime']==regime and r['common_source']==common}
            paired={mode:{(r['snapshot'],r['task_index']):r for r in rows
                          if r['regime']==regime and r['common_source']==common and r['response']==mode}
                    for mode in MODES}
            tests=[]
            for base in ['native','shared_affine']:
                new,old=mr['current'],mr[base]
                full=[k for k,r in paired['current'].items() if r['all_candidates_valid']]
                reduction=1-new['regret']/old['regret'] if old['regret']>0 else None
                passed=reduction is not None and reduction>=.1 and new['accuracy']>=old['accuracy'] and new['selected_contact_rate']<=old['selected_contact_rate']
                tests.append(bool(passed))
                contrasts.append(dict(regime=regime,common_source=common,comparison='current-minus-'+base,
                    differences={k:new[k]-old[k] for k in ['position_rmse','regret','accuracy','selected_contact_rate','contrast_mse','common_mse']},
                    relative_regret_reduction=reduction,
                    changed_choices_all_valid=sum(paired['current'][k]['choice']!=paired[base][k]['choice'] for k in full),
                    changed_to_correct=sum(paired['current'][k]['correct'] and not paired[base][k]['correct'] for k in full),
                    changed_from_correct=sum(not paired['current'][k]['correct'] and paired[base][k]['correct'] for k in full),
                    descriptive_screen_passed=bool(passed)))
            gates.append(dict(regime=regime,common_source=common,passes_both_baselines=all(tests)))
    return contrasts,gates


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=BASE/'attempt01');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    sources={}
    def record(path):
        h=sha(path);rel=str(path.relative_to(ROOT));assert rel not in sources or sources[rel]==h;sources[rel]=h;return h
    def read(path):record(path);return json.loads(path.read_text())
    previous=read(RESPONSE/'sources.json')
    for path,h in previous.items():assert record(ROOT/path)==h
    accepted=read(RESPONSE/'accepted.json');assert accepted['status']=='accepted'
    response_selection=read(RESPONSE/'selection.json')
    assert read(FULL/'accepted.json')['accepted_fits']==84
    record(Path(__file__).resolve());record(BASE/'PROTOCOL.md')
    observer=Estimators('observer',0,'cpu',load_vehicle())
    all_rows=[];baseline_error=0.;mean_error=0.;component_files=[]
    for regime in ['mass_1p4','lag_3']:
        full_fits={};response_fits={};saved={}
        for index,common in enumerate(COMMONS):
            folder=FULL/regime/f'config{index:02d}'
            sel=next(r['selected'] for r in read(folder/'results.json')['records'] if r['objective']=='trajectory')
            path=folder/sel['weight_file'];assert record(path)==sel['weight_sha256']
            with np.load(path) as z:full_fits[common]={k:z[k] for k in z.files}
            old=read(ROOT/'runs/icra_decision_evaluation_20260908'/regime/f'config{EVAL_CONFIGS.index((common,0,"trajectory")):03d}.json')
            assert (old['regime'],old['method'],old['objective'])==(regime,common,'trajectory')
            assert old['predictor_provenance'][str(path.relative_to(ROOT))]==sel['weight_sha256']
            saved[common]={(r['snapshot'],r['task_index']):r for r in old['rows']}
            assert len(saved[common])==108
        for kind in MODES[1:]:
            sel=next(s for s in response_selection if s['regime']==regime and s['method']==kind)
            path=RESPONSE/sel['weight_file'];assert record(path)==sel['weight_sha256']
            with np.load(path) as z:response_fits[kind]={k:z[k] for k in z.files}
        files=sorted((SNAP/regime).glob('*.npz'));assert len(files)==9
        for path in files:
            record(path)
            with np.load(path) as z:a={k:z[k] for k in z.files}
            output,native,response,err=predictions(a,full_fits,response_fits,observer);mean_error=max(mean_error,err)
            forbidden=dict(a)
            for key in ['position_world','reference_world','valid','initial_alive','depth_hist']:
                forbidden[key]=np.full(a[key].shape,np.nan)
            other,_,_,_=predictions(forbidden,full_fits,response_fits,observer)
            for key in output:np.testing.assert_array_equal(output[key],other[key])
            component=args.output/f'{regime}_{path.stem}_components.npz'
            np.savez_compressed(component,native_world=native,response_world=response)
            component_files.append(dict(regime=regime,snapshot=path.stem,file=component.name,sha256=sha(component)))
            for common in COMMONS:
                for mode in MODES:
                    rows=evaluate(output[common,mode],a,regime,common,mode,path.stem)
                    for r in rows:
                        old=saved[common][r['snapshot'],r['task_index']]
                        for k in ['prefix_alive','all_candidates_valid','valid_candidate_records']:
                            assert r[k]==old[k]
                        if r['all_candidates_valid']:
                            np.testing.assert_allclose(r['common_mse'],old['common_mse'],atol=1e-9,rtol=1e-7)
                        if mode=='native':
                            for k in ['choice','selected_contact']:
                                assert r[k]==old[k],(regime,common,path.stem,r['task_index'],k)
                            keys=['position_squared_error']
                            if r['all_candidates_valid']:
                                assert r['correct']==old['correct'];keys+=['regret','common_mse','contrast_mse','total_mse']
                            for k in keys:
                                np.testing.assert_allclose(r[k],old[k],atol=1e-8,rtol=1e-6)
                                baseline_error=max(baseline_error,abs(r[k]-old[k]))
                    all_rows.extend(rows)
    means=[];tiers=[]
    for regime in ['mass_1p4','lag_3']:
        for common in COMMONS:
            for mode in MODES:
                rows=[r for r in all_rows if (r['regime'],r['common_source'],r['response'])==(regime,common,mode)]
                assert len(rows)==108
                metrics=aggregate(rows)
                assert metrics['all_candidates_valid']=={'mass_1p4':98,'lag_3':106}[regime]
                assert metrics['initially_alive']=={'mass_1p4':99,'lag_3':108}[regime]
                means.append(dict(regime=regime,common_source=common,response=mode,metrics=metrics))
                for tier in ['id','wind_extrap','joint_extrap']:
                    rr=[r for r in rows if r['tier']==tier];assert len(rr)==36
                    tiers.append(dict(regime=regime,common_source=common,response=mode,tier=tier,metrics=aggregate(rr)))
    contrasts,gates=comparisons(all_rows,means)
    assert len(all_rows)==1296 and len(means)==12 and len(tiers)==36 and len(contrasts)==8
    result=dict(status='accepted_by_runner',stage='posthoc_unfitted_component_choice_diagnosis',
        configurations=12,unique_states=216,state_configuration_records=1296,
        no_refitting=True,new_rollouts=0,encoder_loaded=False,
        max_native_saved_metric_error=baseline_error,max_common_prediction_change=mean_error,
        future_truth_reference_validity_depth_forecast_invariant=True,
        component_axis_order=dict(native_world=COMMONS,response_world=MODES[1:]),
        component_files=component_files,means=means,tiers=tiers,contrasts=contrasts,gates=gates)
    for name,value in [('summary.json',result),('rows.json',all_rows),('sources.json',sources)]:
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for r in means:print(r['regime'],r['common_source'],r['response'],r['metrics'])
    print('GATES',json.dumps(gates));print('MAX_NATIVE_ERROR',baseline_error,'MAX_COMMON_CHANGE',mean_error)


if __name__=='__main__':main()
