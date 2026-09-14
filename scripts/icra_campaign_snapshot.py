"""Protocol-fixed development cross-evaluation; no fitting or simulator stepping."""
import argparse, hashlib, json, platform, time, traceback
from pathlib import Path
import numpy as np
import torch
from _common import ROOT
from winddyn.models.wm import WorldModel
from icra_adaptation_scratch import make_encoder
from icra_adaptation_fit import features, physics
from icra_confirm_frozen import features as c1_features
from icra_adaptation_ridge import predict
from icra_snapshot_prediction import model_input, cost

BASE = ROOT/'runs/icra_campaign_snapshot_20260909'
DATA = ROOT/'data/icra_campaign_snapshot_20260909'
RECOVER = DATA/'recovered'
REGIMES = ['mass_1p4', 'lag_3']
TIERS = ['id', 'wind_extrap', 'joint_extrap', 'pooled']
METHODS = ['frozen', 'scalar', 'physics_features', 'raw', 'raw_depth', 'adapt_random', 'adapt_pretrained']
METRICS = ['shared_rmse', 'regret', 'accuracy', 'selected_contact']


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def write(p,v):
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def read(p): return json.loads(p.read_text())


def snapshots(reg):
    paths=sorted((ROOT/'runs/icra_snapshot_pilot_20260908'/reg).glob('*.npz'))
    assert len(paths)==9
    out={}
    for p in paths:
        with np.load(p) as z: out[p.stem]={k:z[k] for k in z.files}
    return paths,out


def metadata_audit(inv):
    rows=[];sources=set()
    for c in inv['configurations']:
        p=ROOT/c['metadata'];m=read(p);sources.add(p)
        cm=ROOT/f"runs/icra_calibration_confirmation_20260908/campaign{c['campaign']}/{c['regime']}/cache/metadata.json"
        pool=read(cm);sources.add(cm);split=pool['subsets']['901']['32']
        assert m['campaign']==c['campaign'] and m['regime']==c['regime'] and not m['fixture']
        assert m['model_seed']==c['seed']
        if c['method']=='scalar':
            rec=next(q for q in m['physical_records'] if q['budget']==32 and q['subset_seed']==901 and q['name']=='calibrated')
            assert rec['tau']==c['tau']
        elif c['method'].startswith('adapt_'):
            assert m['status']=='complete' and m['budget']==32 and m['subset_seed']==901
            assert m['initialization']==c['method'].removeprefix('adapt_')
            rec=next(q for q in m['records'] if q['center']=='physical')
            assert rec['weight_sha256']==inv['recover_files'][c['weight']]
            assert m['encoder_sha256']==inv['recover_files'][c['encoder']]
            assert m['nominal_checkpoint_sha256']==sha(RECOVER/f"runs/revision_20260907/checkpoints/supervised_seed{c['seed']}/best.pt")
        else:
            rec=next(q for q in m['records'] if q['budget']==32 and q['subset_seed']==901 and q['center']=='physical')
            assert rec['status']=='complete' and rec['weight_sha256']==inv['recover_files'][c['weight']]
            assert rec['weight_file']==Path(c['weight']).name
            prov=p.with_name('provenance.json');sources.add(prov)
            if c['encoder']: assert read(prov)[c['encoder']]==sha(RECOVER/c['encoder'])
        partitions=m if c['method'].startswith('adapt_') else rec
        assert partitions['fit_episodes']==split['fit'] and partitions['validation_episodes']==split['validation']
        assert len(split['fit'])==24 and len(split['validation'])==8 and not set(split['fit'])&set(split['validation'])
        rows.append(dict(index=c['index'],fit=24,validation=8,fit_windows=partitions['fit_windows'],validation_windows=partitions['validation_windows']))
    return rows,sources


def summarize(rows):
    full=[r for r in rows if r['all_candidates_valid']];alive=[r for r in rows if r['prefix_alive']]
    n=sum(r['valid_candidate_records'] for r in rows)
    return dict(states=len(rows),all_valid=len(full),initially_alive=len(alive),precontact_positions=n,
                shared_rmse=float(np.sqrt(np.mean([r['shared_mse'] for r in full]))) if full else None,
                precontact_rmse=float(np.sqrt(sum(r['position_squared_error'] for r in rows)/n)) if n else None,
                regret=float(np.mean([r['regret'] for r in full])) if full else None,
                accuracy=float(np.mean([r['correct'] for r in full])) if full else None,
                selected_contact=float(np.mean([r['selected_contact'] for r in alive])) if alive else None)


def interval(a,b,counts,geometry,metric):
    reps=20000;rng=np.random.default_rng(30919007);sw=np.zeros((reps,len(counts)),int)
    for g in np.unique(geometry):
        ix=np.flatnonzero(geometry==g);sw[:,ix]=rng.multinomial(len(ix),np.ones(len(ix))/len(ix),size=reps)
    cw=rng.multinomial(5,np.ones(5)/5,size=reps);mw=rng.multinomial(5,np.ones(5)/5,size=reps)
    den=sw@counts;ok=den>0
    def transform(x):return np.sqrt(x) if metric=='shared_rmse' else x
    def draw(x):
        v=transform(np.einsum('msc,rc->rms',x,sw[ok])/den[ok,None,None])
        weights=mw[ok] if x.shape[0]==5 else np.ones((ok.sum(),1))
        return np.einsum('rms,rm,rs->r',v,weights,cw[ok])/(x.shape[0]*5)
    dif=draw(a)-draw(b)
    return dict(difference=float(transform(a.sum(-1)/counts.sum()).mean()-transform(b.sum(-1)/counts.sum()).mean()),
                interval_95=np.quantile(dif,[.025,.975]).tolist(),requested_draws=reps,valid_draws=int(ok.sum()),
                zero_denominator_draws=int((~ok).sum()),seed=30919007,family_size=56,adjustment='none; exploratory')


def aggregate(configs):
    conf=[]
    for d in configs:
        for t in TIERS:
            rr=[r for r in d['rows'] if t=='pooled' or r['tier']==t]
            conf.append({**{k:d[k] for k in ['index','regime','campaign','method','seed']},'tier':t,**summarize(rr)})
    means=[];campaigns=[]
    for reg in REGIMES:
        for method in METHODS:
            for t in TIERS:
                cs=[]
                for campaign in range(5):
                    vals=[x for x in conf if (x['regime'],x['method'],x['tier'],x['campaign'])==(reg,method,t,campaign)]
                    assert len(vals)==(5 if method in ['frozen','adapt_random','adapt_pretrained'] else 1)
                    v={k:(float(np.mean([x[k] for x in vals])) if all(x[k] is not None for x in vals) else None) for k in METRICS+['precontact_rmse']}
                    cs.append(v);campaigns.append(dict(regime=reg,method=method,tier=t,campaign=campaign,**v))
                means.append(dict(regime=reg,method=method,tier=t,**{k:float(np.mean([x[k] for x in cs])) if all(x[k] is not None for x in cs) else None for k in cs[0]}))
    differences=[]
    for reg in REGIMES:
        for c in range(5):
            for t in TIERS:
                f=next(x for x in campaigns if (x['regime'],x['method'],x['tier'],x['campaign'])==(reg,'frozen',t,c))
                for other in METHODS[1:]:
                    b=next(x for x in campaigns if (x['regime'],x['method'],x['tier'],x['campaign'])==(reg,other,t,c))
                    differences.append(dict(regime=reg,campaign=c,tier=t,comparison='frozen minus '+other,**{k:f[k]-b[k] if f[k] is not None and b[k] is not None else None for k in METRICS+['precontact_rmse']}))
    intervals=[]
    pairs=[('frozen',m) for m in METHODS[1:]]+[('adapt_pretrained','adapt_random')]
    for reg in REGIMES:
        models=[d for d in configs if d['regime']==reg];first=models[0]['rows']
        scenes=sorted({r['scene'] for r in first});si={s:i for i,s in enumerate(scenes)}
        geo=np.array([int(any(r['scene']==s and r['tier']=='joint_extrap' for r in first)) for s in scenes])
        for metric in METRICS:
            counts=np.zeros(len(scenes));cubes={m:np.zeros((5 if m in ['frozen','adapt_random','adapt_pretrained'] else 1,5,len(scenes))) for m in METHODS}
            validkey='prefix_alive' if metric=='selected_contact' else 'all_candidates_valid'
            valuekey={'shared_rmse':'shared_mse','accuracy':'correct'}.get(metric,metric)
            for row in first:counts[si[row['scene']]]+=row[validkey]
            for d in models:
                for row in d['rows']:
                    if row[validkey]:cubes[d['method']][d['seed'],d['campaign'],si[row['scene']]]+=row[valuekey]
            for a,b in pairs:
                intervals.append(dict(regime=reg,comparison=a+' minus '+b,metric=metric,eligible_states=int(counts.sum()),scene_clusters=len(scenes),**interval(cubes[a],cubes[b],counts,geo,metric)))
    assert (len(configs),sum(len(d['rows']) for d in configs),len(conf),len(campaigns),len(means),len(intervals))==(190,20520,760,280,56,56)
    return dict(stage='Development calibration-source sensitivity on reused controller states; not task confirmation or full-flight utility.',configuration_tier=conf,campaign_means=campaigns,method_means=means,campaign_differences=differences,comparisons=intervals)


@torch.no_grad()
def run(attempt):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dest=BASE/attempt;dest.mkdir(exist_ok=False);cache=DATA/attempt;cache.mkdir(exist_ok=False)
    started=time.time()
    try:
        inv=read(BASE/'inventory.json');extra=read(BASE/'additional_nominal_recovery.json')['files']
        recovered={**inv['recover_files'],**extra};assert len(recovered)==285
        for p,v in recovered.items():assert sha(RECOVER/p)==v,p
        audits,sources=metadata_audit(inv)
        sources|={BASE/'PROTOCOL.md',BASE/'inventory.json',BASE/'additional_nominal_recovery.json'}
        sources|={RECOVER/p for p in recovered}
        sources|={ROOT/'scripts'/s for s in ['icra_campaign_snapshot.py','icra_adaptation_fit.py','icra_confirm_frozen.py','icra_adaptation_ridge.py','icra_adaptation_scratch.py','icra_snapshot_prediction.py','icra_mpc_policy.py']}
        sources|={ROOT/'src/winddyn/models/wm.py',ROOT/'src/winddyn/train/trainer.py'}
        inputs={};taskmap={}
        for reg in REGIMES:
            paths,inputs[reg]=snapshots(reg);sources.update(paths)
            p=ROOT/'runs/icra_vertical_authority_20260908'/('config001' if reg=='mass_1p4' else 'config005')/'results.json';sources.add(p)
            eps=read(p)['episodes'];taskmap[reg]={}
            for name,a in inputs[reg].items():
                tier=name.split('_step')[0];es=[e for e in eps if e['tier']==tier];assert len(es)==12
                for j,e in enumerate(es):taskmap[reg][name,j]=e
            full=sum(int(a['valid'].all((0,1)).sum()) for a in inputs[reg].values())
            alive=sum(int(a['initial_alive'].sum()) for a in inputs[reg].values())
            assert (full,alive)==((98,99) if reg=='mass_1p4' else (106,108))
        write(dest/'source_lock.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)})
        protected=list((ROOT/'paper/icra_2027/figures/evidence').glob('*'))+[ROOT/'paper/icra_2027/main.tex',ROOT/'paper/icra_2027/supplement.tex',ROOT/'paper/icra_2027/evidence_manifest.json']
        write(dest/'protected_files.json',{str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()})
        write(dest/'preflight.json',dict(recovered_files=285,configuration_splits=audits,tf32=False,cuda=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),context_dtype='float32',forecast_cost_dtype='float64'))
        nominal={s:torch.load(RECOVER/f'runs/revision_20260907/checkpoints/supervised_seed{s}/best.pt',map_location='cpu',weights_only=False) for s in range(5)}
        aud={k:torch.from_numpy(np.concatenate([next(iter(inputs[r].values()))[k] for r in REGIMES])) for k in ['state_hist','action_hist','depth_hist']}
        assert len(aud['state_hist'])==24
        feature_cache={};encoder_audits={};allconfigs=[]
        for c in inv['configurations']:
            reg=c['regime'];model=None
            if c['encoder'] and c['encoder'] not in encoder_audits:
                ck=torch.load(RECOVER/c['encoder'],map_location='cpu',weights_only=False)
                model=WorldModel(ck['cfg']['model'],ck['stats']) if c['method']=='frozen' else make_encoder(ck['stats'])
                model.load_state_dict(ck['model'],strict=True);model.eval();model.requires_grad_(False)
                assert not model.use_wind and not model.privileged and not model.training
                for k,v in ck['stats'].items():
                    for stat in ['mean','std']:
                        assert torch.equal(getattr(model.normalizer,k+'_'+stat),torch.tensor(v[stat])),(c['index'],k,stat)
                for k in ['state_hist','action_hist']:
                    for stat in ['mean','std']:np.testing.assert_array_equal(ck['stats'][k][stat],nominal[c['seed']]['stats'][k][stat])
                cpu=model.encode_context(aud).numpy();model.to('cuda');gpu=model.encode_context({k:v.cuda() for k,v in aud.items()}).cpu().numpy()
                np.testing.assert_allclose(cpu,gpu,atol=5e-4,rtol=2e-5)
                poison={**{k:v.cuda() for k,v in aud.items()},**{k:torch.full((24,30,3),float('nan'),device='cuda') for k in ['target_position','future_state','wind_hist','wind_fut','true_wind','action_fut']}}
                np.testing.assert_array_equal(gpu,model.encode_context(poison).cpu().numpy())
                encoder_audits[c['encoder']]=dict(index=c['index'],strict_load=True,normalizer_matches_checkpoint=True,observed_stats_match_nominal=True,cpu_gpu_max_abs=float(np.max(np.abs(cpu-gpu))),future_privileged_keys_ignored=True)
            elif c['encoder']:
                # Frozen encoders are shared across campaigns; their extracted arrays are reused.
                assert all((c['encoder'],reg,name) in feature_cache for name in inputs[reg]) or c['method']=='frozen'
                if not all((c['encoder'],reg,name) in feature_cache for name in inputs[reg]):
                    ck=nominal[c['seed']];model=WorldModel(ck['cfg']['model'],ck['stats']);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False).cuda()
            fit=None
            if c['weight']:
                with np.load(RECOVER/c['weight']) as z:fit={k:z[k] for k in z.files}
                assert all(np.isfinite(v).all() for v in fit.values()) and (fit['std']>0).all()
            rows=[];worlds=[];contexts=[];feature_max=0.
            for name,a in inputs[reg].items():
                arr=model_input(a);key=(c['encoder'] or c['recipe'],reg,name)
                poisoned={**a,'position_world':np.full_like(a['position_world'],np.nan),'valid':np.zeros_like(a['valid']), 'reference_world':np.full_like(a['reference_world'],np.nan),'future_state':np.array([np.nan]),'true_wind':np.array([np.nan])}
                clean_arr=model_input(poisoned)
                for k in arr:np.testing.assert_array_equal(arr[k],clean_arr[k])
                if c['method']=='scalar':local=physics(arr,'x_',c['tau']);ctx=np.empty((108,0),np.float32)
                else:
                    if key not in feature_cache:
                        f=features(arr,'x_',c['recipe'],model,'cuda');cf=c1_features(arr,'x_',c['recipe'],model,'cuda')
                        np.testing.assert_array_equal(f,cf)
                        pf=features({**arr,'x_future_state':np.array([np.nan]),'x_wind_hist':np.array([np.nan]),'x_target_position':np.array([np.nan])},'x_',c['recipe'],model,'cuda')
                        np.testing.assert_array_equal(f,pf);feature_cache[key]=f
                    f=feature_cache[key];assert f.shape[1]==len(fit['mean']) and fit['coef'].shape==(f.shape[1]+1,90)
                    local=(predict(f,fit)+physics(arr,'x_').reshape(108,-1)).reshape(108,30,3)
                    ctx=f[:,:128].astype(np.float32) if c['encoder'] else np.empty((108,0),np.float32)
                local=local.reshape(12,9,30,3);sf=a['state_hist'][:,-1];co,si=sf[:,11],sf[:,10]
                world=local.copy();world[...,0]=co[:,None,None]*local[...,0]-si[:,None,None]*local[...,1];world[...,1]=si[:,None,None]*local[...,0]+co[:,None,None]*local[...,1];world+=a['initial_position'][:,None,None]
                assert world.dtype==np.float64 and np.isfinite(world).all()
                ca={k:a[k].astype(np.float64) for k in ['candidate_command','reference_world','previous_command']}
                pc=cost(world,ca)
                truth=a['position_world'].transpose(2,1,0,3);valid=a['valid'].transpose(2,1,0);tc=cost(truth.astype(np.float64),ca);choice=pc.argmin(1)
                error=(world-truth)**2;worlds.append(world);contexts.append(ctx)
                for j in range(12):
                    full=bool(valid[j].all());prefix=bool(a['initial_alive'][j]);e=taskmap[reg][name,j]
                    rows.append(dict(snapshot=name,task_index=j,tier=e['tier'],scene=e['scene'],prefix_alive=prefix,all_candidates_valid=full,
                         valid_candidate_records=int(valid[j].sum()),position_squared_error=float(np.where(valid[j],error[j].sum(-1),0).sum()),
                         shared_mse=float(error[j].sum(-1).mean()) if full else None,choice=int(choice[j]),oracle=int(tc[j].argmin()),
                         correct=bool(choice[j]==tc[j].argmin()) if full else None,regret=float(tc[j,choice[j]]-tc[j].min()) if full else None,
                         selected_contact=bool(not valid[j,choice[j],-1]) if prefix else None))
            np.savez_compressed(cache/f"config{c['index']:03d}.npz",world=np.stack(worlds),context=np.stack(contexts))
            result={**c,'rows':rows,'cache':str((cache/f"config{c['index']:03d}.npz").relative_to(ROOT))}
            write(dest/f"config{c['index']:03d}.json",result);allconfigs.append(result)
            if model is not None:del model
            if (c['index']+1)%19==0:print('completed',c['index']+1,'of190',round(time.time()-started,1),'s',flush=True)
        assert len(encoder_audits)==105
        write(dest/'interface_audit.json',dict(unique_encoders=105,encoders=encoder_audits,all_feature_adapters_exact=True,all_future_privileged_poison_checks_exact=True))
        write(dest/'summary.json',aggregate(allconfigs))
        outputs=list(dest.glob('*.json'))+list(cache.glob('*.npz'))
        write(dest/'output_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in outputs})
        write(dest/'completion.json',dict(status='awaiting_independent_acceptance',configurations=190,seconds=time.time()-started))
        print('COMPLETE; independent acceptance pending',flush=True)
    except Exception:
        write(dest/'failure.json',dict(error=traceback.format_exc(),elapsed_s=time.time()-started));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);run(p.parse_args().attempt)
