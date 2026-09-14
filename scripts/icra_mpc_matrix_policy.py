"""Parameter-only extension of accepted MPC pilot; no policy equations changed."""
from icra_mpc_policy import *
from functools import partial
from winddyn.control.tracking import _segment_clear
CONFIGS=[(r,m,0,None) for r in ['mass_1p4','lag_3'] for m in ['feedback','observer']]
CONFIGS += [(r,m,0,s) for r in ['mass_1p4','lag_3'] for m in ['scalar','axis','physics_features','raw'] for s in range(901,906)]
CONFIGS += [(r,'supervised',seed,s) for r in ['mass_1p4','lag_3'] for s in range(901,906) for seed in range(5)]
class MatrixPredictor(Predictor):
    def __init__(self,method,regime,device,model_seed,subset):
        self.method=method;self.device=device;self.model=None;self.hashes={};root=ROOT/'runs/icra_adaptation_20260907'/regime/'formal_v1'
        if method in ['feedback','observer']:return
        recipe={'physics_features':'physics_only','raw':'raw_state','supervised':'supervised'}.get(method,'physics_only');p=root/f'{recipe}_seed{model_seed if method=="supervised" else 0}';m=json.loads((p/'results.json').read_text())
        if method=='scalar':self.tau=next(r['tau'] for r in m['physical_records'] if r['name']=='calibrated' and r['budget']==32 and r['subset_seed']==subset);self.record(p/'results.json')
        elif method=='axis':
            p=ROOT/'runs/icra_sysid_development_20260908'/regime/f'budget32_subset{subset}/results.json';self.selected=json.loads(p.read_text())['selected'];self.record(p)
        else:
            rec=next(r for r in m['records'] if r['budget']==32 and r['subset_seed']==subset and r['center']=='physical');wp=p/rec['weight_file'];assert sha(wp)==rec['weight_sha256'];self.record(wp)
            with np.load(wp) as z:self.fit={k:z[k] for k in z.files}
            self.recipe=recipe
            if method=='supervised':
                ck=TRAIN_BASE/f'checkpoints/supervised_seed{model_seed}/best.pt';self.record(ck);self.model=load_model(ck,device);self.model.eval()
