"""Common-harness closed-loop gate for paired-data trajectory readouts."""
import json
import numpy as np
import icra_mpc_matrix as parent
from icra_mpc_authority_matrix import rollout
from icra_mpc_matrix_policy import MatrixPredictor
from icra_decision_fit import BASE as FIT_BASE,CONFIGS as FIT_CONFIGS
BASE=parent.ROOT/'runs/icra_paired_mpc_20260908'
CONFIGS=[]
for reg in ['mass_1p4','lag_3']:
    CONFIGS += [(reg,m,0,None) for m in ['feedback','observer','scalar','physics_features','raw']]
    CONFIGS += [(reg,'scalar_old',0,s) for s in range(901,906)]
    CONFIGS += [(reg,'supervised',seed,None) for seed in range(5)]

class Predictor(MatrixPredictor):
    def __init__(self,method,regime,device,model_seed,subset):
        super().__init__('scalar' if method=='scalar_old' else method,regime,device,model_seed,subset or 901)
        if method in ['feedback','observer','scalar_old']:return
        self.hashes={k:v for k,v in self.hashes.items() if k.endswith('best.pt')}
        accepted=json.loads((FIT_BASE/'accepted.json').read_text());assert accepted['accepted_fits']==84
        if method=='scalar':
            self.tau=next(r for r in accepted['scalar_new_data'] if r['regime']==regime)['selected']['tau'];self.record(FIT_BASE/'accepted.json')
        else:
            folder=FIT_BASE/regime/f'config{FIT_CONFIGS.index((method,model_seed)):02d}'
            r=next(x for x in json.loads((folder/'results.json').read_text())['records'] if x['objective']=='trajectory')['selected'];path=folder/r['weight_file'];self.record(path)
            assert self.hashes[str(path.relative_to(parent.ROOT))]==r['weight_sha256']
            with np.load(path) as z:self.fit={k:z[k] for k in z.files}

if __name__=='__main__':
    parent.BASE=BASE;parent.CONFIGS=CONFIGS;parent.MatrixPredictor=Predictor;parent.rollout=rollout
    parent.main()
