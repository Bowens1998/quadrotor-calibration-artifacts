"""Episode-budgeted fit/validation splits; no outcome-dependent selection."""
from collections import defaultdict
import numpy as np

BUDGETS=(8,16,32,64)
SUBSET_SEEDS=(901,902,903,904,905)


def calibration_subsets(entries,seed):
    groups=defaultdict(list)
    for e in entries:groups[e['calibration_pair']].append(e)
    assert len(groups)==32
    assert all(len(g)==2 and {e['mode'] for e in g}=={'tracking','perturbation'} for g in groups.values())
    ids=sorted(groups);perm=np.random.default_rng(seed).permutation(ids).tolist();out={}
    for budget in BUDGETS:
        n=budget//2;nfit=3*n//4;selected=perm[:n]
        fit=[e for k in selected[:nfit] for e in groups[k]]
        validation=[e for k in selected[nfit:] for e in groups[k]]
        assert len(fit)+len(validation)==budget
        out[budget]={'fit':fit,'validation':validation}
    return out
