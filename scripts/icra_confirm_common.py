"""Namespaces and job/seed maps for fresh calibration confirmation."""
from _common import ROOT
BASE=ROOT/'runs/icra_calibration_confirmation_20260908'
FIXTURE=BASE/'operational_fixture'
REGIMES=('mass_1p4','lag_3')
BUDGETS=(32,64)
CALIBRATION_JOBS=[(r,c) for c in range(5) for r in REGIMES]
BRANCH_JOBS=[(r,b) for b in range(2) for r in REGIMES]
FROZEN_JOBS=[(r,c,k) for c in range(5) for r in REGIMES for k in range(33)]
ADAPT_JOBS=[(arm,r,b,c,s) for arm in ('random','pretrained') for r in REGIMES for b in BUDGETS for c in range(5) for s in range(5)]

def calibration_seed(campaign):
    assert 0<=campaign<5
    return 12012007+100003*campaign

def branch_seed(batch):
    assert batch in (0,1)
    return (15120007,16120007)[batch]

def verify_plant(entries,regime):
    import numpy as np
    from winddyn.utils.config import load_vehicle
    vp=load_vehicle();mf,lf=(1.4,1.) if regime=='mass_1p4' else (1.,3.)
    values=lambda key:np.array([e['domain_randomization'][key] for e in entries])
    np.testing.assert_array_equal(values('adapt_mass_multiplier'),np.full(len(entries),mf))
    np.testing.assert_array_equal(values('adapt_lag_multiplier'),np.full(len(entries),lf))
    np.testing.assert_allclose(values('realized_mass_kg'),vp.mass*values('mass')*mf,rtol=1e-6)
    np.testing.assert_allclose(values('realized_motor_tau_s'),vp.motor_time_constant*values('tau_m')*lf,rtol=1e-6)
