"""Prespecified eight-comparison confirmation family, paired campaign/seed/scene."""
import numpy as np

PRIMARY=[('lag_3','frozen_supervised','calibrated','position'),('lag_3','frozen_supervised','adapt_random','position'),('lag_3','frozen_supervised','adapt_pretrained','position'),('lag_3','adapt_pretrained','adapt_random','position'),('mass_1p4','frozen_supervised','physics_only','position'),('lag_3','frozen_supervised','calibrated','accuracy'),('lag_3','frozen_supervised','calibrated','contact'),('mass_1p4','frozen_supervised','physics_only','accuracy')]


def interval(a,b,counts,geometry,metric,reps=20000,seed=30812007):
    assert a.shape[1:]==b.shape[1:] and a.shape[1]==5 and a.shape[0] in (1,5) and b.shape[0] in (1,5)
    rng=np.random.default_rng(seed);scene=np.zeros((reps,len(counts)),int)
    for g in np.unique(geometry):
        ix=np.flatnonzero(geometry==g);scene[:,ix]=rng.multinomial(len(ix),np.ones(len(ix))/len(ix),size=reps)
    campaigns=rng.multinomial(5,np.ones(5)/5,size=reps);models=rng.multinomial(5,np.ones(5)/5,size=reps)
    den=scene@counts;ok=den>0
    def transform(v):return np.sqrt(v) if metric=='position' else v
    def samples(x):
        v=transform(np.einsum('msc,rc->rms',x,scene[ok])/den[ok,None,None])
        mw=models[ok] if x.shape[0]==5 else np.ones((ok.sum(),1))
        return np.einsum('rms,rm,rs->r',v,mw,campaigns[ok])/(x.shape[0]*5)
    draws=samples(a)-samples(b)
    point=float(transform(a.sum(-1)/counts.sum()).mean()-transform(b.sum(-1)/counts.sum()).mean())
    return dict(difference=point,adjusted_99_375_interval=np.quantile(draws,[.003125,.996875]).tolist(),descriptive_95_interval=np.quantile(draws,[.025,.975]).tolist(),requested_draws=reps,valid_draws=int(ok.sum()),family_size=8)
