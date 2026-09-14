"""Standardized ridge with an unpenalized intercept and a reusable eigensystem."""
import numpy as np


class RidgePath:
    def __init__(self, x, dual=None):
        x=np.asarray(x,dtype=np.float64)
        self.mean=x.mean(0); self.std=np.maximum(x.std(0),1e-5)
        self.x=(x-self.mean)/self.std
        self.dual=len(x)<x.shape[1] if dual is None else dual
        gram=self.x@self.x.T if self.dual else self.x.T@self.x
        self.values,self.vectors=np.linalg.eigh(gram)
        self.values=np.maximum(self.values,0)

    def fit(self,y,lam):
        y=np.asarray(y,dtype=np.float64); ym=y.mean(0); centered=y-ym
        rhs=centered if self.dual else self.x.T@centered
        solution=self.vectors@((self.vectors.T@rhs)/(self.values[:,None]+len(y)*lam))
        coef=self.x.T@solution if self.dual else solution
        # Correct tiny numerical departures from zero feature means.
        intercept=ym-self.x.mean(0)@coef
        return dict(mean=self.mean,std=self.std,coef=np.vstack([coef,intercept]),lambda_=lam)


def predict(x,fit):
    return ((x-fit['mean'])/fit['std'])@fit['coef'][:-1]+fit['coef'][-1]
