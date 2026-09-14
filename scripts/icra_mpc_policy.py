"""Observed-history predictor adapter and fixed tracking cost for development MPC."""
import json,time
import numpy as np
import torch
from _common import ROOT
from icra_adaptation_fit import features,physics,TRAIN_BASE,sha
from icra_adaptation_ridge import predict
from icra_sysid_development import commands,basis
from winddyn.train.trainer import load_model
METHODS=['feedback','observer','scalar','axis','physics_features','raw','supervised']

def reference_at(wps,t,z,speed=.6):
    wps=np.asarray(wps);t=np.atleast_1d(t);out=np.empty((len(wps),len(t),3));vel=np.zeros_like(out)
    for i,path in enumerate(wps):
        lengths=np.linalg.norm(np.diff(path,axis=0),axis=1);cum=np.r_[0,np.cumsum(lengths)]
        for k,tt in enumerate(t):
            travel=np.clip((tt-2)*speed,0,cum[-1]);j=min(np.searchsorted(cum,travel,side='right')-1,len(lengths)-1);j=max(j,0)
            direction=(path[j+1]-path[j])/max(lengths[j],1e-9)
            out[i,k,:2]=path[j]+direction*(travel-cum[j]);out[i,k,2]=z
            if tt>=2 and travel<cum[-1]:vel[i,k,:2]=speed*direction
    return out,vel

def nominal_command(pos,ref,velocity):
    u=velocity+ref-pos;norm=np.maximum(np.linalg.norm(u[:,:2],axis=1,keepdims=True),1e-8);u[:,:2]*=np.minimum(1,1.5/norm);u[:,2]=np.clip(u[:,2],-.5,.5)
    return np.c_[u,np.zeros(len(u))]

def candidates(nominal):
    offsets=np.array([[x,y] for x in [-.4,0,.4] for y in [-.4,0,.4]])
    u=np.repeat(nominal[:,None],9,axis=1);u[:,:,:2]+=offsets[None];norm=np.maximum(np.linalg.norm(u[:,:,:2],axis=-1,keepdims=True),1e-8);u[:,:,:2]*=np.minimum(1,1.5/norm)
    return u

class Predictor:
    def __init__(self,method,regime,device):
        self.method=method;self.device=device;self.model=None;self.hashes={};root=ROOT/'runs/icra_adaptation_20260907'/regime/'formal_v1'
        if method in ['feedback','observer']:return
        recipe={'physics_features':'physics_only','raw':'raw_state','supervised':'supervised'}.get(method,'physics_only');p=root/f'{recipe}_seed0';m=json.loads((p/'results.json').read_text())
        if method=='scalar':self.tau=next(r['tau'] for r in m['physical_records'] if r['name']=='calibrated' and r['budget']==32 and r['subset_seed']==901);self.record(p/'results.json')
        elif method=='axis':
            p=ROOT/'runs/icra_sysid_development_20260908'/regime/'budget32_subset901/results.json';self.selected=json.loads(p.read_text())['selected'];self.record(p)
        else:
            rec=next(r for r in m['records'] if r['budget']==32 and r['subset_seed']==901 and r['center']=='physical');wp=p/rec['weight_file'];assert sha(wp)==rec['weight_sha256'];self.record(wp)
            with np.load(wp) as z:self.fit={k:z[k] for k in z.files}
            self.recipe=recipe
            if method=='supervised':
                ck=TRAIN_BASE/'checkpoints/supervised_seed0/best.pt';self.record(ck);self.model=load_model(ck,device);self.model.eval()
    def record(self,p):self.hashes[str(p.relative_to(ROOT))]=sha(p)
    @torch.no_grad()
    def plan(self,buf,pos,reference,nominal,previous):
        started=time.perf_counter();u=candidates(nominal);n=len(u);batch=buf.batch();sf=batch['state_hist'][:,-1].cpu().numpy();c,s=sf[:,11],sf[:,10]
        v=buf.vel[-1].cpu().numpy().copy();vx,vy=v[:,0].copy(),v[:,1].copy();v[:,0]=c*vx+s*vy;v[:,1]=-s*vx+c*vy
        arr={'x_'+k:np.repeat(value.cpu().numpy(),9,axis=0) for k,value in batch.items()}
        arr['x_vel_yaw_t']=np.repeat(v,9,axis=0);arr['x_action_fut']=np.repeat(u.reshape(-1,4)[:,None],30,axis=1).astype(np.float32)
        if self.method=='scalar':pred=physics(arr,'x_',self.tau)
        elif self.method=='axis':
            cmd,vel=commands(arr,'x_');parts=[]
            for axis,q in enumerate(self.selected):
                off,x=basis(cmd[...,axis],vel[:,axis],q['tau']);parts.append(off+x@np.array([q['gain'],q['bias']]))
            pred=np.stack(parts,-1)
        else:pred=(predict(features(arr,'x_',self.recipe,self.model,self.device),self.fit)+physics(arr,'x_').reshape(n*9,-1)).reshape(n*9,30,3)
        pred=pred.reshape(n,9,30,3);world=pred.copy();world[...,0]=c[:,None,None]*pred[...,0]-s[:,None,None]*pred[...,1];world[...,1]=s[:,None,None]*pred[...,0]+c[:,None,None]*pred[...,1];world+=pos[:,None,None]
        cost=((world-reference[:,None])**2).sum(-1).mean(-1)+.02*(u[:,:,:3]**2).sum(-1)+.05*((u[:,:,:3]-previous[:,None,:3])**2).sum(-1)
        assert np.isfinite(cost).all();choice=cost.argmin(1)
        return u[np.arange(n),choice],dict(choice=choice,cost=cost,latency_s=time.perf_counter()-started)
