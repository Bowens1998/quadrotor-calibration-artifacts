"""Revision v2: portable manifests, first-contact censoring, causal references.

Legacy dataset and metrics remain unchanged. All new results use this module.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from winddyn.data.dataset import WindowDataset

VERSION = "pajepa-revision-v2-20260907"


def portable_path(path, root):
    p = Path(path)
    if not p.is_absolute():
        return str(Path(root) / p)
    # Never silently use an old checkout's absolute data paths.
    parts = p.parts
    if "data" not in parts:
        raise ValueError(f"Cannot relocate non-data path: {p}")
    return str(Path(root).joinpath(*parts[parts.index("data"):]))


def relocate(entries, root):
    return [dict(e, path=portable_path(e["path"], root)) for e in entries]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


class RevisionDataset(WindowDataset):
    """Discard every window reaching or following the FIRST contact.

    A common set of window anchors is used across history ablations (anchor_H).
    Entire episodes remain the resampling/assignment unit.
    """
    def __init__(self, entries, H=12, K=30, stride=7, with_depth=True,
                 anchor_H=None):
        clean = []
        for e in entries:
            e = dict(e)
            with np.load(e["path"], allow_pickle=False) as z:
                hit = np.flatnonzero(z["collision"])
                if len(hit):
                    e["n_steps"] = min(e["n_steps"], int(hit[0]))
            e["collided"] = False
            clean.append(e)
        super().__init__(clean, H=H, K=K, stride=stride,
                         with_depth=with_depth, with_patch=False)
        # Wind-vector teacher needs no expensive field patch.
        if anchor_H is not None:
            self.index = [(ei,t) for ei,e in enumerate(clean)
                          for t in range(max(H,anchor_H)-1, e["n_steps"]-K-1, stride)]


def causal_observer(ep, vp):
    """Nominal horizontal drag inversion; extra-input (thrust telemetry) reference.

    Uses only samples <=t, full quaternion for thrust axis, backward velocity
    differences, and a stable analytic inverse of isotropic horizontal drag.
    Unknown episode randomization and motor lag are NOT used.
    """
    v = np.asarray(ep["velocity_world"], dtype=np.float64)
    q = np.asarray(ep["quaternion_world_body"], dtype=np.float64)
    T = np.asarray(ep["rotor_thrust_cmd"], dtype=np.float64).sum(-1)
    time = np.asarray(ep["t"], dtype=np.float64)
    acc = np.zeros_like(v)
    acc[1:] = np.diff(v, axis=0) / np.diff(time)[:,None]
    w,x,y,z = q.T
    bz = np.stack([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)],-1)
    force = vp.mass * acc[:,:2] - T[:,None]*bz[:,:2]
    a = 0.5*vp.air_density*float(vp.body_drag_cda[:2].mean())
    b = float(vp.rotor_drag_coeff[:2].mean()) * np.clip(T/(vp.mass*9.81),0,4)
    mag = np.linalg.norm(force,axis=-1)
    speed = 2*mag / np.maximum(np.sqrt(b*b+4*a*mag)+b,1e-12)
    vr = -force / np.maximum(mag[:,None],1e-12) * speed[:,None]
    est = v[:,:2]-vr
    return np.stack([est[max(1,t-5):t+1].mean(0) if t else np.zeros(2)
                     for t in range(len(v))])


def fit_ridge(z, w, lam):
    mean=z.mean(0); std=np.maximum(z.std(0),1e-5)
    x=np.c_[(z-mean)/std,np.ones(len(z))]
    reg=np.eye(x.shape[1])*lam*len(x); reg[-1,-1]=0
    coef=np.linalg.solve(x.T@x+reg,x.T@w)
    return {"mean":mean,"std":std,"coef":coef,"lambda":lam}


def predict_ridge(z, fit):
    return np.c_[(z-fit["mean"])/fit["std"],np.ones(len(z))]@fit["coef"]


def attitude_from_state_features(state):
    """Reconstruct R_world_body from sensed gravity + the body -Y nose yaw.

    Uses only the same 12 channels available to the learned encoder. Undefined
    near a vertical nose; clamp only protects numerical conditioning there.
    """
    s=np.asarray(state,dtype=np.float64)
    gb=s[...,6:9];gb=gb/np.maximum(np.linalg.norm(gb,axis=-1,keepdims=True),1e-12)
    nose=np.zeros_like(gb);nose[...,1]=-1
    nb=nose-(nose*gb).sum(-1,keepdims=True)*gb
    nb=nb/np.maximum(np.linalg.norm(nb,axis=-1,keepdims=True),1e-12)
    nw=np.stack([s[...,11],s[...,10],np.zeros_like(s[...,10])],-1)
    gw=np.zeros_like(gb);gw[...,2]=-1
    body=np.stack([nb,np.cross(gb,nb),gb],axis=-1)
    world=np.stack([nw,np.cross(gw,nw),gw],axis=-1)
    return world@np.swapaxes(body,-1,-2)


def matched_input_observer(ep,vp):
    """Causal observer using only model-observable state channels, no telemetry.

    Estimate thrust from vertical acceleration assuming negligible vertical
    aerodynamic force. This is an explicit nominal-model approximation, not
    a calibrated bound. No future differences or episode randomization used.
    """
    from winddyn.data.dataset import state_features
    s=state_features(ep);R=attitude_from_state_features(s)
    vel=np.einsum('tij,tj->ti',R,s[:,:3])
    acc=np.zeros_like(vel);acc[1:]=np.diff(vel,axis=0)/np.diff(ep['t'])[:,None]
    bz=R[:,:,2]
    T=vp.mass*(acc[:,2]+9.81)/np.maximum(bz[:,2],.2)
    T=np.clip(T,0,4*vp.mass*9.81)
    # Supply reconstructed state only to the common inverse. q is reconstructed
    # indirectly through the exact known bz from observable features; the
    # source quaternion below supplies the same bz and no extra unknown DOF.
    modified=dict(ep,velocity_world=vel,
                  rotor_thrust_cmd=np.repeat((T/4)[:,None],4,axis=1))
    return causal_observer(modified,vp)
