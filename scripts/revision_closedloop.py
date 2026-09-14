"""Corrected development closed-loop evaluator; legacy artifacts unchanged.

Controller computes before recording, as training collection does; each new
20 Hz estimate becomes available at the next 50 Hz control tick. Oracle and
all learned estimators share exactly the H-record warm-up. First SDF contact
is absorbing and all subsequent records are excluded, never scored as flight.
"""
import argparse,json
import numpy as np
import torch
from _common import ROOT
from closed_loop_eval import (HistoryBuffer,state_features_torch,drag_accel_nominal,
    WaypointTracker,sample_waypoints,PlanarGridWind,GeometricController,ToFDepthSensor,
    TorchQuadSim,load_vehicle,load_yaml,load_scenes,load_field,load_model,H,K,DT_REC,DT_CTRL,GRAVITY)
from winddyn.sim.rollout import PHYS_HZ,CTRL_EVERY,REC_EVERY,_box_tensors,_sdf,_randomize
from winddyn.revision.protocol import portable_path,dump_new,attitude_from_state_features,predict_ridge
from revision_v2 import BASE as TRAIN_BASE
BASE=ROOT/'runs/revision_closedloop_20260907'

class Estimators:
    def __init__(self,condition,seed,device,vp):
        self.device=device;self.vp=vp;self.model=None
        if condition not in ['none','true','observer']:
            ck=TRAIN_BASE/'checkpoints'/f'{condition}_seed{seed}'
            self.model=load_model(ck/'best.pt',device)
    @torch.no_grad()
    def estimate(self,name,buf,true_wind):
        if name=='none' or not buf.ready():return torch.zeros_like(true_wind)
        if name=='true':return true_wind
        if name=='observer':
            sf=torch.stack(buf.state[-H:],dim=1).cpu().numpy()
            R=attitude_from_state_features(sf)
            v=np.einsum('ntij,ntj->nti',R,sf[...,:3]);acc=np.diff(v,axis=1)/DT_REC
            bz=R[:,1:,:,2];T=np.clip(self.vp.mass*(acc[...,2]+GRAVITY)/np.maximum(bz[...,2],.2),0,4*self.vp.mass*GRAVITY)
            force=self.vp.mass*acc[...,:2]-T[...,None]*bz[...,:2]
            aa=.5*self.vp.air_density*float(self.vp.body_drag_cda[:2].mean())
            bb=float(self.vp.rotor_drag_coeff[:2].mean())*T/(self.vp.mass*GRAVITY)
            mag=np.linalg.norm(force,axis=-1);speed=2*mag/np.maximum(np.sqrt(bb*bb+4*aa*mag)+bb,1e-12)
            est=v[:,1:,:2]+force/np.maximum(mag[...,None],1e-12)*speed[...,None]
            return torch.tensor(est[:,-6:].mean(1),dtype=torch.float32,device=self.device)
        return self.model.estimate_wind(self.model.encode_context(buf.batch(self.model.use_depth)))

STRESSES={
    'nominal':dict(mass=1.,motor_lag=1.,velocity_gain=1.),
    'payload_1p4':dict(mass=1.4,motor_lag=1.,velocity_gain=1.),
    'motor_lag_3':dict(mass=1.,motor_lag=3.,velocity_gain=1.),
    'gain_0p6':dict(mass=1.,motor_lag=1.,velocity_gain=.6),
    'gain_1p4':dict(mass=1.,motor_lag=1.,velocity_gain=1.4),
}


def apply_stress(sim,controller_cfg,name):
    # Perturb simulated plant only, retaining nominal vp for all estimators
    # and compensation. Controller gain shift is common to all conditions.
    import copy
    spec=STRESSES[name];cfg=copy.deepcopy(controller_cfg)
    sim.mass=sim.mass*spec['mass']
    sim.rotors.time_constant=sim.rotors.time_constant*spec['motor_lag']
    for key in ['velocity_p','velocity_i']:
        cfg['gains'][key]=[v*spec['velocity_gain'] for v in cfg['gains'][key]]
    return cfg


@torch.no_grad()
def run_tier(tasks, vp, estimators, condition, seed, device,
             ref_traj=None, duration_s=12.0, stress="nominal"):
    """One paired rollout batch under `condition`; returns per-env metrics
    and the recorded position trajectories."""
    N = len(tasks)
    d = torch.device(device)
    gen = torch.Generator(device=d).manual_seed(seed)

    scenes = [t["scene"] for t in tasks]
    z_ref = scenes[0].z_ref
    centers, half, mask = _box_tensors(scenes, d)

    zero_wind = condition == "ref"
    fields = [t["field"] for t in tasks]
    grid = PlanarGridWind(fields, N, device=d)
    grid.assign(torch.arange(N))

    class Wind:
        name = "grid"
        def step(self, dt): pass
        def sample(self, positions, sim_time, env_ids=None):
            if zero_wind:
                return torch.zeros_like(positions)
            return grid.sample(positions, sim_time)
        def reset(self, env_ids, generator): pass

    sim = TorchQuadSim(vp, N, Wind(), device=d, physics_dt=1.0 / PHYS_HZ)
    _randomize(sim, vp, gen)

    ctrl_cfg = load_yaml("configs/sim/controller.yaml")["controller"]
    ctrl_cfg = apply_stress(sim,ctrl_cfg,stress)
    ctrl = GeometricController(vp, N, d, ctrl_cfg)

    wps, starts = [], []
    for i, t in enumerate(tasks):
        rng_i = np.random.default_rng(t["wp_seed"])
        w8 = sample_waypoints(t["scene"], rng_i)
        wps.append(w8)
        starts.append(np.array([w8[0][0], w8[0][1], z_ref]))
    tracker = WaypointTracker(
        torch.tensor(np.stack(wps), dtype=torch.float32), z_ref, d)
    wp_t = tracker.wp

    tof = vp.tof or {}
    depth_cam = ToFDepthSensor(
        centers, half, mask, width=64, height=64,
        hfov_deg=float(tof.get("horizontal_fov_deg", 106.0)),
        min_range=float(tof.get("min_range", 0.2)),
        max_range=float(tof.get("max_range", 6.0)),
        offset_body=tuple(tof.get("position", (-0.018, -0.078, 0.003))),
        device=d)

    sim.reset(pos=torch.tensor(np.stack(starts), dtype=torch.float32, device=d),
              generator=gen)
    ctrl.reset(torch.arange(N, device=d), sim.state.quat)

    buf = HistoryBuffer(N, d)
    n_steps = int(duration_s * PHYS_HZ)
    n_rec = n_steps // REC_EVERY
    traj = np.zeros((n_rec, N, 3), np.float32)
    cross_track = np.zeros((n_rec, N), np.float32)
    alt_err = np.zeros((n_rec, N), np.float32)
    collide = np.zeros((n_rec, N), bool)
    w_err = np.full((n_rec, N), np.nan, np.float32)
    energy = np.zeros(N, np.float64)
    wp_count = np.zeros(N, np.int64)
    prev_idx = tracker.idx.clone()

    cmd = torch.zeros(N, 4, device=d)
    thrust_cmd = torch.zeros(N, 4, device=d)
    w_hat = torch.zeros(N, 2, device=d)
    alive=torch.ones(N,dtype=torch.bool,device=d)
    first_hit=np.full(N,-1,dtype=np.int64)
    valid=np.zeros((n_rec,N),bool)
    k = 0
    for step in range(n_steps):
        hit=((_sdf(sim.state.pos,centers,half,mask)<.25)|(sim.state.pos[:,2]<.1)) & alive
        hit_np=hit.cpu().numpy();first_hit[hit_np]=step
        alive &= ~hit
        if step % CTRL_EVERY == 0:
            s = sim.state
            cmd = tracker.command(s.pos, ctrl.nose_yaw(s.quat))
            wp_count += ((tracker.idx != prev_idx) & alive).cpu().numpy().astype(np.int64)
            prev_idx = tracker.idx.clone()
            a_ff = None
            if condition not in ("ref", "none"):
                v_xy = s.lin_vel[:, :2]
                tfac = (thrust_cmd.sum(-1) / (vp.mass * GRAVITY)).clamp(0.0, 4.0)
                a_ff_xy = (drag_accel_nominal(v_xy, tfac, vp)
                           - drag_accel_nominal(v_xy - w_hat, tfac, vp))
                a_ff = torch.cat([a_ff_xy,
                                  torch.zeros(N, 1, device=d)], dim=-1)
            thrust_cmd = ctrl.compute(s.pos, s.quat, s.lin_vel, s.ang_vel,
                                      cmd, DT_CTRL, accel_ff=a_ff)
            energy += ((thrust_cmd ** 2).sum(-1)*alive).cpu().numpy() * DT_CTRL
        if step % REC_EVERY == 0:
            s = sim.state
            sf = state_features_torch(s.pos, s.quat, s.lin_vel, s.ang_vel)
            depth = depth_cam.render(s.pos, s.quat)
            q = s.quat
            bz = torch.stack([2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2]),
                              2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1]),
                              1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)], dim=-1)
            buf.push(sf, cmd.clone(), depth, s.lin_vel.clone(), bz,
                     thrust_cmd.sum(-1).clone())
            true_w = sim.wind.sample(s.pos, sim.time)[:, :2] if not zero_wind \
                else grid.sample(s.pos, sim.time)[:, :2]
            if condition not in ("ref",):
                w_hat = estimators.estimate(condition, buf, true_w)
                w_hat = w_hat.clamp(-15.0, 15.0)
                w_err[k] = (w_hat - true_w).norm(dim=-1).cpu().numpy()
            valid[k] = alive.cpu().numpy()
            traj[k] = s.pos.cpu().numpy()
            sd = _sdf(s.pos, centers, half, mask)
            collide[k] = (sd < 0.25).cpu().numpy() | (s.pos[:, 2] < 0.1).cpu().numpy()
            alt_err[k] = (s.pos[:, 2] - z_ref).abs().cpu().numpy()
            # cross-track: distance to segment prev_wp -> current_wp (xy)
            idx = tracker.idx
            cur = wp_t[torch.arange(N, device=d), idx]
            prev = wp_t[torch.arange(N, device=d), (idx - 1) % tracker.n_wp]
            seg = cur - prev
            L2 = (seg ** 2).sum(-1).clamp_min(1e-6)
            tproj = (((s.pos[:, :2] - prev) * seg).sum(-1) / L2).clamp(0, 1)
            closest = prev + tproj.unsqueeze(-1) * seg
            cross_track[k] = (s.pos[:, :2] - closest).norm(dim=-1).cpu().numpy()
            k += 1
        previous=[getattr(sim.state,key).clone() for key in ['pos','quat','lin_vel','ang_vel']]
        sim.step(thrust_cmd)
        # Absorbing first contact: independent dead environments stay frozen.
        for key,value in zip(['pos','quat','lin_vel','ang_vel'],previous):
            getattr(sim.state,key)[~alive]=value[~alive]

    final_hit=((_sdf(sim.state.pos,centers,half,mask)<.25)|(sim.state.pos[:,2]<.1)) & alive
    first_hit[final_hit.cpu().numpy()]=n_steps
    warm=H-1
    vv=valid[warm:]
    def average(arr):
        count=vv.sum(0)
        return np.divide(np.where(vv,arr[warm:],0).sum(0),count,
                         out=np.full(N,np.nan),where=count>0)
    out={'cross_track_mean':average(cross_track),'alt_err_mean':average(alt_err),
         'collided':first_hit>=0,'first_contact_s':np.where(first_hit>=0,first_hit/PHYS_HZ,duration_s),
         'valid_records':vv.sum(0),'waypoints_reached':wp_count,'energy':energy,
         'wind_est_rmse':np.sqrt(average(w_err**2)),
         'actual_mass_kg':sim.mass.cpu().numpy(),
         'actual_motor_tau_s':sim.rotors.time_constant.cpu().numpy().reshape(-1)}
    return out, {'position':traj,'valid':valid,'cross_track':cross_track,'wind_error':w_err}

def build_tasks(tier,n_tasks,selection_seed=None,waypoint_base=190000):
    scenes={s.scene_id:s for s in load_scenes(ROOT/'data/manifests/scenes.json')}
    fields=json.load(open(ROOT/'data/manifests/wind_fields.json'))['fields']
    tag='train' if tier=='id' else 'ood_extrap'
    cand=sorted([e for e in fields if e['tag']==tag and ((scenes[e['scene_id']].seed>=4)==(tier=='joint_extrap'))],key=lambda e:(e['scene_id'],e['wind_id']))
    rng=np.random.default_rng(selection_seed if selection_seed is not None else 26090711+['id','wind_extrap','joint_extrap'].index(tier))
    sel=rng.choice(len(cand),min(n_tasks,len(cand)),replace=False)
    return [dict(scene=scenes[cand[int(i)]['scene_id']],field=load_field(portable_path(cand[int(i)]['path'],ROOT)),
        wind_id=cand[int(i)]['wind_id'],scene_id=cand[int(i)]['scene_id'],wp_seed=waypoint_base+131*int(i)) for i in sel]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--condition',choices=['none','true','observer','pajepa','direct','supervised_wind','wind_only'],required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--n-tasks',type=int,default=48)
    ap.add_argument('--duration',type=float,default=12);ap.add_argument('--device',default='cuda');ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    vp=load_vehicle();est=Estimators(a.condition,a.seed,a.device,vp);rows=[]
    for ti,tier in enumerate(['id','wind_extrap','joint_extrap']):
        tasks=build_tasks(tier,a.n_tasks)
        for off in range(0,len(tasks),16):
            chunk=tasks[off:off+16]
            out,traj=run_tier(chunk,vp,est,a.condition,1800000+1000*ti+off,a.device,duration_s=a.duration)
            for i,t in enumerate(chunk):
                r=dict(tier=tier,scene=t['scene_id'],wind=t['wind_id'],wp_seed=t['wp_seed'])
                for key,values in out.items():
                    value=float(values[i]);r[key]=value if np.isfinite(value) else None
                rows.append(r)
            print(tier,off+len(chunk),'complete',flush=True)
    dest=BASE/('pilot' if a.pilot else 'metrics')/f'{a.condition}_seed{a.seed}.json'
    dump_new(dest,dict(condition=a.condition,seed=a.seed,duration_s=a.duration,n_tasks=a.n_tasks,episodes=rows))
if __name__=='__main__':main()
