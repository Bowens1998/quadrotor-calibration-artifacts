"""Full development matrix; rollout mechanics copied unchanged from accepted pilot."""
import argparse,hashlib,json,time
from pathlib import Path
from revision_closedloop import *
from winddyn.sim.rollout import _sdf,_box_tensors,_randomize
from icra_mpc_policy import Predictor,METHODS,reference_at,nominal_command
from icra_adaptation_fit import sha
from icra_mpc_matrix_policy import MatrixPredictor,CONFIGS,_segment_clear
BASE=ROOT/'runs/icra_mpc_matrix_20260908'

@torch.no_grad()
def rollout(tasks,regime,method,device,model_seed,subset,seed,duration_s=12.):
    vp=load_vehicle();condition=method;stress='payload_1p4' if regime=='mass_1p4' else 'motor_lag_3'
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
        assert all(_segment_clear(t['scene'],aa,bb,.6) for aa,bb in zip(w8[:-1],w8[1:])), 'Reference clearance failure before simulation'
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
    predictor=MatrixPredictor(method,regime,device,model_seed,subset)
    observer=Estimators('observer',0,device,vp) if method=='observer' else None
    wps_np=np.stack(wps);n_steps=int(duration_s*PHYS_HZ)
    active=torch.zeros(N,4,device=d);thrust=torch.zeros(N,4,device=d);wind_est=torch.zeros(N,2,device=d)
    alive=torch.ones(N,dtype=torch.bool,device=d);first_hit=np.full(N,-1,dtype=int)
    pending=None;pending_step=None;positions=[];commands_log=[];valid=[];errors=[];decisions=[];warm=[hashlib.sha256() for _ in range(N)]
    for step in range(n_steps):
        s=sim.state;hit=((_sdf(s.pos,centers,half,mask)<.25)|(s.pos[:,2]<.1))&alive
        first_hit[hit.cpu().numpy()]=step;alive &= ~hit
        t=step/PHYS_HZ
        if step%CTRL_EVERY==0:
            ref,velocity=reference_at(wps_np,[t],z_ref)
            nominal=nominal_command(s.pos.cpu().numpy(),ref[:,0],velocity[:,0])
            if method in ['feedback','observer'] or t<2:
                active=torch.tensor(nominal,dtype=torch.float32,device=d)
            elif pending is not None and step>=pending_step:
                assert step==pending_step
                active=torch.tensor(pending,dtype=torch.float32,device=d);decisions[-1]['applied_step']=step;pending=None
            a_ff=None
            if method=='observer' and t>=2:
                factor=(thrust.sum(-1)/(vp.mass*GRAVITY)).clamp(0,4)
                ff=drag_accel_nominal(s.lin_vel[:,:2],factor,vp)-drag_accel_nominal(s.lin_vel[:,:2]-wind_est,factor,vp)
                a_ff=torch.cat([ff,torch.zeros(N,1,device=d)],-1)
            thrust=ctrl.compute(s.pos,s.quat,s.lin_vel,s.ang_vel,active,DT_CTRL,accel_ff=a_ff)
        if step%REC_EVERY==0:
            sf=state_features_torch(s.pos,s.quat,s.lin_vel,s.ang_vel);depth=depth_cam.render(s.pos,s.quat)
            q=s.quat;bz=torch.stack([2*(q[:,1]*q[:,3]+q[:,0]*q[:,2]),2*(q[:,2]*q[:,3]-q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)],-1)
            buf.push(sf,active.clone(),depth,s.lin_vel.clone(),bz,thrust.sum(-1).clone())
            positions.append(s.pos.cpu().numpy().copy());commands_log.append(active.cpu().numpy().copy());valid.append(alive.cpu().numpy().copy())
            ref,_=reference_at(wps_np,[t],z_ref);errors.append(np.linalg.norm(positions[-1]-ref[:,0],axis=-1))
            if t<2:
                for i in range(N):
                    for value in [sf[i],active[i],depth[i],s.pos[i],s.quat[i],s.lin_vel[i]]:warm[i].update(value.cpu().numpy().tobytes())
            if t>=2 and method=='observer':wind_est=observer.estimate('observer',buf,torch.zeros(N,2,device=d)).clamp(-15,15)
            if t>=2 and step%20==0 and method not in ['feedback','observer']:
                assert buf.ready();ref,_=reference_at(wps_np,t+.05*np.arange(1,31),z_ref)
                pending,info=predictor.plan(buf,s.pos.cpu().numpy().copy(),ref,nominal,active.cpu().numpy().copy())
                pending_step=step+CTRL_EVERY
                decisions.append(dict(observation_step=step,activate_step=pending_step,applied_step=None,choice=info['choice'].tolist(),latency_s=info['latency_s']))
                # Buffer must retain the command that was active BEFORE planning.
                np.testing.assert_array_equal(buf.action[-1].cpu().numpy(),commands_log[-1])
        previous=[getattr(s,k).clone() for k in ['pos','quat','lin_vel','ang_vel']]
        sim.step(thrust)
        for k,v in zip(['pos','quat','lin_vel','ang_vel'],previous):getattr(sim.state,k)[~alive]=v[~alive]
    hit=((_sdf(sim.state.pos,centers,half,mask)<.25)|(sim.state.pos[:,2]<.1))&alive;first_hit[hit.cpu().numpy()]=n_steps;alive &= ~hit
    terminal_ref,_=reference_at(wps_np,[duration_s],z_ref);terminal=np.linalg.norm(sim.state.pos.cpu().numpy()-terminal_ref[:,0],axis=-1)
    errors=np.array(errors);valid=np.array(valid);post=valid.copy();post[:40]=False
    count=post.sum(0);rmse=np.sqrt(np.divide(np.where(post,errors**2,0).sum(0),count,out=np.full(N,np.nan),where=count>0))
    for decision in decisions:assert decision['applied_step']==decision['activate_step']>decision['observation_step']
    rows=[]
    for i,task in enumerate(tasks):
        rows.append(dict(scene=task['scene_id'],wind=task['wind_id'],tier=task['tier'],wp_seed=task['wp_seed'],collided=bool(first_hit[i]>=0),first_contact_s=float(first_hit[i]/PHYS_HZ if first_hit[i]>=0 else duration_s),valid_records=int(count[i]),tracking_rmse=float(rmse[i]) if np.isfinite(rmse[i]) else None,terminal_error=float(terminal[i]),success=bool(alive[i] and terminal[i]<.5),prefix_sha256=warm[i].hexdigest(),actual_mass_kg=float(sim.mass[i]),actual_motor_tau_s=float(sim.rotors.time_constant.reshape(-1)[i])))
    return dict(method=method,regime=regime,stage='development',episodes=rows,decisions=decisions,predictor_provenance=predictor.hashes),dict(position=np.array(positions),action=np.array(commands_log),valid=valid,tracking_error=errors,reference_waypoints=wps_np)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);a=ap.parse_args();torch.set_num_threads(4)
    regime,method,model_seed,subset=CONFIGS[a.index];d=BASE/'configs'/f'config{a.index:03d}';d.mkdir(parents=True,exist_ok=False)
    episodes=[];decisions=[];hashes={}
    for ti,tier in enumerate(['id','wind_extrap','joint_extrap']):
        tasks=build_tasks(tier,12,selection_seed=55112007+ti,waypoint_base=56112007)
        assert len(tasks)==12
        for t in tasks:t['tier']=tier
        out,arrays=rollout(tasks,regime,method,'cuda',model_seed,subset,57112007+1000*ti)
        np.savez_compressed(d/f'{tier}.npz',**arrays);episodes.extend(out['episodes'])
        for item in out['decisions']:item['tier']=tier
        decisions.extend(out['decisions']);hashes.update(out['predictor_provenance'])
    result=dict(stage='development',regime=regime,method=method,model_seed=model_seed,subset=subset,index=a.index,episodes=episodes,decisions=decisions,predictor_provenance=hashes)
    (d/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('MPC MATRIX COMPLETE',a.index,regime,method,flush=True)
if __name__=='__main__':main()
