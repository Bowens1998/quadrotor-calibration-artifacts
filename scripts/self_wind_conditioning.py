"""Representation-to-prediction conversion: self-estimated wind conditioning.

The trained explicit-wind model (M3) normally consumes the TRUE local wind
over the history and future. Here we replace that input, at inference only,
with wind estimated from PA-JEPA's frozen latent via the linear probe
(constant over the window). Conditions per (M3 seed i, M6-vec seed i):

    true    — oracle upper bound (M3 as evaluated in the main results)
    zero    — wind input zeroed (how much M3 relies on the channel)
    pajepa  — PA-JEPA probe estimate ŵ_t  (the deployable configuration)
    observer— nominal-model disturbance-observer estimate (classical alt.)

If `pajepa` recovers most of the true-vs-zero gap, the learned wind
representation functions as a virtual wind sensor for a downstream predictor.

Output: outputs/metrics/self_wind_conditioning.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from _common import ROOT, device_or_fallback

from train import build_splits
from winddyn.data.dataset import WindowDataset
from winddyn.train.trainer import load_model, make_loader
from winddyn.utils.config import load_vehicle, GRAVITY


def fit_linear_probe(model, entries, device, H=12, K=30):
    ds = WindowDataset(entries, H=H, K=K, stride=7,
                       with_depth=model.use_depth, with_patch=False)
    dl = make_loader(ds, 256, False, workers=2)
    zs, ws = [], []
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            zs.append(model.encode_context(b).cpu().numpy())
            ws.append(b["wind_hist"][:, -1].cpu().numpy())
    Z = np.concatenate(zs); Wd = np.concatenate(ws)
    Zb = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    lam = 1e-3 * np.eye(Zb.shape[1])
    return np.linalg.solve(Zb.T @ Zb + len(Z) * lam, Zb.T @ Wd)


def observer_wind_batch(batch, vp):
    """Per-window wind estimate from the nominal model, using history only.

    Reconstructs world acceleration by finite differences of the recorded
    velocity history (yaw-frame-free: we use the raw stored wind-frame
    convention of state features is not needed — we recompute from raw
    velocity is unavailable in the window; instead use body-frame features).
    For simplicity and parity with the observer baseline, we approximate with
    the last-step drag inversion using state features: v_body ~ state[:, :3],
    a from finite differences of v_body rotated is complex — instead this
    variant uses the horizontal world velocity recovered from v_body and yaw.
    """
    s = batch["state_hist"].cpu().numpy()          # (B, H, 12)
    dt = 0.05
    # world-frame horizontal velocity from body velocity + yaw
    sin_y, cos_y = s[..., 10], s[..., 11]
    vx = cos_y * s[..., 0] - sin_y * s[..., 1]
    vy = sin_y * s[..., 0] + cos_y * s[..., 1]
    v = np.stack([vx, vy], -1)                     # (B, H, 2)
    a = np.gradient(v, dt, axis=1)                 # (B, H, 2)
    rho = vp.air_density
    cda = vp.body_drag_cda[:2].mean()
    krot = vp.rotor_drag_coeff[:2].mean()
    m = vp.mass
    w = np.zeros_like(v)
    for _ in range(25):
        vr = v - w
        sp = np.linalg.norm(vr, axis=-1, keepdims=True)
        coef = (0.5 * rho * cda * sp + krot) / m
        w = 0.5 * w + 0.5 * (v + a / np.clip(coef, 1e-3, None))
    return torch.tensor(w[:, -6:].mean(1), dtype=torch.float32)  # (B, 2)


@torch.no_grad()
def eval_condition(m3, batch, wind_input):
    """wind_input (B, 2) -> pos RMSE per window with constant wind windows."""
    b2 = dict(batch)
    H = batch["wind_hist"].shape[1]
    K = batch["wind_fut"].shape[1]
    b2["wind_hist"] = wind_input.unsqueeze(1).expand(-1, H, -1).contiguous()
    b2["wind_fut"] = wind_input.unsqueeze(1).expand(-1, K, -1).contiguous()
    pred = m3.predict_targets(b2)
    tgt = batch["target"]
    return ((pred[..., 0:3] - tgt[..., 0:3]) ** 2).sum(-1)   # (B, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--splits", nargs="+",
                    default=["id_eval", "wind_ood", "wind_extrap", "joint_ood"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = device_or_fallback(args.device)
    splits = build_splits()
    vp = load_vehicle()
    out = defaultdict(lambda: defaultdict(list))

    per_episode = []   # paired per-episode records for equivalence stats
    for seed in args.seeds:
        m3 = load_model(ROOT / f"outputs/checkpoints/m3_wind_jepa_seed{seed}/best.pt", device)
        m6 = load_model(ROOT / f"outputs/checkpoints/m6_vec_teacher_seed{seed}/best.pt", device)
        Wp = fit_linear_probe(m6, splits["train"][:160], device)
        for sp in args.splits:
            ds = WindowDataset(splits[sp], H=12, K=30, stride=7,
                               with_depth=True, with_patch=False)
            dl = make_loader(ds, 256, False, workers=2)
            ep_ids = np.array([ei for ei, _ in ds.index])
            ep_acc: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
            off = 0
            acc = defaultdict(lambda: [0.0, 0])
            werr = [0.0, 0]
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                # PA-JEPA probe wind
                with torch.no_grad():
                    z = m6.encode_context(batch).cpu().numpy()
                Zb = np.concatenate([z, np.ones((len(z), 1))], 1)
                w_hat = torch.tensor(Zb @ Wp, dtype=torch.float32, device=device)
                w_true_const = batch["wind_hist"][:, -1]
                w_obs = observer_wind_batch(batch, vp).to(device)
                conds = {
                    "true": w_true_const,
                    "zero": torch.zeros_like(w_true_const),
                    "pajepa": w_hat,
                    "observer": w_obs,
                }
                B = len(w_true_const)
                ids_b = ep_ids[off:off + B]
                for name, w in conds.items():
                    se = eval_condition(m3, batch, w)
                    acc[name][0] += float(se.sum()); acc[name][1] += se.numel()
                    se_w = se.sum(-1).cpu().numpy()      # (B,) sum over K
                    for ei in np.unique(ids_b):
                        m_ = ids_b == ei
                        a = ep_acc[int(ei)][name]
                        a[0] += float(se_w[m_].sum())
                        a[1] += int(m_.sum()) * se.shape[1]
                off += B
                werr[0] += float(((w_hat - w_true_const) ** 2).sum(-1).sum())
                werr[1] += len(w_hat)
            for name, (s_, n_) in acc.items():
                out[sp][name].append(float(np.sqrt(s_ / n_)))
            out[sp]["probe_wind_rmse"].append(float(np.sqrt(werr[0] / werr[1])))
            for ei, conds_acc in ep_acc.items():
                e = splits[sp][ei]
                per_episode.append({
                    "seed": seed, "split": sp,
                    "episode": e.get("episode_id", e["path"].split("/")[-1]),
                    **{name: {"sum_sq": a[0], "n": a[1]}
                       for name, a in conds_acc.items()}})
        print(f"seed {seed} done", flush=True)

    res = {sp: {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "per_seed": v} for k, v in d.items()}
           for sp, d in out.items()}
    # recovery fraction: (zero - pajepa) / (zero - true)
    for sp, d in res.items():
        z_, t_, p_ = d["zero"]["mean"], d["true"]["mean"], d["pajepa"]["mean"]
        o_ = d["observer"]["mean"]
        d["recovery_pajepa"] = (z_ - p_) / max(z_ - t_, 1e-9)
        d["recovery_observer"] = (z_ - o_) / max(z_ - t_, 1e-9)
    dest = ROOT / "outputs/metrics/self_wind_conditioning.json"
    with open(dest, "w") as f:
        json.dump(res, f, indent=1)
    with open(ROOT / "outputs/metrics/self_wind_conditioning_episodes.json",
              "w") as f:
        json.dump(per_episode, f)
    print(json.dumps({sp: {k: (round(v["mean"], 4) if isinstance(v, dict) else round(v, 3))
                           for k, v in d.items()} for sp, d in res.items()}, indent=1))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
