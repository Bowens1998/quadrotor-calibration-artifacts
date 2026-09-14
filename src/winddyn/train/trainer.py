"""Training loop for the world-model suite."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from winddyn.data.dataset import WindowDataset, compute_norm_stats
from winddyn.models.wm import WorldModel, compute_loss


def make_loader(ds, batch_size, shuffle, workers=2):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, drop_last=shuffle,
                      persistent_workers=workers > 0)


def horizon_pos_rmse(model: WorldModel, loader, device, max_batches=None) -> dict:
    """Per-horizon position / velocity RMSE (m) from the probe head."""
    model.eval()
    se_p, se_v, n = None, None, 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model.predict_targets(batch)
            tgt = batch["target"]
            ep = ((pred[..., 0:3] - tgt[..., 0:3]) ** 2).sum(-1)   # (B, K)
            ev = ((pred[..., 3:6] - tgt[..., 3:6]) ** 2).sum(-1)
            se_p = ep.sum(0) if se_p is None else se_p + ep.sum(0)
            se_v = ev.sum(0) if se_v is None else se_v + ev.sum(0)
            n += ep.shape[0]
    return {
        "pos_rmse": torch.sqrt(se_p / n).cpu().numpy().tolist(),
        "vel_rmse": torch.sqrt(se_v / n).cpu().numpy().tolist(),
        "n_windows": n,
    }


def train_model(
    cfg: dict,
    splits: dict[str, list[dict]],
    out_dir: str | Path,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    dcfg = cfg.get("data", {})
    H, K = int(dcfg.get("H", 12)), int(dcfg.get("K", 30))
    # data-mixture ablation (spec §19): restrict TRAINING episodes by behavior
    # mode ("tracking" / "perturbation"); evaluation splits stay untouched.
    if dcfg.get("train_modes"):
        allowed = set(dcfg["train_modes"])
        splits = dict(splits)
        splits["train"] = [e for e in splits["train"] if e.get("mode") in allowed]
        print(f"[data] train restricted to modes {sorted(allowed)}: "
              f"{len(splits['train'])} episodes")
    need_depth = bool(cfg["model"].get("use_depth", False))
    need_patch = bool(cfg["model"].get("privileged", False))
    tr = WindowDataset(splits["train"], H=H, K=K,
                       stride=int(dcfg.get("stride", 5)),
                       with_depth=need_depth, with_patch=need_patch)
    va = WindowDataset(splits["id_eval"], H=H, K=K, stride=int(dcfg.get("stride", 5)),
                       with_depth=need_depth, with_patch=need_patch)
    stats_path = out.parent / "norm_stats.npz"
    if stats_path.exists():
        z = np.load(stats_path, allow_pickle=True)
        stats = {k: {"mean": z[f"{k}_mean"], "std": z[f"{k}_std"]}
                 for k in ("state_hist", "action_hist", "target", "wind_hist")}
    else:
        stats = compute_norm_stats(tr)
        np.savez(stats_path, **{f"{k}_{m}": v[m] for k, v in stats.items()
                                for m in ("mean", "std")})

    model = WorldModel(cfg["model"], stats).to(d)
    tcfg = cfg.get("train", {})
    bs = int(tcfg.get("batch_size", 256))
    epochs = int(tcfg.get("epochs", 30))
    lr = float(tcfg.get("lr", 3e-4))
    weights = tcfg.get("loss_weights", {})
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tl = make_loader(tr, bs, True)
    vl = make_loader(va, bs, False)
    best = float("inf")
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        logs_acc: dict[str, float] = {}
        nb = 0
        for batch in tl:
            batch = {k: v.to(d, non_blocking=True) for k, v in batch.items()}
            loss, logs = compute_loss(model, batch, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            model.update_ema()
            for k, v in logs.items():
                logs_acc[k] = logs_acc.get(k, 0.0) + v
            nb += 1
        sched.step()
        val = horizon_pos_rmse(model, vl, d, max_batches=40)
        val_score = float(np.mean(val["pos_rmse"]))
        history.append({"epoch": ep, **{k: v / nb for k, v in logs_acc.items()},
                        "val_pos_rmse_mean": val_score})
        if val_score < best:
            best = val_score
            torch.save({"model": model.state_dict(), "cfg": cfg, "stats": stats,
                        "epoch": ep, "val": val}, out / "best.pt")
        print(f"[{cfg['name']}] epoch {ep}: train {logs_acc.get('total', 0)/nb:.4f} "
              f"val_pos {val_score:.3f} (best {best:.3f})", flush=True)

    result = {"name": cfg["name"], "best_val_pos_rmse_mean": best,
              "epochs": epochs, "train_windows": len(tr), "val_windows": len(va),
              "wall_s": time.time() - t0, "history": history}
    with open(out / "train_log.json", "w") as f:
        json.dump(result, f, indent=1)
    return result


def load_model(ckpt_path: str | Path, device="cpu") -> WorldModel:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WorldModel(ck["cfg"]["model"], ck["stats"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model
