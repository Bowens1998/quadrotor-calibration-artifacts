"""Unified action-conditioned world model covering the ablation ladder (§12, §19).

One architecture, feature flags select the model:

    M1 direct     : jepa=False                      (probe head only)
    M2 state JEPA : jepa=True
    M3 wind JEPA  : jepa=True, use_wind=True        (explicit wind at inference)
    M4 depth JEPA : jepa=True, use_depth=True
    M5 full oracle: jepa=True, use_depth=True, use_wind=True
    M6 PA-JEPA    : jepa=True, use_depth=True, privileged=True
                    (wind patch only as a TRAINING-time latent target)

Structure:
    context:   per-step [state, action (, wind)] -> MLP -> GRU  (+ depth CNN)
               -> z_ctx
    predictor: GRU over future [action (, wind)] embeddings, init from z_ctx;
               per-step hidden -> z_hat_k (latent) and probe(h_k) -> physical
               targets (12) for metrics
    targets:   EMA MLP encoder over the 12-dim future state feature vector
    privileged: E_w(patch) -> z_w with an auxiliary wind decoder (keeps z_w
               informative); P_w(z_ctx) -> z_w prediction with stop-grad target
    anti-collapse: VICReg-style variance floor on z_ctx / z_hat / z_w
    (formulation follows the WindJEPA implementation, adapted; spec §28.1)
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from winddyn.data.dataset import ACTION_DIM, STATE_DIM, TARGET_DIM


def mlp(inp: int, hid: int, out: int, layers: int = 2) -> nn.Sequential:
    seq: list[nn.Module] = []
    d = inp
    for _ in range(layers - 1):
        seq += [nn.Linear(d, hid), nn.SiLU()]
        d = hid
    seq.append(nn.Linear(d, out))
    return nn.Sequential(*seq)


class DepthCNN(nn.Module):
    def __init__(self, out_dim: int = 128, in_ch: int = 2):
        super().__init__()
        ch = (32, 64, 128)
        layers: list[nn.Module] = []
        c = in_ch
        for co in ch:
            layers += [nn.Conv2d(c, co, 3, 2, 1), nn.GroupNorm(8, co), nn.SiLU()]
            c = co
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Linear(ch[-1], out_dim)

    def forward(self, x):
        return self.proj(self.net(x))


class PatchCNN(nn.Module):
    def __init__(self, out_dim: int = 64, in_ch: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.proj(self.net(x))


class Normalizer(nn.Module):
    def __init__(self, stats: dict[str, dict]):
        super().__init__()
        for k, v in stats.items():
            self.register_buffer(f"{k}_mean", torch.tensor(v["mean"]))
            self.register_buffer(f"{k}_std", torch.tensor(v["std"]))

    def norm(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return (x - getattr(self, f"{key}_mean")) / getattr(self, f"{key}_std")

    def denorm(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return x * getattr(self, f"{key}_std") + getattr(self, f"{key}_mean")


class WorldModel(nn.Module):
    def __init__(self, cfg: dict, norm_stats: dict):
        super().__init__()
        self.cfg = cfg
        self.jepa = bool(cfg.get("jepa", True))
        self.use_depth = bool(cfg.get("use_depth", False))
        self.use_wind = bool(cfg.get("use_wind", False))
        self.privileged = bool(cfg.get("privileged", False))
        # Frozen-readout protocol: the probe head trains on DETACHED predictor
        # features, so the trunk is shaped only by the JEPA objectives and the
        # probe measures what the representation encodes (standard JEPA eval).
        # M1 (jepa=False) keeps the probe attached — it is the supervised
        # baseline, not a representation-learning method.
        self.detach_probe = bool(cfg.get("detach_probe", False))
        # input-source ablation: zero the past-action channel after
        # normalization (architecture and parameter count unchanged)
        self.use_past_action = bool(cfg.get("use_past_action", True))
        latent = int(cfg.get("latent_dim", 128))
        hid = int(cfg.get("hidden_dim", 256))
        a_embed = int(cfg.get("action_embed_dim", 64))
        wind_dim = 2 if self.use_wind else 0

        self.normalizer = Normalizer(norm_stats)

        self.step_embed = mlp(STATE_DIM + ACTION_DIM + wind_dim, hid, hid)
        self.ctx_gru = nn.GRU(hid, hid, 1, batch_first=True)
        self.depth_enc = DepthCNN(128) if self.use_depth else None
        ctx_in = hid + (128 if self.use_depth else 0)
        self.ctx_proj = mlp(ctx_in, hid, latent)

        self.act_embed = mlp(ACTION_DIM + wind_dim, a_embed, a_embed)
        self.pred_init = nn.Linear(latent, hid)
        self.pred_gru = nn.GRU(a_embed, hid, 1, batch_first=True)
        self.latent_head = nn.Linear(hid, latent)
        self.probe = mlp(hid, hid, TARGET_DIM)

        if self.jepa:
            self.tgt_enc = mlp(TARGET_DIM, hid, latent)
            self.tgt_enc_ema = copy.deepcopy(self.tgt_enc)
            for p in self.tgt_enc_ema.parameters():
                p.requires_grad_(False)
            self.ema_decay = float(cfg.get("ema_decay", 0.996))

        if self.privileged:
            # wind_objective "latent_teacher" (PA-JEPA): P_w(z_ctx) matches a
            # separately-encoded teacher embedding z_w = E_w(wind), with a
            # decoder keeping z_w informative.  "direct" (M8 baseline): the
            # SAME deployed path decode(P_w(z_ctx)) regresses the raw local
            # wind vector — no teacher encoder, no latent matching — so the two
            # objectives are compared at identical deployed capacity.
            self.wind_objective = str(cfg.get("wind_objective", "latent_teacher"))
            # teacher input ablation (spec §19): local planar patch (default)
            # vs bare local wind vector
            self.wind_teacher = str(cfg.get("wind_teacher", "patch"))
            if self.wind_objective == "latent_teacher":
                self.wind_enc = (PatchCNN(64) if self.wind_teacher == "patch"
                                 else mlp(2, 64, 64))
            else:
                self.wind_enc = None
            self.wind_dec = mlp(64, 64, 2)          # z_w -> local [u, v]
            self.wind_pred = mlp(latent, hid, 64)   # P_w(z_ctx)

        self.latent = latent

    # -------------------------------------------------------------- #
    @torch.no_grad()
    def update_ema(self):
        if not self.jepa:
            return
        d = self.ema_decay
        for pt, po in zip(self.tgt_enc_ema.parameters(), self.tgt_enc.parameters()):
            pt.mul_(d).add_(po.detach(), alpha=1 - d)

    def encode_context(self, batch: dict) -> torch.Tensor:
        n = self.normalizer
        a_hist = n.norm("action_hist", batch["action_hist"])
        if not self.use_past_action:
            a_hist = torch.zeros_like(a_hist)
        parts = [n.norm("state_hist", batch["state_hist"]), a_hist]
        if self.use_wind:
            # wind_in_hist (if present) overrides the wind INPUT channel only;
            # the privileged teacher always reads the true wind_hist (M7
            # probe-in-the-loop: zero / self-estimated input, true target).
            parts.append(n.norm("wind_hist",
                                batch.get("wind_in_hist", batch["wind_hist"])))
        x = self.step_embed(torch.cat(parts, dim=-1))
        _, h = self.ctx_gru(x)
        feats = [h[-1]]
        if self.use_depth:
            feats.append(self.depth_enc(batch["depth_hist"] / 6.0))
        return self.ctx_proj(torch.cat(feats, dim=-1))

    def forward(self, batch: dict) -> dict:
        n = self.normalizer
        z_ctx = self.encode_context(batch)

        a_parts = [n.norm("action_hist", batch["action_fut"])]
        if self.use_wind:
            a_parts.append(n.norm("wind_hist",
                                  batch.get("wind_in_fut", batch["wind_fut"])))
        a = self.act_embed(torch.cat(a_parts, dim=-1))
        h0 = torch.tanh(self.pred_init(z_ctx)).unsqueeze(0)
        hs, _ = self.pred_gru(a, h0)                       # (B, K, hid)

        out = {
            "z_ctx": z_ctx,
            "z_hat": self.latent_head(hs),
            "probe": self.probe(hs.detach() if self.detach_probe else hs),
        }
        if self.jepa and "target" in batch:
            with torch.no_grad():
                out["z_tgt"] = self.tgt_enc_ema(n.norm("target", batch["target"]))
        if self.privileged and ("wind_patch" in batch or "wind_hist" in batch):
            if self.wind_objective == "latent_teacher":
                if self.wind_teacher == "patch":
                    z_w = self.wind_enc(batch["wind_patch"] / 6.0)
                else:
                    z_w = self.wind_enc(n.norm("wind_hist", batch["wind_hist"][:, -1]))
                out["z_w"] = z_w
                out["wind_rec"] = self.wind_dec(z_w)
                out["z_w_hat"] = self.wind_pred(z_ctx)
            else:  # direct regression on the deployed estimate path
                out["wind_dir_hat"] = self.wind_dec(self.wind_pred(z_ctx))
        return out

    def predict_targets(self, batch: dict) -> torch.Tensor:
        """(B, K, 12) denormalized physical predictions for evaluation."""
        return self.normalizer.denorm("target", self.forward(batch)["probe"])

    def estimate_wind(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """(B, 2) internal wind estimate w_hat = decode(P_w(z_ctx)),
        denormalized to world m/s. Privileged models only; trained end-to-end
        by the existing wind_lat + wind_rec losses."""
        assert self.privileged
        return self.normalizer.denorm("wind_hist",
                                      self.wind_dec(self.wind_pred(z_ctx)))


def variance_reg(z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    z = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(target_std - std).mean()


def compute_loss(model: WorldModel, batch: dict, w: dict) -> tuple[torch.Tensor, dict]:
    n = model.normalizer
    # M7 self-conditioning: with probability p per sample, zero the wind INPUT
    # (calm-air token) so the model learns to operate both with and without an
    # external wind channel; the privileged teacher still sees the true wind.
    p_drop = float(model.cfg.get("wind_input_dropout", 0.0))
    if model.use_wind and model.training and p_drop > 0:
        keep = (torch.rand(batch["wind_hist"].shape[0],
                           device=batch["wind_hist"].device) >= p_drop)
        k = keep.float().view(-1, 1, 1)
        batch = dict(batch)
        batch["wind_in_hist"] = batch["wind_hist"] * k
        batch["wind_in_fut"] = batch["wind_fut"] * k
    out = model(batch)
    tgt_n = n.norm("target", batch["target"])
    l_probe = F.smooth_l1_loss(out["probe"], tgt_n, beta=0.5)
    total = w.get("probe", 1.0) * l_probe
    logs = {"probe": float(l_probe)}

    if model.jepa:
        l_dyn = F.mse_loss(out["z_hat"], out["z_tgt"])
        l_var = variance_reg(out["z_hat"]) + variance_reg(out["z_ctx"])
        # target encoder needs gradient too (EMA copy provides stability):
        z_tgt_live = model.tgt_enc(tgt_n)
        l_tgt_var = variance_reg(z_tgt_live)
        l_dyn_live = F.mse_loss(out["z_hat"].detach(), z_tgt_live)
        total = total + w.get("dyn", 1.0) * (l_dyn + 0.5 * l_dyn_live) \
            + w.get("var", 1.0) * (l_var + l_tgt_var)
        logs.update(dyn=float(l_dyn), var=float(l_var),
                    z_std=float(out["z_hat"].reshape(-1, model.latent).std(0).mean()))

    if model.privileged:
        wind_t = batch["wind_hist"][:, -1]              # (B, 2) world [u, v] at t
        if model.wind_objective == "latent_teacher":
            l_wrec = F.mse_loss(out["wind_rec"], n.norm("wind_hist", wind_t))
            l_wlat = F.mse_loss(out["z_w_hat"], out["z_w"].detach())
            l_wvar = variance_reg(out["z_w"])
            total = total + w.get("wind_rec", 1.0) * l_wrec \
                + w.get("wind_lat", 1.0) * l_wlat + w.get("var", 1.0) * l_wvar
            logs.update(wind_rec=float(l_wrec), wind_lat=float(l_wlat))
        else:
            l_wdir = F.mse_loss(out["wind_dir_hat"], n.norm("wind_hist", wind_t))
            total = total + w.get("wind_dir", 1.0) * l_wdir
            logs.update(wind_dir=float(l_wdir))

    logs["total"] = float(total)
    return total, logs
