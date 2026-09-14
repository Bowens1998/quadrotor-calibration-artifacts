"""Specified frozen-feature heads; training never receives branch evaluation data."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap

GRID = [1e-5, 1e-4, 1e-3, 1e-2, .1, 1., 10.]
HEAD_SEEDS = [0, 1, 2]
STEPS = 1000


class Readout(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        if family == 'joint':
            self.joint = nn.Sequential(nn.Linear(338, 128), nn.SiLU(), nn.Linear(128, 90))
        elif family == 'additive':
            self.context = nn.Sequential(nn.Linear(128, 106), nn.SiLU(), nn.Linear(106, 90))
            self.candidate = nn.Sequential(nn.Linear(210, 106), nn.SiLU(), nn.Linear(106, 90))
        else:
            raise ValueError(family)

    def forward(self, x):
        if self.family == 'joint':
            return self.joint(x)
        return self.context(x[..., :128]) + self.candidate(x[..., 128:])


def initial_model(family, seed, device):
    # Float32 default Linear initialization, followed by exact float64 casting.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Readout(family).double().to(device)
    assert sum(p.numel() for p in model.parameters()) == {
        'joint': 55002, 'additive': 55300}[family]
    return model


def predict_saved(x, checkpoint, index, device='cpu'):
    model = Readout(checkpoint['family']).double().to(device)
    model.load_state_dict({k: v[index].to(device) for k, v in checkpoint['parameters'].items()})
    model.eval()
    with torch.no_grad():
        xx = torch.as_tensor((x-checkpoint['mean'])/checkpoint['std'],
                             dtype=torch.float64, device=device)
        return model(xx).cpu().numpy()


def fit_family(x, residual, fit_mask, val_mask, family, device='cpu',
               steps=STEPS, seeds=HEAD_SEEDS, grid=GRID, progress=None):
    """Each seed/grid fit is independent; vectorization only shares execution.

    `residual` is physical displacement minus nominal displacement, not a
    normalized target. No evaluation snapshot or branch label is an argument.
    Optional training settings exist for synthetic verification only; the
    experiment runner requires the protocol's constants for scientific fitting.
    """
    x = np.asarray(x, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    fit_mask, val_mask = np.asarray(fit_mask, bool), np.asarray(val_mask, bool)
    assert x.shape == (len(residual), 338) and residual.shape[1] == 90
    assert fit_mask.any() and val_mask.any() and not np.any(fit_mask & val_mask)
    assert np.isfinite(x).all() and np.isfinite(residual).all()
    mean = x[fit_mask].mean(0)
    std = np.maximum(x[fit_mask].std(0), 1e-5)
    xx = (x-mean)/std
    xf, yf = [torch.tensor(v, dtype=torch.float64, device=device) for v in
              (xx[fit_mask], residual[fit_mask])]
    xv, yv = [torch.tensor(v, dtype=torch.float64, device=device) for v in
              (xx[val_mask], residual[val_mask])]
    records = [dict(head_seed=s, lambda_=lam) for s in seeds for lam in grid]
    models = [initial_model(family, r['head_seed'], device) for r in records]
    params, buffers = stack_module_state(models)
    template = copy.deepcopy(models[0]).to('meta')
    del models
    forward = vmap(lambda p, b, z: functional_call(template, (p, b), (z,)),
                   in_dims=(0, 0, None))
    lam = torch.tensor([r['lambda_'] for r in records], dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(params.values(), lr=.003, betas=(.9, .999), eps=1e-8,
                                 foreach=False)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = forward(params, buffers, xf)
        mse = ((pred-yf)**2).mean((1, 2))
        penalty = sum(v.square().flatten(1).sum(1) for k, v in params.items()
                      if k.endswith('weight'))/90
        loss = mse + lam*penalty
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f'{family}: nonfinite objective at update {step}')
        loss.sum().backward()
        optimizer.step()
        if (step+1) % 100 == 0 or step+1 == steps:
            item = dict(update=step+1, training_objective_before_update=loss.detach().cpu().tolist())
            history.append(item)
            if progress:
                progress(family, step+1, steps)
    with torch.no_grad():
        prediction = forward(params, buffers, xv)
        # mean over XYZ squared norm, exactly C1 position RMSE.
        val_rmse = ((prediction-yv).reshape(len(records), -1, 30, 3).square()
                    .sum(-1).mean((1, 2)).sqrt()).cpu().numpy()
        fit_rmse = ((forward(params, buffers, xf)-yf).reshape(len(records), -1, 30, 3)
                    .square().sum(-1).mean((1, 2)).sqrt()).cpu().numpy()
    assert np.isfinite(val_rmse).all() and np.isfinite(fit_rmse).all()
    for i, record in enumerate(records):
        record.update(index=i, validation_rmse=float(val_rmse[i]), fit_rmse=float(fit_rmse[i]))
    selected = []
    for seed in seeds:
        indices = [i for i, record in enumerate(records) if record['head_seed'] == seed]
        best = min(indices, key=lambda i: (val_rmse[i], i))
        selected.append(dict(records[best]))
    return dict(family=family, mean=mean, std=std, parameters={k: v.detach().cpu()
                for k, v in params.items()}, candidates=records, selected=selected,
                updates=steps, fit_windows=int(fit_mask.sum()), val_windows=int(val_mask.sum()),
                history=history, parameter_count=sum(v[0].numel() for v in params.values()))


def synthetic_checks():
    rng = np.random.default_rng(20260914)
    # Additive cancellation and joint expressivity are different properties.
    q = torch.tensor(rng.normal(size=(9, 210)), dtype=torch.float64)
    z0, z1 = [torch.tensor(rng.normal(size=(1, 128)), dtype=torch.float64) for _ in range(2)]
    x0, x1 = [torch.cat([z.expand(9, -1), q], 1) for z in (z0, z1)]
    max_change = {}
    for family in ['additive', 'joint']:
        model = initial_model(family, 0, 'cpu')
        with torch.no_grad():
            y0, y1 = model(x0), model(x1)
            delta = (y0-y0[4])-(y1-y1[4])
            max_change[family] = float(delta.abs().max())
    assert max_change['additive'] < 1e-12 and max_change['joint'] > 1e-6
    x = rng.normal(size=(18, 338))
    y = rng.normal(scale=.1, size=(18, 90))
    fit, val = np.arange(18) < 12, np.arange(18) >= 12
    all_checks = {}
    for family in ['additive', 'joint']:
        ck = fit_family(x, y, fit, val, family, steps=3, seeds=[0], grid=[.01, 1.])
        # Compare batched independent optimization with a standalone optimizer.
        model = initial_model(family, 0, 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, betas=(.9, .999),
                                     eps=1e-8, foreach=False)
        xx = torch.tensor((x[fit]-ck['mean'])/ck['std'], dtype=torch.float64)
        yy = torch.tensor(y[fit], dtype=torch.float64)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(xx)-yy).square().mean()
            loss = loss + .01*sum(p.square().sum() for n, p in model.named_parameters()
                                  if n.endswith('weight'))/90
            loss.backward()
            optimizer.step()
        err = max(float((v-ck['parameters'][k][0]).abs().max())
                  for k, v in model.state_dict().items())
        assert err < 1e-12, err
        # Validation labels may select lambda but cannot affect candidate fits.
        poisoned = y.copy()
        poisoned[val] += 123.
        second = fit_family(x, poisoned, fit, val, family, steps=3, seeds=[0], grid=[.01, 1.])
        assert all(torch.equal(ck['parameters'][k], second['parameters'][k])
                   for k in ck['parameters'])
        np.testing.assert_array_equal(ck['mean'], x[fit].mean(0))
        for i in range(2):
            assert np.isfinite(predict_saved(x, ck, i)).all()
        all_checks[family] = dict(batched_vs_standalone_max_abs=err,
                                  validation_does_not_train=True,
                                  parameters=ck['parameter_count'])
    return dict(status='passed', synthetic_only=True,
                context_change_in_candidate_difference=max_change, families=all_checks)


if __name__ == '__main__':
    import json
    torch.set_num_threads(4)
    print(json.dumps(synthetic_checks(), indent=2))
