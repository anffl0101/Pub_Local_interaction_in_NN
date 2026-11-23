"""Batched SGLD and AdamW training of M networks at once (weights stacked as (M, out, in), gradients via vmap).

SGLD noise depends on (seed, chunk, step): each chunk index gives its members different noise, and
replay is bit-exact only on the same GPU model. TF32 is turned off because its rounding acts as extra noise.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from src.model import MLP, ModelConfig, orthogonal_init_
from src.train import AdamWConfig, SGLDConfig
from src.utils import disable_tf32, seed_everything


def data_to_device(data: dict, device) -> dict:
    """Copy of a make_modular_addition dict with all tensors on ``device``."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}


def stacked_init(
    model_cfg: ModelConfig, data: dict, init_seeds: list[int], device,
    ortho_init: bool = False,
) -> dict[str, torch.Tensor]:
    """Initial stacked parameters; member m is built on the CPU with seed init_seeds[m]."""
    states = []
    for s in init_seeds:
        seed_everything(int(s))
        model = MLP.for_dataset(model_cfg, data)
        if ortho_init:
            orthogonal_init_(model)
        states.append(model.state_dict())
    return {k: torch.stack([st[k] for st in states]).to(device) for k in states[0]}


def make_template(model_cfg: ModelConfig, data: dict, device) -> MLP:
    """Model that gives ``functional_call`` the layer structure; its own weights are never used."""
    return MLP.for_dataset(model_cfg, data).to(device)


def _make_grad_fn(template: MLP, loss_float64: bool = False):
    """Per-member gradient of the cross-entropy loss over the stacked params.

    ``loss_float64``: loss from float64 scores (AdamW runs).
    """

    def member_loss(params, x, y):
        logits = functional_call(template, params, (x,))
        return F.cross_entropy(logits.double() if loss_float64 else logits, y)

    return vmap(grad(member_loss), in_dims=(0, None, None))


def canonical_keys(params: dict[str, torch.Tensor]) -> list[str]:
    """Sorted parameter names, so the noise draw order does not depend on dict order."""
    return sorted(params)


def _step_noise(
    params: dict[str, torch.Tensor], base_seed: int, step: int, chunk: int = 0
) -> dict[str, torch.Tensor]:
    """Standard-normal noise for every stacked tensor at one step, seeded by (base_seed, chunk, step)."""
    raw = np.random.SeedSequence([base_seed, chunk, step]).generate_state(1)[0]
    dev = next(iter(params.values())).device
    gen = torch.Generator(device=dev).manual_seed(int(raw))
    return {k: torch.empty_like(params[k]).normal_(generator=gen)
            for k in canonical_keys(params)}


@torch.no_grad()
def batched_observables(params: dict[str, torch.Tensor], n_layers: int, float64: bool = False) -> dict:
    """Per-member stable rank ||W||_F^2 / sigma_max^2 and weight norm ||W||_F (full SVD), keys stable_rank_L<i> and weight_norm_L<i>, values (M,).

    ``float64``: from a float64 copy, since float32 sigma_max can be NaN for AdamW weights near 1e-40."""
    out = {}
    for i in range(n_layers):
        W = params[f"linears.{i}.weight"]
        if float64:
            W = W.double()
        fro_sq = (W**2).sum(dim=(-2, -1))
        smax = torch.linalg.matrix_norm(W, ord=2)  # batched largest singular value
        out[f"stable_rank_L{i + 1}"] = (fro_sq / smax**2).cpu().numpy()
        out[f"weight_norm_L{i + 1}"] = torch.sqrt(fro_sq).cpu().numpy()
    return out


# Fast readout: sigma_max by subspace iteration on W^T W (k=24 spans the near-equal top singular values).
SIGMA_FALLBACKS = [0]   # count of exact-SVD fallbacks


def reset_sigma_fallbacks() -> None:
    SIGMA_FALLBACKS[0] = 0


def _subspace_sigma_max(W: torch.Tensor, k: int = 24, iters: int = 20, seed: int = 0):
    """Return (sigma_max, error bound) for each matrix in a batch W (..., n, m).

    The fixed seed gives both copies of a pair the same start, so the estimator bias cancels."""
    g = torch.Generator(device=W.device).manual_seed(seed)
    Q = torch.randn(*W.shape[:-2], W.shape[-1], k,
                    generator=g, device=W.device, dtype=W.dtype)
    Q, _ = torch.linalg.qr(Q)
    for _ in range(iters):
        Q, _ = torch.linalg.qr(W.transpose(-1, -2) @ (W @ Q))   # QR of A Q
    AQ = W.transpose(-1, -2) @ (W @ Q)
    B = Q.transpose(-1, -2) @ AQ                                 # Q^T A Q (k x k)
    evals, evecs = torch.linalg.eigh(B)
    theta = evals[..., -1]                                       # top Ritz value ~ sigma^2
    c = evecs[..., -1]
    y = Q @ c[..., None]
    resid = (AQ @ c[..., None] - theta[..., None, None] * y).norm(dim=-2).squeeze(-1)
    sigma = theta.clamp(min=0).sqrt()
    return sigma, (theta + resid).clamp(min=0).sqrt() - sigma


def fast_observables(params: dict[str, torch.Tensor], n_layers: int,
                     tol: float = 5e-3, k: int = 24, iters: int = 20, float64: bool = False) -> dict:
    """As batched_observables, but sigma_max by subspace iteration (exact SVD where the relative error bound exceeds ``tol``)."""
    out = {}
    for i in range(n_layers):
        W = params[f"linears.{i}.weight"]
        if float64:
            W = W.double()
        fro_sq = (W**2).sum(dim=(-2, -1))
        sigma, err = _subspace_sigma_max(W, k=k, iters=iters)
        bad = (err / sigma.clamp(min=1e-12)) > tol
        if bool(bad.any()):
            SIGMA_FALLBACKS[0] += int(bad.sum())
            sigma = sigma.clone()
            sigma[bad] = torch.linalg.matrix_norm(W[bad], ord=2)
        out[f"stable_rank_L{i + 1}"] = (fro_sq / sigma**2).cpu().numpy()
        out[f"weight_norm_L{i + 1}"] = torch.sqrt(fro_sq).cpu().numpy()
    return out


def fast_stable_ranks(params: dict[str, torch.Tensor], n_layers: int, **kw) -> np.ndarray:
    """(M, n_layers) stable ranks via the fast readout."""
    obs = fast_observables(params, n_layers, **kw)
    return np.stack([obs[f"stable_rank_L{i + 1}"] for i in range(n_layers)], axis=1)


@torch.no_grad()
def batched_loss_acc(
    template: MLP,
    params: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_rows: int = 4096,
):
    """Per-member mean loss and accuracy, shapes (M,), (M,), evaluated in row chunks to save memory."""
    M = next(iter(params.values())).shape[0]
    fwd = vmap(
        lambda prm, xb: functional_call(template, prm, (xb,)), in_dims=(0, None)
    )
    n = x.shape[0]
    loss_sum = torch.zeros(M, device=x.device)
    correct = torch.zeros(M, device=x.device)
    for i in range(0, n, chunk_rows):
        xb, yb = x[i : i + chunk_rows], y[i : i + chunk_rows]
        logits = fwd(params, xb)  # (M, chunk, classes)
        logp = F.log_softmax(logits, dim=-1)
        idx = yb.view(1, -1, 1).expand(M, yb.shape[0], 1)
        loss_sum += -logp.gather(-1, idx).squeeze(-1).sum(dim=1)
        correct += (logits.argmax(-1) == yb.view(1, -1)).float().sum(dim=1)
    return loss_sum / n, correct / n


def _stable_rank_grad(W_stacked: torch.Tensor) -> torch.Tensor:
    """d(stable rank)/dW for every member of a (M, out, in) stack (exact SVD)."""
    W = W_stacked.detach().requires_grad_(True)
    sr = (W**2).sum(dim=(-2, -1)) / torch.linalg.matrix_norm(W, ord=2) ** 2
    (g,) = torch.autograd.grad(sr.sum(), W)
    return g


def _sgld_ensemble_step(
    params: dict[str, torch.Tensor],
    grad_fn,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: SGLDConfig,
    step: int,
    field: float = 0.0,
    field_layer: int = 1,
    chunk: int = 0,
) -> None:
    """One SGLD step for all members in place: -eta*(grad CE + wd*theta - field*d(stable rank)/dW) + N(0, 2*eta*T)."""
    grads = grad_fn(params, x, y)
    field_key = f"linears.{field_layer}.weight"
    fg = _stable_rank_grad(params[field_key]) if field != 0.0 else None
    noise = _step_noise(params, cfg.seed, step, chunk)
    sqrt_noise = (2.0 * cfg.eta * cfg.T) ** 0.5
    with torch.no_grad():
        for k, p in params.items():
            g = grads[k] + cfg.weight_decay * p
            if fg is not None and k == field_key:
                g = g - field * fg
            p.add_(g, alpha=-cfg.eta)
            p.add_(noise[k], alpha=sqrt_noise)


def sgld_ensemble_trajectory(
    model_cfg: ModelConfig,
    data: dict,
    cfg: SGLDConfig,
    n_steps: int,
    log_every: int,
    init_seeds: list[int],
    checkpoint_steps: tuple[int, ...] = (),
    device: str | torch.device = "cpu",
    chunk: int = 0,
    ortho_init: bool = False,
) -> dict:
    """Train M members with SGLD; return log (M, rows), checkpoints {step: state before that step}, final_state."""
    disable_tf32()
    device = torch.device(device)
    data = data_to_device(data, device)
    xt, yt = data["x_train"], data["y_train"]
    xv, yv = data["x_test"], data["y_test"]
    params = stacked_init(model_cfg, data, init_seeds, device, ortho_init=ortho_init)
    template = make_template(model_cfg, data, device)
    grad_fn = _make_grad_fn(template)

    cols: dict[str, list] = {}
    steps_logged: list[int] = []
    checkpoints: dict[int, dict] = {}
    ckpt_set = set(checkpoint_steps)
    for t in range(n_steps):
        if t in ckpt_set:
            checkpoints[t] = {k: p.detach().cpu().clone() for k, p in params.items()}
        if t % log_every == 0:
            tr_loss, tr_acc = batched_loss_acc(template, params, xt, yt)
            te_loss, te_acc = batched_loss_acc(template, params, xv, yv)
            row = {
                "train_loss": tr_loss.cpu().numpy(),
                "train_acc": tr_acc.cpu().numpy(),
                "test_loss": te_loss.cpu().numpy(),
                "test_acc": te_acc.cpu().numpy(),
                **fast_observables(params, model_cfg.n_layers),
            }
            steps_logged.append(t)
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
        _sgld_ensemble_step(params, grad_fn, xt, yt, cfg, t, chunk=chunk)

    log = {k: np.stack(v, axis=1) for k, v in cols.items()}  # (M, rows)
    log["step"] = np.array(steps_logged)
    final_state = {k: p.detach().cpu().clone() for k, p in params.items()}
    return {"log": log, "checkpoints": checkpoints,
            "final_state": final_state, "steps_run": n_steps,
            "chunk": chunk, "n_members": len(init_seeds)}


def sgld_ensemble_kick_trajectory(
    model_cfg: ModelConfig,
    data: dict,
    cfg: SGLDConfig,
    n_steps: int,
    log_every: int,
    init_seeds: list[int],
    kick_steps: list[int],
    kick_len: int,
    window: int,
    kick_fields: tuple[float, ...],
    kick_taus: tuple[int, ...],
    field_layer: int,
    device: str | torch.device = "cpu",
    chunk: int = 0,
    ortho_init: bool = False,
    on_progress=None,   # callable(partial result dict), after every kick time
    dense_every: int = 0,
) -> dict:
    """SGLD trajectory; at each kick step a copy is pushed with strength h on ``field_layer`` for ``kick_len`` steps, with the same noise.

    Returns 'log' and 'kick' (sr_*: stable ranks; la_*: train/test loss and accuracy; *_dense: every ``dense_every`` steps).
    """
    kicks = sorted(int(k) for k in kick_steps)
    if kick_len > window:
        raise ValueError("kick_len must not exceed window")
    if any(b - a < window for a, b in zip(kicks, kicks[1:])):
        raise ValueError("kick times must be at least one window apart (pushed copies must not overlap)")
    if kicks and kicks[-1] + window > n_steps:
        raise ValueError("last kick + window exceeds n_steps")
    if dense_every and (window % dense_every or dense_every % log_every):
        raise ValueError("dense_every must divide window and be a multiple of log_every")
    disable_tf32()
    device = torch.device(device)
    data = data_to_device(data, device)
    xt, yt = data["x_train"], data["y_train"]
    xv, yv = data["x_test"], data["y_test"]
    params = stacked_init(model_cfg, data, init_seeds, device, ortho_init=ortho_init)
    template = make_template(model_cfg, data, device)
    grad_fn = _make_grad_fn(template)
    n_layers = model_cfg.n_layers
    M = len(init_seeds)
    taus = tuple(sorted(set(int(t) for t in kick_taus if 1 <= t <= window)))
    tau_idx = {t: j for j, t in enumerate(taus)}
    F = len(kick_fields)

    cols: dict[str, list] = {}
    steps_logged: list[int] = []

    def maybe_log(t: int) -> None:
        if t % log_every == 0:
            tr_loss, tr_acc = batched_loss_acc(template, params, xt, yt)
            te_loss, te_acc = batched_loss_acc(template, params, xv, yv)
            row = {"train_loss": tr_loss.cpu().numpy(),
                   "train_acc": tr_acc.cpu().numpy(),
                   "test_loss": te_loss.cpu().numpy(),
                   "test_acc": te_acc.cpu().numpy(),
                   **fast_observables(params, n_layers)}
            steps_logged.append(t)
            for k, v in row.items():
                cols.setdefault(k, []).append(v)

    def readout(prm: dict):
        tr_loss, tr_acc = batched_loss_acc(template, prm, xt, yt)
        te_loss, te_acc = batched_loss_acc(template, prm, xv, yv)
        la = np.stack([tr_loss.cpu().numpy(), tr_acc.cpu().numpy(),
                       te_loss.cpu().numpy(), te_acc.cpu().numpy()]).astype(np.float32)
        return fast_stable_ranks(prm, n_layers).astype(np.float32), la

    n_dense = window // dense_every if dense_every else 0

    def run_branch(snap: dict, t1: int, h: float):
        prm = {k: v.detach().clone() for k, v in snap.items()}
        sr = np.zeros((len(taus), M, n_layers), dtype=np.float32)
        la = np.zeros((len(taus), 4, M), dtype=np.float32)
        sr_d = np.zeros((n_dense, M, n_layers), dtype=np.float32)
        la_d = np.zeros((n_dense, 4, M), dtype=np.float32)
        for k in range(window):
            _sgld_ensemble_step(prm, grad_fn, xt, yt, cfg, t1 + k,
                                field=(h if k < kick_len else 0.0),
                                field_layer=field_layer, chunk=chunk)
            if (k + 1) in tau_idx:
                sr[tau_idx[k + 1]], la[tau_idx[k + 1]] = readout(prm)
            if dense_every and (k + 1) % dense_every == 0:
                sr_d[(k + 1) // dense_every - 1], la_d[(k + 1) // dense_every - 1] = readout(prm)
        return sr, la, sr_d, la_d

    def main_log_at(step: int):
        """The main path's own log entry at `step`, in readout() layout."""
        i = steps_logged.index(step)
        sr = np.stack([cols[f"stable_rank_L{j + 1}"][i] for j in range(n_layers)], axis=1)
        la = np.stack([cols[n][i] for n in ("train_loss", "train_acc", "test_loss", "test_acc")])
        return sr.astype(np.float32), la.astype(np.float32)

    done_steps: list[int] = []
    sr_main, sr_kick, la_main, la_kick = [], [], [], []
    sr_kick_d, la_kick_d = [], []
    gate_done = False
    t0 = time.perf_counter()
    total_work = n_steps + len(kicks) * F * window + window     # steps including the gate copy
    t = 0
    qi = 0

    def result() -> dict:
        log = {k: np.stack(v, axis=1) for k, v in cols.items()}
        log["step"] = np.array(steps_logged)
        return {"log": log,
                "kick": {"steps": np.array(done_steps), "taus": np.array(taus),
                         "fields": np.array(kick_fields),
                         "sr_main": np.stack(sr_main) if sr_main else np.zeros((0, len(taus), M, n_layers), np.float32),
                         "sr_kick": np.stack(sr_kick) if sr_kick else np.zeros((0, F, len(taus), M, n_layers), np.float32),
                         "la_main": np.stack(la_main) if la_main else np.zeros((0, len(taus), 4, M), np.float32),
                         "la_kick": np.stack(la_kick) if la_kick else np.zeros((0, F, len(taus), 4, M), np.float32),
                         **({"dense_every": dense_every,
                             "sr_kick_dense": np.stack(sr_kick_d) if sr_kick_d else np.zeros((0, F, n_dense, M, n_layers), np.float32),
                             "la_kick_dense": np.stack(la_kick_d) if la_kick_d else np.zeros((0, F, n_dense, 4, M), np.float32)}
                            if dense_every else {})},
                "chunk": chunk, "n_members": M, "steps_run": t}

    while t < n_steps:
        if qi < len(kicks) and t == kicks[qi]:
            t1 = t
            snap = {k: p.detach().clone() for k, p in params.items()}
            srm = np.zeros((len(taus), M, n_layers), dtype=np.float32)
            lam = np.zeros((len(taus), 4, M), dtype=np.float32)
            for k in range(window):
                maybe_log(t1 + k)
                _sgld_ensemble_step(params, grad_fn, xt, yt, cfg, t1 + k, chunk=chunk)
                if (k + 1) in tau_idx:
                    srm[tau_idx[k + 1]], lam[tau_idx[k + 1]] = readout(params)
            if not gate_done:  # a copy with push strength 0 must reproduce the untouched run bit for bit
                srg, lag, srg_d, lag_d = run_branch(snap, t1, 0.0)
                ok = np.array_equal(srg, srm) and np.array_equal(lag, lam)
                # the state at t1 + window is logged later, so it is not compared here
                for j in range(n_dense - 1):
                    sr_l, la_l = main_log_at(t1 + (j + 1) * dense_every)
                    ok = ok and np.array_equal(srg_d[j], sr_l) and np.array_equal(lag_d[j], la_l)
                if not ok:
                    raise RuntimeError(
                        "same-noise check FAILED: a copy with push strength 0 did not "
                        "reproduce the untouched run bit for bit over the window; "
                        "the pushed copies do not get the same noise.")
                gate_done = True
            outs = [run_branch(snap, t1, h) for h in kick_fields]
            sr_kick.append(np.stack([o[0] for o in outs]))
            la_kick.append(np.stack([o[1] for o in outs]))
            if dense_every:
                sr_kick_d.append(np.stack([o[2] for o in outs]))
                la_kick_d.append(np.stack([o[3] for o in outs]))
            sr_main.append(srm)
            la_main.append(lam)
            done_steps.append(t1)
            t = t1 + window
            qi += 1
            el = time.perf_counter() - t0
            work = t + qi * F * window + window
            print(f"  [kick] {qi}/{len(kicks)} at step {t1} done; step {t}/{n_steps}, "
                  f"{el/3600:.2f} h elapsed, ETA {el*(total_work/work-1)/3600:.2f} h",
                  flush=True)
            if on_progress is not None:
                on_progress(result())
        else:
            maybe_log(t)
            _sgld_ensemble_step(params, grad_fn, xt, yt, cfg, t, chunk=chunk)
            t += 1

    return result()


_ADAMW_BETA1, _ADAMW_BETA2, _ADAMW_EPS = 0.9, 0.999, 1e-8  # torch defaults


def _adamw_update(
    params: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor],
    m_state: dict[str, torch.Tensor],
    v_state: dict[str, torch.Tensor],
    t: int,
    lr: float,
    weight_decay: float,
) -> None:
    """One AdamW update in place, with the same operations as torch.optim.AdamW; ``t`` is 0 at the first update."""
    bc1 = 1.0 - _ADAMW_BETA1 ** (t + 1)
    bc2_sqrt = math.sqrt(1.0 - _ADAMW_BETA2 ** (t + 1))
    for k, p in params.items():
        g = grads[k]
        p.mul_(1.0 - lr * weight_decay)
        m_state[k].lerp_(g, 1.0 - _ADAMW_BETA1)
        v_state[k].mul_(_ADAMW_BETA2).addcmul_(g, g, value=1.0 - _ADAMW_BETA2)
        denom = (v_state[k].sqrt() / bc2_sqrt).add_(_ADAMW_EPS)
        p.addcdiv_(m_state[k], denom, value=-lr / bc1)


def adamw_ensemble_trajectory(
    model_cfg: ModelConfig,
    data: dict,
    cfg: AdamWConfig,
    init_seeds: list[int],
    device: str | torch.device = "cpu",
    ortho_init: bool = False,
    snapshot_every: int | None = None,
    progress_every: int | None = None,
    start_state: dict | None = None,
    start_step: int = 0,
    loss_float64: bool = False,
    readout_float64: bool = False,
    readout_fast: bool = False,
) -> dict:
    """Train M members with AdamW; return log (M, rows), final_state and snapshots {step: {'params', 'm', 'v'}}.

    ``loss_float64``: loss from float64 scores.
    """
    disable_tf32()
    device = torch.device(device)
    data = data_to_device(data, device)
    xt, yt = data["x_train"], data["y_train"]
    xv, yv = data["x_test"], data["y_test"]
    if start_state is None:
        params = stacked_init(model_cfg, data, init_seeds, device, ortho_init=ortho_init)
        m_state = {k: torch.zeros_like(p) for k, p in params.items()}
        v_state = {k: torch.zeros_like(p) for k, p in params.items()}
    else:
        params, m_state, v_state = ({k: x.to(device).clone() for k, x in start_state[name].items()}
                                    for name in ("params", "m", "v"))
    template = make_template(model_cfg, data, device)
    grad_fn = _make_grad_fn(template, loss_float64=loss_float64)

    cols: dict[str, list] = {}
    steps_logged: list[int] = []
    snapshots: dict[int, dict] = {}
    t0 = time.time()
    for t in range(start_step, cfg.steps):
        if snapshot_every and t % snapshot_every == 0:
            snapshots[t] = {name: {k: x.detach().cpu().clone() for k, x in st.items()}
                            for name, st in (("params", params), ("m", m_state), ("v", v_state))}
        if progress_every and t % progress_every == 0:
            print(f"    step {t}/{cfg.steps}  {time.time() - t0:.0f} s", flush=True)
        if t % cfg.log_every == 0:
            tr_loss, tr_acc = batched_loss_acc(template, params, xt, yt)
            te_loss, te_acc = batched_loss_acc(template, params, xv, yv)
            row = {
                "train_loss": tr_loss.cpu().numpy(),
                "train_acc": tr_acc.cpu().numpy(),
                "test_loss": te_loss.cpu().numpy(),
                "test_acc": te_acc.cpu().numpy(),
                **(fast_observables(params, model_cfg.n_layers, float64=readout_float64) if readout_fast
                   else batched_observables(params, model_cfg.n_layers, float64=readout_float64)),
            }
            steps_logged.append(t)
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
        grads = grad_fn(params, xt, yt)
        with torch.no_grad():
            _adamw_update(params, grads, m_state, v_state, t,
                          cfg.lr, cfg.weight_decay)

    if snapshot_every and cfg.steps % snapshot_every == 0:  # the state after the last step
        snapshots[cfg.steps] = {name: {k: x.detach().cpu().clone() for k, x in st.items()}
                                for name, st in (("params", params), ("m", m_state), ("v", v_state))}
    log = {k: np.stack(v, axis=1) for k, v in cols.items()}
    log["step"] = np.array(steps_logged)
    final_state = {k: p.detach().cpu().clone() for k, p in params.items()}
    return {"log": log, "final_state": final_state, "steps_run": cfg.steps, "snapshots": snapshots}
