"""Model, local client update and FedAvg server logic (implemented from scratch, no FL libraries)."""
from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import IMG_PIXELS, NUM_CLASSES, FederatedDataset
from .dp import laplace_perturb

State = "OrderedDict[str, torch.Tensor]"


# ----------------------------------------------------------------------------- model
class MLP(nn.Module):
    """2-layer fully connected network: 784 -> 128 (ReLU) -> 62."""

    def __init__(self, in_dim: int = IMG_PIXELS, hidden: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def make_model(seed: int = 0) -> MLP:
    torch.manual_seed(seed)
    return MLP()


def get_state(model: nn.Module):
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())


def flat(state) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for v in state.values()])


# ----------------------------------------------------------------------------- client side
@dataclass
class LocalConfig:
    epochs: int = 1
    batch_size: int = 32
    lr: float = 0.05
    momentum: float = 0.0
    weight_decay: float = 0.0


def local_train(global_state, x: np.ndarray, y: np.ndarray, cfg: LocalConfig, seed: int):
    """Run E epochs of minibatch SGD starting from the global weights on ONE client's data.
    Returns (new_state, n_samples, stats). Deterministic given `seed`.
    Edge cases: batch_size > n -> one batch of size n; n == 0 -> state returned unchanged with weight 0."""
    model = MLP()
    model.load_state_dict(global_state)
    n = len(y)
    stats = {"loss_before": float("nan"), "loss_after": float("nan"), "steps": 0}
    if n == 0 or cfg.epochs <= 0:
        return get_state(model), n, stats
    xt, yt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.int64))
    with torch.no_grad():
        stats["loss_before"] = float(F.cross_entropy(model(xt), yt))
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(int(seed))
    bs = max(1, min(cfg.batch_size, n))
    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(xt[idx]), yt[idx]).backward()
            opt.step()
            stats["steps"] += 1
    with torch.no_grad():
        stats["loss_after"] = float(F.cross_entropy(model(xt), yt))
    return get_state(model), n, stats


@torch.no_grad()
def evaluate(state_or_model, x: np.ndarray, y: np.ndarray, batch: int = 4096):
    """Return (sum_loss, n_correct, n). Summed (not averaged) so callers can aggregate exactly."""
    if isinstance(state_or_model, nn.Module):
        model = state_or_model
    else:
        model = MLP()
        model.load_state_dict(state_or_model)
    model.eval()
    if len(y) == 0:
        return 0.0, 0, 0
    loss, correct = 0.0, 0
    for i in range(0, len(y), batch):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch], dtype=np.float32))
        yb = torch.from_numpy(np.asarray(y[i:i + batch], dtype=np.int64))
        out = model(xb)
        loss += float(F.cross_entropy(out, yb, reduction="sum"))
        correct += int((out.argmax(1) == yb).sum())
    return loss, correct, len(y)


# ----------------------------------------------------------------------------- server side
def fedavg_aggregate(states: list, weights: list[float]):
    """w_{t+1} = sum_k (n_k / n) w_k. Clients with weight 0 are ignored; all-zero raises."""
    w = np.asarray(weights, dtype=np.float64)
    if len(states) == 0 or w.sum() <= 0:
        raise ValueError("Cannot aggregate: no clients with positive sample count")
    w = w / w.sum()
    out = OrderedDict()
    for k in states[0]:
        acc = torch.zeros_like(states[0][k], dtype=torch.float64)
        for s, wk in zip(states, w):
            if wk > 0:
                acc += wk * s[k].double()
        out[k] = acc.to(states[0][k].dtype)
    return out


def num_selected(num_clients: int, frac: float) -> int:
    """m = max(1, round(C*K)); clipped to K. Guards C*K < 1 and C > 1."""
    if not 0 < frac:
        raise ValueError("Client fraction C must be > 0")
    return int(min(num_clients, max(1, round(frac * num_clients))))


def sample_clients(num_clients: int, m: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(num_clients, size=m, replace=False))


@dataclass
class FedConfig:
    rounds: int = 100
    client_frac: float = 0.1
    clients_per_round: int | None = None  # overrides client_frac when set (Part 1 Q2 uses 4)
    local: LocalConfig = field(default_factory=LocalConfig)
    seed: int = 0
    eval_every: int = 1
    noise_b: float = 0.0          # Laplace scale for Part 2 (0 = no DP)
    noise_clip: bool = False      # optionally clip perturbed pixels back to [0,1]
    eval_val_on_noisy: bool = False
    track_drift: bool = True


def round_seed(seed: int, rnd: int, cid: int) -> int:
    return (seed * 1_000_003 + rnd * 10_007 + cid) % (2 ** 31 - 1)


def perturb_clients(fd: FederatedDataset, b: float, seed: int, clip: bool = False) -> list[np.ndarray]:
    """Each client perturbs its OWN training images once (local DP) with its own RNG stream.
    The server never sees clean or noisy pixels; only model weights leave the client."""
    return [laplace_perturb(c.x_train, b, np.random.default_rng([seed, 7919, i]), clip=clip)
            for i, c in enumerate(fd.clients)]


def global_metrics(state, fd: FederatedDataset, x_train_override=None, x_val_override=None) -> dict:
    """Aggregated metrics of the global model over ALL clients (sample-weighted == pooled)."""
    tl = tc = tn = vl = vc = vn = 0
    per_client_val = []
    for i, c in enumerate(fd.clients):
        xtr = c.x_train if x_train_override is None else x_train_override[i]
        xva = c.x_val if x_val_override is None else x_val_override[i]
        l, k, n = evaluate(state, xtr, c.y_train)
        tl, tc, tn = tl + l, tc + k, tn + n
        l, k, n = evaluate(state, xva, c.y_val)
        vl, vc, vn = vl + l, vc + k, vn + n
        per_client_val.append(k / n if n else np.nan)
    return {"train_loss": tl / max(tn, 1), "train_acc": tc / max(tn, 1),
            "val_loss": vl / max(vn, 1), "val_acc": vc / max(vn, 1),
            "per_client_val_acc": per_client_val}


def run_fedavg(fd: FederatedDataset, cfg: FedConfig, executor=None, log=print, init_state=None) -> dict:
    """Generic FedAvg loop. `executor(global_state, selected_ids, rnd, x_trains) -> [(state, n, stats)]`
    performs the client updates; default is sequential. The Ray version plugs in here, so both
    versions share exactly the same server logic (which makes them directly comparable)."""
    K = fd.num_clients
    m = cfg.clients_per_round or num_selected(K, cfg.client_frac)
    m = min(m, K)
    rng = np.random.default_rng(cfg.seed)
    state = init_state if init_state is not None else get_state(make_model(cfg.seed))
    x_trains = perturb_clients(fd, cfg.noise_b, cfg.seed, cfg.noise_clip) if cfg.noise_b > 0 else [c.x_train for c in fd.clients]
    x_vals = None
    if cfg.noise_b > 0 and cfg.eval_val_on_noisy:
        x_vals = [laplace_perturb(c.x_val, cfg.noise_b, np.random.default_rng([cfg.seed, 31, i]), cfg.noise_clip)
                  for i, c in enumerate(fd.clients)]

    if executor is None:
        def executor(gs, sel, rnd, xs):
            return [local_train(gs, xs[i], fd.clients[i].y_train, cfg.local, round_seed(cfg.seed, rnd, i)) for i in sel]

    hist = {"round": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "selected": [], "client_loss_before": [], "client_loss_after": [], "drift": [],
            "update_norm": [], "per_client_val_acc": [], "round_time": [], "client_timing": []}
    t_start = time.time()
    for rnd in range(1, cfg.rounds + 1):
        sel = sample_clients(K, m, rng)
        t0 = time.time()
        results = executor(state, sel, rnd, x_trains)
        new_states, ns, stats = zip(*[(r[0], r[1], r[2]) for r in results])
        new_global = fedavg_aggregate(list(new_states), list(ns))
        hist["round_time"].append(time.time() - t0)
        if cfg.track_drift:
            g_old = flat(state)
            hist["drift"].append([float((flat(s) - g_old).norm()) for s in new_states])
            hist["update_norm"].append(float((flat(new_global) - g_old).norm()))
        state = new_global
        hist["selected"].append(sel.tolist())
        hist["client_loss_before"].append([s["loss_before"] for s in stats])
        hist["client_loss_after"].append([s["loss_after"] for s in stats])
        hist["client_timing"].append([s.get("timing") for s in stats])
        if rnd % cfg.eval_every == 0 or rnd == cfg.rounds:
            met = global_metrics(state, fd, x_trains if cfg.noise_b > 0 else None, x_vals)
            hist["round"].append(rnd)
            for k in ("train_loss", "train_acc", "val_loss", "val_acc", "per_client_val_acc"):
                hist[k].append(met[k])
            if log:
                log(f"[round {rnd:4d}] train loss {met['train_loss']:.4f} acc {met['train_acc']:.4f} | "
                    f"val loss {met['val_loss']:.4f} acc {met['val_acc']:.4f} | clients {sel.tolist()}")
    hist["wall_time"] = time.time() - t_start
    hist["m"] = m
    return {"state": state, "history": hist}


def evaluate_test(state, fd: FederatedDataset) -> dict:
    l, k, n = evaluate(state, fd.x_test, fd.y_test)
    return {"test_loss": l / max(n, 1), "test_acc": k / max(n, 1), "n_test": n}


def centralized_train(x, y, cfg: LocalConfig, seed: int, init_state):
    """Reference (non-federated) training; used by validation tests."""
    return local_train(init_state, x, y, cfg, seed)[0]


def clone_state(s):
    return copy.deepcopy(s)
