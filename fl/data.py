"""Federated EMNIST loading, per-client splitting and dataset statistics.

The Canvas zip may come in several layouts; `load_femnist` accepts:
  * LEAF json files   (``{"users": [...], "num_samples": [...], "user_data": {u: {"x": [...], "y": [...]}}}``),
    either one train/test pair or many shard files inside ``train/`` and ``test/`` folders;
  * .npy / .npz / .pkl files holding a dict ``{user: {"x":..., "y":...}}`` or a list of such per-user dicts.
Test data can be per-user or pooled; either way it is returned as one pooled held-out set that the
training code never receives.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
from dataclasses import dataclass, field

import numpy as np

NUM_CLASSES = 62
IMG_PIXELS = 784


@dataclass
class ClientData:
    cid: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_val(self) -> int:
        return len(self.y_val)


@dataclass
class FederatedDataset:
    clients: list[ClientData]
    x_test: np.ndarray
    y_test: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def num_clients(self) -> int:
        return len(self.clients)


# ----------------------------------------------------------------------------- raw loading
X_KEYS, Y_KEYS = ("x", "images", "X", "data"), ("y", "labels", "Y", "label")


def _get(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of {keys} found in client record with keys {list(d.keys())}")


def _as_user_dict(obj) -> dict:
    """Normalise any supported container into {user_id: {"x": array, "y": array}}."""
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        obj = obj.item() if obj.shape == () else list(obj)
    if isinstance(obj, dict) and "user_data" in obj:  # LEAF json
        return {str(u): obj["user_data"][u] for u in obj["users"]}
    if isinstance(obj, dict) and set(X_KEYS) & set(obj) and set(Y_KEYS) & set(obj):  # pooled data
        return {"__pooled__": obj}
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = {}
        for i, d in enumerate(obj):
            uid = str(d.get("user", d.get("id", i))) if isinstance(d, dict) else str(i)
            out[uid] = d
        return out
    raise ValueError(f"Unsupported data container of type {type(obj)}")


def _read_file(path: str) -> dict:
    if path.endswith(".json"):
        with open(path) as f:
            return _as_user_dict(json.load(f))
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        if {"x", "y"} <= set(z.files):
            return {"__pooled__": {"x": z["x"], "y": z["y"]}}
        return _as_user_dict({k: z[k].item() if z[k].shape == () else z[k] for k in z.files})
    if path.endswith(".npy"):
        return _as_user_dict(np.load(path, allow_pickle=True))
    if path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            return _as_user_dict(pickle.load(f))
    raise ValueError(f"Unknown file type: {path}")


def _merge(dicts: list[dict]) -> dict:
    out: dict = {}
    for d in dicts:
        for u, v in d.items():
            y = np.asarray(_get(v, Y_KEYS)).astype(np.int64).ravel()
            x = np.asarray(_get(v, X_KEYS), dtype=np.float32).reshape(len(y), IMG_PIXELS)
            if u in out:
                out[u] = {"x": np.concatenate([out[u]["x"], x]), "y": np.concatenate([out[u]["y"], y])}
            else:
                out[u] = {"x": x, "y": y}
    return out


def _find_split_files(root: str, split: str) -> list[str]:
    exts = ("json", "npy", "npz", "pkl", "pickle")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(root, split, "**", f"*.{ext}"), recursive=True)
        files += glob.glob(os.path.join(root, "**", f"*{split}*.{ext}"), recursive=True)
    return sorted(set(files))


def load_raw(root: str) -> tuple[dict, dict]:
    """Return (train_users, test_users) dicts of {"x": (n,784) float32, "y": (n,) int64}."""
    train_files, test_files = _find_split_files(root, "train"), _find_split_files(root, "test")
    test_files = [f for f in test_files if f not in train_files]
    if not train_files:
        raise FileNotFoundError(
            f"No train files found under {root}. Unzip the Canvas archive into ./data "
            "(expected names containing 'train' and 'test').")
    return _merge([_read_file(f) for f in train_files]), _merge([_read_file(f) for f in test_files])


# ----------------------------------------------------------------------------- preprocessing
def preprocess(x: np.ndarray, invert: bool = False) -> np.ndarray:
    """Flatten to (n,784), cast to float32 and scale to [0,1]. LEAF stores 1.0 as background;
    `invert=True` flips to 0-background (purely cosmetic for an MLP, exposed for experiments)."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros((0, IMG_PIXELS), np.float32)
    x = x.reshape(len(x), -1)
    if x.shape[1] != IMG_PIXELS:
        raise ValueError(f"Expected 784 pixels per image, got {x.shape[1]}")
    if x.min() >= 0 and x.max() > 1.0 + 1e-6:  # 0-255 pixel input (signed synthetic features left as-is)
        x = x / 255.0
    return 1.0 - x if invert else x


def split_client(x: np.ndarray, y: np.ndarray, val_frac: float, rng: np.random.Generator):
    """80/20 per-client split. Edge cases: a client with n samples gets
    max(1, round(val_frac*n)) validation samples when n>=2, and 0 when n==1 (all to train),
    so no client ends up with an empty training set."""
    n = len(y)
    if n == 0:
        return x, y, x[:0], y[:0]
    perm = rng.permutation(n)
    n_val = 0 if n < 2 else min(n - 1, max(1, int(round(val_frac * n))))
    vi, ti = perm[:n_val], perm[n_val:]
    return x[ti], y[ti], x[vi], y[vi]


def build_federated(train_users: dict, test_users: dict, val_frac: float = 0.2, seed: int = 0,
                    invert: bool = False, max_clients: int | None = None) -> FederatedDataset:
    rng = np.random.default_rng(seed)
    clients, skipped = [], []
    for uid in sorted(train_users, key=lambda u: (0, int(u), "") if u.isdigit() else (1, 0, u)):
        if uid == "__pooled__":
            continue
        x = preprocess(_get(train_users[uid], X_KEYS), invert)
        y = np.asarray(_get(train_users[uid], Y_KEYS), dtype=np.int64)
        if len(y) == 0:
            skipped.append(uid)  # edge case: empty client is excluded (it would get weight 0 anyway)
            continue
        if y.min() < 0 or y.max() >= NUM_CLASSES:
            raise ValueError(f"Client {uid} has labels outside [0,{NUM_CLASSES})")
        clients.append(ClientData(uid, *split_client(x, y, val_frac, rng)))
        if max_clients and len(clients) >= max_clients:
            break
    if test_users:
        xt = np.concatenate([preprocess(_get(v, X_KEYS), invert) for v in test_users.values()])
        yt = np.concatenate([np.asarray(_get(v, Y_KEYS), dtype=np.int64) for v in test_users.values()])
    else:
        xt, yt = np.zeros((0, IMG_PIXELS), np.float32), np.zeros(0, np.int64)
    return FederatedDataset(clients, xt, yt, meta={"skipped_empty_clients": skipped, "seed": seed})


def load_femnist(root: str = "data", **kw) -> FederatedDataset:
    tr, te = load_raw(root)
    return build_federated(tr, te, **kw)


# ----------------------------------------------------------------------------- synthetic data (tests)
def make_synthetic(num_clients: int = 20, samples_range=(30, 120), num_classes: int = NUM_CLASSES,
                   noniid_alpha: float | None = 0.3, n_test: int = 1000, seed: int = 0,
                   class_sep: float = 2.0) -> FederatedDataset:
    """Gaussian-blob 'images' with a known optimal classifier. With `noniid_alpha` each client's label
    distribution is Dirichlet(alpha) (small alpha = highly non-IID); None means IID."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, class_sep / np.sqrt(IMG_PIXELS) * 10, size=(num_classes, IMG_PIXELS)).astype(np.float32)

    def sample(n, p):
        y = rng.choice(num_classes, size=n, p=p)
        x = centers[y] + rng.normal(0, 1.0, size=(n, IMG_PIXELS)).astype(np.float32)
        return x.astype(np.float32), y.astype(np.int64)

    users = {}
    for k in range(num_clients):
        p = np.full(num_classes, 1 / num_classes) if noniid_alpha is None else rng.dirichlet([noniid_alpha] * num_classes)
        users[f"u{k:03d}"] = dict(zip(("x", "y"), sample(int(rng.integers(*samples_range)), p)))
    xt, yt = sample(n_test, np.full(num_classes, 1 / num_classes))
    fd = build_federated(users, {}, seed=seed)
    fd.x_test, fd.y_test = xt, yt
    return fd


# ----------------------------------------------------------------------------- statistics (Part 1 Q1.1)
def class_distribution(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    return np.bincount(y, minlength=num_classes)


def label_name(c: int) -> str:
    if c < 10:
        return str(c)
    if c < 36:
        return chr(ord("A") + c - 10)
    return chr(ord("a") + c - 36)


def describe(fd: FederatedDataset, users: list[str] | None = None) -> dict:
    all_y = np.concatenate([np.concatenate([c.y_train, c.y_val]) for c in fd.clients])
    counts = class_distribution(all_y)
    sizes = np.array([c.n_train + c.n_val for c in fd.clients])
    p = counts / counts.sum()
    out = {
        "num_clients": fd.num_clients,
        "total_samples": int(sizes.sum()),
        "test_samples": int(len(fd.y_test)),
        "samples_per_client": {"min": int(sizes.min()), "median": float(np.median(sizes)),
                               "mean": float(sizes.mean()), "max": int(sizes.max())},
        "group_share": {"digits": float(p[:10].sum()), "uppercase": float(p[10:36].sum()),
                        "lowercase": float(p[36:].sum())},
        "label_entropy_bits": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
        "classes_missing_globally": [label_name(c) for c in np.where(counts == 0)[0]],
    }
    if users:
        per = {}
        for c in fd.clients:
            if c.cid in users:
                yy = np.concatenate([c.y_train, c.y_val])
                cc = class_distribution(yy)
                q = cc / cc.sum()
                m = 0.5 * (p + q)
                js = 0.5 * (np.sum(q[q > 0] * np.log2(q[q > 0] / m[q > 0])) +
                            np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0])))
                per[c.cid] = {"n": int(len(yy)), "n_classes_present": int((cc > 0).sum()),
                              "top5": [(label_name(i), int(cc[i])) for i in np.argsort(-cc)[:5]],
                              "js_divergence_vs_global": float(js)}
        out["selected_users"] = per
    return out
