"""Shared CLI helpers for the experiment scripts."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import load_femnist, make_synthetic


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data", default="data", help="folder with the unzipped Canvas FEMNIST data")
    p.add_argument("--synthetic", action="store_true", help="use a synthetic 100-client dataset (smoke runs)")
    p.add_argument("--out", default="results")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--C", type=float, default=0.1, help="fraction of clients per round")
    p.add_argument("--E", type=int, default=5, help="local epochs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads for the driver process")
    p.add_argument("--no-invert", dest="invert", action="store_false",
                   help="keep raw LEAF pixels (background=1). Default inverts to background=0, which conditions SGD "
                        "much better (40-round val acc 0.63 vs 0.42 at the best lr of each)")
    return p


def load_data(args):
    import torch
    torch.set_num_threads(args.threads)  # the 784-128-62 MLP is too small to benefit from intra-op threads
    if args.synthetic:
        return make_synthetic(num_clients=100, samples_range=(50, 250), seed=args.seed, n_test=3000)
    return load_femnist(args.data, seed=args.seed, invert=args.invert)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def slim_history(h):
    keep = ("round", "train_loss", "train_acc", "val_loss", "val_acc", "selected", "update_norm",
            "round_time", "wall_time", "m")
    return {k: h[k] for k in keep if k in h}
