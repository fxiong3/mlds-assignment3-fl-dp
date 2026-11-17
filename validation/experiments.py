"""Validation EXPERIMENTS on the real FEMNIST data (quantitative evidence for the report).

python validation/experiments.py --data data --rounds 50

Writes results/validation/*.json|png:
  1. first10_trace.json      - round-by-round trace of the first 10 rounds (selected clients, n_k, weights,
                               local loss before/after, global update norm, global train/val metrics).
  2. iid_vs_noniid.png       - same data re-partitioned IID across users vs the natural (non-IID) split.
  3. label_shuffle.json      - model trained on shuffled labels must be at chance on the test set.
  4. drift_vs_E.png          - client drift grows with E; accuracy vs E.
  5. lr_sanity.json          - lr=0 leaves the model unchanged; too-large lr diverges.
  6. fedsgd_equivalence.json - one FedAvg round (C=1,E=1,B=inf) equals one full-batch GD step on pooled data.
  7. weighted_vs_unweighted.png - FedAvg sample-size weighting vs naive mean.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fl import plots
from fl.cli import base_parser, load_data, save_json
from fl.core import (FedConfig, LocalConfig, fedavg_aggregate, flat, get_state, local_train, make_model, run_fedavg,
                     evaluate_test)
from fl.data import NUM_CLASSES, ClientData


def iid_repartition(fd, seed):
    """Pool all training/val data and redistribute it to clients keeping each client's sample count."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([c.x_train for c in fd.clients]); Y = np.concatenate([c.y_train for c in fd.clients])
    Xv = np.concatenate([c.x_val for c in fd.clients]); Yv = np.concatenate([c.y_val for c in fd.clients])
    p, pv = rng.permutation(len(Y)), rng.permutation(len(Yv))
    out, i, j = copy.copy(fd), 0, 0
    out.clients = []
    for c in fd.clients:
        a, b = p[i:i + c.n_train], pv[j:j + c.n_val]
        out.clients.append(ClientData(c.cid, X[a], Y[a], Xv[b], Yv[b])); i += c.n_train; j += c.n_val
    return out


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    out = os.path.join(args.out, "validation")
    fd = load_data(args)
    lc = LocalConfig(epochs=args.E, batch_size=args.batch_size, lr=args.lr)
    base = lambda **kw: FedConfig(rounds=kw.pop("rounds", args.rounds), client_frac=kw.pop("C", args.C),
                                  seed=args.seed, local=kw.pop("local", lc), **kw)

    # 1. first 10 rounds trace
    h = run_fedavg(fd, base(rounds=10), log=None)["history"]
    trace = []
    for r in range(10):
        sel = h["selected"][r]
        n = [fd.clients[i].n_train for i in sel]
        trace.append({"round": r + 1, "clients": sel, "n_k": n, "weights": list(np.round(np.array(n) / sum(n), 4)),
                      "local_loss_before": np.round(h["client_loss_before"][r], 4).tolist(),
                      "local_loss_after": np.round(h["client_loss_after"][r], 4).tolist(),
                      "client_drift": np.round(h["drift"][r], 4).tolist(), "global_update_norm": h["update_norm"][r],
                      "train_loss": h["train_loss"][r], "val_acc": h["val_acc"][r]})
    save_json(trace, os.path.join(out, "first10_trace.json"))
    print("first-10 trace saved")

    # 2. IID vs non-IID
    res = {}
    for name, data in (("natural (non-IID)", fd), ("IID re-partition", iid_repartition(fd, args.seed))):
        r_ = run_fedavg(data, base(), log=None)
        res[name] = r_["history"]
        res[name]["test_acc_final"] = evaluate_test(r_["state"], fd)["test_acc"]
    plots.plot_sweep_curves(res, os.path.join(out, "iid_vs_noniid.png"), "val_acc", "IID vs non-IID partition")
    plots.plot_client_drift({k: v["drift"] for k, v in res.items()}, os.path.join(out, "iid_vs_noniid_drift.png"))
    save_json({k: {"final_val": v["val_acc"][-1], "test": v["test_acc_final"],
                   "mean_drift": float(np.mean([np.mean(d) for d in v["drift"]]))} for k, v in res.items()},
              os.path.join(out, "iid_vs_noniid.json"))

    # 3. label shuffle
    fs = copy.deepcopy(fd)
    rng = np.random.default_rng(0)
    for c in fs.clients:
        c.y_train = rng.permutation(c.y_train); c.y_val = rng.permutation(c.y_val)
    r_ = run_fedavg(fs, base(), log=None)
    save_json({"train_acc": r_["history"]["train_acc"][-1], "val_acc": r_["history"]["val_acc"][-1],
               "test_acc": evaluate_test(r_["state"], fd)["test_acc"], "chance": 1 / NUM_CLASSES,
               "majority_rate": float(np.bincount(fd.y_test).max() / len(fd.y_test))},
              os.path.join(out, "label_shuffle.json"))

    # 4. E vs drift
    drifts, accs = {}, {}
    for E in (1, 5, 20):
        hh = run_fedavg(fd, base(local=LocalConfig(epochs=E, batch_size=args.batch_size, lr=args.lr)), log=None)["history"]
        drifts[f"E={E}"], accs[f"E={E}"] = hh["drift"], hh
    plots.plot_client_drift(drifts, os.path.join(out, "drift_vs_E.png"))
    plots.plot_sweep_curves(accs, os.path.join(out, "val_acc_vs_E.png"), "val_acc", "Validation accuracy for different E")

    # 5. learning-rate sanity
    init = get_state(make_model(args.seed))
    s0 = run_fedavg(fd, base(rounds=3, local=LocalConfig(epochs=2, lr=0.0)), log=None, init_state=init)["state"]
    hb = run_fedavg(fd, base(rounds=10, local=LocalConfig(epochs=2, lr=50.0)), log=None)["history"]
    save_json({"lr0_max_weight_change": float((flat(s0) - flat(init)).abs().max()),
               "lr50_train_loss": hb["train_loss"], "lr50_val_acc": hb["val_acc"]}, os.path.join(out, "lr_sanity.json"))

    # 6. FedSGD equivalence on real data
    big = LocalConfig(epochs=1, batch_size=10**9, lr=0.5)
    fed = run_fedavg(fd, FedConfig(rounds=1, client_frac=1.0, local=big, track_drift=False), log=None, init_state=init)["state"]
    X = np.concatenate([c.x_train for c in fd.clients]); Y = np.concatenate([c.y_train for c in fd.clients])
    cen = local_train(init, X, Y, big, 0)[0]
    states = [local_train(init, c.x_train, c.y_train, big, 0)[0] for c in fd.clients]
    unw = fedavg_aggregate(states, [1] * len(states))
    save_json({"max_abs_diff_weighted": float((flat(fed) - flat(cen)).abs().max()),
               "max_abs_diff_unweighted": float((flat(unw) - flat(cen)).abs().max())},
              os.path.join(out, "fedsgd_equivalence.json"))

    # 7. weighted vs unweighted aggregation over training
    import fl.core as core
    orig = core.fedavg_aggregate
    hist_w = run_fedavg(fd, base(), log=None)["history"]
    core.fedavg_aggregate = lambda s, w: orig(s, [1] * len(s))
    try:
        hist_u = run_fedavg(fd, base(), log=None)["history"]
    finally:
        core.fedavg_aggregate = orig
    plots.plot_sweep_curves({"weighted by n_k (FedAvg)": hist_w, "unweighted mean": hist_u},
                            os.path.join(out, "weighted_vs_unweighted.png"), "val_acc", "Aggregation weighting")
    print("validation experiments done ->", out)


if __name__ == "__main__":
    main()
