"""Part 1 Q1.4-1.5: compare client fraction C (<=10%) x local epochs E, pick best by VALIDATION
accuracy (test set untouched), retrain with best config and report test accuracy.

python scripts/part1_sweep.py --data data --rounds 150 --Cs 0.02 0.05 0.1 --Es 1 5 10 --seeds 0 1
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from fl import plots
from fl.cli import base_parser, load_data, save_json, slim_history
from fl.core import FedConfig, LocalConfig, run_fedavg, evaluate_test


def main():
    p = base_parser(__doc__)
    p.add_argument("--Cs", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1])
    p.add_argument("--Es", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--final-rounds", type=int, default=None, help="rounds for the final model (default = --rounds)")
    args = p.parse_args()
    if max(args.Cs) > 0.1:
        raise SystemExit("Assignment requires C <= 10%")
    out = os.path.join(args.out, "part1_sweep")
    fd = load_data(args)

    rows, curves, drifts = [], {}, {}
    val_tab = np.zeros((len(args.Cs), len(args.Es)))
    tr_tab = np.zeros_like(val_tab)
    for i, C in enumerate(args.Cs):
        for j, E in enumerate(args.Es):
            vals, trs = [], []
            for s in args.seeds:
                cfg = FedConfig(rounds=args.rounds, client_frac=C, seed=s, eval_every=args.eval_every,
                                local=LocalConfig(epochs=E, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum))
                h = run_fedavg(fd, cfg, log=None)["history"]
                # final metric = mean of last 5 evaluations (less noisy than the last point)
                vals.append(float(np.mean(h["val_acc"][-5:]))); trs.append(float(np.mean(h["train_acc"][-5:])))
                if s == args.seeds[0]:
                    curves[f"C={C:g},E={E}"] = slim_history(h)
                    if C == args.Cs[-1]:
                        drifts[f"E={E}"] = h["drift"]
            val_tab[i, j], tr_tab[i, j] = np.mean(vals), np.mean(trs)
            rows.append({"C": C, "E": E, "m": h["m"], "val_acc_mean": np.mean(vals), "val_acc_std": np.std(vals),
                         "train_acc_mean": np.mean(trs), "train_acc_std": np.std(trs),
                         "gap": np.mean(trs) - np.mean(vals), "wall_time": h["wall_time"]})
            print(rows[-1])

    plots.plot_sweep_heatmap(val_tab, args.Cs, args.Es, os.path.join(out, "val_acc_heatmap.png"), "Final validation accuracy")
    plots.plot_sweep_heatmap(tr_tab, args.Cs, args.Es, os.path.join(out, "train_acc_heatmap.png"), "Final training accuracy")
    plots.plot_sweep_heatmap(tr_tab - val_tab, args.Cs, args.Es, os.path.join(out, "gap_heatmap.png"), "Train - val accuracy gap")
    plots.plot_sweep_curves(curves, os.path.join(out, "val_curves.png"), "val_acc", "Validation accuracy, all (C,E)")
    plots.plot_sweep_curves(curves, os.path.join(out, "train_curves.png"), "train_acc", "Training accuracy, all (C,E)")
    plots.plot_client_drift(drifts, os.path.join(out, "client_drift_vs_E.png"))

    i, j = np.unravel_index(np.argmax(val_tab), val_tab.shape)
    best = {"C": args.Cs[i], "E": args.Es[j]}
    cfg = FedConfig(rounds=args.final_rounds or args.rounds, client_frac=best["C"], seed=args.seeds[0],
                    eval_every=args.eval_every,
                    local=LocalConfig(epochs=best["E"], batch_size=args.batch_size, lr=args.lr, momentum=args.momentum))
    res = run_fedavg(fd, cfg, log=None)
    test = evaluate_test(res["state"], fd)  # first and only time test data is touched
    print("BEST", best, "TEST", test)
    plots.plot_curves(res["history"], os.path.join(out, "best_curves.png"), f"Best config C={best['C']}, E={best['E']}")
    save_json({"grid": rows, "best": best, "test": test, "best_history": slim_history(res["history"])},
              os.path.join(out, "sweep.json"))
    torch.save(res["state"], os.path.join(out, "best_model.pt"))


if __name__ == "__main__":
    main()
