"""Part 1 Q1.1-1.3 (+1.5 test eval): data description, FedAvg with C=10%, loss/accuracy curves.

python scripts/part1_fedavg.py --data data --rounds 200 --C 0.1 --E 5
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from fl import plots
from fl.cli import base_parser, load_data, save_json, slim_history
from fl.core import FedConfig, LocalConfig, run_fedavg, evaluate_test
from fl.data import describe


def main():
    p = base_parser(__doc__)
    p.add_argument("--users", nargs="*", default=None, help="5 user ids to describe (default: 5 spread by size)")
    args = p.parse_args()
    out = os.path.join(args.out, "part1_fedavg")
    fd = load_data(args)

    # ---- Q1.1 data distribution
    sizes = np.array([c.n_train + c.n_val for c in fd.clients])
    order = np.argsort(sizes)
    users = args.users or [fd.clients[i].cid for i in order[np.linspace(0, len(order) - 1, 5).astype(int)]]
    stats = describe(fd, users)
    save_json(stats, os.path.join(out, "data_description.json"))
    plots.plot_class_distribution(fd, users, os.path.join(out, "class_distribution.png"))
    plots.plot_client_sizes(fd, os.path.join(out, "client_sizes.png"))
    plots.plot_label_heatmap(fd, os.path.join(out, "label_heatmap_users_x_classes.png"))
    print({k: v for k, v in stats.items() if k != "selected_users"})

    # ---- Q1.2/1.3 FedAvg
    cfg = FedConfig(rounds=args.rounds, client_frac=args.C, seed=args.seed, eval_every=args.eval_every,
                    local=LocalConfig(epochs=args.E, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum))
    res = run_fedavg(fd, cfg)
    h = res["history"]
    test = evaluate_test(res["state"], fd)
    print("TEST:", test)
    plots.plot_curves(h, os.path.join(out, "curves.png"), f"FedAvg C={args.C}, E={args.E}, B={args.batch_size}, lr={args.lr}")
    plots.plot_participation_heatmap(h, fd.num_clients, os.path.join(out, "participation_heatmap.png"))
    plots.plot_per_client_violin(h, os.path.join(out, "per_client_val_acc_violin.png"))
    plots.plot_client_drift({f"E={args.E}": h["drift"]}, os.path.join(out, "client_drift.png"))
    save_json({"config": vars(args), "test": test, "history": slim_history(h)}, os.path.join(out, "metrics.json"))
    torch.save(res["state"], os.path.join(out, "global_model.pt"))


if __name__ == "__main__":
    main()
