"""Part 1 Q2: FedAvg with Ray actors, 4 clients per round.

python scripts/part1_ray.py --data data --rounds 200 --E 5 --clients-per-round 4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fl import plots
from fl.cli import base_parser, load_data, save_json, slim_history
from fl.core import FedConfig, LocalConfig, evaluate_test
from fl.ray_fed import run_fedavg_ray


def main():
    p = base_parser(__doc__)
    p.add_argument("--clients-per-round", type=int, default=4)
    p.add_argument("--num-cpus", type=int, default=None, help="CPUs given to Ray (default: all)")
    args = p.parse_args()
    out = os.path.join(args.out, "part1_ray")
    fd = load_data(args)
    cfg = FedConfig(rounds=args.rounds, clients_per_round=args.clients_per_round, seed=args.seed,
                    eval_every=args.eval_every,
                    local=LocalConfig(epochs=args.E, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum))
    res = run_fedavg_ray(fd, cfg, num_cpus=args.num_cpus)
    h = res["history"]
    test = evaluate_test(res["state"], fd)
    print("TEST:", test, "| max concurrent actors:", res["ray"]["max_alive_actors"], "| wall time", h["wall_time"])
    plots.plot_curves(h, os.path.join(out, "curves.png"), f"Ray FedAvg, {args.clients_per_round} clients/round, E={args.E}")
    plots.plot_ray_gantt(h, os.path.join(out, "actor_gantt.png"))
    plots.plot_participation_heatmap(h, fd.num_clients, os.path.join(out, "participation_heatmap.png"))
    save_json({"config": vars(args), "test": test, "ray": res["ray"], "history": slim_history(h)},
              os.path.join(out, "metrics.json"))
    torch.save(res["state"], os.path.join(out, "global_model.pt"))


if __name__ == "__main__":
    main()
