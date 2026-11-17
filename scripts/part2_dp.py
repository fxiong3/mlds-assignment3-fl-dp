"""Part 2: FedAvg (C=10%) on Laplace-perturbed local data.

Q1: single run at --b, curves + test accuracy.
Q2: sweep over --bs, plot final training & test accuracy vs b (mean +- std over --seeds).

python scripts/part2_dp.py --data data --rounds 150 --b 0.1 --bs 0 0.01 0.05 0.1 0.2 0.5 1 2 5 --seeds 0 1 2
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fl import plots
from fl.cli import base_parser, load_data, save_json, slim_history
from fl.core import FedConfig, LocalConfig, run_fedavg, evaluate_test
from fl.dp import epsilon_per_image, epsilon_per_pixel, snr_db


def main():
    p = base_parser(__doc__)
    p.add_argument("--b", type=float, default=0.1, help="noise scale for the single run (Q1)")
    p.add_argument("--bs", type=float, nargs="+", default=[0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--clip", action="store_true", help="clip noisy pixels to [0,1] (post-processing)")
    p.add_argument("--skip-single", action="store_true")
    args = p.parse_args()
    out = os.path.join(args.out, "part2_dp")
    fd = load_data(args)

    def cfg_for(b, seed):
        return FedConfig(rounds=args.rounds, client_frac=args.C, seed=seed, eval_every=args.eval_every,
                         noise_b=b, noise_clip=args.clip,
                         local=LocalConfig(epochs=args.E, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum))

    x_example = np.concatenate([c.x_train for c in fd.clients[:3]])
    plots.plot_noise_gallery(x_example, [b for b in args.bs if b <= 5][:10] or [0], os.path.join(out, "noise_gallery.png"))

    if not args.skip_single:
        res = run_fedavg(fd, cfg_for(args.b, args.seeds[0]))
        test = evaluate_test(res["state"], fd)
        print(f"b={args.b} TEST:", test)
        plots.plot_curves(res["history"], os.path.join(out, f"curves_b{args.b:g}.png"),
                          f"FedAvg on Laplace-perturbed data, b={args.b}, C={args.C}")
        save_json({"b": args.b, "test": test, "history": slim_history(res["history"])},
                  os.path.join(out, f"single_b{args.b:g}.json"))

    rows, curves = [], {}
    for b in args.bs:
        tr, va, te = [], [], []
        for s in args.seeds:
            res = run_fedavg(fd, cfg_for(b, s), log=None)
            h = res["history"]
            tr.append(float(np.mean(h["train_acc"][-5:])))  # training acc on the (noisy) data the model saw
            va.append(float(np.mean(h["val_acc"][-5:])))    # clean local validation data
            te.append(evaluate_test(res["state"], fd)["test_acc"])
            if s == args.seeds[0]:
                curves[f"b={b:g}"] = slim_history(h)
        rows.append({"b": b, "eps_per_pixel": epsilon_per_pixel(b), "eps_per_image": epsilon_per_image(b),
                     "snr_db": snr_db(x_example, b),
                     "train_acc_mean": np.mean(tr), "train_acc_std": np.std(tr),
                     "val_acc_mean": np.mean(va), "test_acc_mean": np.mean(te), "test_acc_std": np.std(te)})
        print(rows[-1])

    bs = [r["b"] for r in rows]
    plots.plot_accuracy_vs_noise(bs, [r["train_acc_mean"] for r in rows], [r["test_acc_mean"] for r in rows],
                                 os.path.join(out, "accuracy_vs_noise.png"),
                                 val_acc=[r["val_acc_mean"] for r in rows],
                                 std_train=[r["train_acc_std"] for r in rows], std_test=[r["test_acc_std"] for r in rows])
    plots.plot_privacy_utility(bs, [r["test_acc_mean"] for r in rows], os.path.join(out, "privacy_utility.png"))
    plots.plot_sweep_curves(curves, os.path.join(out, "val_curves_by_b.png"), "val_acc", "Validation accuracy by noise scale")
    plots.plot_sweep_curves(curves, os.path.join(out, "train_curves_by_b.png"), "train_acc", "Training (noisy) accuracy by noise scale")
    save_json({"config": vars(args), "sweep": rows}, os.path.join(out, "noise_sweep.json"))


if __name__ == "__main__":
    main()
