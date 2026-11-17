"""Standard and non-standard plots."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import NUM_CLASSES, class_distribution, label_name
from .dp import laplace_perturb, snr_db

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- data distribution
def plot_class_distribution(fd, users: list[str], path: str):
    all_y = np.concatenate([np.concatenate([c.y_train, c.y_val]) for c in fd.clients])
    fig, axes = plt.subplots(len(users) + 1, 1, figsize=(14, 2.2 * (len(users) + 1)), sharex=True)
    ticks = np.arange(NUM_CLASSES)
    groups = ["tab:blue"] * 10 + ["tab:orange"] * 26 + ["tab:green"] * 26
    axes[0].bar(ticks, class_distribution(all_y), color=groups)
    axes[0].set_title(f"All {fd.num_clients} users (blue=digits, orange=upper, green=lower)")
    by_id = {c.cid: c for c in fd.clients}
    for ax, u in zip(axes[1:], users):
        c = by_id[u]
        yy = np.concatenate([c.y_train, c.y_val])
        ax.bar(ticks, class_distribution(yy), color=groups)
        ax.set_title(f"user {u} (n={len(yy)})", fontsize=9)
    axes[-1].set_xticks(ticks, [label_name(i) for i in ticks], fontsize=7)
    axes[-1].set_xlabel("class")
    return _save(fig, path)


def plot_client_sizes(fd, path):
    sizes = np.array([c.n_train + c.n_val for c in fd.clients])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(sizes, bins=30, color="tab:purple")
    ax.set_xlabel("samples per user"); ax.set_ylabel("# users"); ax.set_title("Local dataset size distribution")
    return _save(fig, path)


def plot_label_heatmap(fd, path):
    """Non-standard: users x classes heatmap of the per-user label distribution (shows non-IID-ness)."""
    mat = np.stack([class_distribution(np.concatenate([c.y_train, c.y_val])) for c in fd.clients]).astype(float)
    mat = mat / mat.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xticks(range(NUM_CLASSES), [label_name(i) for i in range(NUM_CLASSES)], fontsize=6)
    ax.set_xlabel("class"); ax.set_ylabel("user index"); ax.set_title("Per-user label distribution P(y | user)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    return _save(fig, path)


# ----------------------------------------------------------------------------- training curves
def plot_curves(hist, path, title=""):
    r = hist["round"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(r, hist["train_loss"], label="train"); axes[0].plot(r, hist["val_loss"], label="validation")
    axes[0].set_xlabel("communication round"); axes[0].set_ylabel("cross-entropy loss"); axes[0].legend()
    axes[1].plot(r, hist["train_acc"], label="train"); axes[1].plot(r, hist["val_acc"], label="validation")
    axes[1].set_xlabel("communication round"); axes[1].set_ylabel("accuracy"); axes[1].legend()
    fig.suptitle(title)
    return _save(fig, path)


def plot_sweep_curves(results: dict, path, metric="val_acc", title=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, hist in results.items():
        ax.plot(hist["round"], hist[metric], label=label)
    ax.set_xlabel("communication round"); ax.set_ylabel(metric.replace("_", " ")); ax.legend(fontsize=7, ncol=2)
    ax.set_title(title)
    return _save(fig, path)


def plot_sweep_heatmap(table, Cs, Es, path, title):
    """table[i][j] = metric for C=Cs[i], E=Es[j]"""
    t = np.asarray(table)
    fig, ax = plt.subplots(figsize=(1.4 * len(Es) + 2, 1 * len(Cs) + 1.5))
    im = ax.imshow(t, cmap="viridis")
    for i in range(len(Cs)):
        for j in range(len(Es)):
            ax.text(j, i, f"{t[i, j]:.3f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_xticks(range(len(Es)), [f"E={e}" for e in Es]); ax.set_yticks(range(len(Cs)), [f"C={c:g}" for c in Cs])
    ax.set_title(title); ax.grid(False)
    fig.colorbar(im, ax=ax)
    return _save(fig, path)


# ----------------------------------------------------------------------------- non-standard plots
def plot_participation_heatmap(hist, num_clients, path):
    """Non-standard: clients x rounds; cell = relative local loss reduction when selected.
    Visualises sampling fairness (uniform coverage) and which clients' updates helped most."""
    R = len(hist["selected"])
    mat = np.full((num_clients, R), np.nan)
    for r, (sel, lb, la) in enumerate(zip(hist["selected"], hist["client_loss_before"], hist["client_loss_after"])):
        for c, b, a in zip(sel, lb, la):
            mat[c, r] = (b - a) / b if b and b == b else np.nan
    counts = np.sum(~np.isnan(mat), axis=1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [5, 1]})
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm_r", vmin=-0.5, vmax=1, interpolation="nearest")
    ax.set_xlabel("communication round"); ax.set_ylabel("client"); ax.grid(False)
    ax.set_title("Client participation & relative local-loss reduction")
    fig.colorbar(im, ax=ax, fraction=0.02)
    ax2.barh(range(num_clients), counts, color="gray"); ax2.invert_yaxis()
    exp = R * hist["m"] / num_clients
    ax2.axvline(exp, color="r", ls="--", label=f"expected {exp:.1f}"); ax2.legend(fontsize=7)
    ax2.set_title("# times selected")
    return _save(fig, path)


def plot_client_drift(drift_by_label: dict, path):
    """Non-standard: per-round mean L2 distance between local models and the global model they started
    from, for different E. Larger E => more client drift on non-IID data."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, drift in drift_by_label.items():
        d = np.array([np.mean(x) for x in drift])
        s = np.array([np.std(x) for x in drift])
        r = np.arange(1, len(d) + 1)
        ax.plot(r, d, label=label); ax.fill_between(r, d - s, d + s, alpha=0.2)
    ax.set_yscale("log"); ax.set_xlabel("communication round"); ax.set_ylabel("||w_k - w_global||_2")
    ax.set_title("Client drift per round"); ax.legend()
    return _save(fig, path)


def plot_per_client_violin(hist, path, n_snapshots=6):
    """Non-standard: distribution of per-client validation accuracy over training (fairness across users)."""
    pcs = hist["per_client_val_acc"]
    idx = np.unique(np.linspace(0, len(pcs) - 1, n_snapshots).astype(int))
    data = [np.array(pcs[i])[~np.isnan(pcs[i])] for i in idx]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(idx) + 1), [f"round {hist['round'][i]}" for i in idx])
    ax.set_ylabel("per-client validation accuracy"); ax.set_title("Distribution of global-model accuracy across clients")
    return _save(fig, path)


def plot_ray_gantt(hist, path, rounds=5):
    """Non-standard: Gantt chart of actor execution (shows that client updates really overlap in time)."""
    timings = [t for t in hist["client_timing"][:rounds]]
    t0 = min(x["start"] for r in timings for x in r if x)
    fig, ax = plt.subplots(figsize=(10, 0.5 * sum(len(r) for r in timings) + 1))
    y, labels = 0, []
    colors = plt.cm.tab10.colors
    for r, row in enumerate(timings):
        for x in row:
            ax.barh(y, x["end"] - x["start"], left=x["start"] - t0, color=colors[r % 10])
            labels.append(f"r{r + 1} c{x['cid']} pid{x['pid']}"); y += 1
    ax.set_yticks(range(y), labels, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("wall-clock seconds"); ax.set_title("Ray actor execution timeline (one colour per round)")
    return _save(fig, path)


def plot_noise_gallery(x: np.ndarray, bs: list[float], path, seed=0, n_img=4):
    """Non-standard: the same images perturbed with each Laplace scale b, annotated with SNR and per-pixel eps."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(n_img, len(bs), figsize=(1.5 * len(bs), 1.6 * n_img))
    for j, b in enumerate(bs):
        xp = laplace_perturb(x[:n_img], b, rng)
        for i in range(n_img):
            ax = axes[i, j]
            ax.imshow(xp[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1); ax.axis("off")
            if i == 0:
                ax.set_title(f"b={b:g}\nSNR={snr_db(x, b):.1f}dB\nε_px={1 / b if b else np.inf:.2g}", fontsize=7)
    return _save(fig, path)


def plot_accuracy_vs_noise(bs, train_acc, test_acc, path, val_acc=None, std_train=None, std_test=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.array(bs, dtype=float)
    ax.errorbar(x, train_acc, yerr=std_train, marker="o", capsize=3, label="final training accuracy")
    ax.errorbar(x, test_acc, yerr=std_test, marker="s", capsize=3, label="test accuracy")
    if val_acc is not None:
        ax.plot(x, val_acc, marker="^", ls="--", label="validation accuracy")
    ax.axhline(1 / NUM_CLASSES, color="gray", ls=":", label="chance (1/62)")
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlabel("Laplace noise scale b"); ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs. Laplace noise scale"); ax.legend()
    return _save(fig, path)


def plot_privacy_utility(bs, test_acc, path):
    """Non-standard: privacy-utility frontier (test acc vs per-pixel epsilon=1/b on log axis)."""
    bs, acc = np.array(bs, float), np.array(test_acc, float)
    mask = bs > 0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(1 / bs[mask], acc[mask], marker="o")
    for b, e, a in zip(bs[mask], 1 / bs[mask], acc[mask]):
        ax.annotate(f"b={b:g}", (e, a), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if (~mask).any():
        ax.axhline(acc[~mask][0], color="g", ls="--", label="no noise (ε=∞)")
    ax.set_xscale("log"); ax.set_xlabel("per-pixel ε = 1/b  (smaller = more private)"); ax.set_ylabel("test accuracy")
    ax.set_title("Privacy–utility trade-off"); ax.legend()
    return _save(fig, path)
