"""Validation of the Laplace mechanism and its effect on FedAvg (Part 2).

Run:  pytest validation/test_dp.py -v -s
"""
import math

import numpy as np
import pytest
import torch

from fl.core import FedConfig, LocalConfig, flat, perturb_clients, run_fedavg, evaluate_test
from fl.data import NUM_CLASSES
from fl.dp import epsilon_per_pixel, laplace_perturb


def _laplace_cdf(x, b):
    return np.where(x < 0, 0.5 * np.exp(x / b), 1 - 0.5 * np.exp(-x / b))


@pytest.mark.parametrize("b", [0.01, 0.1, 1.0, 5.0])
def test_noise_matches_laplace_distribution(b):
    """Moments (mean 0, var 2b^2, E|e| = b, median |e| = b ln2) and a Kolmogorov-Smirnov test against Lap(0,b)."""
    x = np.zeros((500, 784), np.float32)
    e = laplace_perturb(x, b, np.random.default_rng(0)).ravel().astype(np.float64)
    n = e.size
    assert abs(e.mean()) < 5 * math.sqrt(2) * b / math.sqrt(n)
    assert e.var() == pytest.approx(2 * b * b, rel=0.02)
    assert np.abs(e).mean() == pytest.approx(b, rel=0.01)
    assert np.median(np.abs(e)) == pytest.approx(b * math.log(2), rel=0.02)
    s = np.sort(e[:20000])
    ks = np.max(np.abs(_laplace_cdf(s, b) - np.arange(1, len(s) + 1) / len(s)))
    print(f"b={b}: var={e.var():.4g} (2b^2={2*b*b:.4g}), KS={ks:.4f}")
    assert ks < 1.63 / math.sqrt(len(s))  # 99% critical value


def test_noise_is_additive_iid_and_signal_preserving():
    rng = np.random.default_rng(0)
    x = rng.random((2000, 784)).astype(np.float32)
    xp = laplace_perturb(x, 0.3, np.random.default_rng(1))
    e = (xp - x).astype(np.float64)
    assert abs(np.corrcoef(e.ravel(), x.ravel())[0, 1]) < 0.01           # noise independent of the signal
    assert abs(np.corrcoef(e[:, :-1].ravel(), e[:, 1:].ravel())[0, 1]) < 0.01  # independent across pixels


def test_b_zero_identity_negative_rejected_clip_range():
    x = np.random.default_rng(0).random((4, 784)).astype(np.float32)
    assert np.array_equal(laplace_perturb(x, 0.0, np.random.default_rng(0)), x)
    with pytest.raises(ValueError):
        laplace_perturb(x, -0.1, np.random.default_rng(0))
    c = laplace_perturb(x, 2.0, np.random.default_rng(0), clip=True)
    assert c.min() >= 0 and c.max() <= 1
    assert laplace_perturb(np.zeros((0, 784)), 1.0, np.random.default_rng(0)).shape == (0, 784)
    assert epsilon_per_pixel(0) == float("inf") and epsilon_per_pixel(0.5) == 2


def test_each_client_gets_independent_reproducible_noise(fd_noniid):
    a = perturb_clients(fd_noniid, 0.2, seed=0)
    b = perturb_clients(fd_noniid, 0.2, seed=0)
    c = perturb_clients(fd_noniid, 0.2, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])
    n0 = a[0][:5] - fd_noniid.clients[0].x_train[:5]
    n1 = a[1][:5] - fd_noniid.clients[1].x_train[:5]
    assert not np.allclose(n0, n1)
    assert all(np.array_equal(cl.x_train, fd_noniid.clients[i].x_train) for i, cl in enumerate(fd_noniid.clients))  # originals untouched


def test_zero_noise_run_identical_to_plain_fedavg(fd_noniid):
    cfg = lambda b: FedConfig(rounds=3, client_frac=0.2, seed=0, noise_b=b)
    assert torch.equal(flat(run_fedavg(fd_noniid, cfg(0.0), log=None)["state"]),
                       flat(run_fedavg(fd_noniid, FedConfig(rounds=3, client_frac=0.2, seed=0), log=None)["state"]))


def test_accuracy_degrades_monotonically_with_noise(fd_noniid):
    """Prediction: utility is a decreasing function of b and approaches chance for very large b.
    We check Spearman rank correlation between b and test accuracy is strongly negative."""
    bs = [0, 1, 3, 10, 30, 100]
    accs = []
    for b in bs:
        res = run_fedavg(fd_noniid, FedConfig(rounds=15, client_frac=0.3, seed=0, noise_b=b,
                                              local=LocalConfig(epochs=2, lr=0.02)), log=None)
        accs.append(evaluate_test(res["state"], fd_noniid)["test_acc"])
    print(dict(zip(bs, np.round(accs, 3))))
    rank = lambda v: np.argsort(np.argsort(v))
    rho = np.corrcoef(rank(bs), rank(accs))[0, 1]
    assert rho < -0.8
    assert accs[-1] < 4 / NUM_CLASSES < accs[0]
