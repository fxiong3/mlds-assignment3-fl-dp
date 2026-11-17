"""Algorithmic validation of FedAvg (Part 1 Q1). Each test states the prediction it checks.

Run:  pytest validation/test_fedavg_correctness.py -v -s
"""
import numpy as np
import pytest
import torch

from fl.core import (FedConfig, LocalConfig, fedavg_aggregate, flat, get_state, local_train, make_model,
                     num_selected, round_seed, run_fedavg, sample_clients, evaluate_test)
from fl.data import NUM_CLASSES, build_federated


def _rand_state(seed):
    return get_state(make_model(seed))


# ----------------------------------------------------------------------------- aggregation
def test_aggregation_equals_manual_weighted_mean():
    """Prediction: FedAvg output == sum_k n_k w_k / sum_k n_k computed independently in numpy."""
    states, ns = [_rand_state(s) for s in range(5)], [10, 1, 300, 57, 2]
    agg = fedavg_aggregate(states, ns)
    for k in agg:
        manual = sum(n * s[k].double().numpy() for s, n in zip(states, ns)) / sum(ns)
        np.testing.assert_allclose(agg[k].numpy(), manual, rtol=1e-6, atol=1e-7)


def test_aggregation_permutation_invariant_and_fixed_point():
    states, ns = [_rand_state(s) for s in range(4)], [5, 9, 1, 30]
    a = fedavg_aggregate(states, ns)
    b = fedavg_aggregate(states[::-1], ns[::-1])
    assert torch.allclose(flat(a), flat(b), atol=1e-7)
    same = fedavg_aggregate([states[0]] * 3, [1, 50, 7])  # identical models -> unchanged regardless of weights
    assert torch.allclose(flat(same), flat(states[0]), atol=1e-7)


def test_zero_weight_clients_ignored_and_all_zero_raises():
    s0, s1 = _rand_state(0), _rand_state(1)
    assert torch.allclose(flat(fedavg_aggregate([s0, s1], [10, 0])), flat(s0))
    with pytest.raises(ValueError):
        fedavg_aggregate([s0, s1], [0, 0])
    with pytest.raises(ValueError):
        fedavg_aggregate([], [])


# ----------------------------------------------------------------------------- equivalences with centralized training
def test_fedsgd_equivalence_full_participation(fd_noniid):
    """Prediction (McMahan et al.): with C=1, E=1, B=infinity (full local batch), one FedAvg round equals one
    step of full-batch gradient descent on the POOLED data, because sum_k (n_k/n) grad F_k = grad F.
    This only holds with sample-size weighting, so we also check that UNWEIGHTED averaging breaks it."""
    fd = fd_noniid
    init = _rand_state(3)
    cfg_local = LocalConfig(epochs=1, batch_size=10**9, lr=0.5)
    cfg = FedConfig(rounds=1, client_frac=1.0, local=cfg_local, seed=0, track_drift=False)
    fed = run_fedavg(fd, cfg, log=None, init_state=init)["state"]

    X = np.concatenate([c.x_train for c in fd.clients]); Y = np.concatenate([c.y_train for c in fd.clients])
    central, _, _ = local_train(init, X, Y, cfg_local, seed=0)
    diff = (flat(fed) - flat(central)).abs().max().item()
    print(f"max |FedAvg - centralized GD| = {diff:.2e}")
    assert diff < 1e-5

    states = [local_train(init, c.x_train, c.y_train, cfg_local, 0)[0] for c in fd.clients]
    unweighted = fedavg_aggregate(states, [1] * len(states))
    assert (flat(unweighted) - flat(central)).abs().max().item() > 100 * diff  # weighting matters


def test_single_client_fedavg_equals_sequential_local_training():
    """Prediction: with one client and C=1, R rounds of FedAvg == R chained local updates (no averaging effect)."""
    rng = np.random.default_rng(0)
    users = {"only": {"x": rng.random((80, 784)).astype(np.float32), "y": rng.integers(0, NUM_CLASSES, 80)}}
    fd = build_federated(users, {}, seed=0)
    init = _rand_state(0)
    lc = LocalConfig(epochs=2, batch_size=16, lr=0.1)
    fed = run_fedavg(fd, FedConfig(rounds=4, client_frac=1.0, local=lc, seed=5), log=None, init_state=init)["state"]
    s = init
    for r in range(1, 5):
        s = local_train(s, fd.clients[0].x_train, fd.clients[0].y_train, lc, round_seed(5, r, 0))[0]
    assert torch.allclose(flat(fed), flat(s), atol=1e-6)


def test_no_learning_when_lr_zero_or_zero_epochs(fd_noniid):
    init = _rand_state(0)
    for lc in (LocalConfig(epochs=3, lr=0.0), LocalConfig(epochs=0, lr=0.1)):
        out = run_fedavg(fd_noniid, FedConfig(rounds=3, client_frac=0.2, local=lc), log=None, init_state=init)["state"]
        assert torch.allclose(flat(out), flat(init))


# ----------------------------------------------------------------------------- learning behaviour
def test_learns_well_above_chance_and_loss_decreases(fd_noniid):
    res = run_fedavg(fd_noniid, FedConfig(rounds=30, client_frac=0.2, local=LocalConfig(epochs=2, lr=0.05)), log=None)
    h = res["history"]
    acc = evaluate_test(res["state"], fd_noniid)["test_acc"]
    print(f"synthetic non-IID test acc {acc:.3f}; train loss {h['train_loss'][0]:.3f} -> {h['train_loss'][-1]:.3f}")
    assert acc > 10 / NUM_CLASSES
    assert h["train_loss"][-1] < 0.5 * h["train_loss"][0]


def test_shuffled_labels_give_chance_level_test_accuracy(fd_noniid):
    """Prediction: if labels carry no signal the model can memorise training data but must be at chance
    (~1/62 or the majority-class rate) on the test set. Guards against test leakage / evaluation bugs."""
    import copy
    fd = copy.deepcopy(fd_noniid)
    rng = np.random.default_rng(0)
    for c in fd.clients:
        c.y_train = rng.integers(0, NUM_CLASSES, len(c.y_train)); c.y_val = rng.integers(0, NUM_CLASSES, len(c.y_val))
    res = run_fedavg(fd, FedConfig(rounds=20, client_frac=0.3, local=LocalConfig(epochs=3)), log=None)
    acc = evaluate_test(res["state"], fd)["test_acc"]
    majority = np.bincount(fd.y_test, minlength=NUM_CLASSES).max() / len(fd.y_test)
    print(f"shuffled-label test acc {acc:.3f} (chance {1/NUM_CLASSES:.3f}, majority {majority:.3f})")
    assert acc < max(3 / NUM_CLASSES, 1.5 * majority)


def test_determinism_and_seed_sensitivity(fd_noniid):
    cfg = lambda s: FedConfig(rounds=3, client_frac=0.2, seed=s)
    a = run_fedavg(fd_noniid, cfg(0), log=None)
    b = run_fedavg(fd_noniid, cfg(0), log=None)
    c = run_fedavg(fd_noniid, cfg(1), log=None)
    assert torch.equal(flat(a["state"]), flat(b["state"])) and a["history"]["selected"] == b["history"]["selected"]
    assert not torch.equal(flat(a["state"]), flat(c["state"]))


def test_more_local_epochs_increase_client_drift_on_noniid(fd_noniid):
    """Prediction: on non-IID data local models move further from the global model as E grows."""
    d = {}
    for E in (1, 5, 20):
        h = run_fedavg(fd_noniid, FedConfig(rounds=5, client_frac=0.2, local=LocalConfig(epochs=E)), log=None)["history"]
        d[E] = np.mean([np.mean(x) for x in h["drift"]])
    print("mean drift by E:", d)
    assert d[1] < d[5] < d[20]


def test_iid_has_less_client_disagreement_than_noniid(fd_iid, fd_noniid):
    """Prediction: the spread of client updates (std of drift) is lower when data are IID."""
    spread = {}
    for name, fd in (("iid", fd_iid), ("noniid", fd_noniid)):
        st = run_fedavg(fd, FedConfig(rounds=5, client_frac=0.3, local=LocalConfig(epochs=5)), log=None)["state"]
        # disagreement: loss of the global model on each client's data varies more under non-IID
        from fl.core import evaluate
        losses = [evaluate(st, c.x_train, c.y_train)[0] / c.n_train for c in fd.clients]
        spread[name] = float(np.std(losses))
    print("std of per-client loss:", spread)
    assert spread["iid"] < spread["noniid"]


# ----------------------------------------------------------------------------- client sampling
def test_num_selected_edge_cases():
    assert num_selected(100, 0.1) == 10
    assert num_selected(100, 0.001) == 1      # C*K < 1 -> still one client
    assert num_selected(7, 0.1) == 1
    assert num_selected(10, 5.0) == 10        # never more than K
    with pytest.raises(ValueError):
        num_selected(100, 0.0)


def test_sampling_without_replacement_and_uniform():
    """Prediction: each client is selected with prob m/K per round; a chi-square statistic over 5000 rounds
    should be consistent with uniform (df=99, 99.9% quantile ~ 148)."""
    rng = np.random.default_rng(0)
    K, m, R = 100, 10, 5000
    counts = np.zeros(K)
    for _ in range(R):
        s = sample_clients(K, m, rng)
        assert len(set(s.tolist())) == m
        counts[s] += 1
    exp = R * m / K
    chi2 = ((counts - exp) ** 2 / exp).sum()
    print(f"chi2={chi2:.1f}")
    assert chi2 < 148
