"""Edge cases for data handling and the federated protocol.

Run:  pytest validation/test_edge_cases.py -v -s
"""
import copy
import json

import numpy as np
import pytest
import torch

from fl.core import FedConfig, LocalConfig, evaluate, flat, get_state, global_metrics, local_train, make_model, run_fedavg
from fl.data import NUM_CLASSES, build_federated, load_femnist, preprocess, split_client


def test_split_tiny_clients():
    rng = np.random.default_rng(0)
    x = np.zeros((1, 784), np.float32)
    xtr, ytr, xv, yv = split_client(x, np.array([3]), 0.2, rng)
    assert len(ytr) == 1 and len(yv) == 0          # 1 sample: keep it for training
    xtr, ytr, xv, yv = split_client(np.zeros((2, 784)), np.array([1, 2]), 0.2, rng)
    assert len(ytr) == 1 and len(yv) == 1          # 2 samples: never empty train
    xtr, ytr, xv, yv = split_client(np.zeros((100, 784)), np.arange(100) % 62, 0.2, rng)
    assert (len(ytr), len(yv)) == (80, 20)         # exact 80/20
    assert len(set(ytr.tolist()) | set(yv.tolist())) == len(set((np.arange(100) % 62).tolist()))


def test_split_is_disjoint_and_complete():
    rng = np.random.default_rng(0)
    x = np.arange(57 * 784, dtype=np.float32).reshape(57, 784)
    xtr, _, xv, _ = split_client(x, np.zeros(57, int), 0.2, rng)
    rows = lambda a: {r[0] for r in a}
    assert rows(xtr).isdisjoint(rows(xv)) and len(rows(xtr) | rows(xv)) == 57


def test_empty_client_skipped_and_bad_labels_rejected():
    users = {"a": {"x": np.zeros((0, 784)), "y": np.zeros(0, int)},
             "b": {"x": np.random.rand(10, 784), "y": np.arange(10)}}
    fd = build_federated(users, {})
    assert [c.cid for c in fd.clients] == ["b"] and fd.meta["skipped_empty_clients"] == ["a"]
    with pytest.raises(ValueError):
        build_federated({"c": {"x": np.random.rand(3, 784), "y": [0, 1, 62]}}, {})


def test_preprocess_scaling_shapes():
    x255 = np.full((2, 28, 28), 255, np.uint8)
    assert preprocess(x255).max() == pytest.approx(1.0) and preprocess(x255).shape == (2, 784)
    assert preprocess(np.zeros((1, 784)), invert=True).min() == 1.0
    with pytest.raises(ValueError):
        preprocess(np.zeros((1, 100)))


def test_metrics_with_clients_without_validation_data():
    rng = np.random.default_rng(0)
    users = {"one": {"x": rng.random((1, 784)), "y": [5]},
             "many": {"x": rng.random((50, 784)), "y": rng.integers(0, 62, 50)}}
    fd = build_federated(users, {})
    m = global_metrics(get_state(make_model(0)), fd)
    assert np.isfinite(m["val_acc"]) and np.isnan(m["per_client_val_acc"][[c.cid for c in fd.clients].index("one")])


def test_batch_larger_than_dataset_and_single_sample():
    rng = np.random.default_rng(0)
    s0 = get_state(make_model(0))
    s, n, st = local_train(s0, rng.random((3, 784)), np.array([1, 2, 3]), LocalConfig(epochs=2, batch_size=64), 0)
    assert n == 3 and st["steps"] == 2
    s, n, st = local_train(s0, rng.random((0, 784)), np.zeros(0, int), LocalConfig(), 0)
    assert n == 0 and torch.equal(flat(s), flat(s0))


def test_tiny_C_still_selects_one_client(fd_noniid):
    h = run_fedavg(fd_noniid, FedConfig(rounds=3, client_frac=0.001), log=None)["history"]
    assert h["m"] == 1 and all(len(s) == 1 for s in h["selected"])


def test_extreme_noniid_one_class_per_client():
    """Each client holds a single class. A purely local model can only predict its own class (acc ~1/k);
    FedAvg must combine knowledge and beat that by a wide margin."""
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 0.7, (10, 784))
    users = {}
    for k in range(10):
        users[f"c{k}"] = {"x": (centers[k] + rng.normal(0, 1, (60, 784))).astype(np.float32), "y": np.full(60, k)}
    fd = build_federated(users, {})
    yt = rng.integers(0, 10, 1000)
    fd.x_test, fd.y_test = (centers[yt] + rng.normal(0, 1, (1000, 784))).astype(np.float32), yt
    res = run_fedavg(fd, FedConfig(rounds=40, client_frac=0.5, local=LocalConfig(epochs=1, lr=0.05)), log=None)
    acc = evaluate(res["state"], fd.x_test, fd.y_test)[1] / 1000
    local_only = local_train(get_state(make_model(0)), users["c0"]["x"], users["c0"]["y"], LocalConfig(epochs=5), 0)[0]
    acc_local = evaluate(local_only, fd.x_test, fd.y_test)[1] / 1000
    print(f"one-class-per-client: FedAvg {acc:.3f} vs single-client model {acc_local:.3f}")
    assert acc > 0.6 and acc_local < 0.2


def test_test_set_has_no_influence_on_training(fd_noniid):
    """Corrupting the held-out test set (NaNs, shuffled labels) must not change the trained weights at all."""
    fd2 = copy.deepcopy(fd_noniid)
    fd2.x_test[:] = np.nan; fd2.y_test = fd2.y_test[::-1].copy()
    cfg = FedConfig(rounds=4, client_frac=0.2, seed=3)
    a = run_fedavg(fd_noniid, cfg, log=None)["state"]
    b = run_fedavg(fd2, cfg, log=None)["state"]
    assert torch.equal(flat(a), flat(b))


def test_client_data_isolation(fd_noniid):
    """Changing client j's data must not change client i's local update for the same starting weights."""
    s0 = get_state(make_model(0))
    c0 = fd_noniid.clients[0]
    a = local_train(s0, c0.x_train, c0.y_train, LocalConfig(), 1)[0]
    fd2 = copy.deepcopy(fd_noniid)
    for c in fd2.clients[1:]:
        c.x_train[:] = 0
    b = local_train(s0, fd2.clients[0].x_train, fd2.clients[0].y_train, LocalConfig(), 1)[0]
    assert torch.equal(flat(a), flat(b))


def test_leaf_json_loader(tmp_path):
    """Loader handles the LEAF json layout with sharded train/test folders and pools per-user test data."""
    rng = np.random.default_rng(0)
    for split, n in (("train", 20), ("test", 5)):
        (tmp_path / split).mkdir()
        for shard in range(2):
            users = [f"f{shard}_{i}" for i in range(3)]
            d = {"users": users, "num_samples": [n] * 3,
                 "user_data": {u: {"x": rng.random((n, 784)).tolist(), "y": rng.integers(0, 62, n).tolist()} for u in users}}
            (tmp_path / split / f"all_data_{shard}.json").write_text(json.dumps(d))
    fd = load_femnist(str(tmp_path))
    assert fd.num_clients == 6 and len(fd.y_test) == 30
    assert all(c.n_train == 16 and c.n_val == 4 for c in fd.clients)


def test_npy_dict_loader(tmp_path):
    rng = np.random.default_rng(0)
    tr = {f"u{i}": {"x": rng.random((10, 28, 28)), "y": rng.integers(0, 62, 10)} for i in range(4)}
    te = {"x": rng.random((7, 784)), "y": rng.integers(0, 62, 7)}
    np.save(tmp_path / "train_data.npy", tr, allow_pickle=True)
    np.save(tmp_path / "test_data.npy", te, allow_pickle=True)
    fd = load_femnist(str(tmp_path))
    assert fd.num_clients == 4 and len(fd.y_test) == 7 and fd.clients[0].x_train.shape[1] == 784


def test_course_npy_format(tmp_path):
    """Exact Canvas layout: object array indexed by client_id of {'images': list of 28x28, 'labels': list},
    and test_data[0] holding the pooled held-out set. Client order must follow the numeric client_id."""
    rng = np.random.default_rng(0)
    tr = np.empty(12, dtype=object)
    for i in range(12):
        tr[i] = {"images": [rng.random((28, 28)).astype(np.float32) for _ in range(5 + i)],
                 "labels": [np.int32(i % 62)] * (5 + i)}
    te = np.empty(1, dtype=object)
    te[0] = {"images": [rng.random((28, 28)).astype(np.float32) for _ in range(9)], "labels": list(range(9))}
    np.save(tmp_path / "train_data.npy", tr, allow_pickle=True)
    np.save(tmp_path / "test_data.npy", te, allow_pickle=True)
    fd = load_femnist(str(tmp_path))
    assert [c.cid for c in fd.clients] == [str(i) for i in range(12)]
    assert all(c.n_train + c.n_val == 5 + i for i, c in enumerate(fd.clients))
    assert len(fd.y_test) == 9 and fd.x_test.shape == (9, 784)
