"""Validation of the Ray implementation (Part 1 Q2).

Run:  pytest validation/test_ray_parallel.py -v -s
"""
import numpy as np
import pytest
import ray
import torch

from fl.core import FedConfig, LocalConfig, flat, run_fedavg
from fl.ray_fed import ClientActor, resources_per_actor, run_fedavg_ray


@pytest.fixture(scope="module")
def ray_ctx():
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    yield
    ray.shutdown()


def _cfg(**kw):
    return FedConfig(rounds=kw.pop("rounds", 4), clients_per_round=4, seed=7, local=LocalConfig(epochs=2), **kw)


def test_ray_matches_sequential_exactly(ray_ctx, fd_noniid):
    """Prediction: parallel execution changes WHEN updates run, not WHAT they compute. With identical
    per-(round,client) seeds the Ray and sequential global models must coincide."""
    seq = run_fedavg(fd_noniid, _cfg(), log=None)
    par = run_fedavg_ray(fd_noniid, _cfg(), log=None)
    d = (flat(seq["state"]) - flat(par["state"])).abs().max().item()
    print(f"max |ray - sequential| = {d:.2e}")
    assert d < 1e-6 and seq["history"]["selected"] == par["history"]["selected"]
    assert np.allclose(seq["history"]["val_acc"], par["history"]["val_acc"])


def test_actors_only_for_selected_clients_and_freed(ray_ctx, fd_noniid):
    out = run_fedavg_ray(fd_noniid, _cfg(rounds=3), log=None)
    assert out["ray"]["max_alive_actors"] == 4
    assert all(r["n_actors"] == 4 for r in out["ray"]["actor_log"])
    # every CPU reserved by an actor is returned once the actor is killed -> no actors leak across rounds
    import time
    for _ in range(50):
        if ray.available_resources().get("CPU", 0) == ray.cluster_resources()["CPU"]:
            break
        time.sleep(0.1)
    assert ray.available_resources().get("CPU", 0) == ray.cluster_resources()["CPU"], "actors must be killed after each round"


def test_resource_limits(ray_ctx):
    r = resources_per_actor(4)
    assert 0 < r["num_cpus"] <= 1 and 0 <= r["num_gpus"] <= 1
    r = resources_per_actor(100)  # more clients than CPUs -> fractional CPUs, still <= 1
    assert r["num_cpus"] <= 1


def test_actor_holds_only_its_own_data(ray_ctx, fd_noniid):
    for i in (0, 5):
        c = fd_noniid.clients[i]
        a = ClientActor.remote(i, c.x_train, c.y_train, LocalConfig())
        cid, n, s = ray.get(a.data_fingerprint.remote())
        assert (cid, n) == (i, c.n_train) and s == pytest.approx(float(c.x_train.astype(np.float64).sum()))
        ray.kill(a)


def test_client_updates_overlap_in_time(ray_ctx, fd_noniid):
    """Prediction: with 4 CPUs the 4 actors of a round run concurrently (distinct PIDs, overlapping intervals)."""
    cfg = FedConfig(rounds=2, clients_per_round=4, seed=0, local=LocalConfig(epochs=30))
    h = run_fedavg_ray(fd_noniid, cfg, log=None)["history"]
    row = h["client_timing"][-1]
    assert len({t["pid"] for t in row}) == 4
    latest_start, earliest_end = max(t["start"] for t in row), min(t["end"] for t in row)
    print(f"overlap window {earliest_end - latest_start:.3f}s")
    assert latest_start < earliest_end
