"""Part 1 Q2: parallel client updates with Ray Core Actors.

* One Actor instance is created for each client selected in a round and destroyed (ray.kill) right
  after it returns, so at most m actors exist at any time (Actors have large memory overhead).
* Each actor gets num_cpus <= 1 and num_gpus <= 1 (the GPU is used only when one is available).
* An actor only ever receives its own client's data, so local data is never shared between clients.
"""
from __future__ import annotations

import os
import time

import numpy as np
import ray
import torch

from .core import FedConfig, LocalConfig, local_train, round_seed, run_fedavg
from .data import FederatedDataset


@ray.remote
class ClientActor:
    def __init__(self, cid: int, x: np.ndarray, y: np.ndarray, cfg: LocalConfig):
        torch.set_num_threads(1)  # 1 CPU per actor => deterministic and no oversubscription
        self.cid, self.x, self.y, self.cfg = cid, x, y, cfg
        self.pid = os.getpid()

    def train(self, global_state, seed: int):
        t0 = time.time()
        state, n, stats = local_train(global_state, self.x, self.y, self.cfg, seed)
        stats["timing"] = {"cid": self.cid, "pid": self.pid, "start": t0, "end": time.time()}
        return state, n, stats

    def data_fingerprint(self):
        """Used by tests to prove each actor holds only its own client's data."""
        return self.cid, int(len(self.y)), float(np.asarray(self.x, dtype=np.float64).sum())


def resources_per_actor(num_selected: int) -> dict:
    total_cpu = ray.cluster_resources().get("CPU", 1)
    total_gpu = ray.cluster_resources().get("GPU", 0)
    cpu = min(1.0, total_cpu / max(num_selected, 1))
    gpu = min(1.0, total_gpu / max(num_selected, 1)) if total_gpu > 0 else 0
    return {"num_cpus": cpu, "num_gpus": gpu}


class RayExecutor:
    def __init__(self, fd: FederatedDataset, cfg: FedConfig):
        self.fd, self.cfg = fd, cfg
        self.max_alive = 0
        self.actor_log = []

    def __call__(self, global_state, selected, rnd, x_trains):
        res = resources_per_actor(len(selected))
        gs_ref = ray.put(global_state)  # broadcast once per round
        actors = [ClientActor.options(**res).remote(int(i), x_trains[i], self.fd.clients[i].y_train, self.cfg.local)
                  for i in selected]
        self.max_alive = max(self.max_alive, len(actors))
        futures = [a.train.remote(gs_ref, round_seed(self.cfg.seed, rnd, int(i))) for a, i in zip(actors, selected)]
        out = ray.get(futures)
        for a in actors:
            ray.kill(a)  # free actor memory: actors only live for the round they are selected in
        self.actor_log.append({"round": rnd, "n_actors": len(actors), "resources": res})
        return out


def run_fedavg_ray(fd: FederatedDataset, cfg: FedConfig, log=print, num_cpus: int | None = None):
    started = False
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
        started = True
    ex = RayExecutor(fd, cfg)
    try:
        out = run_fedavg(fd, cfg, executor=ex, log=log)
    finally:
        if started:
            ray.shutdown()
    out["ray"] = {"max_alive_actors": ex.max_alive, "actor_log": ex.actor_log}
    return out
