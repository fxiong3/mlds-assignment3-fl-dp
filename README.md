# MLDS Assignment 3 — Federated Learning (FedAvg + Ray) and Differential Privacy

FedAvg, Ray-parallel clients and Laplace input perturbation on Federated EMNIST (100 users, 62 classes),
implemented from scratch with PyTorch + Ray Core (no FL libraries). Model: 784 → 128 (ReLU) → 62.

## Setup
```bash
pip install -r requirements.txt
# put train_data.npy and test_data.npy from Canvas (Assignment3-data) into ./data
# file names must contain "train" / "test", or be in train/ and test/ folders)
```
Every script also accepts `--synthetic` to run on a generated 100-client dataset (no download needed).

## Run
| Task | Command | Output (`results/…`) |
|---|---|---|
| P1 Q1.1–1.3, 1.5: data description, FedAvg C=10%, curves, test acc | `python scripts/part1_fedavg.py --rounds 200 --C 0.1 --E 5` | `part1_fedavg/` |
| P1 Q1.4–1.5: C × E sweep, best config by **validation** acc → test acc | `python scripts/part1_sweep.py --rounds 150 --Cs 0.01 0.02 0.05 0.1 --Es 1 2 5 10 20 --seeds 0 1` | `part1_sweep/` |
| P1 Q2: Ray actors, 4 clients/round | `python scripts/part1_ray.py --rounds 200 --clients-per-round 4` | `part1_ray/` |
| P2 Q1–Q3: Laplace noise single run + noise sweep | `python scripts/part2_dp.py --rounds 150 --b 0.1 --bs 0 0.01 0.05 0.1 0.2 0.3 0.5 1 2 5 --seeds 0 1 2` | `part2_dp/` |
| Validation experiments on real data | `python validation/experiments.py --rounds 50` | `validation/` |
| Validation test suite (40 checks, ~2 min) | `pytest validation -v -s` | console |
| Everything | `bash run_all.sh` | |

Common flags (defaults): `--lr 0.1 --batch-size 32 --seed 0 --eval-every 1 --data data --out results`. Pixels are inverted to background 0 by default (`--no-invert` keeps raw LEAF pixels); this conditions SGD much better (40-round val acc 0.63 vs 0.42).

## Code layout
```
fl/data.py      loading (LEAF json / npy / npz / pkl), 80/20 per-client split, synthetic data, statistics
fl/core.py      MLP, local SGD update, weighted FedAvg aggregation, client sampling, training loop, metrics
fl/ray_fed.py   ClientActor (1 per selected client per round, killed after the round), Ray executor
fl/dp.py        Laplace mechanism X' = X + Lap(0,b), per-pixel / per-image epsilon, SNR
fl/plots.py     standard + non-standard plots
scripts/        one script per assignment part
validation/     pytest validation suite + experiments.py (real-data evidence)
```
Design choices that matter for correctness:
* Sequential and Ray versions share one server loop (`run_fedavg`), only the executor differs, and every
  (round, client) update uses its own seed. Ray results are therefore bit-for-bit comparable to sequential ones.
* Aggregation weights are `n_k / Σ n_k` over the **selected** clients (the server knows each client's sample count).
* Reported train/val metrics are evaluated with the global model on **all** 100 clients (sample-weighted).
* The test set is used once, after training; the best (C, E) is chosen on validation accuracy.
* Part 2: each client perturbs its own training images once with its own RNG stream (local DP on inputs);
  training accuracy is measured on the noisy data the model trained on, validation/test on clean data.

## Non-standard plots
* `participation_heatmap.png`: clients × rounds, coloured by relative local-loss reduction, plus selection counts vs the expected count.
* `client_drift*.png`: ‖w_k − w_global‖ per round for different E, and for IID vs non-IID data.
* `per_client_val_acc_violin.png`: how the global model's accuracy is spread across users over training.
* `actor_gantt.png`: Ray actor execution timeline, which shows the client updates really run in parallel.
* `label_heatmap_users_x_classes.png`: P(y | user) for all users.
* `noise_gallery.png`: the same digits under each b, with SNR and ε. `privacy_utility.png`: test accuracy vs ε = 1/b.

## Validation tests: what each checks and why it validates
**FedAvg (`validation/test_fedavg_correctness.py`)**
| Test | Prediction checked |
|---|---|
| `aggregation_equals_manual_weighted_mean` | aggregate == Σ n_k w_k / Σ n_k computed independently |
| `permutation_invariant_and_fixed_point` | client order irrelevant; averaging identical models returns that model |
| `zero_weight_clients_ignored_and_all_zero_raises` | n_k = 0 contributes nothing; no clients → error |
| `fedsgd_equivalence_full_participation` | C=1, E=1, B=∞: one FedAvg round == one full-batch GD step on pooled data (diff ~1e-8); unweighted averaging breaks it |
| `single_client_fedavg_equals_sequential_local_training` | with one client, FedAvg reduces to chained local SGD |
| `no_learning_when_lr_zero_or_zero_epochs` | weights unchanged |
| `learns_well_above_chance_and_loss_decreases` | separable data → test acc ≫ 1/62, loss halves |
| `shuffled_labels_give_chance_level_test_accuracy` | no signal → test acc at chance (catches leakage / eval bugs) |
| `determinism_and_seed_sensitivity` | same seed → identical model and client draws; different seed → different |
| `more_local_epochs_increase_client_drift_on_noniid` | drift(E=1) < drift(E=5) < drift(E=20) |
| `iid_has_less_client_disagreement_than_noniid` | per-client loss spread smaller under IID |
| `num_selected_edge_cases`, `sampling_without_replacement_and_uniform` | m = max(1, round(CK)) ≤ K; uniform selection (χ² test over 5000 rounds) |

**Edge cases (`validation/test_edge_cases.py`)**: 1- and 2-sample clients (never an empty train split), exact 80/20 split,
disjoint train/val split, empty clients skipped, invalid labels rejected, 0–255 vs [0,1] inputs, wrong image size,
clients with no validation data, batch > local data, empty local data, C·K < 1, one class per client (FedAvg ≫ any
local model), corrupting the test set leaves training bit-identical (no leakage), changing other clients' data leaves a
client's update unchanged (isolation), and the LEAF-json and npy-dict loaders.

**Ray (`validation/test_ray_parallel.py`)**: Ray == sequential (diff ~1e-8, same client draws and curves); exactly 4
actors per round with all CPUs released afterwards (no leaked actors); ≤1 CPU/≤1 GPU per actor; each actor holds only
its own client's data (fingerprint); the 4 updates overlap in wall-clock time with 4 distinct PIDs.

**DP (`validation/test_dp.py`)**: the noise matches Lap(0,b) in mean, variance 2b², E|ε| = b, median b·ln2 and a KS test,
for several b; noise is independent of the signal and across pixels; b=0 is the identity; b<0 is rejected; clipping stays in [0,1];
noise is independent per client and reproducible per seed while clean data stay untouched; a b=0 run equals plain FedAvg;
test accuracy decreases with b (Spearman ρ < −0.8) and reaches chance for very large b.

**Real-data experiments (`validation/experiments.py`)**: first-10-round trace (selected clients, n_k, weights, local loss
before/after, drift, update norm), IID re-partition vs natural split, shuffled labels, E vs drift, lr = 0 / lr = 50
sanity, FedSGD equivalence on FEMNIST, weighted vs unweighted aggregation.
