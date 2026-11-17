#!/usr/bin/env bash
# Reproduce every result. Add --synthetic to any line to run without the Canvas data.
set -euo pipefail
DATA=${DATA:-data}
pytest validation -v -s | tee results_validation_tests.log
python scripts/part1_fedavg.py --data "$DATA" --rounds 200 --C 0.1 --E 5
python scripts/part1_sweep.py  --data "$DATA" --rounds 150 --Cs 0.01 0.02 0.05 0.1 --Es 1 2 5 10 20 --seeds 0 1 --eval-every 5
python scripts/part1_ray.py    --data "$DATA" --rounds 200 --clients-per-round 4 --E 5
python scripts/part2_dp.py     --data "$DATA" --rounds 150 --b 0.1 --bs 0 0.01 0.05 0.1 0.2 0.3 0.5 1 2 5 --seeds 0 1 2 --eval-every 5
python validation/experiments.py --data "$DATA" --rounds 50
