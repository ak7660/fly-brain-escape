#!/usr/bin/env bash
# P3: 3 variants × 3 seeds; one round per seed runs the three variants concurrently (2 torch threads each).
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=2
mkdir -p results/logs
for seed in 0 1 2; do
  for v in real readout_only shuffled; do
    uv run python -m flybrain.train --variant $v --seed $seed > results/logs/train_${v}_s${seed}.log 2>&1 &
  done
  wait
  echo "ROUND seed=$seed finished" >> results/logs/train_all.log
done
echo "ALL TRAINING DONE" >> results/logs/train_all.log
