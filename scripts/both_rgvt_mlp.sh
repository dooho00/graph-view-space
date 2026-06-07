#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --mode both \
  --datasetA 27_ogbn_arxiv \
  --checkpoint checkpoints/mlp/seed_42.pth \
  --learning_rate 0.005 \
  --predictor_type mlp
