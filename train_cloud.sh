#!/usr/bin/env bash
# Cloud / headless training entry point for LT-PINN Milestone 1
# Run on servers without GUI (no display required).
set -euo pipefail

cd "$(dirname "$0")"

# Use CPU if no CUDA is available; override with --device cpu if needed.
python src/main.py \
    --layout center \
    --epochs 6000 \
    --lbfgs-steps 800 \
    --no-plot \
    "$@"
