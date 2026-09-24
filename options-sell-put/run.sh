#!/usr/bin/env bash
# One-click execution script for Sell Put Options Deep Learning System
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "======================================================================"
echo " Launching Sell Put Quantitative Pipeline (MPS / CUDA / CPU)"
echo "======================================================================"

# Run full pipeline with uv
uv run --with "torch" --with "numpy" python3 train.py "$@"
