#!/usr/bin/env bash
# Run the whole project. Results go to outputs/. Extra options are passed on:
#   bash run.sh --sample     (made-up data, for CI)
#   bash run.sh --data-dir D (look for the four data files in folder D)
# Stage 6: checks the data files and their layout, builds the dataset, splits it by time, audits for leakage and train-versus-validation shift, scores three baselines and an untuned model ladder on validation. Tuning and the final choice come later.
# This never opens the test split.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python3
fi

"$PYTHON" src/pipeline.py "$@"
