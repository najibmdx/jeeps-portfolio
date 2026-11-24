#!/usr/bin/env bash
# Run the whole project. Results go to outputs/. Extra options are passed on:
#   bash run.sh --sample     (made-up data, for CI)
#   bash run.sh --data-dir D (look for the four data files in folder D)
# Stage 0: this only checks the data files and their layout. The steps that use data come in later stages.
# This never opens the test split.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python3
fi

"$PYTHON" src/pipeline.py "$@"
