#!/usr/bin/env bash
# Run the whole project: check the data, then every step that uses the train and validation piles.
# Results go to outputs/. Extra options are passed on:  bash run.sh --sample   (made-up data, for CI)
#                                                       bash run.sh --skip-tune (skip the slow tuning re-run)
# This never opens the test pile (it was opened once and is locked).
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python3
fi

"$PYTHON" src/pipeline.py "$@"
