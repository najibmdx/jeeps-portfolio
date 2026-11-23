#!/usr/bin/env bash
# Run the whole project: get the data if missing, then every step that uses
# training and validation data. Results go to outputs/.
# This never runs the one-time test run (src/final.py).
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python3
fi

"$PYTHON" src/pipeline.py
