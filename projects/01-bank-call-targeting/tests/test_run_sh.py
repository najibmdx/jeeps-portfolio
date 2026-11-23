"""Runs run.sh for real and checks that it exits 0 and leaves the one-time test result alone.

This takes about 2 minutes and downloads the UCI file if data/bank.db is missing, so it only
runs when RUN_SLOW=1 (CI sets this). To run it on your machine:

    RUN_SLOW=1 python -m pytest tests/test_run_sh.py
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
TEST_RESULTS = PROJECT / "outputs" / "test_results.json"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="slow (about 2 minutes, needs internet on a fresh clone); set RUN_SLOW=1 to run it",
)


def snapshot(path: Path):
    """What the file looks like now: does not exist, or (size, last change, content)."""
    if not path.exists():
        return None
    return (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())


def test_run_sh_exits_zero_and_never_writes_or_changes_the_test_result():
    assert shutil.which("bash"), "bash is needed to run run.sh"
    before = snapshot(TEST_RESULTS)
    done = subprocess.run(["bash", "run.sh"], cwd=PROJECT, capture_output=True, text=True, timeout=1200)
    assert done.returncode == 0, f"run.sh failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
    assert "The test set was not used" in done.stdout
    assert snapshot(TEST_RESULTS) == before     # not created, not changed
