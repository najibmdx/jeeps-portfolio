"""Runs run.sh --sample for real and checks that it exits 0 and leaves everything committed (and the test pile) alone.

This takes under a minute and needs no internet (the data is made up), but it needs bash, so it only runs
when RUN_SLOW=1 (CI sets this). To run it on your machine, from Git Bash:

    RUN_SLOW=1 python -m pytest tests/test_run_sh.py
"""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import settings

PROJECT = settings.PROJECT_DIR
GUARDED = ["config.yaml", "tuned_params.yaml", "frozen_choice.yaml"]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="slow and needs bash; set RUN_SLOW=1 to run it",
)


def snapshot(path: Path):
    """What the file looks like now: does not exist, or (size, last change, content hash)."""
    if not path.exists():
        return None
    return (path.stat().st_size, path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())


def everything_guarded():
    cfg = settings.load_config()
    lock = settings.DATA_DIR / cfg["final"]["test_lock_file"]
    result = settings.PROJECT_DIR / cfg["evaluation"]["outputs_folder"] / "final_test.json"
    return {p: snapshot(PROJECT / p) for p in GUARDED} | {"lock": snapshot(lock), "final_test.json": snapshot(result)}


def test_run_sh_sample_exits_zero_and_leaves_committed_files_and_the_test_pile_alone():
    assert shutil.which("bash"), "bash is needed to run run.sh"
    before = everything_guarded()
    done = subprocess.run(["bash", "run.sh", "--sample"], cwd=PROJECT, capture_output=True, text=True, timeout=1200)
    assert done.returncode == 0, f"run.sh failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
    assert "SAMPLE MODE" in done.stdout
    assert "verified: the re-run reproduces tuned_params.yaml and frozen_choice.yaml" in done.stdout
    assert "The test pile was not opened by this run" in done.stdout
    assert everything_guarded() == before          # nothing committed changed, no lock created, no test result written
