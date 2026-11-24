import os
import subprocess

import pytest

import settings

pytestmark = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="slow: set RUN_SLOW=1")


def test_run_sh_sample():
    before = settings.CONFIG_PATH.read_bytes()
    r = subprocess.run(["bash", "run.sh", "--sample"], cwd=settings.PROJECT_DIR, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SAMPLE MODE" in r.stdout
    assert "The test split was not opened" in r.stdout
    assert settings.CONFIG_PATH.read_bytes() == before
