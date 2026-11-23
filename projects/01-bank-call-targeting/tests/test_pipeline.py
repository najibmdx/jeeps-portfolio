"""Tests for pipeline.py and run.sh: the whole run works, and it never touches the test set."""

import ast
from pathlib import Path

import pytest

import ablation
import cutoff
import get_data
import leakage_audit
import pipeline
import split
import train
import tune
import uncertainty
from settings import load_config
from test_ablation import make_frame
from test_tune import make_timed

PROJECT = Path(__file__).resolve().parent.parent


class NoTest(dict):
    """A sets dictionary that fails loudly if anyone reaches for the test set in any way."""

    def _refuse(self, *args, **kwargs):
        raise AssertionError("the pipeline must never read the test set")

    def __getitem__(self, key):
        if key == "test":
            self._refuse()
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "test":
            self._refuse()
        return super().get(key, default)

    items = values = copy = _refuse


def small_config():
    """The real settings, made small so the whole pipeline runs in seconds."""
    config = load_config()
    config["models"] = {"logistic_regression": {"max_iter": 300}}
    config["evaluation"] = {"top_fraction": 0.10, "bootstrap_draws": 10, "cutoff_share_step": 0.1}
    config["tuning"] = {"min_train_months": 5, "grids": {"logistic_regression": {"C": [0.1, 1.0]}}}
    return config


@pytest.fixture
def small_pipeline(tmp_path, monkeypatch):
    """Point every step at small made-up data, a temporary outputs folder and no downloads."""
    config = small_config()
    sets = NoTest(train=make_timed(2400, 0), validation=make_frame(800, 1), test=make_frame(800, 2))
    for module in (train, uncertainty, leakage_audit, ablation, cutoff, tune):
        monkeypatch.setattr(module, "load_config", lambda config=config: config)
    monkeypatch.setattr(split, "prepare", lambda config=None: sets)
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(train, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(leakage_audit, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(get_data, "main", lambda: print("(download skipped in this test)"))
    return outputs


def test_the_whole_pipeline_runs_and_writes_every_result_file(small_pipeline):
    done = pipeline.run_steps()
    assert [name for name, _ in done] == [name for name, _ in pipeline.STEPS]
    written = {p.name for p in small_pipeline.iterdir()}
    assert written == {"metrics.json", "wobble.json", "leakage_audit.csv", "ablation.json",
                       "scenarios.json", "tuning_results.json"}


def test_the_pipeline_never_reads_the_test_set_and_never_writes_test_results(small_pipeline):
    pipeline.run_steps()   # NoTest raises if any step asks for the test set
    assert not (small_pipeline / "test_results.json").exists()


def test_the_step_list_does_not_contain_the_one_time_test_run():
    assert "final" not in [name for name, _ in pipeline.STEPS]
    assert all(module.__name__ != "final" for _, module in pipeline.STEPS)


def test_pipeline_py_does_not_import_final():
    tree = ast.parse((PROJECT / "src" / "pipeline.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "final" not in imported


def test_run_sh_calls_the_pipeline_and_never_the_final_run():
    text = (PROJECT / "run.sh").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "src/pipeline.py" in code
    assert "final" not in code and "test_results" not in code
    assert "set -euo pipefail" in code          # any failing step makes run.sh fail
    assert "\r" not in text                      # Windows line endings would break bash


def test_a_failing_step_stops_the_run_and_names_the_step(capsys):
    class Boom:
        __name__ = "boom"

        @staticmethod
        def main():
            raise RuntimeError("broken")

    class Never:
        __name__ = "never"

        @staticmethod
        def main():
            raise AssertionError("must not run after a failure")

    with pytest.raises(RuntimeError, match="broken"):
        pipeline.run_steps([("boom", Boom), ("never", Never)])
    assert "FAILED in step 1 (boom)" in capsys.readouterr().out


def test_the_requirements_file_pins_every_package_exactly():
    lines = [line.strip() for line in (PROJECT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    packages = [line for line in lines if line and not line.startswith("#")]
    assert packages and all("==" in line for line in packages)
    names = {line.split("==")[0].lower() for line in packages}
    assert {"pandas", "numpy", "scikit-learn", "pyyaml", "pytest"} <= names
