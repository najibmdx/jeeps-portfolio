"""Tests for Stage 7: the made-up pipeline sample, pipeline.py in real and sample mode, and the walls around the test pile."""

import ast
import contextlib
import dataclasses
import hashlib
import io
import math
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import leakage_audit
import pipeline
import safe_pickle
import sample
import settings

PROJECT = settings.PROJECT_DIR
SRC = PROJECT / "src"
GUARDED = ["config.yaml", "tuned_params.yaml", "frozen_choice.yaml"]


def snapshot_committed():
    """Hashes of the committed files a sample run must never touch, plus the real outputs listing and lock."""
    state = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in GUARDED}
    outputs = PROJECT / "outputs"
    state["outputs"] = sorted((p.name, p.stat().st_mtime_ns) for p in outputs.iterdir()) if outputs.exists() else None
    lock = settings.DATA_DIR / settings.load_config()["final"]["test_lock_file"]
    state["lock"] = (lock.exists(), lock.stat().st_mtime_ns if lock.exists() else None)
    return state


@pytest.fixture(scope="module")
def sample_run(tmp_path_factory):
    """One full pipeline run on made-up data, shared by the tests below."""
    work = tmp_path_factory.mktemp("sample")
    before = snapshot_committed()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ctx = pipeline.build_context(True, work)
        ok, done = pipeline.run_steps(ctx)
    return SimpleNamespace(ctx=ctx, ok=ok, done=done, out=buffer.getvalue(), work=work,
                           before=before, after=snapshot_committed())


# ---------------------------------------------------------------- the made-up sample
def test_pipeline_sample_is_repeatable_and_leaves_the_old_sample_alone():
    a, la = sample.make_pipeline_sample(seed=3)
    b, lb = sample.make_pipeline_sample(seed=3)
    assert la == lb and [list(x) for x in a] == [list(x) for x in b]
    old, _ = sample.make_sample()
    assert len(old) == 80 and "group1_migrate_year" not in old[0]      # the old sample is exactly as before


def test_pipeline_sample_has_all_four_groups_and_the_clock_columns():
    features, labels = sample.make_pipeline_sample()
    cols = set(features[0])
    for group in ("group1", "group2", "group3", "group4"):
        assert any(c.startswith(group + "_") for c in cols)
    cfg = settings.load_config()
    assert set(cfg["ablation"]["never_use"]) <= cols
    assert len(features) - len(labels) == 20
    assert all(c["mint_address"].startswith("SAMPLE") for c in features)


def test_pipeline_sample_has_both_sides_everywhere_it_matters():
    features, labels = sample.make_pipeline_sample()
    crash = np.array([l["min_ratio"] < 0.3 for l in labels])
    assert 0.5 < crash.mean() < 0.8
    scope = settings.load_config()["scope"]
    kept = [c for c in features if c[scope["holders_column"]] >= scope["min_holders"]
            and c[scope["valid_swap_column"]] >= scope["min_valid_swap_seconds"]]
    assert 600 < len(kept) < len(features)


def test_pipeline_sample_notes_cover_every_feature_column_and_pass_the_audit_check():
    features, _ = sample.make_pipeline_sample()
    notes = sample.pipeline_sample_notes(features)
    assert set(notes) == {c for c in features[0] if c.startswith("group")}
    leakage_audit.check_notes(notes)
    assert all(notes[c]["known_at_migration"] == "unverified" for c in notes if c.startswith("group4"))


def test_pipeline_sample_files_load_through_the_safe_loader(tmp_path):
    feature_path, label_path, _ = sample.write_pipeline_sample(tmp_path)
    assert feature_path.name == "feature.pkl" and label_path.name == "label.csv"
    rows = safe_pickle.load_file(feature_path)
    assert len(rows) == 1500 and rows[0]["ts"].shape == (360, 5)


# ---------------------------------------------------------------- config for the sample
def test_sample_config_uses_the_real_hashes_of_the_made_up_files(tmp_path):
    feature_path, label_path, _ = sample.write_pipeline_sample(tmp_path)
    real = settings.load_config()
    cfg = pipeline.sample_config(real, tmp_path, feature_path, label_path)
    assert cfg["data"]["feature"]["bytes"] == feature_path.stat().st_size
    assert cfg["data"]["feature"]["sha256"] == hashlib.sha256(feature_path.read_bytes()).hexdigest()
    assert cfg["data"]["label"]["sha256"] == hashlib.sha256(label_path.read_bytes()).hexdigest()
    assert cfg["verified_on_real_data"] == {}
    assert real["verified_on_real_data"] and real["data"]["feature"]["bytes"] == 517642519    # the real config is untouched


def test_sample_mode_refuses_to_work_inside_the_project_folder():
    with pytest.raises(ValueError, match="outside the project folder"):
        pipeline.build_context(True, PROJECT / "outputs" / "sample_here")
    with pytest.raises(ValueError, match="outside the project folder"):
        pipeline.build_context(True, PROJECT)


def test_real_mode_never_writes_to_the_committed_settings_files():
    ctx = pipeline.build_context(False)
    assert ctx.params_path.name == "tuned_params_rerun.yaml"
    assert ctx.params_path != ctx.reference_params
    assert ctx.reference_params == PROJECT / "tuned_params.yaml"
    assert ctx.reference_frozen == PROJECT / "frozen_choice.yaml"
    assert ctx.data_dir == settings.DATA_DIR and not ctx.sample_mode


# ---------------------------------------------------------------- step order
def names(ctx):
    return [n for n, _ in pipeline.steps_for(ctx)]


def test_real_step_order():
    assert names(pipeline.build_context(False)) == [
        "get_data", "prepare", "leakage_audit", "train", "uncertainty", "tune", "ablation", "verify"]


def test_skip_tune_drops_tune_and_verify_only():
    assert names(pipeline.build_context(False, skip_tune=True)) == [
        "get_data", "prepare", "leakage_audit", "train", "uncertainty", "ablation"]


def test_sample_mode_adds_the_reference_step_right_after_prepare(tmp_path):
    n = names(pipeline.build_context(True, tmp_path))
    assert n[:3] == ["get_data", "prepare", "reference (sample only)"] and n[-1] == "verify"


def test_pipeline_never_imports_the_step_that_opens_the_test_pile():
    text = (SRC / "pipeline.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not {"final_test", "holdout"} & imported
    assert "open_test_pile" not in text and "test_features" not in text


# ---------------------------------------------------------------- stopping at a failure
def test_run_steps_stops_at_the_first_failing_step_and_names_it(capsys):
    ran = []
    steps = [("one", lambda c: ran.append("one") or 0), ("two", lambda c: ran.append("two") or 1),
             ("three", lambda c: ran.append("three") or 0)]
    ok, done = pipeline.run_steps(None, steps)
    assert not ok and ran == ["one", "two"] and [n for n, _ in done] == ["one"]
    assert "STOPPED: step two did not succeed" in capsys.readouterr().out


def test_run_steps_stops_when_a_step_raises(capsys):
    def boom(ctx):
        raise RuntimeError("kaput")
    ran = []
    ok, done = pipeline.run_steps(None, [("a", boom), ("b", lambda c: ran.append("b") or 0)])
    assert not ok and ran == [] and "step a raised an error" in capsys.readouterr().out


def test_main_returns_1_and_skips_later_steps_when_a_step_fails(monkeypatch, capsys, tmp_path):
    ran = []
    monkeypatch.setattr(pipeline, "step_prepare", lambda ctx: ran.append("prepare") or 1)
    monkeypatch.setattr(pipeline, "step_leakage_audit", lambda ctx: ran.append("audit") or 0)
    monkeypatch.setattr(pipeline, "step_get_data", lambda ctx: 0)
    assert pipeline.main(["--sample", "--work-dir", str(tmp_path)]) == 1
    assert ran == ["prepare"]
    assert "test pile was not opened" not in capsys.readouterr().out


def test_work_dir_without_sample_is_refused(tmp_path):
    assert pipeline.main(["--work-dir", str(tmp_path)]) == 2


# ---------------------------------------------------------------- comparing two results
def test_differences_ignores_tiny_noise_and_the_note():
    a = {"note": "x", "m": 1.0, "n": {"cut": [0.5, 0.25]}, "s": "rf"}
    b = {"note": "different", "m": 1.0 + 1e-12, "n": {"cut": [0.5, 0.25 - 1e-12]}, "s": "rf"}
    assert pipeline.differences(a, b) == []


def test_differences_names_every_real_difference():
    a = {"model": "rf", "cut": [0.5, 0.25], "only_new": 1, "x": {"y": 0.3}}
    b = {"model": "gb", "cut": [0.5, 0.26], "only_old": 1, "x": {"y": 0.3}}
    found = pipeline.differences(a, b)
    joined = "\n".join(found)
    assert "model" in joined and "cut[1]" in joined and "only_new" in joined and "only_old" in joined
    assert "x.y" not in joined


def test_differences_handles_infinity_lengths_and_types():
    assert pipeline.differences({"c": math.inf}, {"c": math.inf}) == []
    assert pipeline.differences({"c": math.inf}, {"c": 1.0})
    assert pipeline.differences({"c": [1, 2]}, {"c": [1, 2, 3]})
    assert pipeline.differences({"c": 1}, {"c": "1"})


def test_differences_tolerance_is_tight():
    assert pipeline.differences({"c": 0.5}, {"c": 0.5 + 1e-6})


# ---------------------------------------------------------------- one full run on made-up data
def test_sample_run_succeeds_and_runs_every_step_in_order(sample_run):
    assert sample_run.ok
    assert [n for n, _ in sample_run.done] == names(sample_run.ctx)
    for step in ("get_data", "prepare", "leakage_audit", "train", "uncertainty", "tune", "ablation", "verify"):
        assert f"===== {step} =====" in sample_run.out


def test_sample_run_writes_its_results_to_the_temp_folder(sample_run):
    out = sample_run.ctx.out_dir
    for name in ("metrics.json", "wobble.json", "tuning.json", "tuned_metrics.json", "tuned_wobble.json",
                 "ablation.json", "leakage_summary.json", "select_final.json", "tuned_params_rerun.yaml",
                 "validation_cutoff_curve.csv"):
        assert (out / name).exists(), name
    assert sample_run.ctx.reference_params.exists() and sample_run.ctx.reference_frozen.exists()
    assert not (sample_run.work / "outputs" / "final_test.json").exists()


def test_sample_run_never_touches_the_committed_files_or_real_outputs(sample_run):
    assert sample_run.before == sample_run.after


def test_sample_run_never_opens_the_test_pile(sample_run):
    lock = sample_run.ctx.data_dir / settings.load_config()["final"]["test_lock_file"]
    assert not lock.exists()
    assert "MISMATCH" not in sample_run.out


def test_sample_run_says_what_it_verified(sample_run):
    assert "verified: the re-run reproduces tuned_params.yaml and frozen_choice.yaml" in sample_run.out


def test_main_in_sample_mode_exits_zero_and_prints_timings(tmp_path, capsys):
    assert pipeline.main(["--sample", "--skip-tune", "--work-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "SAMPLE MODE" in out and "===== timings =====" in out and "The test pile was not opened by this run" in out
    assert "===== tune =====" not in out and "===== verify =====" not in out
    assert not (tmp_path / "outputs" / "tuned_params_rerun.yaml").exists()


# ---------------------------------------------------------------- the verify step must really verify
def test_verify_passes_on_identical_files(sample_run):
    with contextlib.redirect_stdout(io.StringIO()):
        assert pipeline.step_verify(sample_run.ctx) == 0


def altered(sample_run, tmp_path, params=None, frozen=None):
    """A context whose reference files are changed copies of the real sample references."""
    ctx = sample_run.ctx
    ref_p = yaml.safe_load(ctx.reference_params.read_text())
    ref_f = yaml.safe_load(ctx.reference_frozen.read_text())
    if params:
        params(ref_p)
    if frozen:
        frozen(ref_f)
    p, f = tmp_path / "p.yaml", tmp_path / "f.yaml"
    p.write_text(yaml.safe_dump(ref_p))
    f.write_text(yaml.safe_dump(ref_f))
    return dataclasses.replace(ctx, reference_params=p, reference_frozen=f)


def test_verify_catches_a_changed_tuned_setting(sample_run, tmp_path, capsys):
    ctx = altered(sample_run, tmp_path, params=lambda p: p["models"]["logistic_regression"].update(C=123.0))
    assert pipeline.step_verify(ctx) == 1
    out = capsys.readouterr().out
    assert "VERIFY FAILED" in out and "tuned settings" in out and "Nothing was overwritten" in out


def test_verify_catches_a_moved_cutoff(sample_run, tmp_path, capsys):
    ctx = altered(sample_run, tmp_path, frozen=lambda f: f["scenarios"]["equal"].update(
        cutoff=f["scenarios"]["equal"]["cutoff"] + 0.01))
    assert pipeline.step_verify(ctx) == 1
    assert "scenarios.equal.cutoff" in capsys.readouterr().out


def test_verify_catches_a_different_model_in_the_frozen_file(sample_run, tmp_path, capsys):
    ctx = altered(sample_run, tmp_path, frozen=lambda f: f.update(model="gradient_boosting"))
    assert pipeline.step_verify(ctx) == 1
    assert "frozen choice: model" in capsys.readouterr().out


def test_verify_tolerates_float_noise_below_the_tolerance(sample_run, tmp_path):
    ctx = altered(sample_run, tmp_path, frozen=lambda f: f["scenarios"]["equal"].update(
        cutoff=f["scenarios"]["equal"]["cutoff"] + 1e-12))
    with contextlib.redirect_stdout(io.StringIO()):
        assert pipeline.step_verify(ctx) == 0


def test_verify_never_writes_the_reference_files(sample_run, tmp_path):
    before = (sample_run.ctx.reference_params.read_bytes(), sample_run.ctx.reference_frozen.read_bytes())
    with contextlib.redirect_stdout(io.StringIO()):
        pipeline.step_verify(sample_run.ctx)
    assert before == (sample_run.ctx.reference_params.read_bytes(), sample_run.ctx.reference_frozen.read_bytes())


# ---------------------------------------------------------------- run.sh and the CI workflow
WORKFLOW = PROJECT.parent.parent / ".github" / "workflows" / "p02-ci.yml"


def test_run_sh_only_calls_the_pipeline_and_passes_options_on():
    text = (PROJECT / "run.sh").read_text(encoding="utf-8")
    assert "src/pipeline.py" in text and '"$@"' in text and "set -euo pipefail" in text
    assert "final_test" not in text and "select_final" not in text


def test_ci_workflow_runs_the_slow_sample_test_on_pinned_python_without_downloads():
    flow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = flow.get("on", flow.get(True))        # PyYAML reads a bare `on` as True
    for event in ("push", "pull_request"):
        assert "projects/02-memecoin-launch-risk/**" in triggers[event]["paths"]
    job = flow["jobs"]["tests-and-run"]
    assert job["defaults"]["run"]["working-directory"] == "projects/02-memecoin-launch-risk"
    steps = job["steps"]
    assert any(s.get("with", {}).get("python-version") == "3.14" for s in steps)
    assert any(s.get("run") == "python -m pip install -r requirements.txt" for s in steps)
    test_step = next(s for s in steps if "pytest" in s.get("run", ""))
    assert test_step["env"]["RUN_SLOW"] == "1"
    text = WORKFLOW.read_text(encoding="utf-8").replace("# ", "")
    assert "curl" not in text and "get_data" not in text and "final_test" not in text and "drive.google" not in text
    assert flow["permissions"] == {"contents": "read"}
