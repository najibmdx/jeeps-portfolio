"""Stage 9 tests: the settings, the lock and the one reader of the test split, the order of the opening, the checks that stop it
before anything is opened, known answers on hand-made tables, the three readings, the test-window check, and the command line."""

import copy
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import baselines
import final_test as ft
import freeze
import models
import pipeline
import settings
import split
from test_ablate import _frozen_file, loss
from test_tune import CONFIG, EPS, _split_dir, frame

SRC = Path(settings.PROJECT_DIR) / "src"
READINGS = CONFIG["final_test"]["readings"]


def mk(prices, labels, events=None, t0="2025-02-01", close_days=None, category=None, seed=0):
    """A table with every column the test reads, with known prices and outcomes."""
    n = len(prices)
    d = frame(n, seed=seed, last_close=np.asarray(prices, dtype=float), label=np.asarray(labels, dtype="int8"))
    d["event_key"] = events if events is not None else [f"E{seed}-{i}" for i in range(n)]
    t = pd.Timestamp(t0, tz="UTC") + pd.to_timedelta(np.arange(n), unit="h")
    d["t"] = t
    d["close_time"] = t + pd.to_timedelta(close_days if close_days is not None else [1.0] * n, unit="D")
    if category is not None:
        d["category"] = category
    return d


@pytest.fixture(autouse=True)
def identity_correction(monkeypatch):
    """The price-only correction is the identity in the hand-made tables, so every baseline is known by hand."""
    monkeypatch.setattr(baselines, "fit_correction", lambda price, y, eps: {"intercept": 0.0, "slope": 1.0})


# hand-made tables: 6 contested markets priced 0.5 and 4 near-decided priced 0.01
TRAIN = mk([0.5] * 8 + [0.01] * 2, [1, 1, 1, 0, 0, 0, 0, 0, 1, 0], seed=1)
VAL = mk([0.5] * 6 + [0.01] * 4, [1, 1, 0, 0, 0, 0, 0, 0, 0, 1], seed=2)
TEST = mk([0.5] * 6 + [0.01] * 4, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0], seed=3)
BASE = 0.4  # the train Yes share


def const(p):
    return lambda df: np.full(len(df), p)


# ---- the settings

def _bad(edit):
    c = copy.deepcopy(CONFIG)
    edit(c)
    return c


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda c: c["final_test"].update(expected_test_markets=0), "whole number above 0"),
        (lambda c: c["final_test"].update(expected_test_markets=True), "whole number above 0"),
        (lambda c: c["final_test"].update(expected_test_markets="8991"), "whole number above 0"),
        (lambda c: c["final_test"].update(expected_test_markets=8991.5), "whole number above 0"),
        (lambda c: c["final_test"].update(reproduce_tolerance=0), "small number above 0"),
        (lambda c: c["final_test"].update(reproduce_tolerance=0.01), "small number above 0"),
        (lambda c: c["final_test"].update(reproduce_tolerance="1e-7"), "small number above 0"),
        (lambda c: c["final_test"].update(headline_slice="both"), "all or contested"),
        (lambda c: c["final_test"].update(window_percentiles=[]), "different numbers above 0 and below 100"),
        (lambda c: c["final_test"].update(window_percentiles=[0, 50]), "different numbers above 0 and below 100"),
        (lambda c: c["final_test"].update(window_percentiles=[50, 100]), "different numbers above 0 and below 100"),
        (lambda c: c["final_test"].update(window_percentiles=[90, 50]), "smallest first"),
        (lambda c: c["final_test"].update(window_percentiles=[50, 50]), "different numbers"),
        (lambda c: c["final_test"].update(window_percentiles=["50"]), "different numbers above 0 and below 100"),
        (lambda c: c["final_test"].update(window_percentiles=[True]), "different numbers above 0 and below 100"),
        (lambda c: c["final_test"].update(window_percentiles=50), "different numbers above 0 and below 100"),
        (lambda c: c["final_test"]["readings"].pop("worse"), "exactly one non-empty sentence"),
        (lambda c: c["final_test"]["readings"].update(extra="x"), "exactly one non-empty sentence"),
        (lambda c: c["final_test"]["readings"].update(better="  "), "exactly one non-empty sentence"),
        (lambda c: c["final_test"]["readings"].update(better=3), "exactly one non-empty sentence"),
        (lambda c: c["final_test"].update(readings=["a", "b", "c"]), "exactly one non-empty sentence"),
    ],
)
def test_bad_final_test_settings_are_refused_in_plain_words(edit, message):
    with pytest.raises(ValueError, match=message):
        settings.validate_config(_bad(edit))


def test_the_agreed_rules_are_in_the_config():
    f = CONFIG["final_test"]
    assert f["expected_test_markets"] == 8991 and f["headline_slice"] == "contested" and f["window_percentiles"] == [50, 90, 99]
    assert set(f["readings"]) == {"better", "cannot be told apart", "worse"} and f["readings"]["cannot be told apart"].startswith("No measurable difference")
    assert CONFIG["models"]["test_lock_file"] == "test_opened.lock"
    settings.validate_config(copy.deepcopy(CONFIG))


def test_the_readings_are_keyed_by_exactly_the_verdicts_the_scoring_produces():
    assert {baselines.verdict(-2, -1), baselines.verdict(1, 2), baselines.verdict(-1, 1)} == set(READINGS)


# ---- the lock and the one reader of the test split

def _frozen_bytes(tmp_path):
    p = tmp_path / "frozen_choice.yaml"
    p.write_bytes(b"a: 1\n")                      # bytes, so Windows does not turn the newline into \r\n
    return p


def test_the_lock_records_the_time_the_frozen_hash_and_k(tmp_path):
    frozen = _frozen_bytes(tmp_path)
    when = pd.Timestamp("2026-11-26 09:00", tz="UTC").to_pydatetime()
    path = ft.write_lock(tmp_path / "data", CONFIG, 7, frozen, now=when)
    assert path == tmp_path / "data" / "test_opened.lock"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body == {"opened_at": when.isoformat(), "k": 7, "frozen_file": "frozen_choice.yaml", "frozen_sha256": hashlib.sha256(b"a: 1\n").hexdigest()}


def test_the_lock_cannot_be_written_twice_and_the_first_one_is_kept(tmp_path):
    frozen = _frozen_bytes(tmp_path)
    ft.write_lock(tmp_path / "data", CONFIG, 7, frozen)
    first = (tmp_path / "data" / "test_opened.lock").read_bytes()
    frozen.write_text("a: 2\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ft.write_lock(tmp_path / "data", CONFIG, 7, frozen)
    assert (tmp_path / "data" / "test_opened.lock").read_bytes() == first


def test_the_test_split_cannot_be_read_without_the_lock_and_is_read_with_it(tmp_path):
    data_dir = _split_dir(tmp_path)
    with pytest.raises(PermissionError, match="only be read after the lock file is written"):
        ft.read_test(data_dir, CONFIG, 7)
    ft.write_lock(data_dir, CONFIG, 7, _frozen_bytes(tmp_path))
    assert len(ft.read_test(data_dir, CONFIG, 7)) == 14
    with pytest.raises(PermissionError):                          # the ordinary reader still refuses, lock or no lock
        split.read_split(data_dir, CONFIG, "test", 7)


def test_only_final_test_reads_the_test_split_and_nothing_imports_it():
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name == "final_test.py":
            continue
        assert not re.search(r"^\s*(import|from)\s+final_test", text, re.M), path.name
        assert not re.search(r'split_path\([^)]*"test"', text), path.name
        assert "read_test" not in text, path.name


def test_the_pipeline_and_run_sh_never_open_the_test_split(tmp_path):
    assert "final_test" not in (SRC / "pipeline.py").read_text(encoding="utf-8")
    commands = [l for l in (Path(settings.PROJECT_DIR) / "run.sh").read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#")]
    assert not any("final_test" in l for l in commands)
    lock = settings.DATA_DIR / "test_opened.lock"
    before = lock.read_bytes() if lock.exists() else None   # after the one-time test (25 Nov 2026) the real lock is there, and must stay
    assert pipeline.main(["--sample"]) == 0
    after = lock.read_bytes() if lock.exists() else None
    assert after == before   # a sample run neither creates the lock nor changes or removes an existing one


# ---- is the frozen file committed (a real git repository)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True)


@needs_git
def test_the_frozen_file_must_be_committed_and_unchanged(tmp_path):
    _git(tmp_path, "init", "-q")
    frozen = _frozen_bytes(tmp_path)
    assert "not committed in git" in ft.frozen_committed(frozen, tmp_path)           # untracked
    _git(tmp_path, "add", "frozen_choice.yaml")
    assert "has a change that is not committed" in ft.frozen_committed(frozen, tmp_path)   # staged only
    _git(tmp_path, "commit", "-q", "-m", "x")
    assert ft.frozen_committed(frozen, tmp_path) is None                             # committed and clean
    frozen.write_text("a: 2\n", encoding="utf-8")
    assert "has a change that is not committed" in ft.frozen_committed(frozen, tmp_path)   # edited afterwards


def test_a_missing_git_or_a_git_that_fails_does_not_pass_the_check(tmp_path, monkeypatch):
    frozen = _frozen_bytes(tmp_path)

    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(subprocess, "run", no_git)
    assert "git was not found" in ft.frozen_committed(frozen, tmp_path)

    calls = []

    def half(cmd, **k):
        calls.append(cmd[1])
        return subprocess.CompletedProcess(cmd, 0 if cmd[1] == "ls-files" else 128, stdout="", stderr="fatal: boom")
    monkeypatch.setattr(subprocess, "run", half)
    assert "git could not say" in ft.frozen_committed(frozen, tmp_path) and "boom" in ft.frozen_committed(frozen, tmp_path)
    assert calls[:2] == ["ls-files", "status"]


# ---- fitting the frozen model: train only, the frozen settings

def test_the_frozen_model_is_fitted_on_train_rows_and_labels_only_with_the_frozen_settings(monkeypatch):
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append((list(X.columns), len(X), [int(v) for v in y]))
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.7), np.full(len(X), 0.3)])
    built = []
    monkeypatch.setattr(models, "build_model", lambda name, cfg, columns, params=None: (built.append((name, params)), Spy())[1])
    predict = ft.fit_frozen(TRAIN, CONFIG, "random_forest", {"min_samples_leaf": 50})
    assert built == [("random_forest", {"min_samples_leaf": 50})]
    assert seen == [(models.input_columns(CONFIG)[0], 10, [int(v) for v in TRAIN["label"]])]
    assert predict(VAL).tolist() == [0.3] * 10                     # clipped P(Yes) of the fitted thing, for any table


def test_the_price_only_rung_is_fitted_as_the_price_correction_on_train(monkeypatch):
    seen = []
    monkeypatch.setattr(baselines, "fit_correction", lambda price, y, eps: (seen.append((list(price), list(y), eps)), {"intercept": 0.0, "slope": 1.0})[1])
    predict = ft.fit_frozen(TRAIN, CONFIG, "lr_price_only", {})
    assert seen == [(list(TRAIN["last_close"]), list(TRAIN["label"]), EPS)]
    assert predict(VAL) == pytest.approx([0.5] * 6 + [0.01] * 4, abs=1e-12)


@pytest.mark.parametrize("model,params", [("lr_all_features", {"C": 1.0}), ("random_forest", {"min_samples_leaf": 50}), ("gradient_boosting", {"learning_rate": 0.1, "min_samples_leaf": 100})])
def test_every_kind_of_frozen_model_fits_and_forecasts(model, params):
    train, val = frame(600, seed=0), frame(300, seed=1)
    p = ft.fit_frozen(train, CONFIG, model, params)(val)
    assert p.shape == (300,) and ((p >= EPS) & (p <= 1 - EPS)).all()


def test_the_logistic_frozen_model_is_exactly_the_stage_6_and_7_model():
    train, val = frame(600, seed=0), frame(300, seed=1)
    fitted, columns, _ = models.fit_models(train, CONFIG)
    want = models.forecast_all(fitted, columns, 0.0, val, CONFIG)["lr_all_features"]
    assert ft.fit_frozen(train, CONFIG, "lr_all_features", {"C": 1.0})(val) == pytest.approx(want, abs=1e-9)


# ---- the frozen model must reproduce its validation score

FROZEN_VAL = {"log_loss_all": loss([0.3] * 10, VAL["label"]), "log_loss_contested": loss([0.3] * 6, VAL["label"][:6])}


def test_the_reproduction_reports_the_real_difference_and_both_scores():
    off = FROZEN_VAL["log_loss_all"] + 3e-8
    out = ft.check_reproduces(TRAIN, VAL, CONFIG, const(0.3), dict(FROZEN_VAL, log_loss_all=off))
    assert out["all"]["difference"] == pytest.approx(-3e-8, abs=1e-12) and out["all"]["frozen"] == off and out["all"]["this_run"] == pytest.approx(FROZEN_VAL["log_loss_all"], abs=1e-12)


def test_the_number_of_test_events_is_counted_in_the_test_table_not_validation():
    test = TEST.copy()
    test["event_key"] = ["same"] * 5 + ["other"] * 5
    r = ft.score_test(TRAIN, VAL, test, CONFIG, const(0.3), "lr_all_features", {})
    assert r["test_events"] == 2 and r["test"]["events"] == 2 and r["validation"]["events"] == 10


def test_the_frozen_model_reproducing_its_validation_score_passes_and_is_reported():
    out = ft.check_reproduces(TRAIN, VAL, CONFIG, const(0.3), FROZEN_VAL)
    assert out["all"]["difference"] == pytest.approx(0, abs=1e-12) and out["contested"]["frozen"] == FROZEN_VAL["log_loss_contested"]


@pytest.mark.parametrize("key,sname", [("log_loss_all", "all"), ("log_loss_contested", "contested")])
@pytest.mark.parametrize("sign", [1, -1])
def test_a_frozen_model_that_does_not_reproduce_is_refused_on_either_slice_in_either_direction(key, sname, sign):
    tol = CONFIG["final_test"]["reproduce_tolerance"]
    bad = dict(FROZEN_VAL, **{key: FROZEN_VAL[key] + sign * tol * 2})
    with pytest.raises(ValueError, match=rf"scores \d\.\d+ on validation \({sname}\) but frozen_choice.yaml says .*the test is not opened"):
        ft.check_reproduces(TRAIN, VAL, CONFIG, const(0.3), bad)
    ok = dict(FROZEN_VAL, **{key: FROZEN_VAL[key] + sign * tol / 2})
    ft.check_reproduces(TRAIN, VAL, CONFIG, const(0.3), ok)


def test_the_reproduction_needs_a_validation_score_to_compare_with():
    with pytest.raises(ValueError, match="holds no validation score"):
        ft.check_reproduces(TRAIN, VAL, CONFIG, const(0.3), {})
    out = ft.check_reproduces(TRAIN, VAL, CONFIG, const(0.3), {"log_loss_all": FROZEN_VAL["log_loss_all"]})
    assert "contested" not in out


# ---- known answers

def test_every_forecast_is_scored_by_hand_on_both_slices():
    r = ft.score_test(TRAIN, VAL, TEST, CONFIG, const(0.3), "lr_all_features", {"C": 1.0})
    y = TEST["label"].to_numpy()
    price = np.array([0.5] * 6 + [0.01] * 4)
    for sname, idx in (("all", list(range(10))), ("contested", list(range(6)))):
        sc = r["test"]["slices"][sname]["scores"]
        assert sc["raw_price"]["log_loss"] == pytest.approx(loss(price[idx], y[idx]), abs=1e-12)
        assert sc["price_only_correction"]["log_loss"] == pytest.approx(loss(price[idx], y[idx]), abs=1e-12)      # identity correction
        assert sc["base_rate"]["log_loss"] == pytest.approx(loss([BASE] * len(idx), y[idx]), abs=1e-12)
        assert sc["frozen_model"]["log_loss"] == pytest.approx(loss([0.3] * len(idx), y[idx]), abs=1e-12)
        assert sc["frozen_model"]["brier"] == pytest.approx(float(np.mean((0.3 - y[idx]) ** 2)), abs=1e-12)
        d = r["test"]["slices"][sname]["difference_vs_raw_price"]["frozen_model"]["log_loss"]
        assert d["point"] == pytest.approx(loss([0.3] * len(idx), y[idx]) - loss(price[idx], y[idx]), abs=1e-12)
        assert r["test"]["slices"][sname]["frozen_minus_price_only_correction"]["point"] == pytest.approx(d["point"], abs=1e-12)
    assert r["test"]["slices"]["all"]["markets"] == 10 and r["test"]["slices"]["contested"]["markets"] == 6


def test_the_same_numbers_are_computed_for_validation_beside_the_test():
    r = ft.score_test(TRAIN, VAL, TEST, CONFIG, const(0.3), "lr_all_features", {"C": 1.0})
    y, price = VAL["label"].to_numpy(), np.array([0.5] * 6 + [0.01] * 4)
    d = r["validation"]["slices"]["contested"]["difference_vs_raw_price"]["frozen_model"]["log_loss"]
    assert d["point"] == pytest.approx(loss([0.3] * 6, y[:6]) - loss(price[:6], y[:6]), abs=1e-12)
    assert r["headline"]["validation"]["point"] == pytest.approx(d["point"], abs=1e-12)


def test_only_the_frozen_model_and_the_three_baselines_are_scored_on_test():
    r = ft.score_test(TRAIN, VAL, TEST, CONFIG, const(0.3), "lr_all_features", {"C": 1.0})
    for part in ("test", "validation"):
        assert list(r[part]["slices"]["all"]["scores"]) == ["raw_price", "price_only_correction", "base_rate", "frozen_model"]
    assert "random_forest" not in json.dumps(r["test"]) and "gradient_boosting" not in json.dumps(r["test"])


def test_the_base_rate_and_the_price_correction_come_from_train_not_test(monkeypatch):
    seen = []
    monkeypatch.setattr(baselines, "fit_correction", lambda price, y, eps: (seen.append((list(price), list(y))), {"intercept": 0.0, "slope": 1.0})[1])
    r = ft.score_test(TRAIN, VAL, TEST, CONFIG, const(0.3), "lr_all_features", {})
    assert all(s == (list(TRAIN["last_close"]), list(TRAIN["label"])) for s in seen) and seen
    y = TEST["label"].to_numpy()
    assert r["test"]["slices"]["all"]["scores"]["base_rate"]["log_loss"] == pytest.approx(loss([BASE] * 10, y), abs=1e-12)


def test_the_report_names_what_it_did_and_can_be_written_as_it_is():
    r = ft.score_test(TRAIN, VAL, TEST, CONFIG, const(0.3), "lr_all_features", {"C": 1.0}, reproduced={"all": {}}, frozen_sha="abc", opened_at="2026-11-26T01:00:00+00:00")
    assert r["model"] == "lr_all_features" and r["params"] == {"C": 1.0} and r["fitted_on"].startswith("train only") and "not refitted" in r["fitted_on"]
    assert (r["frozen_sha256"], r["opened_at"], r["reproduces_frozen_on_validation"]) == ("abc", "2026-11-26T01:00:00+00:00", {"all": {}})
    assert (r["train_markets"], r["validation_markets"], r["test_markets"], r["test_events"]) == (10, 10, 10, 10)
    assert r["settings"] == {"seed": 42, "forest_trees": 300, "logistic_max_iter": 1000} and r["rules"]["readings"] == READINGS
    assert any("opened once" in n for n in r["notes"]) and any("choosing after seeing test" in n for n in r["notes"]) and any("not the shift between periods" in n for n in r["notes"])
    json.dumps(r)


def test_the_wobble_uses_the_configured_draws_seed_interval_and_the_same_draws_for_every_pair(monkeypatch):
    seen = {"w": [], "p": [], "s": []}
    real_w, real_p, real_s = baselines.resample_weights, baselines.paired_difference, baselines.score_slice
    monkeypatch.setattr(baselines, "resample_weights", lambda *a: (seen["w"].append(a), real_w(*a))[1])
    monkeypatch.setattr(baselines, "paired_difference", lambda *a, **k: (seen["p"].append(a), real_p(*a, **k))[1])
    monkeypatch.setattr(baselines, "score_slice", lambda *a, **k: (seen["s"].append((a[3], a[6])), real_s(*a, **k))[1])
    ft.score_test(TRAIN, VAL, TEST, CONFIG, const(0.3), "lr_all_features", {})
    ev = CONFIG["evaluation"]
    assert seen["w"] == [(10, ev["bootstrap_draws"], CONFIG["seed"])] * 2        # test, then validation: events 10 each
    assert len(seen["s"]) == 4 and len(seen["p"]) == 4
    assert all(i == ev["interval"] for _, i in seen["s"]) and all(a[7] == ev["interval"] for a in seen["p"])
    assert seen["s"][0][0] is seen["s"][1][0] and seen["p"][0][4] is seen["s"][0][0] and seen["p"][1][4] is seen["s"][0][0]   # one draw set per table


def test_the_calibration_table_is_kept_for_the_frozen_model_and_the_raw_price_on_each_slice():
    r = ft.score_test(TRAIN, VAL, TEST, CONFIG, const(0.3), "lr_all_features", {})
    for sname, n in (("all", 10), ("contested", 6)):
        cal = r["test"]["slices"][sname]["calibration"]
        assert set(cal) == {"frozen_model", "raw_price"} and len(cal["frozen_model"]) == len(CONFIG["metrics"]["calibration_bands"]) - 1
        assert sum(b["markets"] for b in cal["frozen_model"]) == n
        assert [b for b in cal["frozen_model"] if b["markets"]][0]["mean_forecast"] == pytest.approx(0.3)


def test_a_slice_with_no_markets_in_test_is_left_empty_not_invented_and_has_no_headline():
    near = mk([0.01] * 10, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], seed=4)
    r = ft.score_test(TRAIN, VAL, near, CONFIG, const(0.3), "lr_all_features", {})
    assert r["test"]["slices"]["contested"]["markets"] == 0 and r["headline"] is None
    text = "\n".join(ft.summary_lines(r))
    assert "contested: no markets in test" in text and "HEADLINE" not in text


@pytest.mark.parametrize("which", ["train", "validation", "test"])
def test_missing_columns_and_empty_tables_are_refused_in_plain_words(which):
    t = {"train": TRAIN, "validation": VAL, "test": TEST}
    for drop in ("close_time", "t", "event_key"):
        bad = dict(t, **{which: t[which].drop(columns=[drop])})
        with pytest.raises(ValueError, match=f"the {which} table lacks the columns the test needs"):
            ft.score_test(bad["train"], bad["validation"], bad["test"], CONFIG, const(0.3), "m", {})
    bad = dict(t, **{which: t[which].iloc[0:0]})
    with pytest.raises(ValueError, match=f"the {which} table has no markets"):
        ft.score_test(bad["train"], bad["validation"], bad["test"], CONFIG, const(0.3), "m", {})


# ---- the three readings, fixed before the test was opened

def _predict_by_label(good):
    return lambda df: np.where(df["label"].to_numpy() == 1, 0.9 if good else 0.1, 0.1 if good else 0.9)


@pytest.mark.parametrize("predict,verdict", [(_predict_by_label(True), "better"), (_predict_by_label(False), "worse"), (lambda df: df["last_close"].to_numpy(), "cannot be told apart")])
def test_the_headline_is_the_verdict_against_the_raw_price_read_with_the_agreed_sentence(predict, verdict):
    c = copy.deepcopy(CONFIG)
    c["final_test"]["readings"] = {"better": "SENTENCE-BETTER", "cannot be told apart": "SENTENCE-TIE", "worse": "SENTENCE-WORSE"}
    r = ft.score_test(TRAIN, VAL, TEST, c, predict, "lr_all_features", {})
    h = r["headline"]
    assert h["slice"] == "contested" and h["test"]["verdict"] == verdict
    assert h["reading"] == {"better": "SENTENCE-BETTER", "cannot be told apart": "SENTENCE-TIE", "worse": "SENTENCE-WORSE"}[verdict]
    d = r["test"]["slices"]["contested"]["difference_vs_raw_price"]["frozen_model"]["log_loss"]
    assert (h["test"]["point"], h["test"]["low"], h["test"]["high"]) == (d["point"], d["low"], d["high"])
    assert any(h["reading"] in line for line in ft.summary_lines(r))


def test_the_headline_slice_follows_the_config():
    c = copy.deepcopy(CONFIG)
    c["final_test"]["headline_slice"] = "all"
    r = ft.score_test(TRAIN, VAL, TEST, c, const(0.3), "lr_all_features", {})
    assert r["headline"]["slice"] == "all"
    assert r["headline"]["test"]["point"] == r["test"]["slices"]["all"]["difference_vs_raw_price"]["frozen_model"]["log_loss"]["point"]
    assert "HEADLINE (all," in "\n".join(ft.summary_lines(r))


def _view(markets, point, low, high, verdict):
    d = {"point": point, "low": low, "high": high, "verdict": verdict, "extra": 1}
    return {"slices": {"contested": {"markets": markets, "difference_vs_raw_price": {"frozen_model": {"log_loss": d}}}}}


@pytest.mark.parametrize(
    "test_d,val_d,agree",
    [
        ((-0.01, -0.02, -0.005, "better"), (-0.02, -0.03, -0.01, "better"), "agree"),
        ((0.01, -0.01, 0.02, "cannot be told apart"), (0.005, -0.01, 0.02, "cannot be told apart"), "agree"),
        ((0.01, -0.01, 0.02, "cannot be told apart"), (-0.005, -0.02, 0.01, "cannot be told apart"), "same verdict, opposite direction"),
        ((-0.01, -0.02, -0.005, "better"), (0.02, 0.01, 0.03, "worse"), "different verdicts"),
        ((0.01, -0.01, 0.02, "cannot be told apart"), (0.02, 0.01, 0.03, "worse"), "different verdicts"),
    ],
)
def test_validation_and_test_are_compared_and_the_disagreement_is_named(test_d, val_d, agree):
    h = ft.headline(_view(5, *test_d), _view(5, *val_d), CONFIG)
    assert h["validation_and_test"] == agree and h["test"] == dict(zip(("point", "low", "high", "verdict"), test_d)) and h["validation"]["verdict"] == val_d[3]
    assert "extra" not in h["test"]


def test_a_headline_without_validation_markets_says_nothing_about_validation_and_none_without_test_markets():
    h = ft.headline(_view(5, 0.0, -1, 1, "cannot be told apart"), _view(0, 0, 0, 0, "x"), CONFIG)
    assert "validation" not in h and "validation_and_test" not in h
    assert ft.headline(_view(0, 0, 0, 0, "x"), _view(5, 0, 0, 0, "x"), CONFIG) is None


def test_the_printout_says_which_of_the_three_validation_and_test_cases_it_is():
    r = ft.score_test(TRAIN, VAL, TEST, CONFIG, const(0.3), "lr_all_features", {})
    for state, sentence in (("agree", "Validation and test agree."), ("same verdict, opposite direction", "point estimates point in opposite directions"),
                            ("different verdicts", "give DIFFERENT verdicts")):
        r["headline"]["validation_and_test"] = state
        assert sentence in "\n".join(ft.summary_lines(r))


# ---- the test-window check (censoring), by hand

def test_the_window_check_is_computed_by_hand():
    df = mk([0.5, 0.5, 0.9, 0.01], [1, 0, 0, 0], close_days=[0.5, 1.0, 2.0, 10.0], category=["A", "A", None, "B"])
    w = ft.window_check({"test": df}, CONFIG)["test"]
    days = [0.5, 1.0, 2.0, 10.0]
    assert w["days_t_to_close"] == {"p50": pytest.approx(np.percentile(days, 50)), "p90": pytest.approx(np.percentile(days, 90)), "p99": pytest.approx(np.percentile(days, 99))}
    assert w["days_t_to_close"]["p50"] == pytest.approx(1.5) and w["days_t_to_close"]["p90"] == pytest.approx(7.6)
    assert w["share_closing_within_1_day"] == 0.5                     # 0.5 and exactly 1.0 count
    assert (w["markets"], w["yes_rate"], w["share_contested"], w["mean_last_close"]) == (4, 0.25, 0.75, pytest.approx((0.5 + 0.5 + 0.9 + 0.01) / 4))
    assert w["category_share"] == {"A": 0.5, "(missing)": 0.25, "B": 0.25}
    assert w["t_first"].startswith("2025-02-01T00:00") and w["t_last"].startswith("2025-02-01T03:00")


def test_the_window_check_uses_the_configured_percentiles_and_covers_every_part():
    c = copy.deepcopy(CONFIG)
    c["final_test"]["window_percentiles"] = [25, 75.5]
    w = ft.window_check({"train": TRAIN, "validation": VAL, "test": TEST}, c)
    assert list(w) == ["train", "validation", "test"] and list(w["test"]["days_t_to_close"]) == ["p25", "p75.5"]
    r = ft.score_test(TRAIN, VAL, TEST, CONFIG, const(0.3), "lr_all_features", {})
    assert list(r["window_check"]) == ["train", "validation", "test"]


def test_a_short_window_and_a_long_window_are_told_apart():
    quick = mk([0.5] * 4, [1, 0, 1, 0], close_days=[0.1, 0.2, 0.3, 0.4])
    slow = mk([0.5] * 4, [1, 0, 1, 0], close_days=[5, 6, 7, 8])
    w = ft.window_check({"quick": quick, "slow": slow}, CONFIG)
    assert w["quick"]["share_closing_within_1_day"] == 1.0 and w["slow"]["share_closing_within_1_day"] == 0.0
    assert w["quick"]["days_t_to_close"]["p50"] < 1 < w["slow"]["days_t_to_close"]["p50"]


# ---- the printout

def test_the_printout_shows_the_headline_every_forecast_the_calibration_and_the_window_check():
    r = ft.score_test(TRAIN, VAL, TEST, CONFIG, const(0.3), "lr_all_features", {"C": 1.0}, reproduced=ft.check_reproduces(TRAIN, VAL, CONFIG, const(0.3), FROZEN_VAL))
    text = "\n".join(ft.summary_lines(r))
    assert "FINAL TEST on 10 markets (10 events). Model: lr_all_features (C=1)" in text and "fitted on train only (10 markets). Nothing was retuned." in text
    assert "the middle 90% of 200 re-drawn sets of whole events" in text
    assert "the frozen model reproduced its validation score" in text and "HEADLINE (contested, frozen model minus raw price, log loss)" in text
    for n in ft.NAMES:
        assert re.search(rf"^    {n} ", text, re.M)
    assert f"{loss([0.3] * 10, TEST['label']):.4f}" in text and f"{loss([0.3] * 6, TEST['label'][:6]):.4f}" in text
    assert text.count("frozen_model minus price_only_correction") == 2 and text.count("calibration, frozen model") == 2
    for part in ("train", "validation", "test"):
        assert re.search(rf"^    {part} +10 markets, t 2025-02-01 to 2025-02-01; p50", text, re.M)
    assert "within a day" in text and "leans towards quick-closing markets" in text and "opened once: this was the only run" in text
    assert "Validation, same slice:" in text


def test_the_plan_printed_without_the_flag_lists_the_steps_and_the_no_second_run_rule(tmp_path):
    text = "\n".join(ft.plan_lines(CONFIG, 7, tmp_path))
    assert "opens the test split ONCE" in text and "Nothing was opened or written" in text and "8,991" in text
    assert str(tmp_path / "test_opened.lock") in text and "frozen_choice.yaml" in text and "No second run" in text and "not scored on test" in text


# ---- the command line

@pytest.fixture
def project(tmp_path, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    cfg = copy.deepcopy(CONFIG)
    cfg["final_test"]["expected_test_markets"] = 14
    monkeypatch.setattr(settings, "load_config", lambda: cfg)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "split_report_k7.json").write_text(json.dumps({"final": {"sizes": {"train": 36, "validation": 11, "test": 14}}}), encoding="utf-8")
    train, val = split.read_split(data_dir, cfg, "train", 7), split.read_split(data_dir, cfg, "validation", 7)
    predict = ft.fit_frozen(train, cfg, "lr_all_features", {"C": 1.0})
    p = predict(val)
    masks = baselines.slice_masks(val, cfg)
    _frozen_file(tmp_path / "frozen_choice.yaml", baselines.log_loss(val["label"].to_numpy(dtype=float), p), params={"C": 1.0})
    # _frozen_file writes a contested score of 0.5; replace it by the true one so both slices reproduce
    text = (tmp_path / "frozen_choice.yaml").read_text(encoding="utf-8")
    true_c = baselines.log_loss(val["label"].to_numpy(dtype=float)[masks["contested"]], p[masks["contested"]])
    (tmp_path / "frozen_choice.yaml").write_text(text.replace("0.5", repr(true_c)), encoding="utf-8")
    monkeypatch.setattr(ft, "frozen_committed", lambda path, cwd: None)
    return tmp_path, data_dir


def _opened(tmp_path, data_dir):
    return (data_dir / "test_opened.lock").exists() or (tmp_path / "outputs" / "final_test_k7.json").exists()


def test_without_the_flag_nothing_is_read_or_written_and_the_plan_is_printed(project, capsys, monkeypatch):
    tmp_path, data_dir = project
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("nothing may be read without --confirm-open-test"))
    assert ft.main(["--data-dir", str(data_dir)]) == 2
    out = capsys.readouterr().out
    assert "opens the test split ONCE" in out and not _opened(tmp_path, data_dir)


def test_the_opening_in_order_train_validation_then_the_lock_then_the_test_and_the_report(project, capsys, monkeypatch):
    tmp_path, data_dir = project
    order = []
    real_read, real_lock, real_test = split.read_split, ft.write_lock, ft.read_test
    monkeypatch.setattr(split, "read_split", lambda d, c, part, k: (order.append(part), real_read(d, c, part, k))[1])
    monkeypatch.setattr(ft, "write_lock", lambda *a, **k: (order.append("lock"), real_lock(*a, **k))[1])
    monkeypatch.setattr(ft, "read_test", lambda *a, **k: (order.append("test"), real_test(*a, **k))[1])
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir)]) == 0
    assert order == ["train", "validation", "lock", "test"]
    out = capsys.readouterr().out
    assert "Lock written" in out and out.index("Lock written") < out.index("FINAL TEST") and "opened once: this was the only run" in out
    report = json.loads((tmp_path / "outputs" / "final_test_k7.json").read_text(encoding="utf-8"))
    lock = json.loads((data_dir / "test_opened.lock").read_text(encoding="utf-8"))
    assert report["test_markets"] == 14 and report["model"] == "lr_all_features" and report["params"] == {"C": 1.0}
    assert report["opened_at"] == lock["opened_at"] and report["frozen_sha256"] == lock["frozen_sha256"] == hashlib.sha256((tmp_path / "frozen_choice.yaml").read_bytes()).hexdigest()
    assert report["reproduces_frozen_on_validation"]["all"]["difference"] == pytest.approx(0, abs=1e-9)


def test_there_is_no_second_run_and_the_first_result_is_kept(project, capsys, monkeypatch):
    tmp_path, data_dir = project
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir)]) == 0
    report, lock = (tmp_path / "outputs" / "final_test_k7.json").read_bytes(), (data_dir / "test_opened.lock").read_bytes()
    capsys.readouterr()
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("a second run must read nothing"))
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir)]) == 1
    assert "There is no second run" in capsys.readouterr().out
    assert (tmp_path / "outputs" / "final_test_k7.json").read_bytes() == report and (data_dir / "test_opened.lock").read_bytes() == lock


def test_freeze_refuses_once_the_test_has_been_opened(project):
    tmp_path, data_dir = project
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir)]) == 0
    (tmp_path / "outputs" / "tuning_k7.json").write_text("{}", encoding="utf-8")      # so the lock, not a missing report, is what stops it
    before = (tmp_path / "frozen_choice.yaml").read_bytes()
    assert freeze.main(["--data-dir", str(data_dir)]) == 1
    assert (tmp_path / "frozen_choice.yaml").read_bytes() == before


def test_main_never_changes_the_frozen_file(project):
    tmp_path, data_dir = project
    before = (tmp_path / "frozen_choice.yaml").read_bytes()
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir)]) == 0
    assert (tmp_path / "frozen_choice.yaml").read_bytes() == before


def _refused(project, capsys, code, text):
    tmp_path, data_dir = project
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir)]) == code
    out = capsys.readouterr().out
    assert text in out and not _opened(tmp_path, data_dir)       # nothing was opened


def test_nothing_is_opened_without_a_frozen_file(project, capsys):
    (project[0] / "frozen_choice.yaml").unlink()
    _refused(project, capsys, 1, "Nothing was opened.")


def test_nothing_is_opened_with_a_frozen_file_that_fails_its_checks(project, capsys):
    p = project[0] / "frozen_choice.yaml"
    p.write_text(p.read_text(encoding="utf-8").replace("not opened", "opened"), encoding="utf-8")
    _refused(project, capsys, 1, "Nothing was opened.")


def test_nothing_is_opened_when_the_frozen_file_is_not_committed(project, capsys, monkeypatch):
    monkeypatch.setattr(ft, "frozen_committed", lambda path, cwd: "frozen_choice.yaml is not committed in git. It must be committed before the test split is opened.")
    _refused(project, capsys, 1, "is not committed in git")


def test_nothing_is_opened_without_a_split_report(project, capsys):
    (project[0] / "outputs" / "split_report_k7.json").unlink()
    _refused(project, capsys, 1, "there is no split report")


def test_nothing_is_opened_when_the_split_report_gives_another_test_size(project, capsys):
    (project[0] / "outputs" / "split_report_k7.json").write_text(json.dumps({"final": {"sizes": {"test": 9071}}}), encoding="utf-8")
    _refused(project, capsys, 1, "says the test split has 9,071 markets, not the 14 that were agreed")


def test_nothing_is_opened_when_the_frozen_model_does_not_reproduce(project, capsys):
    p = project[0] / "frozen_choice.yaml"
    text = p.read_text(encoding="utf-8")
    p.write_text(re.sub(r"log_loss_all: [0-9.]+", "log_loss_all: 0.1", text), encoding="utf-8")
    _refused(project, capsys, 1, "the test is not opened")


@pytest.mark.parametrize("part", ["train", "validation", "test"])
def test_nothing_is_opened_when_a_split_file_is_missing(project, capsys, part):
    tmp_path, data_dir = project
    split.split_path(data_dir, CONFIG, part, 7).unlink()
    _refused(project, capsys, 2, f"No {part} split")


def test_k_is_the_one_given(project, capsys):
    tmp_path, data_dir = project
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir), "--k", "3"]) == 2
    assert "No train split" in capsys.readouterr().out and not _opened(tmp_path, data_dir)


def test_a_lock_that_appears_while_the_run_starts_stops_it_before_the_test_is_read(project, capsys, monkeypatch):
    tmp_path, data_dir = project
    monkeypatch.setattr(ft, "write_lock", lambda *a, **k: (_ for _ in ()).throw(FileExistsError("x")))
    monkeypatch.setattr(ft, "read_test", lambda *a, **k: pytest.fail("the test split must not be read"))
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir)]) == 1
    assert "appeared while this run was starting" in capsys.readouterr().out and not (tmp_path / "outputs" / "final_test_k7.json").exists()


def test_a_crash_after_the_lock_keeps_the_lock_says_what_to_do_and_writes_no_report(project, capsys, monkeypatch):
    tmp_path, data_dir = project
    monkeypatch.setattr(ft, "score_test", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir)]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM after the lock was written: RuntimeError: boom" in out and "The lock stays." in out and "decision_log.md" in out
    assert (data_dir / "test_opened.lock").exists() and not (tmp_path / "outputs" / "final_test_k7.json").exists()
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir)]) == 1       # and a rerun is refused until the lock is deleted by hand


def test_a_test_file_that_disagrees_with_the_split_report_stops_the_run_after_the_lock(project, capsys, monkeypatch):
    tmp_path, data_dir = project
    real = ft.read_test
    monkeypatch.setattr(ft, "read_test", lambda *a, **k: real(*a, **k).iloc[:5])
    assert ft.main(["--confirm-open-test", "--data-dir", str(data_dir)]) == 1
    out = capsys.readouterr().out
    assert "holds 5 markets but the split report says 14" in out and (data_dir / "test_opened.lock").exists() and not (tmp_path / "outputs" / "final_test_k7.json").exists()
