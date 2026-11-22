"""Tests for the bootstrap wobble measurement, on small made-up data."""

import numpy as np
import pytest

import uncertainty


def make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    informative = y * 0.4 + rng.random(n)      # a model that does carry some signal
    noise = rng.random(n)                       # a model with no signal
    return y, {"informative": informative, "noise": noise}


def test_the_same_seed_gives_the_same_result():
    y, scores = make_data()
    first = uncertainty.bootstrap(y, scores, k=40, n_draws=30, seed=1)
    second = uncertainty.bootstrap(y, scores, k=40, n_draws=30, seed=1)
    assert first == second


def test_ranges_are_ordered_low_median_high():
    y, scores = make_data()
    out = uncertainty.bootstrap(y, scores, k=40, n_draws=50, seed=1)
    for name in scores:
        r = out["pr_auc"][name]
        assert r["low"] <= r["median"] <= r["high"]


def test_a_perfect_model_has_no_wobble():
    y, _ = make_data()
    out = uncertainty.bootstrap(y, {"perfect": y.astype(float)}, k=40, n_draws=30, seed=1)
    r = out["pr_auc"]["perfect"]
    assert r["low"] == r["high"] == 1.0


def test_an_imperfect_model_does_wobble():
    y, scores = make_data()
    out = uncertainty.bootstrap(y, scores, k=40, n_draws=50, seed=1)
    r = out["pr_auc"]["informative"]
    assert r["high"] > r["low"]   # re-drawn exams really differ from each other


def test_the_reference_model_compared_with_itself_is_exactly_zero():
    y, scores = make_data()
    out = uncertainty.bootstrap(y, {"a": scores["informative"], "b": scores["informative"]}, k=40, n_draws=30, seed=1)
    d = out["pr_auc_minus_reference"]["b"]
    assert d["low"] == d["median"] == d["high"] == 0.0


def test_a_model_with_signal_beats_one_without():
    y, scores = make_data()
    out = uncertainty.bootstrap(y, scores, k=40, n_draws=100, seed=1, reference="noise")
    d = out["pr_auc_minus_reference"]["informative"]
    assert d["low"] > 0
    assert uncertainty.verdict(d) == "better in nearly every re-drawn exam"


def test_verdict_says_cannot_tell_when_the_range_includes_zero():
    assert uncertainty.verdict({"low": -0.01, "median": 0.0, "high": 0.02}).startswith("cannot be told apart")
    assert uncertainty.verdict({"low": -0.05, "median": -0.03, "high": -0.01}).startswith("worse")
