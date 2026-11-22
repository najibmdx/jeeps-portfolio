"""Tests for the top-10% rule, recall, the always-no baseline, and train.run."""

import numpy as np
import pytest

import evaluate
import train
from test_ablation import make_frame


def test_top_k_rounds_up_and_is_never_zero():
    assert evaluate.top_k_from_fraction(1000, 0.10) == 100
    assert evaluate.top_k_from_fraction(5793, 0.10) == 580   # 579.3 rounds up
    assert evaluate.top_k_from_fraction(3, 0.10) == 1


def test_recall_at_k_counts_the_share_of_all_yes_clients_reached():
    y = np.array([1, 1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    assert evaluate.recall_at_k(y, scores, 2) == pytest.approx(2 / 3)   # top 2 hold 2 of the 3 yes
    assert evaluate.recall_at_k(y, scores, 6) == pytest.approx(1.0)     # phoning everyone reaches everyone


def test_recall_is_refused_when_nobody_said_yes():
    with pytest.raises(ValueError):
        evaluate.recall_at_k(np.zeros(5), np.arange(5), 2)


def test_always_no_looks_accurate_but_finds_nobody():
    out = evaluate.always_no_baseline(np.array([1, 0, 0, 0]))
    assert out["accuracy"] == 0.75
    assert out["precision"] == 0.0 and out["recall"] == 0.0
    assert out["pr_auc_blind_guess"] == 0.25


def small_config():
    return {
        "seed": 1,
        "features": {"groups": ["client", "contact", "history"]},
        "models": {"logistic_regression": {"max_iter": 500}},
        "evaluation": {"top_fraction": 0.10, "bootstrap_draws": 30},
    }


def test_run_uses_the_top_10_percent_of_validation_and_never_reads_test():
    sets = {"train": make_frame(1500, 0), "validation": make_frame(600, 1), "test": None}   # test would crash
    metrics = train.run(small_config(), sets)
    assert metrics["top_k"] == 60            # 10% of 600 validation clients
    m = metrics["models"]["logistic_regression"]
    assert {"validation_precision_at_k", "validation_recall_at_k", "validation_lift_at_k"} <= set(m)
    assert metrics["always_no"]["recall"] == 0.0


def test_precision_and_recall_at_k_agree_with_each_other():
    """precision x k = yes found = recall x all yes. Same count, two views."""
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(int)
    s = y * 0.3 + rng.random(500)
    k = 50
    found_from_precision = evaluate.precision_at_k(y, s, k) * k
    found_from_recall = evaluate.recall_at_k(y, s, k) * y.sum()
    assert found_from_precision == pytest.approx(found_from_recall)
