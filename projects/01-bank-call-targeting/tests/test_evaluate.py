"""Tests for the scoring functions, on tiny examples worked out by hand."""

import numpy as np
import pytest

import evaluate


def test_a_perfect_ranking_has_pr_auc_one():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert evaluate.pr_auc(y, scores) == pytest.approx(1.0)


def test_a_blind_guesser_scores_the_yes_rate():
    y = [0, 0, 0, 1]            # yes-rate 0.25
    same_score_for_everyone = [0.5, 0.5, 0.5, 0.5]
    assert evaluate.pr_auc(y, same_score_for_everyone) == pytest.approx(0.25)


def test_precision_at_k_looks_only_at_the_top_k():
    y = [1, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    assert evaluate.precision_at_k(y, scores, 2) == pytest.approx(0.5)   # top 2: yes, no
    assert evaluate.precision_at_k(y, scores, 3) == pytest.approx(2 / 3)  # top 3: yes, no, yes


def test_lift_is_precision_over_the_yes_rate():
    y = [1, 0, 1, 0, 0]           # yes-rate 0.4
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    assert evaluate.lift_at_k(y, scores, 2) == pytest.approx(0.5 / 0.4)


def test_ranking_order_not_score_size_decides_the_result():
    y = np.array([1, 0, 1, 0])
    assert evaluate.pr_auc(y, [0.9, 0.1, 0.8, 0.2]) == evaluate.pr_auc(y, [90, 10, 80, 20])
