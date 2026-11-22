"""Scores that tell us how well a model ranks clients.

Every score here is read next to the blind-guess baseline of the SAME set:
for PR-AUC the baseline is the share of yes in that set.
"""

import math

import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc(y_true, scores) -> float:
    """Precision-recall area (average precision). A blind guesser scores the yes-rate."""
    return float(average_precision_score(y_true, scores))


def precision_at_k(y_true, scores, k: int) -> float:
    """Phone the k clients with the highest scores: what share of them said yes?"""
    y = np.asarray(y_true)
    order = np.argsort(-np.asarray(scores), kind="stable")
    return float(y[order[:k]].mean())


def lift_at_k(y_true, scores, k: int) -> float:
    """How many times better than phoning k clients at random."""
    baseline = float(np.mean(y_true))
    return precision_at_k(y_true, scores, k) / baseline


def top_k_from_fraction(n_clients: int, fraction: float) -> int:
    """How many clients are in the top `fraction` of n_clients. Rounds up, never less than 1."""
    return max(1, math.ceil(round(n_clients * fraction, 9)))


def recall_at_k(y_true, scores, k: int) -> float:
    """Phone the k clients with the highest scores: what share of ALL the yes clients did we reach?"""
    y = np.asarray(y_true)
    if y.sum() == 0:
        raise ValueError("recall is undefined when nobody said yes")
    order = np.argsort(-np.asarray(scores), kind="stable")
    return float(y[order[:k]].sum() / y.sum())


def always_no_baseline(y_true) -> dict:
    """The lazy model that predicts 'no' for everyone and phones nobody.

    It looks good on accuracy and is useless: it finds no yes at all.
    """
    yes_rate = float(np.mean(y_true))
    return {"accuracy": 1 - yes_rate, "precision": 0.0, "recall": 0.0, "pr_auc_blind_guess": yes_rate}
