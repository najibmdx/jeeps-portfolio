"""Scores that tell us how well a model ranks clients.

Every score here is read next to the blind-guess baseline of the SAME set:
for PR-AUC the baseline is the share of yes in that set.
"""

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
