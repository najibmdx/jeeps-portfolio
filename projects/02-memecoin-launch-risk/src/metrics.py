"""Scores that compare a scorer with the lazy guess, on both sides.

Two sides, because a trader cares about both:
  crash side     find the coins that crash          (the class we predict, 1)
  survivor side  find the coins that do NOT crash   (the rare group, 0)
The survivor side uses the same scorer, flipped: a coin the scorer thinks is least likely
to crash is the one it recommends as a survivor.

Every score is shown next to its baseline: the lazy guess "everything is this side".
For PR-AUC, the lazy guess scores exactly the share of that side in the pile.
"""

import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve. A scorer with no information gets the share of 1s."""
    return float(average_precision_score(y_true, scores))


def precision_at_fraction(y_true, scores, fraction: float) -> float:
    """Of the coins the scorer is most sure about (the top `fraction`), the share that really are 1.

    Ties are broken by row order (oldest coin first), so the result never depends on chance.
    """
    y = np.asarray(y_true)
    n = len(y)
    k = max(1, int(round(fraction * n)))
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(y[order].mean())


def side_metrics(y_true, crash_scores, fraction: float) -> dict:
    """PR-AUC and precision in the top fraction, each next to the lazy-guess baseline, for both sides."""
    y = np.asarray(y_true)
    s = np.asarray(crash_scores, dtype=float)
    if len(y) == 0 or y.min() == y.max():
        raise ValueError("both crashes and survivors are needed to score a pile")
    result = {}
    for side, (target, score) in {"crash": (y, s), "survivor": (1 - y, 1 - s)}.items():
        baseline = float(target.mean())
        area = pr_auc(target, score)
        precision = precision_at_fraction(target, score, fraction)
        result[side] = {
            "baseline": round(baseline, 4),
            "pr_auc": round(area, 4),
            "pr_auc_times_baseline": round(area / baseline, 2),
            "precision_top": round(precision, 4),
            "lift_top": round(precision / baseline, 2),
        }
    return result
