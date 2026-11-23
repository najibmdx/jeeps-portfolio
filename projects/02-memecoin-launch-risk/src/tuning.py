"""Time-ordered tuning: pick model settings using only the TRAIN pile.

Train is cut into consecutive blocks (oldest first). Trial k trains on the blocks before k and is
graded on block k, so settings are always judged on coins that come AFTER the coins they learned from.
A coin's label is only known 20 minutes after migration, so coins that migrated in the last 20
minutes before a graded block starts are dropped from that trial's training rows.
"""

import itertools

import numpy as np

import metrics
import models


def time_ordered(pile):
    """The pile sorted by migration time (stable, so ties keep their saved order)."""
    return pile.sort_values("migration_time", kind="stable").reset_index(drop=True)


def make_folds(migration_time, n_blocks: int, window_seconds: float):
    """Expanding-window folds over a time-ordered pile. Returns a list of (train_idx, grade_idx)."""
    t = np.asarray(migration_time)
    if np.any(np.diff(t) < 0):
        raise ValueError("the pile must be sorted by migration time first")
    if n_blocks < 2:
        raise ValueError("need at least 2 blocks")
    edges = np.linspace(0, len(t), n_blocks + 1).astype(int)
    folds = []
    for k in range(1, n_blocks):
        grade = np.arange(edges[k], edges[k + 1])
        start = t[grade[0]]
        train = np.arange(0, edges[k])
        train = train[t[train] <= start - window_seconds]     # label must be known when the block starts
        folds.append((train, grade))
    return folds


def grid_points(grid: dict) -> list:
    """Every combination of the grid, in a fixed order."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def fold_score(y_true, crash_scores) -> dict:
    """Times better than the lazy guess on each side (unrounded), and their plain average."""
    y = np.asarray(y_true)
    s = np.asarray(crash_scores, dtype=float)
    if y.min() == y.max():
        raise ValueError("a graded block needs both crashes and survivors")
    crash = metrics.pr_auc(y, s) / y.mean()
    survivor = metrics.pr_auc(1 - y, 1 - s) / (1 - y).mean()
    return {"crash": crash, "survivor": survivor, "average": (crash + survivor) / 2}


def tune_model(name: str, base_params: dict, grid: dict, X, y, folds, seed: int) -> dict:
    """Try every grid point on every fold. The best average score wins; ties keep the earliest point."""
    y = np.asarray(y)
    trials = []
    best = None
    for point in grid_points(grid):
        params = {**base_params, **point}
        per_fold = []
        for train_idx, grade_idx in folds:
            if y[train_idx].min() == y[train_idx].max():
                raise ValueError("a training block needs both crashes and survivors")
            model = models.build_model(name, params, seed)
            model.fit(X.iloc[train_idx], y[train_idx])
            per_fold.append(fold_score(y[grade_idx], model.predict_proba(X.iloc[grade_idx])[:, 1]))
        trial = {
            "grid_point": point,
            "score": float(np.mean([f["average"] for f in per_fold])),
            "crash": float(np.mean([f["crash"] for f in per_fold])),
            "survivor": float(np.mean([f["survivor"] for f in per_fold])),
        }
        trials.append(trial)
        if best is None or trial["score"] > best["score"]:
            best = trial
    return {"best": best["grid_point"], "best_score": best["score"], "trials": trials}
