"""Tune the models with time-aware practice exams, using the TRAINING set only.

Tuning = try different settings for a model and keep the ones that score best.
The danger is fooling ourselves, so:

1. Practice exams (folds) are cut by month with a growing window. Fold 1 learns on the
   first `min_train_months` months and is tested on the next whole month; fold 2 learns on
   one more month and is tested on the one after, and so on. A learning row is always
   older than every exam row, and no month is shared (Decision 1 of this project).
2. Every setting combination in config.yaml is scored on the SAME folds, with PR-AUC.
3. The best combination is the one with the highest mean score over the folds.
4. Then, once, default and tuned settings are fitted on the whole training set and scored on
   the validation set. Whether tuning helped is judged there, with the bootstrap range.
   If the range includes 0, we say tuning did not clearly help.

The test set is not read.

Limit: with only a few folds of uneven size, the fold scores are noisy. The mean over folds
and the number of folds that improved are both shown, so a lucky fold cannot hide.
"""

import itertools
import json

import numpy as np

import columns
import evaluate
import features
import models
import split
import train
import uncertainty
from settings import load_config


def month_folds(df, min_train_months: int) -> list:
    """Growing-window folds by month. df must have a `period` column (YYYYMM) in time order.

    Returns a list of (learning row positions, exam row positions).
    """
    periods = sorted(df["period"].unique())
    if len(periods) <= min_train_months:
        raise ValueError(f"{len(periods)} months of data is too few for min_train_months={min_train_months}")
    values = df["period"].to_numpy()
    return [
        (np.flatnonzero(np.isin(values, periods[:i])), np.flatnonzero(values == periods[i]))
        for i in range(min_train_months, len(periods))
    ]


def expand_grid(grid: dict) -> list:
    """Every combination of the listed values, in a fixed order."""
    names = list(grid)
    return [dict(zip(names, values)) for values in itertools.product(*grid.values())]


def cv_scores(df, y, cols, model_name: str, params: dict, seed: int, folds: list) -> list:
    """PR-AUC of one settings combination on every fold."""
    scores = []
    for learn, exam in folds:
        if y.iloc[exam].sum() == 0:
            raise ValueError("a practice exam has no yes clients, so PR-AUC is undefined")
        model = models.build_model(model_name, cols, params, seed)
        model.fit(df.iloc[learn][cols], y.iloc[learn])
        proba = model.predict_proba(df.iloc[exam][cols])[:, 1]
        scores.append(evaluate.pr_auc(y.iloc[exam], proba))
    return scores


def _summary(scores) -> dict:
    return {
        "cv_mean": round(float(np.mean(scores)), 4),
        "cv_sd": round(float(np.std(scores, ddof=1)), 4),
        "cv_scores": [round(float(s), 4) for s in scores],
    }


def run(config: dict, sets: dict) -> dict:
    """Tune every model. Reads sets["train"] to tune and sets["validation"] to judge. Not test."""
    settings = config["tuning"]
    train_df, valid_df = sets["train"], sets["validation"]
    cols = columns.feature_columns(config["features"]["groups"])
    y_train, y_valid = features.make_target(train_df), features.make_target(valid_df)
    seed = config["seed"]
    folds = month_folds(train_df, settings["min_train_months"])
    k = evaluate.top_k_from_fraction(len(y_valid), config["evaluation"]["top_fraction"])
    draws = config["evaluation"]["bootstrap_draws"]

    result = {
        "fold_exam_months": [int(train_df["period"].iloc[exam[0]]) for _, exam in folds],
        "models": {},
    }
    for name, base in config["models"].items():
        default_summary = _summary(cv_scores(train_df, y_train, cols, name, base, seed, folds))

        tried = []
        for combo in expand_grid(settings["grids"][name]):
            scores = cv_scores(train_df, y_train, cols, name, {**base, **combo}, seed, folds)
            tried.append({"params": combo, **_summary(scores)})
        best = max(tried, key=lambda row: row["cv_mean"])       # first best if there is a tie
        best_params = {**base, **best["params"]}

        valid_scores = {}
        for label, params in (("default", base), ("tuned", best_params)):
            model = models.build_model(name, cols, params, seed)
            model.fit(train_df[cols], y_train)
            valid_scores[label] = model.predict_proba(valid_df[cols])[:, 1]
        boot = uncertainty.bootstrap(y_valid, valid_scores, k, draws, seed, reference="default")
        diff = boot["pr_auc_minus_reference"]["tuned"]

        result["models"][name] = {
            "default_params": base,
            "best_params": best["params"],
            "default_cv": default_summary,
            "tuned_cv": {k_: best[k_] for k_ in ("cv_mean", "cv_sd", "cv_scores")},
            # both lists are rounded the same way, so identical settings never count as an improvement
            "folds_improved": int(sum(t > d for t, d in zip(best["cv_scores"], default_summary["cv_scores"]))),
            "n_folds": len(folds),
            "validation_pr_auc": {
                "default": round(evaluate.pr_auc(y_valid, valid_scores["default"]), 4),
                "tuned": round(evaluate.pr_auc(y_valid, valid_scores["tuned"]), 4),
            },
            "validation_change": {**diff, "verdict": uncertainty.verdict(diff)},
            "all_tried": tried,
        }
    return result


def main() -> None:
    config = load_config()
    result = run(config, split.prepare(config))

    months = ", ".join(str(m) for m in result["fold_exam_months"])
    print(f"Practice exams (one whole month each): {months}")
    print("Scores are PR-AUC. Tuned settings are chosen on the training folds only.\n")
    for name, r in result["models"].items():
        d, t, v = r["default_cv"], r["tuned_cv"], r["validation_pr_auc"]
        c = r["validation_change"]
        print(name)
        print(f"  best settings tried:      {r['best_params']}")
        print(f"  practice exams, default:  {d['cv_mean']:.3f} +/- {d['cv_sd']:.3f}")
        print(f"  practice exams, tuned:    {t['cv_mean']:.3f} +/- {t['cv_sd']:.3f}   "
              f"(better in {r['folds_improved']} of {r['n_folds']} folds)")
        print(f"  validation, default -> tuned: {v['default']:.3f} -> {v['tuned']:.3f}   "
              f"change {c['median']:+.3f} ({c['low']:+.3f} to {c['high']:+.3f})  {c['verdict']}\n")

    train.OUTPUT_DIR.mkdir(exist_ok=True)
    path = train.OUTPUT_DIR / "tuning_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
