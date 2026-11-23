"""The one-time test run.

Rules, all decided BEFORE looking at the test set:
- The model and settings come from config.yaml ("final"). They are not changed after the run.
- The model is fitted on the training set only.
- The cut-off for each cost scenario (the share of clients to phone) is chosen on the
  VALIDATION set with cutoff.best_share. The test set only ever grades that choice.
- The script refuses to run a second time. A test set that has been looked at, and then
  used to change something, is no longer an honest exam.

Read the results with care:
- The yes-rate is very different in each period (train 6.3%, validation 9.0%, test 44.5%).
  Precision and net value on test are therefore NOT comparable with validation. The
  PR-AUC has to be read next to the test yes-rate, which is what a blind guess scores.
- Costs are the invented assumptions from config.yaml, not the bank's real costs.
"""

import json

import numpy as np

import columns
import cutoff
import evaluate
import features
import models
import split
import train
import uncertainty
from settings import load_config

RESULT_FILE = "test_results.json"


def check_not_run_before(path) -> None:
    """Refuse to touch the test set a second time."""
    if path.exists():
        raise RuntimeError(
            f"{path.name} already exists: the test set has been used once. "
            "Running again after seeing the result would make the test set part of the tuning."
        )


def final_model(config: dict) -> tuple:
    """Name and full settings of the final model: the base settings plus the chosen ones."""
    name = config["final"]["model"]
    return name, {**config["models"][name], **config["final"].get("params", {})}


def at_share(y_true, scores, share: float, cost_per_call: float) -> dict:
    """Phone the top `share` of clients by score: what happens on THIS set?"""
    y = np.asarray(y_true)
    n = len(y)
    k = evaluate.top_k_from_fraction(n, share) if share > 0 else 0
    order = np.argsort(-np.asarray(scores), kind="stable")
    found, total_yes = int(y[order[:k]].sum()), int(y.sum())
    return {
        "share": share,
        "calls": k,
        "precision": round(found / k, 4) if k else None,
        "recall": round(found / total_yes, 4) if total_yes else None,
        "net_per_100_clients": round(100 * (found - cost_per_call * k) / n, 2),
        "call_everyone_net_per_100_clients": round(100 * (total_yes - cost_per_call * n) / n, 2),
    }


def run(config: dict, sets: dict) -> dict:
    """Fit on train, choose cut-offs on validation, grade once on test."""
    name, params = final_model(config)
    cols = columns.feature_columns(config["features"]["groups"])
    train_df, valid_df, test_df = sets["train"], sets["validation"], sets["test"]
    y_train, y_valid, y_test = (features.make_target(d) for d in (train_df, valid_df, test_df))

    model = models.build_model(name, cols, params, config["seed"])
    model.fit(train_df[cols], y_train)
    valid_scores = model.predict_proba(valid_df[cols])[:, 1]
    test_scores = model.predict_proba(test_df[cols])[:, 1]

    step = config["evaluation"]["cutoff_share_step"]
    scenarios = {}
    for scenario, settings in config["scenarios"].items():
        cost = settings["cost_per_call"]
        chosen = cutoff.best_share(y_valid, valid_scores, cost, step)      # validation only
        scenarios[scenario] = {
            "cost_per_call": cost,
            "validation": chosen,
            "test": at_share(y_test, test_scores, chosen["share"], cost),  # graded, not chosen, on test
        }

    fraction = config["evaluation"]["top_fraction"]
    k = evaluate.top_k_from_fraction(len(y_test), fraction)
    base_test = float(y_test.mean())
    pr = evaluate.pr_auc(y_test, test_scores)
    boot = uncertainty.bootstrap(y_test, {"final": test_scores}, k, config["evaluation"]["bootstrap_draws"], config["seed"])

    return {
        "model": name,
        "settings": params,
        "yes_rate": {"train": round(float(y_train.mean()), 4), "validation": round(float(y_valid.mean()), 4), "test": round(base_test, 4)},
        "test_clients": len(y_test),
        "test_pr_auc": round(pr, 4),
        "test_pr_auc_range": boot["pr_auc"]["final"],
        "test_times_blind_guess": round(pr / base_test, 2),
        "top_fraction": fraction,
        "top_k": k,
        "test_precision_at_k": round(evaluate.precision_at_k(y_test, test_scores, k), 4),
        "test_recall_at_k": round(evaluate.recall_at_k(y_test, test_scores, k), 4),
        "test_lift_at_k": round(evaluate.lift_at_k(y_test, test_scores, k), 2),
        "always_no": {key: round(v, 4) for key, v in evaluate.always_no_baseline(y_test).items()},
        "scenarios": scenarios,
    }


def main() -> None:
    path = train.OUTPUT_DIR / RESULT_FILE
    check_not_run_before(path)

    config = load_config()
    r = run(config, split.prepare(config))

    y = r["yes_rate"]
    print(f"Final model: {r['model']} {r['settings']}, fitted on the training set only.")
    print(f"Yes-rate: train {y['train']:.3f}, validation {y['validation']:.3f}, test {y['test']:.3f}")
    print("Precision and net value depend on the yes-rate, so compare them across sets only with those rates in mind.\n")
    pr_range = r["test_pr_auc_range"]
    print(f"Test PR-AUC {r['test_pr_auc']:.3f} (middle 90%: {pr_range['low']:.3f} to {pr_range['high']:.3f}); "
          f"a blind guess scores {y['test']:.3f}, so this is {r['test_times_blind_guess']:.2f}x a blind guess.")
    print(f"Top {r['top_fraction']:.0%} ({r['top_k']:,} calls): precision {r['test_precision_at_k']:.3f}, "
          f"recall {r['test_recall_at_k']:.3f}, lift {r['test_lift_at_k']:.2f}")
    a = r["always_no"]
    print(f"Always 'no': accuracy {a['accuracy']:.3f}, finds nobody.\n")

    print("Cut-off shares were chosen on validation. Costs are ASSUMED. net = units per 100 clients.")
    print(f"  {'scenario':16} {'share':>6} {'calls':>6} {'precision':>10} {'recall':>7} {'net':>7} {'call everyone':>14}")
    for name, s in r["scenarios"].items():
        t = s["test"]
        print(f"  {name:16} {t['share']:6.0%} {t['calls']:6,} {t['precision']:10.3f} {t['recall']:7.3f} "
              f"{t['net_per_100_clients']:7.2f} {t['call_everyone_net_per_100_clients']:14.2f}")

    train.OUTPUT_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"\nsaved {path}. The test set is now used up: do not run this again.")


if __name__ == "__main__":
    main()
