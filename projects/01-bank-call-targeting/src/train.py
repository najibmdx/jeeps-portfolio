"""Train the three models on the training set and score them on the validation set.

The test set is never used here. It is kept for one final check at the end.
"""

import json
from pathlib import Path

import columns
import evaluate
import features
import models
import split
from settings import load_config

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def score_models(config: dict, sets: dict) -> dict:
    """Fit every model in the config on train. Return the scores on train and validation.

    Only sets["train"] and sets["validation"] are read; the test set is not touched.
    """
    feature_cols = columns.feature_columns(config["features"]["groups"])
    train, valid = sets["train"], sets["validation"]
    y_train, y_valid = features.make_target(train), features.make_target(valid)

    scores = {}
    for name, params in config["models"].items():
        model = models.build_model(name, feature_cols, params, config["seed"])
        model.fit(train[feature_cols], y_train)
        scores[name] = {
            "train": model.predict_proba(train[feature_cols])[:, 1],
            "validation": model.predict_proba(valid[feature_cols])[:, 1],
        }
    return {"y_train": y_train, "y_validation": y_valid, "scores": scores}


def run(config: dict, sets: dict) -> dict:
    """Score every model and return the metrics."""
    scored = score_models(config, sets)
    y_train, y_valid = scored["y_train"], scored["y_validation"]
    k = config["evaluation"]["top_k"]
    base_train, base_valid = float(y_train.mean()), float(y_valid.mean())

    results = {}
    for name, s in scored["scores"].items():
        train_pr = evaluate.pr_auc(y_train, s["train"])
        valid_pr = evaluate.pr_auc(y_valid, s["validation"])
        results[name] = {
            "train_pr_auc": round(train_pr, 4),
            "train_times_blind_guess": round(train_pr / base_train, 2),
            "validation_pr_auc": round(valid_pr, 4),
            "validation_times_blind_guess": round(valid_pr / base_valid, 2),
            "validation_precision_at_k": round(evaluate.precision_at_k(y_valid, s["validation"], k), 4),
            "validation_lift_at_k": round(evaluate.lift_at_k(y_valid, s["validation"], k), 2),
        }

    return {
        "train_baseline": round(base_train, 4),
        "validation_baseline": round(base_valid, 4),
        "top_k": k,
        "models": results,
    }


def main() -> None:
    config = load_config()
    metrics = run(config, split.prepare(config))

    print(f"blind guess (yes-rate):  train {metrics['train_baseline']:.3f}   validation {metrics['validation_baseline']:.3f}")
    print(f"top_k = {metrics['top_k']} calls (assumption)\n")
    print(f"{'model':20} {'train PR-AUC':>18} {'valid PR-AUC':>18} {'precision@k':>12} {'lift@k':>7}")
    for name, m in metrics["models"].items():
        train_text = f"{m['train_pr_auc']:.3f} ({m['train_times_blind_guess']:.1f}x)"
        valid_text = f"{m['validation_pr_auc']:.3f} ({m['validation_times_blind_guess']:.1f}x)"
        print(f"{name:20} {train_text:>18} {valid_text:>18} "
              f"{m['validation_precision_at_k']:12.3f} {m['validation_lift_at_k']:7.2f}")
    print("(x) = times better than a blind guess on the SAME set")

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nsaved {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
