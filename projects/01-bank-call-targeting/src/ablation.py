"""Ablation study: what happens to the score when a group of columns is added or removed?

Each setup in config.yaml ("ablations") is a different list of columns. For every model
and every setup we fit on the training set and score the validation set, then ask how
much the score moved compared with the first setup (the baseline).

A change is judged with the bootstrap from uncertainty.py: the same re-drawn exams are
given to every setup, and the range of the difference says whether it is real.
If the range includes 0, the change cannot be told apart from luck.

The setup that adds `duration` is a deliberate LEAKAGE DEMO. duration is only known after
the call, so it is never allowed in a real model. It is here to show what leakage looks like.

Only the training and validation sets are read. The test set is not touched.
"""

import json

import columns
import features
import models
import split
import train
import uncertainty
from settings import load_config


def feature_list(setup: dict) -> list[str]:
    """Columns for one setup: its groups, then any extra single columns."""
    extras = list(setup.get("extra_columns", []))
    allowed = set(columns.all_known_columns()) - {columns.ROW_ID, columns.TARGET}
    unknown = [c for c in extras if c not in allowed]
    if unknown:
        raise ValueError(f"unknown extra columns in setup {setup['name']}: {unknown}")
    cols = columns.feature_columns(setup["groups"])
    return cols + [c for c in extras if c not in cols]


def score_setups(config: dict, sets: dict) -> dict:
    """Fit every model on every setup. Only sets["train"] and sets["validation"] are read."""
    train_df, valid_df = sets["train"], sets["validation"]
    y_train, y_valid = features.make_target(train_df), features.make_target(valid_df)

    scores = {}
    for model_name, params in config["models"].items():
        scores[model_name] = {}
        for setup in config["ablations"]:
            cols = feature_list(setup)
            model = models.build_model(model_name, cols, params, config["seed"])
            model.fit(train_df[cols], y_train)
            scores[model_name][setup["name"]] = model.predict_proba(valid_df[cols])[:, 1]
    return {"y_validation": y_valid, "scores": scores}


def run(config: dict, sets: dict) -> dict:
    """Score every setup, then measure each change against the baseline with the bootstrap."""
    scored = score_setups(config, sets)
    y_valid = scored["y_validation"]
    setups = config["ablations"]
    reference = setups[0]["name"]
    k = config["evaluation"]["top_k"]
    draws = config["evaluation"]["bootstrap_draws"]

    result = {"reference": reference, "setups": {s["name"]: feature_list(s) for s in setups}, "models": {}}
    for model_name, by_setup in scored["scores"].items():
        boot = uncertainty.bootstrap(y_valid, by_setup, k, draws, config["seed"], reference=reference)
        change = {
            name: {**d, "verdict": uncertainty.verdict(d)}
            for name, d in boot["pr_auc_minus_reference"].items()
        }
        result["models"][model_name] = {"pr_auc": boot["pr_auc"], "change_vs_baseline": change}
    return result


def main() -> None:
    config = load_config()
    result = run(config, split.prepare(config))
    reference = result["reference"]

    print(f"Reference setup: {reference}. Ranges are the middle 90% of re-drawn validation exams.")
    print("Change = PR-AUC of the setup minus PR-AUC of the reference.\n")
    for model_name, m in result["models"].items():
        print(model_name)
        base = m["pr_auc"][reference]
        print(f"  {reference:30} {base['median']:.3f} ({base['low']:.3f} to {base['high']:.3f})")
        for name, d in m["change_vs_baseline"].items():
            print(f"  {name:30} {d['median']:+.3f} ({d['low']:+.3f} to {d['high']:+.3f})  {d['verdict']}")
        print()

    train.OUTPUT_DIR.mkdir(exist_ok=True)
    path = train.OUTPUT_DIR / "ablation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
