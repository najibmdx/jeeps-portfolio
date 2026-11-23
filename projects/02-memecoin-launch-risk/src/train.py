"""Train the three scorers on the TRAIN pile and grade them on the VALIDATION pile.

The test pile is never opened here. Nothing is tuned yet: the settings in config.yaml are
first guesses. Every score is shown next to the lazy guess for the same pile.

Run from the projects folder:  python 02-memecoin-launch-risk/src/train.py
"""

import json
import sys
from pathlib import Path

import metrics
import models
import piles
import settings

PARTS = piles.MODELLING_PARTS


def fit_models(config: dict, pile_set: dict) -> dict:
    """Fit every model in the config on train. Return its scores on train and validation."""
    cols = piles.select_features(pile_set["train"].columns, config)
    train, valid = pile_set["train"], pile_set["validation"]
    scores = {}
    for name, params in config["models"].items():
        model = models.build_model(name, params, config["seed"])
        model.fit(train[cols], train["crash"])
        scores[name] = {
            "train": model.predict_proba(train[cols])[:, 1],
            "validation": model.predict_proba(valid[cols])[:, 1],
        }
    return {"columns": cols, "y_train": train["crash"].to_numpy(),
            "y_validation": valid["crash"].to_numpy(), "scores": scores}


def run(config: dict, pile_set: dict) -> dict:
    """Fit and grade every model. Returns the metrics."""
    fitted = fit_models(config, pile_set)
    fraction = config["evaluation"]["top_fraction"]
    result = {
        "feature_groups": config["features"]["groups"],
        "n_features": len(fitted["columns"]),
        "top_fraction": fraction,
        "models": {},
    }
    for name, s in fitted["scores"].items():
        result["models"][name] = {
            "train": metrics.side_metrics(fitted["y_train"], s["train"], fraction),
            "validation": metrics.side_metrics(fitted["y_validation"], s["validation"], fraction),
        }
    first = next(iter(result["models"].values()))
    result["baselines"] = {part: {side: first[part][side]["baseline"] for side in ("crash", "survivor")}
                           for part in PARTS}
    return result


def format_table(result: dict) -> str:
    b = result["baselines"]
    lines = [
        f"columns used: {result['n_features']} (groups {', '.join(result['feature_groups'])})",
        "lazy guess (the share of that side in the pile):",
        f"  train       crash {b['train']['crash']:.3f}   survivor {b['train']['survivor']:.3f}",
        f"  validation  crash {b['validation']['crash']:.3f}   survivor {b['validation']['survivor']:.3f}",
        "(x) = times better than the lazy guess on the SAME pile. Not tuned yet.\n",
        f"{'model':20} {'train crash PR-AUC':>20} {'valid crash PR-AUC':>20} {'valid survivor PR-AUC':>23}",
    ]
    for name, m in result["models"].items():
        def cell(part, side):
            x = m[part][side]
            return f"{x['pr_auc']:.3f} ({x['pr_auc_times_baseline']:.2f}x)"
        lines.append(f"{name:20} {cell('train', 'crash'):>20} {cell('validation', 'crash'):>20} "
                     f"{cell('validation', 'survivor'):>23}")
    pct = int(round(result["top_fraction"] * 100))
    lines.append(f"\nprecision in the top {pct}% on validation (lift = times the lazy guess):")
    lines.append(f"{'model':20} {'crash side':>18} {'survivor side':>18}")
    for name, m in result["models"].items():
        v = m["validation"]
        lines.append(f"{name:20} {v['crash']['precision_top']:.3f} ({v['crash']['lift_top']:.2f}x) "
                     f"{v['survivor']['precision_top']:>9.3f} ({v['survivor']['lift_top']:.2f}x)")
    return "\n".join(lines)


def main(argv=None, config=None, data_dir=None, out_dir=None) -> int:
    config = config or settings.load_config()
    data_dir = Path(data_dir) if data_dir else settings.DATA_DIR
    out_dir = Path(out_dir) if out_dir else settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    try:
        pile_set = piles.load_piles(data_dir, config)
    except FileNotFoundError as error:
        print(error)
        return 1
    result = run(config, pile_set)
    print(format_table(result))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
