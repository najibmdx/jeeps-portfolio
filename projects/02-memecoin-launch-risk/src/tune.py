"""Stage 5: tune each model on TRAIN with time-ordered blocks, then grade once on VALIDATION.

Saves the chosen settings to tuned_params.yaml (kept in the repo so later stages repeat exactly
the same models) and the numbers to outputs/. The test pile is never opened.

Run from the project folder:  python src/tune.py
"""

import json
import sys
from pathlib import Path

import yaml

import metrics
import piles
import settings
import train
import tuning
import uncertainty


def tuned_models_config(config: dict, tuned: dict) -> dict:
    """A copy of the config whose model settings are the tuned ones."""
    merged = {name: {**params, **tuned[name]["best"]} for name, params in config["models"].items()}
    return {**config, "models": merged}


def run_tuning(config: dict, train_pile) -> dict:
    ordered = tuning.time_ordered(train_pile)
    cols = piles.select_features(ordered.columns, config)
    window = config["label"]["window_minutes"] * 60
    folds = tuning.make_folds(ordered["migration_time"], config["tuning"]["n_blocks"], window)
    return {
        name: tuning.tune_model(name, config["models"][name], grid, ordered[cols], ordered["crash"].to_numpy(),
                                folds, config["seed"])
        for name, grid in config["tuning"]["grids"].items()
    }


def save_params(tuned: dict, config: dict, path: Path) -> None:
    out = {"note": "written by tune.py; the settings chosen by time-ordered tuning on the train pile",
           "feature_groups": config["features"]["groups"],
           "models": {name: {**config["models"][name], **t["best"]} for name, t in tuned.items()}}
    path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")


def format_tuning(tuned: dict) -> str:
    lines = ["tuning on train (time-ordered blocks). score = average of crash and survivor 'times the lazy guess'"]
    for name, t in tuned.items():
        worst = min(x["score"] for x in t["trials"])
        lines.append(f"  {name:20} best {t['best_score']:.3f} (worst trial {worst:.3f}, {len(t['trials'])} trials)  {t['best']}")
    return "\n".join(lines)


def format_comparison(before: dict, after: dict) -> str:
    lines = ["\nvalidation PR-AUC, untuned -> tuned (times the lazy guess in brackets):",
             f"{'model':20} {'crash side':>30} {'survivor side':>30}"]
    for name in after["models"]:
        row = [f"{name:20}"]
        for side in ("crash", "survivor"):
            b, a = before["models"][name]["validation"][side], after["models"][name]["validation"][side]
            row.append(f"{b['pr_auc']:.3f} ({b['pr_auc_times_baseline']:.2f}x) -> {a['pr_auc']:.3f} ({a['pr_auc_times_baseline']:.2f}x)")
        lines.append(f"{row[0]} {row[1]:>30} {row[2]:>30}")
    return "\n".join(lines)


def main(argv=None, config=None, data_dir=None, out_dir=None, params_path=None) -> int:
    config = config or settings.load_config()
    data_dir = Path(data_dir) if data_dir else settings.DATA_DIR
    out_dir = Path(out_dir) if out_dir else settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    params_path = Path(params_path) if params_path else settings.PROJECT_DIR / config["tuning"]["tuned_params_file"]
    try:
        pile_set = piles.load_piles(data_dir, config)
    except FileNotFoundError as error:
        print(error)
        return 1

    tuned = run_tuning(config, pile_set["train"])
    print(format_tuning(tuned))

    tuned_config = tuned_models_config(config, tuned)
    before = train.run(config, pile_set)
    after = train.run(tuned_config, pile_set)
    print(format_comparison(before, after))

    fitted = train.fit_models(tuned_config, pile_set)
    valid_scores = {name: s["validation"] for name, s in fitted["scores"].items()}
    wobble = uncertainty.bootstrap(fitted["y_validation"], valid_scores, config["evaluation"]["bootstrap_draws"],
                                   config["seed"], config["evaluation"]["reference_model"])
    print("\n" + uncertainty.format_result(wobble, len(fitted["y_validation"])))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tuning.json").write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    (out_dir / "tuned_metrics.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    (out_dir / "tuned_wobble.json").write_text(json.dumps(wobble, indent=2), encoding="utf-8")
    save_params(tuned, config, params_path)
    print(f"\nsaved tuning.json, tuned_metrics.json, tuned_wobble.json in {out_dir} and {params_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
