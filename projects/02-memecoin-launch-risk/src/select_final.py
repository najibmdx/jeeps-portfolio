"""Stage 6, part 1: choose the final model and the cut-offs on VALIDATION, then FREEZE them in a file.

  - The three tuned models (settings from tuned_params.yaml) are refitted on TRAIN only, exactly as validated.
  - Model rule, fixed before looking: the highest average of crash-side and survivor-side PR-AUC on validation.
  - One cut-off per cost scenario: the one with the lowest invented cost on validation.
  - Everything is written to frozen_choice.yaml. Commit that file BEFORE running final_test.py.

The test pile is never opened here. Once the test pile has been opened this script refuses to run,
so the choices cannot be changed after seeing the test result.

Run from the project folder:  python src/select_final.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import decision
import metrics
import piles
import settings
import holdout
import train


def tuned_config(config: dict, params_path) -> dict:
    """The config with the tuned model settings from tuned_params.yaml. Stops if the file does not fit."""
    tuned = yaml.safe_load(Path(params_path).read_text(encoding="utf-8"))
    if tuned["feature_groups"] != config["features"]["groups"]:
        raise ValueError("tuned_params.yaml was made for other feature groups than config.yaml; run tune.py again")
    missing = set(config["models"]) - set(tuned["models"])
    if missing:
        raise ValueError(f"tuned_params.yaml has no settings for {sorted(missing)}; run tune.py again")
    return {**config, "models": {name: tuned["models"][name] for name in config["models"]}}


def columns_hash(cols) -> str:
    return hashlib.sha256("\n".join(cols).encode("utf-8")).hexdigest()


def choose_model(y_valid, valid_scores: dict, candidates) -> tuple:
    """Highest average of crash-side and survivor-side PR-AUC. A tie goes to the earlier candidate."""
    y = np.asarray(y_valid)
    table = {}
    for name in candidates:
        s = np.asarray(valid_scores[name], dtype=float)
        crash, survivor = metrics.pr_auc(y, s), metrics.pr_auc(1 - y, 1 - s)
        table[name] = {"crash_pr_auc": float(crash), "survivor_pr_auc": float(survivor),
                       "average": float((crash + survivor) / 2)}
    best = None
    for name in candidates:
        if best is None or table[name]["average"] > table[best]["average"]:
            best = name
    return best, table


def scenario_choices(y_valid, scores, config: dict) -> dict:
    """One cut-off per scenario, with the validation counts and cost next to the two lazy options."""
    out = {}
    nothing, everything = decision.flag_nothing(y_valid), decision.flag_everything(y_valid)
    for name, scenario in config["final"]["scenarios"].items():
        best = decision.best_cutoff(y_valid, scores, scenario)
        out[name] = {
            "costs": dict(scenario),
            "cutoff": best["cutoff"],
            "validation": {
                "counts": best["counts"],
                "cost_per_coin": round(decision.cost_per_coin(best["counts"], scenario), 4),
                "cost_per_coin_flag_nothing": round(decision.cost_per_coin(nothing, scenario), 4),
                "cost_per_coin_flag_everything": round(decision.cost_per_coin(everything, scenario), 4),
            },
        }
    return out


def build_frozen(config: dict, tuned_cfg: dict, pile_set: dict) -> tuple:
    """Fit on train, choose model and cut-offs on validation. Returns (frozen dict, extras for the printout)."""
    fitted = train.fit_models(tuned_cfg, pile_set)
    y_valid = fitted["y_validation"]
    valid_scores = {name: s["validation"] for name, s in fitted["scores"].items()}
    model, table = choose_model(y_valid, valid_scores, config["final"]["candidates"])
    scores = valid_scores[model]
    side = metrics.side_metrics(y_valid, scores, config["evaluation"]["top_fraction"])
    frozen = {
        "note": "written by select_final.py from TRAIN and VALIDATION only. Commit this file before opening the test pile.",
        "rule": config["final"]["selection"],
        "model": model,
        "model_params": dict(tuned_cfg["models"][model]),
        "fit_on": "train only (not train + validation), exactly as validated",
        "feature_groups": list(config["features"]["groups"]),
        "n_features": len(fitted["columns"]),
        "columns_sha256": columns_hash(fitted["columns"]),
        "seed": config["seed"],
        "train_rows": int(len(fitted["y_train"])),
        "validation": {
            "rows": int(len(y_valid)),
            "crash_share": round(float(np.mean(y_valid)), 4),
            "selection_table": table,
            "top_fraction": config["evaluation"]["top_fraction"],
            "side_metrics": side,
        },
        "scenarios": scenario_choices(y_valid, scores, config),
    }
    return frozen, {"y_valid": y_valid, "scores": scores}


def curve_table(y_valid, scores, config: dict) -> pd.DataFrame:
    """Every distinct cut-off on validation, both mistakes, and the invented cost per coin in each scenario."""
    c = decision.curve(y_valid, scores)
    frame = pd.DataFrame({"cutoff": c["cutoff"], "flagged": c["flagged"],
                          "skipped_good": c["skipped_good"], "held_crasher": c["held_crasher"]})
    for name, sc in config["final"]["scenarios"].items():
        frame[f"cost_per_coin_{name}"] = (sc["holding_crasher"] * frame["held_crasher"]
                                          + sc["skipped_good"] * frame["skipped_good"]) / c["n"]
    return frame


def format_choice(frozen: dict, extras: dict, config: dict) -> str:
    v = frozen["validation"]
    lines = ["model choice on validation (rule: highest average of crash-side and survivor-side PR-AUC):",
             f"{'model':20} {'crash PR-AUC':>13} {'survivor PR-AUC':>16} {'average':>9}"]
    for name, t in v["selection_table"].items():
        mark = "   <- chosen" if name == frozen["model"] else ""
        lines.append(f"{name:20} {t['crash_pr_auc']:13.4f} {t['survivor_pr_auc']:16.4f} {t['average']:9.4f}{mark}")
    ranked = sorted(v["selection_table"].items(), key=lambda kv: -kv[1]["average"])
    if len(ranked) > 1:
        gap = ranked[0][1]["average"] - ranked[1][1]["average"]
        lines.append(f"margin over the runner-up ({ranked[1][0]}): {gap:.4f}. "
                     "The rule picks the top average even when the margin is tiny.")
    y, s = extras["y_valid"], extras["scores"]
    lines.append(f"\nflag = score at or above the cut-off = 'likely to crash' (risk flag, not a trade signal). "
                 f"Validation: {v['rows']:,} coins, {100 * v['crash_share']:.1f}% crash.")
    lines.append("both mistakes at a grid of cut-offs (P = precision, R = recall, in %):")
    lines.append("  " + decision.describe("flag nothing", np.inf, decision.flag_nothing(y)))
    for share in config["final"]["grid_shares"]:
        cut = decision.cutoff_at_flag_share(s, share)
        lines.append("  " + decision.describe(f"flag top ~{int(round(share * 100))}%", cut, decision.counts_at(y, s, cut)))
    lines.append("  " + decision.describe("flag everything", 0.0, decision.flag_everything(y)))
    lines.append("\nchosen cut-off per INVENTED cost scenario (costs are assumptions, not measured):")
    for name, sc in frozen["scenarios"].items():
        c = sc["costs"]
        lines.append(f"  {name}  (holding a crasher = {c['holding_crasher']}, skipping a good coin = {c['skipped_good']})")
        lines.append("    " + decision.describe("chosen", sc["cutoff"], sc["validation"]["counts"]))
        val = sc["validation"]
        lines.append(f"    cost per coin: chosen {val['cost_per_coin']:.4f}   flag nothing "
                     f"{val['cost_per_coin_flag_nothing']:.4f}   flag everything {val['cost_per_coin_flag_everything']:.4f}")
    return "\n".join(lines)


def main(argv=None, config=None, data_dir=None, out_dir=None, params_path=None, frozen_path=None) -> int:
    config = config or settings.load_config()
    data_dir = Path(data_dir) if data_dir else settings.DATA_DIR
    out_dir = Path(out_dir) if out_dir else settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    params_path = Path(params_path) if params_path else settings.PROJECT_DIR / config["tuning"]["tuned_params_file"]
    frozen_path = Path(frozen_path) if frozen_path else settings.PROJECT_DIR / config["final"]["frozen_file"]

    if holdout.is_locked(data_dir, config):
        print("REFUSED: the test pile was already opened, so the choices are final. Nothing was changed.")
        return 1
    try:
        pile_set = piles.load_piles(data_dir, config)
    except FileNotFoundError as error:
        print(error)
        return 1

    frozen, extras = build_frozen(config, tuned_config(config, params_path), pile_set)
    print(format_choice(frozen, extras, config))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "select_final.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    curve_table(extras["y_valid"], extras["scores"], config).to_csv(out_dir / "validation_cutoff_curve.csv", index=False)
    frozen_path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    print(f"\nFROZEN in {frozen_path.name}. Commit it now, before the test pile is opened.")
    print(f"also saved select_final.json and validation_cutoff_curve.csv in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
