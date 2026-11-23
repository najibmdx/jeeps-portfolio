"""Stage 5: do the held-back column groups help? Same tuned model, different column sets.

Every set is trained on TRAIN and graded on VALIDATION, with the same re-drawn exams for all sets
so differences are fair. A group that helps is NOT automatically a win: group1 (calendar and SOL
price) carries clock risk and group4 (cluster columns) has unverified timing, so a gain there could
be leakage. The two calendar columns the audit flagged are never used.

Run from the project folder (after tune.py):  python src/ablation.py
"""

import json
import sys
from pathlib import Path

import yaml

import metrics
import models
import piles
import settings
import uncertainty


def column_sets(all_columns, config: dict) -> dict:
    """Name -> column list for every ablation set."""
    base = list(config["features"]["groups"])
    drop = config["ablation"]["never_use"]
    return {name: piles.select_features(all_columns, config, groups=base + list(extra), drop=drop)
            for name, extra in config["ablation"]["sets"].items()}


def load_tuned_params(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run tune.py first.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))["models"]


def run(config: dict, pile_set: dict, params: dict) -> dict:
    name = config["ablation"]["model"]
    train_df, valid = pile_set["train"], pile_set["validation"]
    sets = column_sets(train_df.columns, config)
    fraction = config["evaluation"]["top_fraction"]
    scores, result_sets = {}, {}
    for set_name, cols in sets.items():
        model = models.build_model(name, params[name], config["seed"])
        model.fit(train_df[cols], train_df["crash"])
        scores[set_name] = model.predict_proba(valid[cols])[:, 1]
        result_sets[set_name] = {"n_columns": len(cols),
                                 "validation": metrics.side_metrics(valid["crash"].to_numpy(), scores[set_name], fraction)}
    reference = next(iter(sets))
    wobble = uncertainty.bootstrap(valid["crash"].to_numpy(), scores, config["evaluation"]["bootstrap_draws"],
                                   config["seed"], reference)
    return {"model": name, "reference": reference, "sets": result_sets, "wobble": wobble,
            "flags": flags(wobble)}


def flags(wobble: dict) -> dict:
    """Sets whose gain over the reference is clear on either side get a warning, not a medal."""
    out = {}
    for set_name in wobble["difference_crash"]:
        helps = [side for side in ("crash", "survivor") if wobble[f"difference_{side}"][set_name]["low"] > 0]
        if helps:
            out[set_name] = ("helps on the " + " and ".join(helps) + " side on validation, but these columns carry "
                             "clock risk or unverified timing: treat as possible leakage, not a win")
    return out


def format_result(result: dict) -> str:
    w = result["wobble"]
    lines = [f"model: {result['model']} (tuned). reference set: {result['reference']}",
             f"{'set':26} {'cols':>5} {'crash PR-AUC':>14} {'survivor PR-AUC':>17}"]
    for name, s in result["sets"].items():
        v = s["validation"]
        lines.append(f"{name:26} {s['n_columns']:>5} {v['crash']['pr_auc']:>9.3f} ({v['crash']['pr_auc_times_baseline']:.2f}x)"
                     f" {v['survivor']['pr_auc']:>8.3f} ({v['survivor']['pr_auc_times_baseline']:.2f}x)")
    for side in ("crash", "survivor"):
        lines.append(f"\n{side} side: PR-AUC difference compared with {result['reference']} (middle 90% of re-drawn exams):")
        for name, d in w[f"difference_{side}"].items():
            lines.append(f"{name:26} {d['median']:+.3f} ({d['low']:+.3f} to {d['high']:+.3f})  {uncertainty.verdict(d)}")
    for name, text in result["flags"].items():
        lines.append(f"\nWARNING {name}: {text}")
    return "\n".join(lines)


def main(argv=None, config=None, data_dir=None, out_dir=None, params_path=None) -> int:
    config = config or settings.load_config()
    data_dir = Path(data_dir) if data_dir else settings.DATA_DIR
    out_dir = Path(out_dir) if out_dir else settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    params_path = Path(params_path) if params_path else settings.PROJECT_DIR / config["tuning"]["tuned_params_file"]
    try:
        params = load_tuned_params(params_path)
        pile_set = piles.load_piles(data_dir, config)
    except FileNotFoundError as error:
        print(error)
        return 1
    result = run(config, pile_set, params)
    print(format_result(result))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved {out_dir / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
