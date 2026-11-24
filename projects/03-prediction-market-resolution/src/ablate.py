"""Stage 8: column-group ablations. Fitted on TRAIN, scored on VALIDATION. The test split is never read.

    python src/ablate.py [--data-dir DIR] [--k 7]       (needs the split files and the committed frozen_choice.yaml)

Rules fixed with Jeeps on 25 Nov 2026 (the Stage 8 design discussion):
  - The model is the FROZEN choice (frozen_choice.yaml): the same model and the same settings in every row, refitted on
    train on a subset of the columns. Nothing is retuned: retuning per row would measure tuning effort, not columns.
  - The rows are in config.yaml (ablation.rows). Every row keeps the price path. The row all_columns is the frozen model
    itself, and its validation score must reproduce the one written in frozen_choice.yaml; the run stops if it does not.
  - Rows are compared with the price-only correction (lr_price_only) and with each other, in pairs (ablation.pairs),
    A minus B in log loss, on the same re-drawn sets of whole events, so a pair shares its noise.
  - The sibling-count answers are two of those pairs: price_path_plus_sibling_count minus price_path_only, and
    all_columns minus all_minus_sibling_count.
  - Nothing here can change the frozen choice. If a subset beat the frozen model on validation it would be a finding to
    report, not a reason to switch: switching now would be choosing twice on the same validation markets.
  - sibling_count_at_t drifts upwards with t (+0.295 against t in the Stage 4 screen), so a gain from it could partly be the calendar.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import baselines
import freeze
import models
import settings
import split
import tune

SLICES = ("all", "contested")
BASELINE = "lr_price_only"
FULL = "all_columns"


def row_names(config) -> list:
    return list(config["ablation"]["rows"])


def run(train, validation, config, model, params, frozen_log_loss_all=None) -> dict:
    """Fit `model` with `params` on train once per row (on that row's columns), score everything on validation."""
    if model == "lr_price_only" or model not in config["models"]["ladder"]:
        raise ValueError(f"the frozen model is {model}: it needs a ladder model that reads several columns, so there is nothing to ablate")
    for part, df in (("train", train), ("validation", validation)):
        miss = [c for c in baselines.NEEDED if c not in df.columns]
        if miss:
            raise ValueError(f"the {part} table lacks the columns the models need: {miss}")
        if len(df) == 0:
            raise ValueError(f"the {part} table has no markets")
    eps, ev = config["metrics"]["price_clip"], config["evaluation"]
    rows = config["ablation"]["rows"]
    y_train = train["label"].to_numpy()
    correction = baselines.fit_correction(train["last_close"].to_numpy(), y_train, eps)  # refuses a train set with one outcome
    fc = {
        "raw_price": baselines.clip_probability(validation["last_close"].to_numpy(dtype="float64"), eps),
        BASELINE: baselines.apply_correction(validation["last_close"].to_numpy(dtype="float64"), correction, eps),
    }
    for name, columns in rows.items():
        fitted = models.build_model(model, config, columns, params).fit(models.model_inputs(train, columns), y_train)
        fc[name] = baselines.clip_probability(fitted.predict_proba(models.model_inputs(validation, columns))[:, 1], eps)
    names = ["raw_price", BASELINE] + list(rows)
    codes, uniques = pd.factorize(validation["event_key"])
    weights = baselines.resample_weights(len(uniques), ev["bootstrap_draws"], config["seed"])
    masks = baselines.slice_masks(validation, config)
    scored, pairs = {}, {}
    for sname, mask in masks.items():
        scored[sname] = baselines.score_slice(validation, {n: fc[n] for n in names}, mask, weights, codes, len(uniques), ev["interval"], names=names)
        pairs[sname] = []
        if mask.any():
            for a, b in config["ablation"]["pairs"]:
                d = baselines.paired_difference(validation, fc[a], fc[b], mask, weights, codes, len(uniques), ev["interval"])
                pairs[sname].append({"a": a, "b": b, **d})
    reproduced = None
    if frozen_log_loss_all is not None:
        got = scored["all"]["scores"][FULL]["log_loss"]
        reproduced = {"frozen_log_loss_all": frozen_log_loss_all, "this_run": got, "difference": got - frozen_log_loss_all}
        if abs(got - frozen_log_loss_all) > config["ablation"]["reproduce_tolerance"]:
            raise ValueError(f"the all_columns row scores {got:.10f} on validation but frozen_choice.yaml says {frozen_log_loss_all:.10f}: "
                             "the frozen model is not reproduced, so these rows would not be comparable with it")
    return {
        "model": model, "params": dict(params), "train_markets": len(train), "validation_markets": len(validation), "validation_events": len(uniques),
        "price_clip": eps, "bootstrap_draws": ev["bootstrap_draws"], "interval": ev["interval"],
        "settings": {"seed": config["seed"], "forest_trees": config["models"]["forest_trees"], "logistic_max_iter": config["models"]["logistic_max_iter"]},
        "groups": {g: list(c) for g, c in config["ablation"]["groups"].items()},
        "rows": {n: list(c) for n, c in rows.items()}, "order": names, "reproduces_frozen": reproduced,
        "validation": scored, "pairs": pairs,
        "notes": [
            "Same model and same settings in every row, refitted on train on a subset of the columns. Nothing was retuned.",
            "Scored on validation only. The test split was not read. Nothing here changes the frozen choice.",
            "Pairs are A minus B in log loss on the same re-drawn sets of whole events; negative means A has the lower log loss.",
            "The ranges cover sampling noise inside validation, not the change between periods.",
            "sibling_count_at_t rises with t, so a gain from it could partly be the calendar.",
        ],
    }


# ---- printing

def _f(x) -> str:
    return f"{x:.4f}"


def summary_lines(r) -> list:
    lines = [
        f"Column-group ablations on validation ({r['validation_markets']:,} markets, {r['validation_events']:,} events). Model: {r['model']} ({tune.describe(r['params'])}), "
        f"the frozen choice, the same settings in every row, refitted on train ({r['train_markets']:,} markets) on each row's columns. Nothing is retuned.",
        f"  Ranges are the middle {r['interval']:.0%} of {r['bootstrap_draws']} re-drawn sets of whole events. A pair is A minus B in log loss; negative means A is lower (better).",
    ]
    lines += [f"  {g}: {', '.join(c)}" for g, c in r["groups"].items()]
    if r["reproduces_frozen"]:
        rp = r["reproduces_frozen"]
        lines.append(f"  all_columns reproduces the frozen validation score: {rp['this_run']:.6f} against {rp['frozen_log_loss_all']:.6f} in frozen_choice.yaml")
    for sname in SLICES:
        s = r["validation"][sname]
        if s["markets"] == 0:
            lines.append(f"  {sname}: no markets in validation")
            continue
        lines.append(f"  {sname} (validation {s['markets']:,} markets, {s['yes_rate']:.1%} Yes)")
        lines.append(f"    {'row':<32} {'columns':>7} {'log loss (range)':>26}   minus raw_price (range)")
        for n in r["order"]:
            w = s["wobble"][n]["log_loss"]
            n_cols = str(len(r["rows"][n])) if n in r["rows"] else "-"
            if n == "raw_price":
                tail = ""
            else:
                d = s["difference_vs_raw_price"][n]["log_loss"]
                tail = f"{d['point']:+.4f} ({d['low']:+.4f} to {d['high']:+.4f}): {d['verdict']}"
            lines.append(f"    {n:<32} {n_cols:>7} {_f(s['scores'][n]['log_loss']) + ' (' + _f(w['low']) + ' to ' + _f(w['high']) + ')':>26}   {tail}")
        lines.append(f"    pairs, log loss, A minus B ({sname}):")
        for p in r["pairs"][sname]:
            lines.append(f"      {p['a']} minus {p['b']}: {p['point']:+.4f} ({p['low']:+.4f} to {p['high']:+.4f}): {p['verdict']}")
    sib = [p for p in r["pairs"]["all"] if (p["a"], p["b"]) in (("price_path_plus_sibling_count", "price_path_only"), ("all_columns", "all_minus_sibling_count"))]
    if sib:
        lines.append("  Sibling count, all markets: " + "; ".join(f"{p['a']} minus {p['b']} {p['point']:+.4f} ({p['verdict']})" for p in sib))
    lines.append("  Nothing here changes the frozen choice. sibling_count_at_t rises with t, so a gain from it could partly be the calendar.")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Column-group ablations of the frozen model, on validation")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--k", type=int, default=None, help="days after open (default: from config.yaml)")
    args = parser.parse_args(argv)
    config = settings.load_config()
    k = args.k or config["decision_time"]["days_after_open"]
    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_DIR
    for part in ("train", "validation"):
        if not split.split_path(data_dir, config, part, k).exists():
            print(f"No {part} split in {data_dir}. Run src/pipeline.py (or src/split.py) first.")
            return 2
    try:
        frozen = freeze.read_frozen(settings.PROJECT_DIR / config["models"]["frozen_file"], config)
    except ValueError as e:
        print(f"PROBLEM {e}")
        return 2
    train = split.read_split(data_dir, config, "train", k)
    validation = split.read_split(data_dir, config, "validation", k)
    try:
        report = run(train, validation, config, frozen["model"], frozen["params"], frozen["validation"]["log_loss_all"])
    except ValueError as e:
        print(f"PROBLEM {e}")
        return 1
    out_dir = settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"ablation_k{k}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(summary_lines(report)))
    print("Test was not read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
