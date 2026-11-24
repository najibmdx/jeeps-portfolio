"""Stage 7: light tuning and the choice. Fitted on TRAIN, tuned and chosen on VALIDATION. The test split is never read.

    python src/tune.py [--data-dir DIR] [--k 7]       (needs the split files written by split.py)

Then `python src/freeze.py` writes frozen_choice.yaml from the report of this step. Both are committed BEFORE the test split is opened.

Rules fixed with Jeeps on 24 Nov 2026 (decision 7 of the design record, and the Stage 7 discussion):
  - Same budget for every model: 6 settings each (config.yaml models.tuning). Every grid holds the library default,
    so the best setting cannot be worse than the untuned model of Stage 6 on validation. lr_price_only has one setting.
  - Every setting is fitted on train only and scored on validation, for all markets and for the contested slice.
  - The choice is the lowest validation log loss on ALL markets, compared at 4 decimals (the decimals printed).
    A tie between settings goes to the first one listed in the grid (the grids are listed simplest first), a tie between
    models to the earlier model in the ladder. Only the four ladder models can win; raw_price is a yardstick, never a candidate.
  - The winner's margin over the runner-up is shown next to the wobble (whole events re-drawn from validation, the same
    draws for both). The wobble is never used to decide: the point estimate decides.
  - No refit on train + validation: the chosen model stays fitted on train only, so the test measures exactly the model that was chosen and scored.
  - The winner's validation score is slightly flattering: it was picked on these very markets. The one-time test is the honest number.

HistGradientBoostingClassifier's library default stops early on a 10% hold-out taken from the TRAIN rows when train has more
than 10,000 rows. That hold-out is inside train and never touches validation; it is part of "library defaults" and was already so in Stage 6.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import models
import settings
import split

DECIMALS = 4
SLICES = ("all", "contested")


# ---- the settings to try

def grid_settings(config, name) -> list:
    """Every setting to try for `name`, in the order that breaks ties (first of equals wins). lr_price_only has one, empty."""
    if name not in settings.TUNABLE:
        return [{}]
    grid = config["models"]["tuning"][name]
    keys = settings.TUNABLE[name]
    return [{k: settings.PARAM_TYPES[k](v) for k, v in zip(keys, combo)} for combo in itertools.product(*(grid[k] for k in keys))]


def is_library_default(name, params) -> bool:
    return name not in settings.TUNABLE or dict(params) == settings.LIBRARY_DEFAULTS[name]


def describe(params) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in params.items()) if params else "no setting to tune"


def pick_first_lowest(values, decimals=DECIMALS) -> int:
    """Index of the lowest value, compared at `decimals` decimals. Among equals the earliest wins."""
    r = [round(float(v), decimals) for v in values]
    if not r or any(np.isnan(r)):
        raise ValueError("cannot choose between scores that are missing")
    return r.index(min(r))


# ---- the run

def _by_slice(y, p, masks) -> dict:
    return {s: (baselines.log_loss(y[m], p[m]) if m.any() else None) for s, m in masks.items()}


def _gap(val, tr) -> dict:
    return {s: (val[s] - tr[s] if val[s] is not None and tr[s] is not None else None) for s in val}


def run(train, validation, config) -> dict:
    """Fit every setting on train, score it on validation, choose. Needs nothing but the train and validation rows."""
    for part, df in (("train", train), ("validation", validation)):
        miss = [c for c in baselines.NEEDED if c not in df.columns]
        if miss:
            raise ValueError(f"the {part} table lacks the columns the models need: {miss}")
        if len(df) == 0:
            raise ValueError(f"the {part} table has no markets")
    ladder = list(config["models"]["ladder"])
    if len(ladder) < 2:
        raise ValueError("the ladder needs at least two models to choose between")
    columns, dropped = models.input_columns(config)
    ev = config["evaluation"]
    correction = baselines.fit_correction(train["last_close"].to_numpy(), train["label"].to_numpy(), config["metrics"]["price_clip"])
    X, y_fit = models.model_inputs(train, columns), train["label"].to_numpy()
    base = float(train["label"].mean())
    y_tr, y_va = train["label"].to_numpy(dtype="float64"), validation["label"].to_numpy(dtype="float64")
    tr_masks, va_masks = baselines.slice_masks(train, config), baselines.slice_masks(validation, config)

    report_models, best_fc, bests = {}, {}, []
    for name in ladder:
        rows, fcs = [], []
        for i, params in enumerate(grid_settings(config, name)):
            thing = correction if name == "lr_price_only" else models.build_model(name, config, columns, params).fit(X, y_fit)
            f_tr = models.forecast_all({name: thing}, columns, base, train, config)[name]
            f_va = models.forecast_all({name: thing}, columns, base, validation, config)[name]
            tr, va = _by_slice(y_tr, f_tr, tr_masks), _by_slice(y_va, f_va, va_masks)
            rows.append({"index": i, "params": params, "is_library_default": is_library_default(name, params), "train": tr, "validation": va, "gap": _gap(va, tr)})
            fcs.append(f_va)
        best = pick_first_lowest([r["validation"]["all"] for r in rows])
        default = next(r for r in rows if r["is_library_default"])
        report_models[name] = {
            "tunable": name in settings.TUNABLE, "settings": rows, "best_index": best, "best_params": rows[best]["params"],
            "library_default_validation_log_loss": default["validation"],
            "best_minus_library_default_all": rows[best]["validation"]["all"] - default["validation"]["all"],
        }
        best_fc[name] = fcs[best]
        bests.append({"model": name, "params": rows[best]["params"], "validation": rows[best]["validation"], "train": rows[best]["train"]})

    win = pick_first_lowest([b["validation"]["all"] for b in bests])
    rest = [i for i in range(len(bests)) if i != win]
    run_up = rest[pick_first_lowest([bests[i]["validation"]["all"] for i in rest])]
    winner, runner = bests[win], bests[run_up]

    # The wobble: whole events re-drawn from validation, the same draws for every forecast, so differences are paired.
    codes, uniques = pd.factorize(validation["event_key"])
    weights = baselines.resample_weights(len(uniques), ev["bootstrap_draws"], config["seed"])
    fc = models.forecast_all({}, columns, base, validation, config)  # raw_price and base_rate
    fc.update(best_fc)
    names = list(models.REFERENCE) + ladder
    scored, margin, winner_vs_raw = {}, {}, {}
    for sname, mask in va_masks.items():
        scored[sname] = baselines.score_slice(validation, {n: fc[n] for n in names}, mask, weights, codes, len(uniques), ev["interval"], names=names)
        if mask.any():
            d = baselines.paired_difference(validation, fc[runner["model"]], fc[winner["model"]], mask, weights, codes, len(uniques), ev["interval"])
            d["reading"] = {"worse": "the winner is ahead of the runner-up by more than the wobble",
                            "better": "the range favours the runner-up; the winner was chosen on the point estimate",
                            "cannot be told apart": "the winner and the runner-up cannot be told apart"}[d["verdict"]]
            margin[sname] = d
            winner_vs_raw[sname] = scored[sname]["difference_vs_raw_price"][winner["model"]]["log_loss"]
    lowest_contested = bests[pick_first_lowest([b["validation"]["contested"] for b in bests])]["model"] if va_masks["contested"].any() else None

    return {
        "train_markets": len(train), "train_events": int(train["event_key"].nunique()),
        "validation_markets": len(validation), "validation_events": len(uniques),
        "price_clip": config["metrics"]["price_clip"], "bootstrap_draws": ev["bootstrap_draws"], "interval": ev["interval"],
        "inputs": columns, "dropped_inputs": dropped,
        "rule": {
            "decides": "lowest validation log loss on all markets, compared at 4 decimals",
            "tie_between_settings": "the first one listed in the grid (grids are listed simplest first)",
            "tie_between_models": "the earlier model in the ladder",
            "wobble": "shown next to the margin, never used to decide",
            "refit_on_train_plus_validation": False,
        },
        "seed": config["seed"], "forest_trees": config["models"]["forest_trees"], "logistic_max_iter": config["models"]["logistic_max_iter"],
        "grids": {n: config["models"]["tuning"][n] for n in ladder if n in settings.TUNABLE},
        "order": names, "models": report_models, "best_per_model": bests,
        "choice": {
            "winner": winner["model"], "params": winner["params"], "validation": winner["validation"], "train": winner["train"],
            "runner_up": runner["model"], "runner_up_params": runner["params"], "runner_up_validation": runner["validation"],
            "tie_at_4_decimals": round(winner["validation"]["all"], DECIMALS) == round(runner["validation"]["all"], DECIMALS),
            "margin_runner_up_minus_winner": margin, "winner_minus_raw_price": winner_vs_raw,
            "lowest_contested_model": lowest_contested,
        },
        "validation": scored,
        "notes": [
            "Fitted on train only, tuned and chosen on validation. The test split was not read. Validation labels were used to score, never to fit.",
            "The winner's validation score is slightly flattering: it was picked on these markets. The one-time test is the honest number.",
            "The ranges resample whole events inside validation. They cover sampling noise, not the change between periods.",
            "No refit on train + validation: the chosen model stays fitted on train only.",
        ],
    }


# ---- printing

def _f(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _s(x) -> str:
    return "n/a" if x is None else f"{x:+.4f}"


def summary_lines(r) -> list:
    lines = [
        f"Light tuning. Every setting fitted on train ({r['train_markets']:,} markets), scored on validation ({r['validation_markets']:,} markets, {r['validation_events']:,} events). Seed {r['seed']}.",
        f"  inputs ({len(r['inputs'])}): {', '.join(r['inputs'])}",
        f"  Choice rule: {r['rule']['decides']}; a tie goes to the first listed setting, then to the earlier model in the ladder. The range never decides.",
        f"  Ranges are the middle {r['interval']:.0%} of {r['bootstrap_draws']} re-drawn sets of whole events. Overfit gap = validation log loss minus train log loss.",
    ]
    for name, m in r["models"].items():
        n = len(m["settings"])
        lines.append(f"  {name}: {n} setting{'s' if n != 1 else ''}" + (" (* = library default, the Stage 6 model)" if m["tunable"] else ""))
        lines.append(f"      {'setting':<34} {'validation all':>14} {'validation contested':>21} {'train all':>10} {'gap (all)':>10}")
        for row in m["settings"]:
            mark = ("*" if row["is_library_default"] and m["tunable"] else " ") + " "
            tail = "  <- best" if row["index"] == m["best_index"] else ""
            lines.append(f"    {mark}{describe(row['params']):<34} {_f(row['validation']['all']):>14} {_f(row['validation']['contested']):>21} {_f(row['train']['all']):>10} {_s(row['gap']['all']):>10}{tail}")
        if m["tunable"]:
            lines.append(f"      best minus library default, validation all markets: {m['best_minus_library_default_all']:+.4f}")
    bests = {b["model"]: b for b in r["best_per_model"]}
    for sname in SLICES:
        s = r["validation"][sname]
        if s["markets"] == 0:
            lines.append(f"  {sname}: no markets in validation")
            continue
        lines.append(f"  best of each model, {sname} (validation {s['markets']:,} markets, {s['yes_rate']:.1%} Yes)")
        lines.append(f"    {'model':<18} {'log loss (range)':<30} {'minus raw_price (range)'}")
        for n in r["order"]:
            w = s["wobble"][n]["log_loss"]
            ll = f"{_f(s['scores'][n]['log_loss'])} ({_f(w['low'])} to {_f(w['high'])})"
            if n == "raw_price":
                diff = ""
            else:
                d = s["difference_vs_raw_price"][n]["log_loss"]
                diff = f"{d['point']:+.4f} ({d['low']:+.4f} to {d['high']:+.4f}): {d['verdict']}"
            tag = f"  [{describe(bests[n]['params'])}]" if n in bests and r["models"][n]["tunable"] else ""
            lines.append(f"    {n:<18} {ll:<30} {diff}{tag}")
    c = r["choice"]
    lines.append("  Choice (validation log loss, all markets)")
    lines.append(f"    winner:    {c['winner']} [{describe(c['params'])}], log loss {_f(c['validation']['all'])}")
    lines.append(f"    runner-up: {c['runner_up']} [{describe(c['runner_up_params'])}], log loss {_f(c['runner_up_validation']['all'])}")
    for sname in SLICES:
        d = c["margin_runner_up_minus_winner"].get(sname)
        if d:
            lines.append(f"    margin, {sname} (runner-up minus winner, positive = winner ahead): {d['point']:+.4f} ({d['low']:+.4f} to {d['high']:+.4f}): {d['reading']}")
    if c["tie_at_4_decimals"]:
        lines.append("    The winner and the runner-up are equal at 4 decimals; the earlier model in the ladder won the tie.")
    d = c["winner_minus_raw_price"].get("all")
    if d:
        lines.append(f"    winner minus raw_price, all markets: {d['point']:+.4f} ({d['low']:+.4f} to {d['high']:+.4f}): {d['verdict']}")
    if c["lowest_contested_model"] and c["lowest_contested_model"] != c["winner"]:
        lines.append(f"    On the contested slice alone {c['lowest_contested_model']} has the lowest log loss; the rule decides on all markets.")
    lines.append("    The winner's validation score is slightly flattering: it was picked on these markets. The one-time test is the honest number.")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Light tuning of the model ladder on validation, and the choice")
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
    train = split.read_split(data_dir, config, "train", k)
    validation = split.read_split(data_dir, config, "validation", k)
    try:
        report = run(train, validation, config)
    except ValueError as e:
        print(f"PROBLEM {e}")
        return 1
    out_dir = settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"tuning_k{k}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(summary_lines(report)))
    print("Nothing is frozen yet. Read the choice above, then run src/freeze.py. Test was not read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
