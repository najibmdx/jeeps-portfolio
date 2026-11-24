"""Stage 6: the model ladder, untuned. Fitted on TRAIN, scored on train and on VALIDATION. Nothing is chosen here.

    python src/models.py [--data-dir DIR] [--k 7]       (needs the split files written by split.py)

The ladder (decision 7 of the design record):
  lr_price_only     the Stage 5 price correction (intercept and slope on the log-odds of the last close). Nothing new to build.
  lr_all_features   logistic regression on all model inputs.
  random_forest     random forest on all model inputs.
  gradient_boosting histogram gradient boosting on all model inputs.

Untuned means library defaults and the fixed seed. The only settings written in config.yaml are 300 trees for the
forest and 1,000 steps for the logistic regression. Light tuning and the choice of a winner are Stage 7.

Rules fixed with Jeeps on 24 Nov 2026:
  - has_fee is not a model input: it is True for 6 train markets and about none of validation, so nothing can be learned
    from it. It stays in the leakage table; config.yaml models.dropped_features records the reason.
  - category becomes one yes/no column per category (one-hot) for every model. A category not seen in train becomes all zeros.
    A market with no category is a category of its own. A missing number is never filled in: the run stops.
  - Stale prices get no special handling: days_since_last_bar is an ordinary number.
  - The logistic regression sees the price columns as log-odds (the five price levels), so that it starts from the same
    input as the price-only rung and can only add to it. The forests see the raw prices (a monotone change makes no difference to a tree).
  - Every forecast is clipped into [price_clip, 1 - price_clip] before scoring, as in Stage 5.

Train and validation are read here. The test split is never read. Validation labels are used only to score, never to fit.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import baselines
import leakage
import settings
import split

CATEGORY = "category"
MISSING_CATEGORY = "(missing)"
REFERENCE = ("raw_price", "base_rate")  # scored next to the ladder


# ---- which columns go in

def input_columns(config):
    """(columns the models read, {dropped column: reason}). Order as in config.yaml."""
    f = config["features"]
    listed = list(f["price_path"]) + list(f["activity"]) + list(f["context"])
    dropped = dict(config["models"]["dropped_features"])
    return [c for c in listed if c not in dropped], dropped


def model_inputs(df, columns):
    """The model's input table: yes/no columns become 0/1 numbers; a missing category becomes its own category; a missing number is refused, not filled."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"the table lacks model input columns: {missing}")
    X = df[columns].copy()
    for c in columns:
        if leakage._is_yes_no(X[c]):
            X[c] = X[c].astype("float64")
    if CATEGORY in columns:  # a market with no category is a category of its own, as in the leakage screens
        X[CATEGORY] = X[CATEGORY].map(lambda v: MISSING_CATEGORY if pd.isna(v) else str(v))
    holes = [c for c in columns if X[c].isna().any()]
    if holes:
        raise ValueError(f"missing values in model inputs, not filled in silently: {holes}")
    return X


def _log_odds(x, eps):
    x = np.clip(np.asarray(x, dtype="float64"), eps, 1 - eps)
    return np.log(x / (1 - x))


def build_model(name, config, columns, params=None):
    """One model (a scikit-learn pipeline) for the ladder step `name`. lr_price_only is not built here.

    `params` sets the tunable settings of Stage 7 (see settings.TUNABLE); None or {} gives the untuned model of Stage 6."""
    params = dict(params or {})
    unknown = [p for p in params if p not in settings.TUNABLE.get(name, ())]
    if unknown:
        raise ValueError(f"{name} cannot tune {unknown}; it can tune {list(settings.TUNABLE.get(name, ()))}")
    eps, seed = config["metrics"]["price_clip"], config["seed"]
    numbers = [c for c in columns if c != CATEGORY]
    category = [c for c in columns if c == CATEGORY]
    one_hot = ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), category)
    if name == "lr_all_features":
        odds = [c for c in config["models"]["log_odds_columns"] if c in numbers]
        rest = [c for c in numbers if c not in odds]
        pre = ColumnTransformer([
            ("price_levels", Pipeline([("log_odds", FunctionTransformer(_log_odds, kw_args={"eps": eps}, feature_names_out="one-to-one")), ("scale", StandardScaler())]), odds),
            ("number", StandardScaler(), rest), one_hot,
        ])
        return Pipeline([("prepare", pre), ("model", LogisticRegression(max_iter=config["models"]["logistic_max_iter"], random_state=seed, **params))])
    pre = ColumnTransformer([("number", "passthrough", numbers), one_hot])
    if name == "random_forest":
        return Pipeline([("prepare", pre), ("model", RandomForestClassifier(n_estimators=config["models"]["forest_trees"], random_state=seed, n_jobs=-1, **params))])
    if name == "gradient_boosting":
        return Pipeline([("prepare", pre), ("model", HistGradientBoostingClassifier(random_state=seed, **params))])
    raise ValueError(f"no model is built for {name}")


# ---- fitting and forecasting

def fit_models(train, config):
    """Fit every ladder step on the train rows only. Returns ({name: fitted thing}, columns, correction)."""
    columns, _ = input_columns(config)
    eps = config["metrics"]["price_clip"]
    correction = baselines.fit_correction(train["last_close"].to_numpy(), train["label"].to_numpy(), eps)
    X, y = model_inputs(train, columns), train["label"].to_numpy()  # a one-outcome train table was already refused by fit_correction above
    fitted = {}
    for name in config["models"]["ladder"]:
        fitted[name] = correction if name == "lr_price_only" else build_model(name, config, columns).fit(X, y)
    return fitted, columns, correction


def forecast_all(fitted, columns, train_base_rate, df, config):
    """P(Yes) for every market of `df`, clipped. Uses only the input columns and last_close, never the label."""
    eps = config["metrics"]["price_clip"]
    out = {
        "raw_price": baselines.clip_probability(df["last_close"].to_numpy(dtype="float64"), eps),
        "base_rate": baselines.clip_probability(np.full(len(df), train_base_rate), eps),
    }
    X = model_inputs(df, columns)
    for name, thing in fitted.items():
        if name == "lr_price_only":
            out[name] = baselines.apply_correction(df["last_close"].to_numpy(dtype="float64"), thing, eps)
        else:
            out[name] = baselines.clip_probability(thing.predict_proba(X)[:, 1], eps)
    return out


# ---- the whole run

def _train_scores(train, fc, config):
    y = train["label"].to_numpy(dtype="float64")
    out = {}
    for sname, mask in baselines.slice_masks(train, config).items():
        out[sname] = {"markets": int(mask.sum()), "scores": {}}
        if mask.sum():
            out[sname]["scores"] = {n: {"log_loss": baselines.log_loss(y[mask], p[mask]), "brier": baselines.brier(y[mask], p[mask])} for n, p in fc.items()}
    return out


def run(train, validation, config) -> dict:
    """Fit on train, score on train and on validation. Needs nothing but the train and validation rows."""
    for part, df in (("train", train), ("validation", validation)):
        miss = [c for c in baselines.NEEDED if c not in df.columns]
        if miss:
            raise ValueError(f"the {part} table lacks the columns the models need: {miss}")
        if len(df) == 0:
            raise ValueError(f"the {part} table has no markets")
    fitted, columns, correction = fit_models(train, config)
    base = float(train["label"].mean())
    fc_train = forecast_all(fitted, columns, base, train, config)
    fc_val = forecast_all(fitted, columns, base, validation, config)
    names = list(REFERENCE) + list(config["models"]["ladder"])
    ev = config["evaluation"]
    codes, uniques = pd.factorize(validation["event_key"])
    weights = baselines.resample_weights(len(uniques), ev["bootstrap_draws"], config["seed"])
    masks = baselines.slice_masks(validation, config)
    val = {}
    for sname, mask in masks.items():
        s = baselines.score_slice(validation, {n: fc_val[n] for n in names}, mask, weights, codes, len(uniques), ev["interval"], names=names)
        val[sname] = s
    train_part = _train_scores(train, {n: fc_train[n] for n in names}, config)
    gap = {}
    for sname in val:
        if val[sname]["markets"] and train_part[sname]["markets"]:
            gap[sname] = {n: val[sname]["scores"][n]["log_loss"] - train_part[sname]["scores"][n]["log_loss"] for n in names}
    dropped = input_columns(config)[1]
    return {
        "train_markets": len(train), "train_events": int(train["event_key"].nunique()),
        "validation_markets": len(validation), "validation_events": len(uniques),
        "price_clip": config["metrics"]["price_clip"], "bootstrap_draws": ev["bootstrap_draws"], "interval": ev["interval"],
        "inputs": columns, "dropped_inputs": dropped,
        "category_levels_in_train": sorted(map(str, train[CATEGORY].unique())) if CATEGORY in columns else [],
        "category_levels_only_in_validation": sorted(set(map(str, validation[CATEGORY].unique())) - set(map(str, train[CATEGORY].unique()))) if CATEGORY in columns else [],
        "settings": {"seed": config["seed"], "forest_trees": config["models"]["forest_trees"], "logistic_max_iter": config["models"]["logistic_max_iter"],
                     "other": "library defaults"},
        "order": names, "correction": correction, "base_rate_train": base,
        "train": train_part, "validation": val, "validation_minus_train_log_loss": gap,
        "notes": [
            "Untuned. Nothing is chosen here; the choice is Stage 7.",
            "Scored on train and validation only. The test split was not read. Validation labels were used to score, never to fit.",
            "The ranges resample whole events inside validation. They cover sampling noise, not the change between periods.",
            "lr_price_only is the Stage 5 price correction under its ladder name.",
        ],
    }


# ---- printing

def _f(x) -> str:
    return f"{x:.4f}"


def summary_lines(r) -> list:
    lines = [
        f"Model ladder, untuned (library defaults, seed {r['settings']['seed']}). Fitted on train ({r['train_markets']:,} markets), "
        f"scored on train and on validation ({r['validation_markets']:,} markets, {r['validation_events']:,} events)",
        f"  inputs ({len(r['inputs'])}): {', '.join(r['inputs'])}",
    ]
    lines += [f"  not a model input: {c} ({why})" for c, why in r["dropped_inputs"].items()]
    only = r["category_levels_only_in_validation"]
    if only:
        lines.append(f"  categories in validation but not in train (all zeros for the models): {', '.join(only)}")
    lines.append(f"  Ranges are the middle {r['interval']:.0%} of {r['bootstrap_draws']} re-drawn sets of whole events. Overfit gap = validation log loss minus train log loss.")
    for sname, s in r["validation"].items():
        t = r["train"][sname]
        if s["markets"] == 0 or t["markets"] == 0:
            lines.append(f"  {sname}: no markets in train or validation")
            continue
        lines.append(f"  {sname} (validation {s['markets']:,} markets, {s['yes_rate']:.1%} Yes; train {t['markets']:,} markets)")
        lines.append(f"    {'model':<18} {'train log loss':>14} {'validation log loss (range)':>34} {'overfit gap':>12} {'validation Brier':>17}")
        for n in r["order"]:
            w = s["wobble"][n]["log_loss"]
            ll = f"{_f(s['scores'][n]['log_loss'])} ({_f(w['low'])} to {_f(w['high'])})"
            lines.append(f"    {n:<18} {_f(t['scores'][n]['log_loss']):>14} {ll:>34} {r['validation_minus_train_log_loss'][sname][n]:>+12.4f} {_f(s['scores'][n]['brier']):>17}")
        for n in r["order"]:
            if n == "raw_price":
                continue
            d = s["difference_vs_raw_price"][n]["log_loss"]
            lines.append(f"    log loss, {n} minus raw_price: {d['point']:+.4f} ({d['low']:+.4f} to {d['high']:+.4f}): {d['verdict']}")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The untuned model ladder, scored on train and validation")
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
    (out_dir / f"models_k{k}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(summary_lines(report)))
    print("Test was not read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
