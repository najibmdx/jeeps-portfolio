"""Stage 5: the three baselines, scored on VALIDATION only.

    python src/baselines.py [--data-dir DIR] [--k 7]       (needs the split files written by split.py)

The baselines are the yardstick every later model must beat (decision 5 of the design record):
  raw_price              the last close before t, used as the probability of Yes.
  price_only_correction  the same price, passed through a rule fitted on TRAIN: sigmoid(intercept + slope * logit(price)).
                         Slope below 1 means prices were too extreme, above 1 too timid; the intercept moves every price up or down.
  base_rate              the share of Yes in TRAIN, for every market.

Rules fixed with Jeeps on 24 Nov 2026:
  - Prices (and every forecast) are clipped into [price_clip, 1 - price_clip] before scoring, so a price of exactly 0 or 1
    that meets the other outcome costs a large number, not infinity. Same clip for all three baselines.
  - Log loss decides. Brier is shown beside it. Both are shown for all markets and for the contested slice.
  - The calibration table uses fixed price bands (config metrics.calibration_bands).
  - The wobble resamples whole EVENTS (never single markets), the same draws for every baseline, so differences are paired.

Train and validation are read here. The test split is never read.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import settings
import split

BASELINES = settings.KNOWN_BASELINES
METRICS = ("log_loss", "brier")
NEEDED = ("label", "last_close", "event_key")


# ---- scores

def log_loss(y, p) -> float:
    y, p = np.asarray(y, dtype="float64"), np.asarray(p, dtype="float64")
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("log loss needs probabilities strictly between 0 and 1 (clip them first)")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier(y, p) -> float:
    y, p = np.asarray(y, dtype="float64"), np.asarray(p, dtype="float64")
    return float(np.mean((p - y) ** 2))


def _loss_each(metric, y, p):
    """The score of every single market (so they can be summed per event)."""
    if metric == "log_loss":
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return (p - y) ** 2


# ---- the forecasts

def clip_probability(p, eps):
    return np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)


def _logit(p):
    return np.log(p / (1 - p))


def fit_correction(price, y, eps, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Intercept and slope of a logistic regression of the outcome on logit(clipped price), no penalty (Newton steps)."""
    z = _logit(clip_probability(price, eps))
    y = np.asarray(y, dtype="float64")
    if len(y) == 0 or y.min() == y.max():
        raise ValueError("the correction needs train markets with both outcomes")
    if np.ptp(z) == 0:
        raise ValueError("the correction needs train prices that are not all the same")
    X = np.column_stack([np.ones_like(z), z])

    def loglik(b):
        eta = X @ b
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    beta = np.array([0.0, 1.0])  # start from "the price is right"
    current = loglik(beta)
    for _ in range(max_iter):
        p = expit(X @ beta)
        w = p * (1 - p)
        try:
            step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - p))
        except np.linalg.LinAlgError:
            raise ValueError("the correction could not be fitted (the train prices may separate the outcomes perfectly)") from None
        scale = 1.0
        while loglik(beta + scale * step) < current - 1e-12 and scale > 1e-12:  # halve a step that makes the fit worse
            scale /= 2
        beta = beta + scale * step
        current = loglik(beta)
        if np.max(np.abs(scale * step)) < tol:
            return {"intercept": float(beta[0]), "slope": float(beta[1])}
    raise ValueError("the correction did not settle (the train prices may separate the outcomes perfectly)")


def apply_correction(price, params, eps):
    z = _logit(clip_probability(price, eps))
    return clip_probability(expit(params["intercept"] + params["slope"] * z), eps)


def forecasts(train, validation, config):
    """The three baselines for the validation markets, already clipped. Returns (forecasts, fitted-things)."""
    eps = config["metrics"]["price_clip"]
    params = fit_correction(train["last_close"].to_numpy(), train["label"].to_numpy(), eps)
    base = float(train["label"].mean())
    price = validation["last_close"].to_numpy(dtype="float64")
    out = {
        "raw_price": clip_probability(price, eps),
        "price_only_correction": apply_correction(price, params, eps),
        "base_rate": clip_probability(np.full(len(validation), base), eps),
    }
    return out, {"correction": params, "base_rate_train": base}


# ---- slices and the calibration table

def slice_masks(validation, config) -> dict:
    """all markets, and contested: contested_low <= last close < contested_high (the Stage 1 definition)."""
    u = config["universe"]
    price = validation["last_close"].to_numpy(dtype="float64")
    return {"all": np.ones(len(validation), dtype=bool), "contested": (price >= u["contested_low"]) & (price < u["contested_high"])}


def band_labels(edges) -> list:
    out = [f"{lo:g} to {hi:g}" for lo, hi in zip(edges[:-2], edges[1:-1])]
    return out + [f"{edges[-2]:g} and above"]


def calibration_rows(p, y, edges) -> list:
    """Per price band [low, high): markets, mean forecast, actual Yes rate. The last band includes 1."""
    p, y = np.asarray(p, dtype="float64"), np.asarray(y, dtype="float64")
    band = np.digitize(p, edges[1:-1])  # edges[1:-1] are the inner lines; 0 .. len(edges)-2
    rows = []
    for i, label in enumerate(band_labels(edges)):
        m = band == i
        n = int(m.sum())
        rows.append({"band": label, "markets": n,
                     "mean_forecast": float(p[m].mean()) if n else None,
                     "yes_rate": float(y[m].mean()) if n else None})
    return rows


# ---- the wobble: resample whole events

def resample_weights(n_events, draws, seed) -> np.ndarray:
    """draws x n_events: how many times each event appears in each re-drawn validation set."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_events, size=(draws, n_events))
    return np.stack([np.bincount(row, minlength=n_events) for row in idx]).astype("float64")


def weighted_scores(weights, event_sums, event_counts) -> np.ndarray:
    """The score of every re-drawn set: total loss over total markets, events counted as often as they were drawn."""
    total, n = weights @ event_sums, weights @ event_counts
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / n, np.nan)


def verdict(low, high) -> str:
    """For a difference in a score where LOWER is better, against the raw price."""
    if low > 0:
        return "worse"
    if high < 0:
        return "better"
    return "cannot be told apart"


def _range(draws, interval) -> dict:
    lo = (1 - interval) / 2
    a, b = np.nanquantile(draws, [lo, 1 - lo])
    return {"low": float(a), "high": float(b)}


def score_slice(validation, fc, mask, weights, codes, n_events, interval, names=BASELINES) -> dict:
    """Scores, wobble and paired differences against raw_price for the forecasts called `names` (raw_price must be one)."""
    if "raw_price" not in names:
        raise ValueError("the forecasts to score must include raw_price: every difference is taken against it")
    y_all = validation["label"].to_numpy(dtype="float64")
    n = int(mask.sum())
    if n == 0:
        return {"markets": 0, "events": 0, "yes_rate": None, "note": "no markets in this slice"}
    y = y_all[mask]
    result = {"markets": n, "events": int(len(np.unique(codes[mask]))), "yes_rate": float(y.mean()), "scores": {}, "wobble": {}, "difference_vs_raw_price": {}}
    draws = {}
    for name in names:
        p = fc[name][mask]
        result["scores"][name] = {"log_loss": log_loss(y, p), "brier": brier(y, p)}
        result["wobble"][name] = {}
        for metric in METRICS:
            loss = _loss_each(metric, y, p)
            sums = np.bincount(codes[mask], weights=loss, minlength=n_events)
            counts = np.bincount(codes[mask], minlength=n_events).astype("float64")
            draws[(name, metric)] = weighted_scores(weights, sums, counts)
            result["wobble"][name][metric] = _range(draws[(name, metric)], interval)
    for name in [n for n in names if n != "raw_price"]:
        result["difference_vs_raw_price"][name] = {}
        for metric in METRICS:
            diff = draws[(name, metric)] - draws[("raw_price", metric)]
            r = _range(diff, interval)
            r["point"] = result["scores"][name][metric] - result["scores"]["raw_price"][metric]
            r["verdict"] = verdict(r["low"], r["high"])
            result["difference_vs_raw_price"][name][metric] = r
    return result


def paired_difference(validation, p_a, p_b, mask, weights, codes, n_events, interval, metric="log_loss") -> dict:
    """Forecast A minus forecast B in one score (lower is better), with the paired wobble: both are scored on the same re-drawn sets of whole events.

    Returns point, low, high and a verdict about A against B ("better" if the whole range is below 0, "worse" if above, else "cannot be told apart")."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if int(mask.sum()) == 0:
        raise ValueError("no markets in this slice")
    score = {"log_loss": log_loss, "brier": brier}[metric]
    y = validation["label"].to_numpy(dtype="float64")[mask]
    a, b = np.asarray(p_a, dtype="float64")[mask], np.asarray(p_b, dtype="float64")[mask]
    counts = np.bincount(codes[mask], minlength=n_events).astype("float64")
    draws = [weighted_scores(weights, np.bincount(codes[mask], weights=_loss_each(metric, y, p), minlength=n_events), counts) for p in (a, b)]
    r = _range(draws[0] - draws[1], interval)
    r["point"] = score(y, a) - score(y, b)
    r["verdict"] = verdict(r["low"], r["high"])
    return r


def run(train, validation, config) -> dict:
    """Fit on train, score on validation. Needs nothing but the train and validation rows."""
    for part, df in (("train", train), ("validation", validation)):
        miss = [c for c in NEEDED if c not in df.columns]
        if miss:
            raise ValueError(f"the {part} table lacks the columns the baselines need: {miss}")
        if len(df) == 0:
            raise ValueError(f"the {part} table has no markets")
    m, ev = config["metrics"], config["evaluation"]
    fc, fitted = forecasts(train, validation, config)
    codes, uniques = pd.factorize(validation["event_key"])
    n_events = len(uniques)
    weights = resample_weights(n_events, ev["bootstrap_draws"], config["seed"])
    masks = slice_masks(validation, config)
    y = validation["label"].to_numpy(dtype="float64")
    slices = {}
    for name, mask in masks.items():
        s = score_slice(validation, fc, mask, weights, codes, n_events, ev["interval"])
        s["calibration"] = {b: calibration_rows(fc[b][mask], y[mask], m["calibration_bands"]) for b in ("raw_price", "price_only_correction")}
        slices[name] = s
    return {
        "train_markets": len(train), "train_events": int(train["event_key"].nunique()),
        "validation_markets": len(validation), "validation_events": n_events,
        "price_clip": m["price_clip"], "bootstrap_draws": ev["bootstrap_draws"], "interval": ev["interval"],
        "base_rate_train": fitted["base_rate_train"], "correction": fitted["correction"],
        "slices": slices,
        "notes": [
            "Scored on validation only. The test split was not read.",
            "The wobble resamples whole events inside validation. It covers sampling noise, not the shift between periods.",
            "The correction and the base rate are learned on train; validation can have a different Yes rate (see the Stage 3 report).",
        ],
    }


# ---- printing

def _fmt(x) -> str:
    return f"{x:.4f}"


def summary_lines(r) -> list:
    c = r["correction"]
    lines = [
        f"Baselines on validation ({r['validation_markets']:,} markets, {r['validation_events']:,} events); "
        f"correction and base rate fitted on train ({r['train_markets']:,} markets)",
        f"  correction: intercept {c['intercept']:+.3f}, slope {c['slope']:.3f} (slope 1 and intercept 0 would mean the price is right)",
        f"  base rate from train: {r['base_rate_train']:.1%}.  Prices clipped to [{r['price_clip']}, {1 - r['price_clip']}].  "
        f"Ranges are the middle {r['interval']:.0%} of {r['bootstrap_draws']} re-drawn sets of whole events.",
    ]
    for sname, s in r["slices"].items():
        if s["markets"] == 0:
            lines.append(f"  {sname}: no markets")
            continue
        lines.append(f"  {sname} ({s['markets']:,} markets, {s['events']:,} events, {s['yes_rate']:.1%} Yes)")
        lines.append(f"    {'baseline':<22} {'log loss (range)':<28} Brier (range)")
        for b in BASELINES:
            sc, w = s["scores"][b], s["wobble"][b]
            ll = f"{_fmt(sc['log_loss'])} ({_fmt(w['log_loss']['low'])} to {_fmt(w['log_loss']['high'])})"
            br = f"{_fmt(sc['brier'])} ({_fmt(w['brier']['low'])} to {_fmt(w['brier']['high'])})"
            lines.append(f"    {b:<22} {ll:<28} {br}")
        for b in BASELINES[1:]:
            d = s["difference_vs_raw_price"][b]["log_loss"]
            lines.append(f"    log loss, {b} minus raw_price: {d['point']:+.4f} ({d['low']:+.4f} to {d['high']:+.4f}): {d['verdict']}")
    all_ = r["slices"]["all"]
    if all_["markets"]:
        lines.append("  calibration, all markets (price band: markets, mean forecast, actual Yes rate)")
        for b in ("raw_price", "price_only_correction"):
            lines.append(f"    {b}")
            for row in all_["calibration"][b]:
                if row["markets"]:
                    lines.append(f"      {row['band']:<16} {row['markets']:>6,}  {row['mean_forecast']:.3f}  {row['yes_rate']:.3f}")
                else:
                    lines.append(f"      {row['band']:<16} {0:>6,}")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Baselines scored on the validation split")
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
    (out_dir / f"baselines_k{k}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(summary_lines(report)))
    print("Test was not read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
