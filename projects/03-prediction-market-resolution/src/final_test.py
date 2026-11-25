"""Stage 9: the one-time test. The ONLY place in the project that reads the test split.

    python src/final_test.py --confirm-open-test [--data-dir DIR] [--k 7]

Rules fixed with Jeeps on 25 Nov 2026, before the test split was opened:
  - Without --confirm-open-test the script only prints what it would do, reads nothing and exits.
  - Before anything is opened it checks, in this order: the lock file does not exist; the train and validation splits exist;
    frozen_choice.yaml passes its own checks (freeze.read_frozen) and is committed in git with no uncommitted change;
    the split report says the test split has the expected number of markets; the frozen model, refitted on TRAIN ONLY,
    reproduces the validation score written in frozen_choice.yaml. If any of these fails, nothing is opened.
  - Then it writes the lock file (exclusive create: it cannot be written twice), and only then reads the test split,
    through read_test, which refuses to read without the lock. The lock records the time and a hash of frozen_choice.yaml.
  - The frozen model is NOT refitted on train + validation (Stage 7 decision) and nothing is retuned.
  - Scored on test: the frozen model and the three baselines (raw price, price-only correction fitted on train, base rate).
    The other ladder models are NOT scored on test: that would be choosing after seeing test.
  - Both slices, log loss decides, Brier and the calibration table beside it, wobble from whole events re-drawn
    (the same draws for every forecast, so differences are paired). The frozen model is compared in pairs with the raw price
    and with the price-only correction.
  - The headline is the slice in config final_test.headline_slice (contested); its verdict against the raw price is read with
    the sentence for that verdict in config final_test.readings. The same numbers are shown for validation; if validation and
    test disagree, both are reported.
  - The test-window check shows days from t to close in test next to train and validation (the censoring effect), and how the
    test markets differ from the others. It is descriptive and changes nothing.
  - If the script stops AFTER the lock is written and BEFORE any score was computed or printed (a typo, say), the lock may be
    deleted by hand, and that must be written in decision_log.md. Once the report file exists, the lock is permanent.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import freeze
import models
import settings
import split

FROZEN = "frozen_model"
SLICES = ("all", "contested")
NAMES = ("raw_price", "price_only_correction", "base_rate", FROZEN)


# ---- the lock and the one reader of the test split

def lock_path(data_dir, config) -> Path:
    return Path(data_dir) / config["models"]["test_lock_file"]


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_lock(data_dir, config, k, frozen_path, now=None) -> Path:
    """Exclusive create: writing the lock twice is impossible, even by two runs at the same moment."""
    path = lock_path(data_dir, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(timezone.utc)
    body = {"opened_at": now.isoformat(), "k": k, "frozen_file": Path(frozen_path).name, "frozen_sha256": sha256_of(frozen_path)}
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(body, indent=2))
    return path


def read_test(data_dir, config, k) -> pd.DataFrame:
    """The only reader of the test split. Refuses unless the lock file exists."""
    if not lock_path(data_dir, config).exists():
        raise PermissionError("the test split can only be read after the lock file is written")
    return pd.read_parquet(split.split_path(data_dir, config, "test", k))


def frozen_committed(frozen_path, cwd) -> "str | None":
    """None if frozen_choice.yaml is committed in git with no uncommitted change; otherwise what is wrong."""
    name = Path(frozen_path).name
    try:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", name], cwd=cwd, capture_output=True, text=True)
        if tracked.returncode != 0:
            return f"{name} is not committed in git. It must be committed before the test split is opened."
        status = subprocess.run(["git", "status", "--porcelain", "--", name], cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError:
        return f"git was not found, so I cannot check that {name} was committed before the test split is opened"
    if status.returncode != 0:
        return f"git could not say whether {name} is committed: {status.stderr.strip()}"
    if status.stdout.strip():
        return f"{name} has a change that is not committed ({status.stdout.strip()}). Commit it before the test split is opened."
    return None


# ---- fitting and checking the frozen model (train and validation only)

def fit_frozen(train, config, model, params):
    """The frozen model fitted on the train rows only. Returns a function: table -> clipped P(Yes)."""
    columns, _ = models.input_columns(config)
    eps = config["metrics"]["price_clip"]
    base = float(train["label"].mean())
    if model == "lr_price_only":
        thing = baselines.fit_correction(train["last_close"].to_numpy(), train["label"].to_numpy(), eps)
    else:
        thing = models.build_model(model, config, columns, params).fit(models.model_inputs(train, columns), train["label"].to_numpy())
    return lambda df: models.forecast_all({model: thing}, columns, base, df, config)[model]


def all_forecasts(train, df, config, predict) -> dict:
    fc, _ = baselines.forecasts(train, df, config)
    fc[FROZEN] = predict(df)
    return fc


def _slice_loss(df, p, mask) -> float:
    return baselines.log_loss(df["label"].to_numpy(dtype="float64")[mask], p[mask])


def check_reproduces(train, validation, config, predict, frozen_validation) -> dict:
    """The frozen model, refitted on train, must give the validation score that frozen_choice.yaml holds, on both slices."""
    tol = config["final_test"]["reproduce_tolerance"]
    p = predict(validation)
    masks = baselines.slice_masks(validation, config)
    out = {}
    for sname, key in (("all", "log_loss_all"), ("contested", "log_loss_contested")):
        want = frozen_validation.get(key)
        if want is None or not masks[sname].any():
            continue
        got = _slice_loss(validation, p, masks[sname])
        out[sname] = {"frozen": want, "this_run": got, "difference": got - want}
        if abs(got - want) > tol:
            raise ValueError(f"the frozen model, refitted on train, scores {got:.10f} on validation ({sname}) but frozen_choice.yaml says {want:.10f}: "
                             "it is not the model that was frozen, so the test is not opened")
    if "all" not in out:
        raise ValueError("frozen_choice.yaml holds no validation score to reproduce, so the test is not opened")
    return out


# ---- the report

def _compact(diff) -> dict:
    return {k: diff[k] for k in ("point", "low", "high", "verdict")}


def _view(df, fc, config, names) -> dict:
    """Scores, wobble, differences against the raw price and calibration, per slice, for `df` (validation or test)."""
    m, ev = config["metrics"], config["evaluation"]
    codes, uniques = pd.factorize(df["event_key"])
    n_events = len(uniques)
    weights = baselines.resample_weights(n_events, ev["bootstrap_draws"], config["seed"])
    y = df["label"].to_numpy(dtype="float64")
    slices = {}
    for sname, mask in baselines.slice_masks(df, config).items():
        s = baselines.score_slice(df, fc, mask, weights, codes, n_events, ev["interval"], names=list(names))
        if s["markets"] > 0:
            s["calibration"] = {b: baselines.calibration_rows(fc[b][mask], y[mask], m["calibration_bands"]) for b in (FROZEN, "raw_price")}
            s["frozen_minus_price_only_correction"] = baselines.paired_difference(df, fc[FROZEN], fc["price_only_correction"], mask, weights, codes, n_events, ev["interval"])
        slices[sname] = s
    return {"events": n_events, "slices": slices}


def window_check(parts: dict, config) -> dict:
    """Days from t to close, and how the markets look, for each part. Descriptive only."""
    out = {}
    for name, df in parts.items():
        t, close = pd.to_datetime(df["t"], utc=True), pd.to_datetime(df["close_time"], utc=True)
        days = ((close - t).dt.total_seconds() / 86400.0).to_numpy()
        mask = baselines.slice_masks(df, config)["contested"]
        cat = df["category"].fillna(models.MISSING_CATEGORY).value_counts(normalize=True) if "category" in df.columns else pd.Series(dtype="float64")
        out[name] = {
            "markets": int(len(df)), "t_first": t.min().isoformat(), "t_last": t.max().isoformat(),
            "days_t_to_close": {f"p{p:g}": float(np.percentile(days, p)) for p in config["final_test"]["window_percentiles"]},
            "share_closing_within_1_day": float((days <= 1.0).mean()),
            "yes_rate": float(df["label"].mean()), "share_contested": float(mask.mean()),
            "mean_last_close": float(df["last_close"].mean()),
            "category_share": {str(c): float(v) for c, v in cat.items()},
        }
    return out


def headline(test_view, validation_view, config) -> "dict | None":
    f = config["final_test"]
    sname = f["headline_slice"]
    t = test_view["slices"][sname]
    if t["markets"] == 0:
        return None
    d = t["difference_vs_raw_price"][FROZEN]["log_loss"]
    out = {"slice": sname, "test": _compact(d), "reading": f["readings"][d["verdict"]]}
    v = validation_view["slices"][sname]
    if v["markets"] > 0:
        vd = v["difference_vs_raw_price"][FROZEN]["log_loss"]
        out["validation"] = _compact(vd)
        same_verdict, same_sign = vd["verdict"] == d["verdict"], bool(np.sign(vd["point"]) == np.sign(d["point"]))
        out["validation_and_test"] = "agree" if same_verdict and same_sign else ("same verdict, opposite direction" if same_verdict else "different verdicts")
    return out


def score_test(train, validation, test, config, predict, model, params, reproduced=None, frozen_sha=None, opened_at=None) -> dict:
    """Everything the test reports. Pure: needs only the three tables and the fitted model."""
    for part, df in (("train", train), ("validation", validation), ("test", test)):
        miss = [c for c in ("label", "last_close", "event_key", "t", "close_time") if c not in df.columns]
        if miss:
            raise ValueError(f"the {part} table lacks the columns the test needs: {miss}")
        if len(df) == 0:
            raise ValueError(f"the {part} table has no markets")
    fc_test = all_forecasts(train, test, config, predict)
    fc_val = all_forecasts(train, validation, config, predict)
    tv, vv = _view(test, fc_test, config, NAMES), _view(validation, fc_val, config, NAMES)
    ev = config["evaluation"]
    return {
        "model": model, "params": dict(params), "fitted_on": "train only (not refitted on train + validation)",
        "frozen_sha256": frozen_sha, "opened_at": opened_at,
        "settings": {"seed": config["seed"], "forest_trees": config["models"]["forest_trees"], "logistic_max_iter": config["models"]["logistic_max_iter"]},
        "train_markets": len(train), "validation_markets": len(validation), "test_markets": len(test), "test_events": tv["events"],
        "price_clip": config["metrics"]["price_clip"], "bootstrap_draws": ev["bootstrap_draws"], "interval": ev["interval"],
        "reproduces_frozen_on_validation": reproduced,
        "headline": headline(tv, vv, config),
        "test": tv, "validation": vv,
        "window_check": window_check({"train": train, "validation": validation, "test": test}, config),
        "rules": {"headline_slice": config["final_test"]["headline_slice"], "readings": config["final_test"]["readings"]},
        "notes": [
            "The test split was opened once, by this run. The frozen model was fitted on train only and nothing was retuned.",
            "Only the frozen model and the three baselines are scored on test. The other ladder models are not: that would be choosing after seeing test.",
            "The wobble resamples whole events inside the test split. It covers sampling noise, not the shift between periods.",
            "The test window is one stretch of weeks. Markets still open when the data ends are absent, so test leans towards markets that close quickly (see the window check).",
        ],
    }


# ---- printing

def _f(x) -> str:
    return f"{x:.4f}"


def summary_lines(r) -> list:
    lines = [
        f"FINAL TEST on {r['test_markets']:,} markets ({r['test_events']:,} events). Model: {r['model']} ({_describe(r['params'])}), the frozen choice, fitted on train only "
        f"({r['train_markets']:,} markets). Nothing was retuned. Ranges are the middle {r['interval']:.0%} of {r['bootstrap_draws']} re-drawn sets of whole events.",
    ]
    if r["reproduces_frozen_on_validation"]:
        rp = r["reproduces_frozen_on_validation"]["all"]
        lines.append(f"  Before the test was opened, the frozen model reproduced its validation score: {rp['this_run']:.6f} against {rp['frozen']:.6f}.")
    h = r["headline"]
    if h:
        t = h["test"]
        lines.append(f"  HEADLINE ({h['slice']}, frozen model minus raw price, log loss): {t['point']:+.4f} ({t['low']:+.4f} to {t['high']:+.4f}): {t['verdict']}")
        lines.append(f"  {h['reading']}")
        if "validation" in h:
            v = h["validation"]
            lines.append(f"  Validation, same slice: {v['point']:+.4f} ({v['low']:+.4f} to {v['high']:+.4f}): {v['verdict']}. "
                         + {"agree": "Validation and test agree.",
                            "same verdict, opposite direction": "Validation and test give the same verdict, but the point estimates point in opposite directions: both are reported.",
                            "different verdicts": "Validation and test give DIFFERENT verdicts: both are reported, neither is hidden."}[h["validation_and_test"]])
    for sname in SLICES:
        s = r["test"]["slices"][sname]
        if s["markets"] == 0:
            lines.append(f"  {sname}: no markets in test")
            continue
        lines.append(f"  {sname} (test {s['markets']:,} markets, {s['yes_rate']:.1%} Yes)")
        lines.append(f"    {'forecast':<24} {'log loss (range)':>26}   {'Brier':>7}   minus raw_price (range)")
        for n in NAMES:
            w = s["wobble"][n]["log_loss"]
            tail = "" if n == "raw_price" else "{point:+.4f} ({low:+.4f} to {high:+.4f}): {verdict}".format(**s["difference_vs_raw_price"][n]["log_loss"])
            lines.append(f"    {n:<24} {_f(s['scores'][n]['log_loss']) + ' (' + _f(w['low']) + ' to ' + _f(w['high']) + ')':>26}   {_f(s['scores'][n]['brier']):>7}   {tail}")
        p = s["frozen_minus_price_only_correction"]
        lines.append(f"    frozen_model minus price_only_correction: {p['point']:+.4f} ({p['low']:+.4f} to {p['high']:+.4f}): {p['verdict']}")
        lines.append(f"    calibration, frozen model ({sname}): band, markets, mean forecast, actual Yes rate")
        for row in s["calibration"][FROZEN]:
            if row["markets"]:
                lines.append(f"      {row['band']:<16} {row['markets']:>6}   {row['mean_forecast']:.3f}   {row['yes_rate']:.3f}")
    lines.append("  Test-window check (days from t to close; share closing within a day; Yes rate; share contested):")
    for part, w in r["window_check"].items():
        pct = ", ".join(f"{k} {v:.2f}" for k, v in w["days_t_to_close"].items())
        lines.append(f"    {part:<11} {w['markets']:>6,} markets, t {w['t_first'][:10]} to {w['t_last'][:10]}; {pct}; within a day {w['share_closing_within_1_day']:.1%}; "
                     f"Yes {w['yes_rate']:.1%}; contested {w['share_contested']:.1%}")
    lines.append("  One window only. Markets still open when the data ends are absent, so test leans towards quick-closing markets.")
    lines.append("  The test split has now been opened. It is opened once: this was the only run.")
    return lines


def _describe(params) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in params.items()) if params else "no setting to tune"


def plan_lines(config, k, data_dir) -> list:
    return [
        "This step opens the test split ONCE. Nothing was opened or written because --confirm-open-test was not given.",
        "With --confirm-open-test it will, in this order:",
        f"  1. stop if the lock file ({lock_path(data_dir, config)}) already exists;",
        f"  2. check that {config['models']['frozen_file']} passes its checks and is committed in git with no change;",
        f"  3. check that the split report says the test split has {config['final_test']['expected_test_markets']:,} markets;",
        "  4. refit the frozen model on train only and check that it reproduces its validation score;",
        "  5. write the lock file, then read the test split, score the frozen model and the three baselines, and write the report.",
        "No second run is possible, nothing is refitted or retuned, and the other ladder models are not scored on test.",
    ]


# ---- the command line

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The one-time test: opens the test split once, with a lock file")
    parser.add_argument("--confirm-open-test", action="store_true", help="really open the test split (once)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--k", type=int, default=None, help="days after open (default: from config.yaml)")
    args = parser.parse_args(argv)
    config = settings.load_config()
    k = args.k or config["decision_time"]["days_after_open"]
    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_DIR
    if not args.confirm_open_test:
        print("\n".join(plan_lines(config, k, data_dir)))
        return 2
    if lock_path(data_dir, config).exists():
        print(f"PROBLEM the lock file {lock_path(data_dir, config)} exists: the test split was already opened. There is no second run.")
        return 1
    for part in ("train", "validation", "test"):
        if not split.split_path(data_dir, config, part, k).exists():
            print(f"No {part} split in {data_dir}. Run src/pipeline.py (or src/split.py) first. Nothing was opened.")
            return 2
    frozen_path = settings.PROJECT_DIR / config["models"]["frozen_file"]
    try:
        frozen = freeze.read_frozen(frozen_path, config)
    except ValueError as e:
        print(f"PROBLEM {e}. Nothing was opened.")
        return 1
    problem = frozen_committed(frozen_path, settings.PROJECT_DIR)
    if problem:
        print(f"PROBLEM {problem} Nothing was opened.")
        return 1
    report_path = settings.PROJECT_DIR / config["evaluation"]["outputs_folder"] / f"split_report_k{k}.json"
    if not report_path.exists():
        print(f"PROBLEM there is no split report at {report_path}, so I cannot check the size of the test split. Run src/split.py. Nothing was opened.")
        return 1
    want = config["final_test"]["expected_test_markets"]
    have = json.loads(report_path.read_text(encoding="utf-8"))["final"]["sizes"]["test"]
    if have != want:
        print(f"PROBLEM the split report says the test split has {have:,} markets, not the {want:,} that were agreed. Nothing was opened.")
        return 1
    train = split.read_split(data_dir, config, "train", k)
    validation = split.read_split(data_dir, config, "validation", k)
    try:
        predict = fit_frozen(train, config, frozen["model"], frozen["params"])
        reproduced = check_reproduces(train, validation, config, predict, frozen["validation"])
    except ValueError as e:
        print(f"PROBLEM {e}.")
        return 1
    # ---- from here on the test split is opened: the lock is written first
    try:
        lock = write_lock(data_dir, config, k, frozen_path)
    except FileExistsError:
        print("PROBLEM the lock file appeared while this run was starting: the test split was already opened. Nothing was read.")
        return 1
    print(f"Lock written: {lock}. The test split is being opened now.")
    try:
        test = read_test(data_dir, config, k)
        if len(test) != have:
            raise ValueError(f"the test file holds {len(test):,} markets but the split report says {have:,}")
        report = score_test(train, validation, test, config, predict, frozen["model"], frozen["params"], reproduced,
                            sha256_of(frozen_path), json.loads(lock.read_text(encoding="utf-8"))["opened_at"])
        out_dir = settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"final_test_k{k}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\n".join(summary_lines(report)))
    except Exception as e:  # the lock stays; the module docstring says what that means
        print(f"PROBLEM after the lock was written: {type(e).__name__}: {e}")
        print("The lock stays. If NO score was computed or printed, you may delete the lock by hand and must write that in decision_log.md. Tell Claude.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
