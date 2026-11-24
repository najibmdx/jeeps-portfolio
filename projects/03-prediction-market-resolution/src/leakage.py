"""Stage 4: the leakage audit, on the TRAIN part only.

    python src/leakage.py [--data-dir DIR] [--k 7]       (needs the split files written by split.py)

1. The table: every dataset column, and every field kept out, has a row in config.yaml (leakage_table) with its
   source and a "known at t" decision. This module checks the table against the real columns.
2. Screen 1: groups of train markets whose outcome never varies (all Yes or all No). Only groups with enough
   markets AND enough events are tested, because one huge event can be 50 markets.
3. Screen 2: numeric features that move with the calendar (rank correlation with t).
4. Screen 3 is the final-bar leak. It needs bars after t, which the dataset does not hold, so it is cited from
   the audit, not recomputed.

A flag is a finding to discuss. Nothing is dropped automatically. Validation and test are never read here.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import settings
import split

REQUIRED_COLUMNS = ("event_key", "t", "label", "category", "has_fee", "sibling_count_at_t", "days_since_last_bar")


def table_rows(config) -> list:
    return [{"name": name, **row} for name, row in config["leakage_table"].items()]


def check_table(config, columns) -> list:
    """Plain-words problems between the table in config.yaml and the columns that really exist."""
    table = config["leakage_table"]
    meta = set(config["dataset"]["meta"])
    problems = []
    for col in columns:
        if col not in table:
            problems.append(f"dataset column {col} has no row in leakage_table")
        elif table[col]["role"] in ("excluded", "held_back") and col not in meta:
            problems.append(f"{col} is {table[col]['role']} but sits in the dataset as a column")
    for name, row in table.items():
        if row["role"] == "feature" and name not in columns:
            problems.append(f"feature {name} is listed in leakage_table but is not a dataset column")
    return problems


def sibling_labels(edges) -> list:
    """For edges [1, 5, 15]: ['1', '2-5', '6-15', '16+']."""
    lows = [1] + [e + 1 for e in edges]
    highs = list(edges) + [None]
    return [f"{lo}+" if hi is None else (f"{lo}" if hi == lo else f"{lo}-{hi}") for lo, hi in zip(lows, highs)]


def sibling_groups(series, edges) -> pd.Series:
    return pd.cut(series, bins=[0] + list(edges) + [np.inf], labels=sibling_labels(edges))


def _group(train, key, name, min_markets, min_events) -> list:
    rows = []
    for value, g in train.groupby(key, sort=True, observed=True, dropna=False):
        n, events, rate = len(g), int(g["event_key"].nunique()), float(g["label"].mean())
        tested = n >= min_markets and events >= min_events
        rows.append({
            "cut": name,
            "value": "(missing)" if pd.isna(value) else str(value),
            "markets": n, "events": events, "yes_rate": rate,
            "tested": tested, "flagged": tested and (rate == 0.0 or rate == 1.0),
        })
    return rows


def segment_rows(train, config) -> list:
    """Screen 1. Every group of every cut, with whether it was big enough to test and whether it was flagged."""
    lk = config["leakage"]
    lo, ev = lk["min_markets_in_segment"], lk["min_events_in_segment"]
    cuts = [
        ("category", train["category"]),
        ("has_fee", train["has_fee"]),
        ("sibling_count_group", sibling_groups(train["sibling_count_at_t"], lk["sibling_bucket_edges"])),
        ("days_since_last_bar", train["days_since_last_bar"]),
    ]
    rows = []
    for name, key in cuts:
        rows += _group(train, key, name, lo, ev)
    return rows


def _as_number(series):
    """The column as numbers, or None if it is not numeric (booleans count as 0 and 1)."""
    if series.dtype.kind in "biuf":
        return series.astype("float64")
    if series.dtype == object:
        values = series.dropna()
        if len(values) and all(isinstance(v, (bool, np.bool_)) for v in values):
            return series.map({True: 1.0, False: 0.0}).astype("float64")
    return None


def clock_like_rows(train, config) -> list:
    """Screen 2. Rank correlation of each feature with t (the decision time), on train."""
    f = config["features"]
    threshold = config["leakage"]["clock_like_correlation"]
    t = train["t"].dt.as_unit("ns").astype("int64").astype("float64")
    rows = []
    for name in list(f["price_path"]) + list(f["activity"]) + list(f["context"]):
        x = _as_number(train[name])
        if x is None:
            rows.append({"feature": name, "rank_correlation_with_t": None, "flagged": False, "note": "not numeric, not tested"})
            continue
        rho = x.corr(t, method="spearman") if x.nunique() > 1 else np.nan
        if pd.isna(rho):
            rows.append({"feature": name, "rank_correlation_with_t": None, "flagged": False, "note": "constant on train, not tested"})
            continue
        rho = round(float(rho), 12)
        rows.append({"feature": name, "rank_correlation_with_t": rho, "flagged": abs(rho) >= threshold})
    return rows


def audit(train, config) -> dict:
    """All of Stage 4, from the train rows only."""
    missing = [c for c in REQUIRED_COLUMNS if c not in train.columns]
    if missing:
        raise ValueError(f"the train table lacks the columns the screens need: {missing}")
    return {
        "train_markets": len(train),
        "train_events": int(train["event_key"].nunique()),
        "table": table_rows(config),
        "problems": check_table(config, list(train.columns)),
        "segments": segment_rows(train, config),
        "clock_like": clock_like_rows(train, config),
        "final_bar_leak": {
            "value": config["audit_expected"]["final_bar_predicts_outcome"],
            "note": "cited from audit.py (reproduced on real data in Stage 1); not recomputed here because it needs bars after t",
        },
    }


def summary_lines(r, config) -> list:
    lk = config["leakage"]
    tested = [s for s in r["segments"] if s["tested"]]
    flagged = [s for s in r["segments"] if s["flagged"]]
    clock = [c for c in r["clock_like"] if c["flagged"]]
    numeric = [c for c in r["clock_like"] if c["rank_correlation_with_t"] is not None]
    lines = [
        f"Leakage audit (train only: {r['train_markets']:,} markets, {r['train_events']:,} events)",
        f"  table: {len(r['table'])} rows checked against the dataset columns, {len(r['problems'])} problems",
        f"  screen 1, groups where the outcome never varies: {len(tested)} of {len(r['segments'])} groups big enough to test "
        f"(at least {lk['min_markets_in_segment']} markets and {lk['min_events_in_segment']} events), {len(flagged)} flagged",
    ]
    lines += [
        f"    FLAG {s['cut']} = {s['value']}: {s['markets']:,} markets, {s['events']:,} events, {s['yes_rate']:.0%} Yes" for s in flagged
    ]
    lines.append(
        f"  screen 2, clock-like features (rank correlation with t at least {lk['clock_like_correlation']} either way): "
        f"{len(clock)} of {len(numeric)} numeric features flagged"
    )
    lines += [f"    FLAG {c['feature']}: {c['rank_correlation_with_t']:+.3f}" for c in clock]
    lines.append(f"  screen 3, final-bar leak: {r['final_bar_leak']['value']:.2%} (cited from audit.py, not recomputed here)")
    lines += [f"  PROBLEM {p}" for p in r["problems"]]
    return lines


def table_lines(r) -> list:
    width = max(len(row["name"]) for row in r["table"])
    lines = [f"{'column'.ljust(width)}  {'role':9}  known at t"]
    lines += [f"{row['name'].ljust(width)}  {row['role']:9}  {row['known_at_t']}" for row in r["table"]]
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Leakage audit on the train split")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--k", type=int, default=None, help="days after open (default: from config.yaml)")
    args = parser.parse_args(argv)
    config = settings.load_config()
    k = args.k or config["decision_time"]["days_after_open"]
    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_DIR
    if not split.split_path(data_dir, config, "train", k).exists():
        print(f"No train split in {data_dir}. Run src/pipeline.py (or src/split.py) first.")
        return 2
    report = audit(split.read_split(data_dir, config, "train", k), config)
    out_dir = settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"leakage_audit_k{k}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(table_lines(report)))
    print()
    print("\n".join(summary_lines(report, config)))
    print("Validation and test were not read.")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
