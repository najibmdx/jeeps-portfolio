"""Stage 4: the leakage audit. Screens 1 to 3 read the TRAIN part only; screen 4 also reads validation features.

    python src/leakage.py [--data-dir DIR] [--k 7]       (needs the split files written by split.py)

1. The table: every dataset column, and every field kept out, has a row in config.yaml (leakage_table) with its
   source and a "known at t" decision. This module checks the table against the real columns.
2. Screen 1: groups of train markets whose outcome never varies (all Yes or all No). Only groups with enough
   markets AND enough events are tested, because one huge event can be 50 markets.
3. Screen 2: numeric features that move with the calendar (rank correlation with t).
4. Screen 3 is the final-bar leak. It needs bars after t, which the dataset does not hold, so it is cited from
   the audit, not recomputed.

4. Screen 4: do the features look different in validation than in train? Features only: the label column of
   validation is dropped before anything is computed. A flag says the model will meet inputs unlike its training
   inputs (expected with a split by time). It is not leakage.

A flag is a finding to discuss. Nothing is dropped automatically. Test is never read here.
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


def _is_yes_no(series) -> bool:
    if series.dtype.kind == "b":
        return True
    values = series.dropna()
    return series.dtype == object and len(values) > 0 and all(isinstance(v, (bool, np.bool_)) for v in values)


def _shares(series) -> pd.Series:
    return series.map(lambda v: "(missing)" if pd.isna(v) else str(v)).value_counts(normalize=True)


def shift_rows(train, validation, config) -> list:
    """Screen 4. For each feature: how far validation sits from train. Features only; the label is never used."""
    lk = config["leakage"]
    f = config["features"]
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("the shift screen needs at least one train and one validation market")
    rows = []
    for name in list(f["price_path"]) + list(f["activity"]) + list(f["context"]):
        a, b = train[name], validation[name]
        if _is_yes_no(a) and _is_yes_no(b):
            ta, tb = float(a.astype(float).mean()), float(b.astype(float).mean())
            change = round(tb - ta, 12)
            rows.append({"feature": name, "kind": "yes_no", "train": ta, "validation": tb, "change": change,
                         "value": "True", "flagged": abs(change) >= lk["shift_share_change"], "note": None})
            continue
        x, y = _as_number(a), _as_number(b)
        if x is not None and y is not None:
            sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
            ta, tb = float(x.mean()), float(y.mean())
            if sd == 0 or np.isnan(sd):
                moved = tb != ta
                rows.append({"feature": name, "kind": "number", "train": ta, "validation": tb, "change": None, "value": None,
                             "flagged": moved, "note": "constant on train: flagged if validation differs, no size can be given"})
                continue
            change = round((tb - ta) / sd, 12)
            rows.append({"feature": name, "kind": "number", "train": ta, "validation": tb, "change": change,
                         "value": None, "flagged": abs(change) >= lk["shift_number_difference"], "note": None})
            continue
        sa, sb = _shares(a), _shares(b)
        values = sorted(set(sa.index) | set(sb.index))
        diffs = {v: round(float(sb.get(v, 0.0) - sa.get(v, 0.0)), 12) for v in values}
        top = max(values, key=lambda v: abs(diffs[v]))
        rows.append({"feature": name, "kind": "category", "train": float(sa.get(top, 0.0)), "validation": float(sb.get(top, 0.0)),
                     "change": diffs[top], "value": top, "flagged": abs(diffs[top]) >= lk["shift_share_change"],
                     "note": "only in validation: " + ", ".join(v for v in values if v not in sa.index) if any(v not in sa.index for v in values) else None})
    return rows


def audit(train, validation, config) -> dict:
    """All of Stage 4. Screens 1 to 3 use the train rows only; screen 4 compares the features of train and validation."""
    missing = [c for c in REQUIRED_COLUMNS if c not in train.columns]
    if missing:
        raise ValueError(f"the train table lacks the columns the screens need: {missing}")
    validation = validation.drop(columns=["label"], errors="ignore")  # screen 4 never sees an outcome
    f = config["features"]
    gone = [c for c in list(f["price_path"]) + list(f["activity"]) + list(f["context"]) if c not in validation.columns]
    if gone:
        raise ValueError(f"the validation table lacks feature columns: {gone}")
    return {
        "train_markets": len(train),
        "train_events": int(train["event_key"].nunique()),
        "table": table_rows(config),
        "problems": check_table(config, list(train.columns)),
        "segments": segment_rows(train, config),
        "clock_like": clock_like_rows(train, config),
        "validation_markets": len(validation),
        "shift": shift_rows(train, validation, config),
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
    shift = [c for c in r["shift"] if c["flagged"]]
    lines.append(
        f"  screen 4, train versus validation, features only ({r['validation_markets']:,} validation markets; number features "
        f"at least {lk['shift_number_difference']} train standard deviations, shares at least {lk['shift_share_change'] * 100:g} points): "
        f"{len(shift)} of {len(r['shift'])} features flagged"
    )
    for c in shift:
        if c["kind"] == "number":
            size = "constant on train" if c["change"] is None else f"{c['change']:+.2f} standard deviations"
            lines.append(f"    FLAG {c['feature']}: mean {c['train']:.4g} in train, {c['validation']:.4g} in validation ({size})")
        else:
            lines.append(f"    FLAG {c['feature']} = {c['value']}: {c['train']:.1%} of train, {c['validation']:.1%} of validation ({c['change']*100:+.1f} points)")
    lines.append(f"  screen 3, final-bar leak: {r['final_bar_leak']['value']:.2%} (cited from audit.py, not recomputed here)")
    lines += [f"  PROBLEM {p}" for p in r["problems"]]
    return lines


def table_lines(r) -> list:
    width = max(len(row["name"]) for row in r["table"])
    lines = [f"{'column'.ljust(width)}  {'role':9}  known at t"]
    lines += [f"{row['name'].ljust(width)}  {row['role']:9}  {row['known_at_t']}" for row in r["table"]]
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Leakage audit on the train split, plus the train-versus-validation feature shift")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--k", type=int, default=None, help="days after open (default: from config.yaml)")
    args = parser.parse_args(argv)
    config = settings.load_config()
    k = args.k or config["decision_time"]["days_after_open"]
    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_DIR
    if not split.split_path(data_dir, config, "train", k).exists():
        print(f"No train split in {data_dir}. Run src/pipeline.py (or src/split.py) first.")
        return 2
    if not split.split_path(data_dir, config, "validation", k).exists():
        print(f"No validation split in {data_dir}. Run src/pipeline.py (or src/split.py) first.")
        return 2
    train = split.read_split(data_dir, config, "train", k)
    validation = split.read_split(data_dir, config, "validation", k)
    report = audit(train, validation, config)
    out_dir = settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"leakage_audit_k{k}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(table_lines(report)))
    print()
    print("\n".join(summary_lines(report, config)))
    print("Test was not read. Validation was read for its features only, not its outcomes.")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
