"""Check every feature column for cheating, using the TRAIN pile only.

A column "cheats" if it gives away the answer, or if it was not really known at the moment
the coin migrated. This audit does not remove any column. It flags columns for a human
decision (yours, in decision_log.md) and saves the evidence.

Checks, per column:
  extreme_bin   a band of the column's values has a crash rate of exactly 0% or 100%
                (bands smaller than audit.min_bin_size are ignored: too small to trust)
  mostly_zero   more than audit.zero_share_flag of coins have exactly 0 (0 can hide "no data")
  constant      the column has a single value
  strong_alone  on its own the column ranks crashes this well, or this badly (AUC)
  clock_like    the column's rank moves strongly with migration time (it may tell WHEN, not WHAT)

Also saved: how the crash share drifts over time blocks, and how well migration time alone
ranks crashes. Only counts and column names are printed. Validation and test are never opened.

Run from the projects folder:  python 02-memecoin-launch-risk/src/leakage_audit.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import columns
import prepare
import settings

NOTES_PATH = settings.PROJECT_DIR / "column_notes.yaml"
KNOWN_VALUES = ("code_read", "unverified")
NOTE_FIELDS = ("group", "meaning", "known_at_migration", "why", "watch")
FLAGS = ("extreme_bin", "mostly_zero", "constant", "strong_alone", "clock_like")


# ---------- notes ----------

def load_notes(path=None) -> dict:
    with open(path or NOTES_PATH, "r", encoding="utf-8") as f:
        return check_notes(yaml.safe_load(f))


def check_notes(notes: dict) -> dict:
    """Every column needs every field, and a known_at_migration value we recognise."""
    for name, note in notes.items():
        missing = [f for f in NOTE_FIELDS if f not in note]
        if missing:
            raise ValueError(f"{name}: note is missing {missing}")
        if note["known_at_migration"] not in KNOWN_VALUES:
            raise ValueError(f"{name}: known_at_migration must be one of {KNOWN_VALUES}")
        if not str(note["meaning"]).strip():
            raise ValueError(f"{name}: the meaning is empty")
    return notes


# ---------- the checks ----------

def auc_alone(values, y) -> float:
    """How well the column's ranking separates crashes (1) from non-crashes (0). 0.5 = no information."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(values)).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank_corr(a, b) -> float:
    """Spearman correlation: how strongly two columns move together in rank."""
    ra = pd.Series(np.asarray(a)).rank(method="average")
    rb = pd.Series(np.asarray(b)).rank(method="average")
    if ra.nunique() < 2 or rb.nunique() < 2:
        return 0.0
    return float(ra.corr(rb))


def band_rates(values: pd.Series, y: pd.Series, bins: int, min_bin_size: int) -> dict:
    """Crash rate in each band of the column. Bands under min_bin_size are ignored."""
    values = pd.Series(np.asarray(values))
    y = pd.Series(np.asarray(y))
    if values.nunique() <= bins:
        groups = values
    else:
        groups = pd.qcut(values, bins, duplicates="drop")
    table = pd.DataFrame({"band": groups, "crash": y}).groupby("band", observed=True)["crash"].agg(["size", "mean"])
    trusted = table[table["size"] >= min_bin_size]
    if trusted.empty:
        return {"bands_used": len(table), "min_rate": np.nan, "max_rate": np.nan, "extreme_bin": False}
    return {
        "bands_used": len(table),
        "min_rate": float(trusted["mean"].min()),
        "max_rate": float(trusted["mean"].max()),
        "extreme_bin": bool((trusted["mean"] <= 0.0).any() or (trusted["mean"] >= 1.0).any()),
    }


def column_stats(values: pd.Series, y: pd.Series, time: pd.Series, settings_: dict) -> dict:
    """All checks for one column."""
    a = settings_
    bands = band_rates(values, y, a["bins"], a["min_bin_size"])
    zero_share = float((values == 0).mean())
    auc = auc_alone(values, y)
    corr = rank_corr(values, time)
    flags = []
    if bands["extreme_bin"]:
        flags.append("extreme_bin")
    if values.nunique() <= 1:
        flags.append("constant")
    elif zero_share > a["zero_share_flag"]:
        flags.append("mostly_zero")
    if not np.isnan(auc) and (auc >= a["alone_auc_flag"] or auc <= 1 - a["alone_auc_flag"]):
        flags.append("strong_alone")
    if abs(corr) >= a["clock_corr_flag"]:
        flags.append("clock_like")
    return {
        "n_unique": int(values.nunique()),
        "zero_share": round(zero_share, 4),
        "bands_used": bands["bands_used"],
        "min_band_crash_rate": None if np.isnan(bands["min_rate"]) else round(bands["min_rate"], 4),
        "max_band_crash_rate": None if np.isnan(bands["max_rate"]) else round(bands["max_rate"], 4),
        "auc_alone": None if np.isnan(auc) else round(auc, 4),
        "rank_corr_with_time": round(corr, 4),
        "flags": ";".join(flags),
    }


def audit_train(train: pd.DataFrame, notes: dict, config: dict) -> pd.DataFrame:
    """One row per feature column, in the order the columns come in the file."""
    for needed in ("crash", "migration_time"):
        if needed not in train:
            raise ValueError(f"the train file needs a {needed} column")
    feature_cols = columns.feature_columns(train.columns, config)
    columns.check_no_banned(feature_cols, config)

    missing_notes = [c for c in feature_cols if c not in notes]
    unused_notes = [c for c in notes if c not in feature_cols]
    if missing_notes or unused_notes:
        raise ValueError(
            f"columns and notes do not match. Columns with no note: {missing_notes[:5]} "
            f"({len(missing_notes)} in all). Notes with no column: {unused_notes[:5]} ({len(unused_notes)} in all)."
        )

    rows = []
    for name in feature_cols:
        stats = column_stats(train[name], train["crash"], train["migration_time"], config["audit"])
        rows.append({"column": name, "group": notes[name]["group"],
                     "known_at_migration": notes[name]["known_at_migration"], **stats,
                     "watch": notes[name]["watch"]})
    return pd.DataFrame(rows)


# ---------- time drift ----------

def _utc_minute(seconds) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def crash_by_period(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Crash share in consecutive time blocks of audit.period_days days."""
    seconds = config["audit"]["period_days"] * 86400
    t = train["migration_time"]
    block = ((t - t.min()) // seconds).astype(int)
    out = (train.assign(block=block).groupby("block")
           .agg(coins=("crash", "size"), crash_share=("crash", "mean"), first_migration=("migration_time", "min"))
           .reset_index())
    out["crash_share"] = out["crash_share"].round(4)
    out["first_migration"] = out["first_migration"].map(_utc_minute)
    return out


# ---------- summary ----------

def summarise(table: pd.DataFrame, train: pd.DataFrame) -> dict:
    flagged = {f: table.loc[table["flags"].str.contains(f), "column"].tolist() for f in FLAGS}
    return {
        "train_coins": len(train),
        "train_crash_share": round(float(train["crash"].mean()), 4),
        "columns_audited": len(table),
        "columns_unverified_timing": int((table["known_at_migration"] == "unverified").sum()),
        "migration_time_alone_auc": round(auc_alone(train["migration_time"], train["crash"]), 4),
        "flag_counts": {f: len(v) for f, v in flagged.items()},
        "flagged_columns": flagged,
    }


def format_summary(summary: dict, table: pd.DataFrame, top: int = 10) -> str:
    s = summary
    lines = [
        f"train coins: {s['train_coins']:,}   crash share: {s['train_crash_share']:.1%}   columns audited: {s['columns_audited']}",
        f"columns whose timing could only be read from code, not checked: {s['columns_unverified_timing']}",
        f"migration time ALONE ranks crashes with AUC {s['migration_time_alone_auc']} (0.5 = no information)",
        "flags: " + ", ".join(f"{k} {v}" for k, v in s["flag_counts"].items()),
    ]
    ranked = table.assign(strength=(table["auc_alone"].fillna(0.5) - 0.5).abs()).sort_values("strength", ascending=False).head(top)
    lines.append(f"\nthe {top} columns that rank crashes best or worst on their own (AUC):")
    for _, r in ranked.iterrows():
        lines.append(f"  {r['column']:45} AUC {r['auc_alone']:.3f}   flags: {r['flags'] or '-'}")
    return "\n".join(lines)


# ---------- saving and running ----------

def save(table, weekly, summary, out_dir) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "leakage_audit.csv", out_dir / "crash_by_period.csv", out_dir / "leakage_summary.json"]
    table.to_csv(paths[0], index=False)
    weekly.to_csv(paths[1], index=False)
    paths[2].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths


def main(argv=None, config=None, data_dir=None, out_dir=None, notes=None) -> int:
    """Audit the train pile and save the evidence. Returns 1 if columns and notes do not match."""
    config = config or settings.load_config()
    data_dir = Path(data_dir) if data_dir else settings.DATA_DIR
    out_dir = Path(out_dir) if out_dir else settings.PROJECT_DIR / config["audit"]["outputs_folder"]
    notes = notes if notes is not None else load_notes()

    train_path = data_dir / config["data"]["prepared_folder"] / "train_features.csv"
    if not train_path.exists():
        print(f"{train_path} is missing. Run prepare.py first.")
        return 1
    train = prepare.load_features(train_path)   # the only data file this audit opens

    try:
        table = audit_train(train, notes, config)
    except ValueError as error:
        print(f"STOPPED: {error}")
        return 1
    weekly = crash_by_period(train, config)
    summary = summarise(table, train)
    print(format_summary(summary, table))
    print("\ncrash share by time block (train only):")
    print(weekly.to_string(index=False))
    paths = save(table, weekly, summary, out_dir)
    print(f"\nsaved {len(paths)} files in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
