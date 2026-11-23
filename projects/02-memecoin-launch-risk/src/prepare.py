"""Turn the two raw files into three clean piles: practice (train), tuning (validation)
and final exam (test). No model is used here.

Steps, in order:
  1. Rebuild a table from feature.pkl (a list with one dict per coin).
  2. Match each coin to its result in label.csv. Coins with no result are set aside and counted.
  3. Our label: crash = 1 when min_ratio is below the crash line. The file's own `label`
     column is never used.
  4. Keep only coins that pass the authors' scope filter.
  5. Order coins by migration time (mint_ts + group3_time_span), oldest first, and cut 60/20/20.
  6. Compare every count with the numbers verified on the real files. If one differs,
     stop and save nothing.

Run from the projects folder:  python 02-memecoin-launch-risk/src/prepare.py
Only counts are printed. No coin addresses are ever printed.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import columns
import get_data
import safe_pickle
import settings

PARTS = ("train", "validation", "test")
SERIES_KEYS = ("ts", "ts_len")


# ---------- step 1: rows -> table ----------

def rows_to_table(rows: list) -> pd.DataFrame:
    """One row per coin, without the price series. `row_id` remembers each coin's place in `rows`."""
    if not rows:
        raise ValueError("the feature file has no coins")
    keys = set(rows[0])
    if any(set(r) != keys for r in rows):
        raise ValueError("coins in the feature file do not all have the same columns")
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "ts"} for r in rows])
    if table["mint_address"].duplicated().any():
        raise ValueError("duplicate coin ids in the feature file")
    table.insert(0, "row_id", np.arange(len(rows)))
    return table


def stack_series(rows: list, row_ids) -> np.ndarray:
    """The price series of the chosen coins, in the order given. Shape (coins, 360, 5)."""
    return np.stack([rows[i]["ts"] for i in row_ids]).astype(np.float32)


# ---------- steps 2-3: labels ----------

def attach_labels(table: pd.DataFrame, labels: pd.DataFrame, config: dict):
    """Match coins to results. Returns (matched table, counts). Coins without a result are dropped."""
    source = config["label"]["source_column"]
    if "mint_address" not in labels or source not in labels:
        raise ValueError(f"the label file needs the columns mint_address and {source}")
    if labels["mint_address"].duplicated().any():
        raise ValueError("duplicate coin ids in the label file")
    clash = (set(table.columns) & set(labels.columns)) - {"mint_address"}
    if clash:
        raise ValueError(f"column names appear in both files: {sorted(clash)}")

    without_feature = len(set(labels["mint_address"]) - set(table["mint_address"]))
    matched = table.merge(labels, on="mint_address", how="inner", validate="one_to_one")
    matched = matched.sort_values("row_id").reset_index(drop=True)
    return matched, {
        "unlabelled": len(table) - len(matched),
        "label_without_feature": without_feature,
    }


def make_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add `crash` = 1 when min_ratio is below the crash line. Never uses the file's `label`."""
    source = config["label"]["source_column"]
    if df[source].isna().any():
        raise ValueError(f"{source} has missing values; a crash cannot be decided for those coins")
    out = df.copy()
    out["crash"] = (out[source] < config["label"]["crash_line"]).astype(int)
    return out


# ---------- step 4: scope ----------

def apply_scope(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep coins with at least the minimum holders AND at least the minimum valid-swap seconds."""
    scope = config["scope"]
    keep = (df[scope["holders_column"]] >= scope["min_holders"]) & (
        df[scope["valid_swap_column"]] >= scope["min_valid_swap_seconds"]
    )
    return df[keep].copy()


# ---------- step 5: time split ----------

def add_migration_time(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add `migration_time` = creation time + seconds from creation to migration."""
    out = df.copy()
    first, second = config["split"]["order_columns"]
    out["migration_time"] = out[first] + out[second]
    if (out[second] < 0).any():
        raise ValueError(f"{second} has negative values")
    return out


def part_names(values, config: dict) -> np.ndarray:
    """For each value, which part its coin falls in when coins are ordered by that value."""
    values = np.asarray(values)
    n = len(values)
    split = config["split"]
    train_end = int(round(n * split["train"]))
    validation_end = int(round(n * (split["train"] + split["validation"])))
    order = np.argsort(values, kind="stable")
    names = np.empty(n, dtype=object)
    names[order[:train_end]] = "train"
    names[order[train_end:validation_end]] = "validation"
    names[order[validation_end:]] = "test"
    return names


def split_by_time(df: pd.DataFrame, config: dict) -> dict:
    """Three piles ordered oldest to newest by migration time."""
    ordered = df.sort_values("migration_time", kind="stable")
    names = part_names(ordered["migration_time"].to_numpy(), config)
    parts = {name: ordered[names == name].copy() for name in PARTS}
    if any(len(p) == 0 for p in parts.values()):
        raise ValueError("a pile came out empty; there are too few coins for this split")
    return parts


def boundary_counts(parts: dict, window_minutes: int) -> list:
    """Coins whose result was not yet known when the next pile starts.

    A coin's result is known `window_minutes` after it migrates. So a coin is unknown
    at a boundary if it migrated less than that long before the first coin of the next pile.
    Returns [train->validation, validation->test].
    """
    window = window_minutes * 60
    counts = []
    for earlier, later in (("train", "validation"), ("validation", "test")):
        start = parts[later]["migration_time"].min()
        counts.append(int((parts[earlier]["migration_time"] > start - window).sum()))
    return counts


def moved_by_creation_order(df: pd.DataFrame, config: dict) -> int:
    """How many coins land in a different pile if ordered by creation time instead."""
    by_creation = part_names(df["mint_ts"].to_numpy(), config)
    by_migration = part_names(df["migration_time"].to_numpy(), config)
    return int((by_creation != by_migration).sum())


# ---------- the whole pipeline ----------

def _utc_minute(seconds) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def build_parts(rows: list, labels: pd.DataFrame, config: dict):
    """Run steps 1 to 5. Returns (parts, report). The report holds counts only."""
    table = rows_to_table(rows)
    matched, counts = attach_labels(table, labels, config)
    matched = make_target(matched, config)
    crash_before = int(matched["crash"].sum())

    kept = add_migration_time(apply_scope(matched, config), config)
    parts = split_by_time(kept, config)

    report = {
        "rows_feature": len(table),
        "rows_label": len(labels),
        "matched_before_filter": len(matched),
        "unlabelled": counts["unlabelled"],
        "label_without_feature": counts["label_without_feature"],
        "crash_before_filter": crash_before,
        "crash_share_before_filter_percent": round(float(100 * crash_before / len(matched)), 1),
        "after_filter": len(kept),
        "crash_share_after_filter_percent": round(float(100 * kept["crash"].mean()), 1),
        "sizes": [len(parts[p]) for p in PARTS],
        "crash_share_percent": [round(float(100 * parts[p]["crash"].mean()), 1) for p in PARTS],
        "boundary": boundary_counts(parts, config["label"]["window_minutes"]),
        "moved_by_creation_order": moved_by_creation_order(kept, config),
        "migration_ranges": [
            [_utc_minute(parts[p]["migration_time"].min()), _utc_minute(parts[p]["migration_time"].max())]
            for p in PARTS
        ],
    }
    return parts, report


# ---------- step 6: compare with the verified numbers ----------

def compare_with_verified(report: dict, verified: dict) -> list:
    """Return (name, expected, actual, ok) for every verified fact."""
    results = []
    for name, expected in verified.items():
        actual = report.get(name)
        results.append((name, expected, actual, actual == expected))
    return results


def format_report(report: dict) -> str:
    r = report
    lines = [
        f"coins in feature file: {r['rows_feature']:,}   label rows: {r['rows_label']:,}",
        f"coins with a result: {r['matched_before_filter']:,}   set aside (no result): {r['unlabelled']:,}   "
        f"result rows with no coin: {r['label_without_feature']:,}",
        f"crash share before the filter: {r['crash_share_before_filter_percent']}% "
        f"({r['crash_before_filter']:,} coins)",
        f"after the scope filter: {r['after_filter']:,} coins, crash share {r['crash_share_after_filter_percent']}%",
    ]
    for i, name in enumerate(PARTS):
        start, end = r["migration_ranges"][i]
        lines.append(f"  {name:10} {r['sizes'][i]:6,} coins   migrations {start} to {end}   "
                     f"crash share {r['crash_share_percent'][i]}%")
    lines.append(f"coins whose result is not yet known at a boundary: {r['boundary'][0]} (train/validation), "
                 f"{r['boundary'][1]} (validation/test)")
    lines.append(f"coins that change pile when ordered by creation time instead: {r['moved_by_creation_order']}")
    return "\n".join(lines)


# ---------- saving ----------

def save_parts(parts: dict, rows: list, report: dict, out_dir, config: dict) -> list:
    """Save the three piles. Feature files hold NO outcome columns: those go to reference.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outcome_cols = [c for c in config["columns"]["banned_as_features"]]
    all_cols = list(next(iter(parts.values())).columns)
    feature_cols = columns.feature_columns(all_cols, config)
    keep = ["mint_address", "mint_ts", "migration_time", "ts_len", "crash"] + feature_cols
    written = []

    for name, df in parts.items():
        path = out_dir / f"{name}_features.csv"
        df[keep].to_csv(path, index=False)
        series_path = out_dir / f"{name}_series.npy"
        np.save(series_path, stack_series(rows, df["row_id"].to_numpy()), allow_pickle=False)
        written += [path, series_path]

    reference = pd.concat(
        [df[["mint_address"] + outcome_cols].assign(part=name) for name, df in parts.items()],
        ignore_index=True,
    )
    reference.to_csv(out_dir / "reference.csv", index=False)
    (out_dir / "prepared_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return written + [out_dir / "reference.csv", out_dir / "prepared_report.json"]


def load_features(path) -> pd.DataFrame:
    """Read a saved feature file back.

    With the pinned pandas (3.1.1) the default reader gives every float back bit for bit.
    Older pandas versions did not, so the test that checks this assumes the pinned version.
    """
    return pd.read_csv(path)


# ---------- command line ----------

def main(argv=None, config=None, data_dir=None) -> int:
    """Return 0 if every verified number is reproduced and the piles are saved, otherwise 1."""
    config = config or settings.load_config()
    data_dir = Path(data_dir) if data_dir else settings.DATA_DIR

    ok, message = get_data.check_feature(config, data_dir)
    if not ok:
        print(message)
        return 1
    entry = config["data"]["label"]
    status = get_data.check_file(data_dir / entry["file"], entry["bytes"], entry["sha256"])
    if status != "ok":
        print(f"{entry['file']}: {status}. Run get_data.py first.")
        return 1

    print("opening feature.pkl through the safe loader (this can take a minute)...")
    rows = safe_pickle.load_file(data_dir / config["data"]["feature"]["file"])
    labels = pd.read_csv(data_dir / entry["file"])
    parts, report = build_parts(rows, labels, config)

    print(format_report(report))
    print("\nchecks against the numbers verified on 23 Nov:")
    checks = compare_with_verified(report, config["verified_on_real_data"])
    for name, expected, actual, good in checks:
        print(f"  {'OK      ' if good else 'MISMATCH'} {name}: expected {expected}, got {actual}")
    if not all(c[3] for c in checks):
        print("\nSTOPPED. At least one number differs. Nothing was saved. Find out why before going on.")
        return 1

    written = save_parts(parts, rows, report, data_dir / config["data"]["prepared_folder"], config)
    print(f"\nall numbers match. saved {len(written)} files in {data_dir / config['data']['prepared_folder']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
