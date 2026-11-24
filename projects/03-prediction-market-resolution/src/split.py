"""Split the dataset by decision time, keep events whole, purge markets whose outcome is not yet known.

    python src/split.py [--data-dir DIR] [--k 7]       (needs the dataset written by build.py)

The rules are decision 4 of the design record. None is tuned.
  1. Order by t. Boundary 1 is the 60% quantile of t and boundary 2 the 80% quantile (the same rule
     audit.py uses). t before boundary 1 is train, before boundary 2 validation, otherwise test.
  2. An event goes wholly to the split of its earliest market (the one with the smallest t).
  3. Purge: a train market whose close_time is after boundary 1, or a validation market whose close_time
     is after boundary 2, is dropped. Its outcome was not yet known when the next period began.
     Test markets are never purged: nothing follows them.

The test file is written and never read here, and nothing in this module computes a number from the
labels of test rows. Stage 9 opens it once.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import build
import settings

PARTS = ("train", "validation", "test")
# Columns this module adds. They are not written to the split files.
HELPERS = ["split_by_t", "split", "purged"]


def _ns(series) -> np.ndarray:
    return series.dt.as_unit("ns").astype("int64").to_numpy()


def boundaries(t, config):
    """The two boundary times (UTC timestamps): the train share and the train+validation share of t, as quantiles."""
    if len(t) < 3:
        raise ValueError(f"need at least 3 markets to split, got {len(t)}")
    s = config["split"]
    t_ns = _ns(t)
    b1 = np.quantile(t_ns, s["train"])
    b2 = np.quantile(t_ns, s["train"] + s["validation"])
    return pd.Timestamp(int(b1), tz="UTC"), pd.Timestamp(int(b2), tz="UTC")


def assign(dataset, config):
    """Returns (table, report). The table is the dataset plus the columns in HELPERS.

    split_by_t: the part by the market's own t.  split: the part after events are kept whole.
    purged: True for a train or validation market that must be dropped (its outcome was not known in time).
    """
    df = dataset.reset_index(drop=True).copy()
    b1, b2 = boundaries(df["t"], config)
    t_ns, close_ns = _ns(df["t"]), _ns(df["close_time"])

    # 1. by t alone
    df["split_by_t"] = np.where(t_ns < b1.value, "train", np.where(t_ns < b2.value, "validation", "test"))

    # 2. events whole: every market takes the part of its event's earliest market
    earliest = df.sort_values(["t", "market_id"], kind="stable").drop_duplicates("event_key")
    df["split"] = df["event_key"].map(earliest.set_index("event_key")["split_by_t"])

    # 3. purge, judged on the part after step 2
    df["purged"] = ((df["split"] == "train") & (close_ns > b1.value)) | ((df["split"] == "validation") & (close_ns > b2.value))

    return df, _report(df, b1, b2, close_ns)


def _sizes(df, col) -> dict:
    return {p: int((df[col] == p).sum()) for p in PARTS}


def _events(df, col) -> dict:
    return {p: int(df.loc[df[col] == p, "event_key"].nunique()) for p in PARTS}


def _report(df, b1, b2, close_ns) -> dict:
    by_t = df["split_by_t"]
    kept = df[~df["purged"]]
    # Only train and validation labels are used for a number. The test labels are not touched.
    yes_rate = {
        p: (float(kept.loc[kept["split"] == p, "label"].mean()) if (kept["split"] == p).any() else None)
        for p in ("train", "validation")
    }
    return {
        "boundaries": [b.strftime("%Y-%m-%d") for b in (b1, b2)],
        "boundary_times": [b.isoformat() for b in (b1, b2)],
        "rows": len(df),
        "by_t": {
            "sizes": _sizes(df, "split_by_t"),
            "events": _events(df, "split_by_t"),
            "purge_train": int(((by_t == "train") & (close_ns > b1.value)).sum()),
            "purge_validation": int(((by_t == "validation") & (close_ns > b2.value)).sum()),
            "events_straddling_a_boundary": int((df.groupby("event_key")["split_by_t"].nunique() > 1).sum()),
        },
        "events_whole": {
            "sizes": _sizes(df, "split"),
            "events": _events(df, "split"),
            "markets_moved": int((df["split"] != by_t).sum()),
        },
        "purged": {p: int(((df["split"] == p) & df["purged"]).sum()) for p in ("train", "validation")},
        "final": {"sizes": _sizes(kept, "split"), "events": _events(kept, "split"), "yes_rate": yes_rate},
    }


def summary_lines(r) -> list:
    def line(d):
        return ", ".join(f"{p} {d[p]:,}" for p in PARTS)

    return [
        f"Split by t: boundaries {r['boundaries'][0]} and {r['boundaries'][1]}",
        f"  by t alone:        {line(r['by_t']['sizes'])}",
        f"  events kept whole: {line(r['events_whole']['sizes'])} ({r['events_whole']['markets_moved']:,} markets moved)",
        f"  purged:            train {r['purged']['train']:,}, validation {r['purged']['validation']:,}",
        f"  final:             {line(r['final']['sizes'])}",
    ]


def split_path(data_dir, config, part, k) -> Path:
    d = config["dataset"]
    return Path(data_dir) / d["folder"] / d["split_file"].format(part=part, k=k)


def write_splits(table, data_dir, config, k) -> dict:
    """One file per part, without the purged markets and without the helper columns. Returns the paths."""
    cols = [c for c in table.columns if c not in HELPERS]
    paths = {}
    for part in PARTS:
        rows = table[(table["split"] == part) & ~table["purged"]]
        path = split_path(data_dir, config, part, k)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows[cols].reset_index(drop=True).to_parquet(path, index=False)
        paths[part] = path
    return paths


def read_split(data_dir, config, part, k) -> pd.DataFrame:
    """Train or validation rows. The test split is not readable here: Stage 9 opens it once."""
    if part == "test":
        raise PermissionError("the test split is opened once, by the final test step, with a lock file")
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    return pd.read_parquet(split_path(data_dir, config, part, k))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Split the built dataset by decision time")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--k", type=int, default=None, help="days after open (default: from config.yaml)")
    args = parser.parse_args(argv)
    config = settings.load_config()
    k = args.k or config["decision_time"]["days_after_open"]
    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_DIR
    source = build.dataset_path(data_dir, config, k)
    if not source.exists():
        print(f"No dataset at {source}. Run src/build.py first.")
        return 2
    table, report = assign(pd.read_parquet(source), config)
    write_splits(table, data_dir, config, k)
    out_dir = settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"split_report_k{k}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\n".join(summary_lines(report)))
    print("The test split was written. Nothing here reads it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
