"""Build the dataset: one row per market in the universe, features from bars strictly before t.

    python src/build.py [--data-dir DIR] [--k 7]

Every rule below is a decision from the design record; none is tuned.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import settings

BAR_COLUMNS = ["prediction_id", "timestamp", "open", "high", "low", "close", "trade_count"]
MARKET_COLUMNS = ["market_id", "event_id", "category", "outcome_yes", "market_start_time", "close_time", "expiration_time", "has_fee"]
PRED_COLUMNS = ["prediction_id", "market_id", "outcome", "n_outcomes", "winner"]


def feature_columns(config) -> list:
    f = config["features"]
    return list(f["price_path"]) + list(f["activity"]) + list(f["context"])


def build_from_tables(markets, preds, bars, config, k=None):
    """Returns (dataset, funnel). Inputs are the markets, predictions and daily-bar tables as DataFrames."""
    k = config["decision_time"]["days_after_open"] if k is None else k
    min_trades = config["universe"]["min_trades_before_t"]
    funnel = {}

    # 1. Yes/No markets with exactly one winning token. The label is the Yes token's `winner`.
    preds = preds.assign(is_winner=preds["winner"].eq(True))
    yes = preds[(preds["outcome"] == "Yes") & (preds["n_outcomes"] == 2)][["market_id", "prediction_id", "is_winner"]]
    yes = yes.rename(columns={"prediction_id": "token", "is_winner": "yes_wins"})
    no = preds[(preds["outcome"] == "No") & (preds["n_outcomes"] == 2)][["market_id", "is_winner"]]
    no = no.rename(columns={"is_winner": "no_wins"})
    m = yes.merge(no, on="market_id")
    m = m[m["yes_wins"] ^ m["no_wins"]].merge(markets, on="market_id", how="left")
    funnel["binary_markets"] = len(m)

    # The label must agree with markets.outcome_yes wherever that is filled in.
    known = m[m["outcome_yes"].notna()]
    wrong = int((known["outcome_yes"].astype(bool) != known["yes_wins"]).sum())
    if wrong:
        raise ValueError(f"label cross-check failed: {wrong} markets where outcome_yes disagrees with the winning token")

    # 2. A decision time needs an open time and a close time (close_time is used only for the open-at-t check).
    m = m[m["market_start_time"].notna() & m["close_time"].notna()].copy()
    funnel["with_start_and_close"] = len(m)
    m["open_time"] = m["market_start_time"].dt.as_unit("ns")
    m["t"] = m["open_time"].dt.floor("D") + pd.Timedelta(days=k)
    m = m[m["close_time"] > m["t"]]
    funnel["open_at_t"] = len(m)

    # 3. Daily bars of the Yes token dated strictly before t, and nothing else.
    b = bars[BAR_COLUMNS].merge(m[["token", "t"]], left_on="prediction_id", right_on="token")
    b = b[b["timestamp"] < b["t"]].sort_values("timestamp", kind="stable")
    g = b.groupby("token")
    agg = g.agg(
        last_close=("close", "last"),
        first_open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        mean_close=("close", "mean"),
        n_bars=("close", "size"),
        trades_before_t=("trade_count", "sum"),
        trades_last_bar=("trade_count", "last"),
        last_bar_day=("timestamp", "last"),
    )
    d = m.merge(agg, left_on="token", right_index=True)
    funnel["with_bars_before_t"] = len(d)

    # 4. Enough trading before t.
    d = d[d["trades_before_t"] >= min_trades].copy()
    funnel["universe"] = len(d)

    # 5. Derived columns.
    d["change_first_to_last"] = d["last_close"] - d["first_open"]
    d["days_since_last_bar"] = (d["t"] - d["last_bar_day"]).dt.days
    d["event_key"] = d["event_id"].where(d["event_id"].notna(), "market:" + d["market_id"].astype(str))
    d["sibling_count_at_t"] = _sibling_counts(d, markets)
    d["label"] = d["yes_wins"].astype("int8")

    cols = list(config["dataset"]["meta"]) + feature_columns(config)
    out = d[cols].sort_values(["t", "market_id"], kind="stable").reset_index(drop=True)
    return out, funnel


def _sibling_counts(d, markets) -> pd.Series:
    """Markets of the same event, this one included, that opened on or before this market's t."""
    mk = markets[markets["market_start_time"].notna() & markets["event_id"].notna()]
    starts = {
        event: np.sort(g.dt.as_unit("ns").astype("int64").to_numpy())
        for event, g in mk.groupby("event_id")["market_start_time"]
    }
    t_ns = d["t"].dt.as_unit("ns").astype("int64").to_numpy()
    counts = np.ones(len(d), dtype="int64")  # a market with no event is its own event
    for i, event in enumerate(d["event_id"].to_numpy()):
        if event is not None and not pd.isna(event):
            counts[i] = int(np.searchsorted(starts[event], t_ns[i], side="right"))
    return pd.Series(counts, index=d.index)


def read_tables(data_dir):
    data_dir = Path(data_dir)
    markets = pd.read_parquet(data_dir / "markets.parquet", columns=MARKET_COLUMNS)
    preds = pd.read_parquet(data_dir / "predictions.parquet", columns=PRED_COLUMNS)
    bars = pd.read_parquet(data_dir / "ohlcv_1d.parquet", columns=BAR_COLUMNS)
    return markets, preds, bars


def build_dataset(data_dir, config, k=None):
    markets, preds, bars = read_tables(data_dir)
    return build_from_tables(markets, preds, bars, config, k)


def dataset_path(data_dir, config, k) -> Path:
    d = config["dataset"]
    return Path(data_dir) / d["folder"] / d["file"].format(k=k)


def write_dataset(dataset, funnel, data_dir, config, k) -> Path:
    path = dataset_path(data_dir, config, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(path, index=False)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the dataset from the four data files")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--k", type=int, default=None, help="days after open (default: from config.yaml)")
    args = parser.parse_args(argv)
    config = settings.load_config()
    k = args.k or config["decision_time"]["days_after_open"]
    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_DIR
    dataset, funnel = build_dataset(data_dir, config, k)
    path = write_dataset(dataset, funnel, data_dir, config, k)
    out_dir = settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"build_funnel_k{k}.json").write_text(json.dumps(funnel, indent=2), encoding="utf-8")
    print(json.dumps(funnel, indent=2))
    print(f"Wrote {len(dataset):,} rows to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
