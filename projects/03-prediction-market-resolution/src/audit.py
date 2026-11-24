"""Audit the four real tables: reproduce the numbers the design record rests on.

Reads only. Writes outputs/audit.json. Every number here must come out the same on
every machine. Run:  python src/audit.py [--data-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import settings


def _read(data_dir, table, columns):
    return pd.read_parquet(Path(data_dir) / f"{table}.parquet", columns=columns)


def compute_audit(data_dir, config) -> dict:
    """Every audit number, from the four tables."""
    k = config["decision_time"]["days_after_open"]
    min_trades = config["universe"]["min_trades_before_t"]
    low, high = config["universe"]["contested_low"], config["universe"]["contested_high"]
    split = config["split"]
    out = {}

    events = _read(data_dir, "events", ["event_id"])
    markets = _read(
        data_dir, "markets",
        ["market_id", "event_id", "category", "outcome_yes", "market_start_time", "close_time", "expiration_time", "has_fee"],
    )
    preds = _read(data_dir, "predictions", ["prediction_id", "market_id", "outcome", "n_outcomes", "winner"])
    bars = _read(data_dir, "ohlcv_1d", ["prediction_id", "timestamp", "close", "trade_count"])
    out["rows"] = {"events": len(events), "markets": len(markets), "predictions": len(preds), "ohlcv_1d": len(bars)}

    # --- which markets have exactly one winning token
    preds["is_winner"] = preds["winner"].eq(True)
    winners = preds.groupby("market_id")["is_winner"].sum()
    out["markets_with_exactly_one_winner"] = int((winners == 1).sum())
    out["markets_with_no_winner"] = int((winners == 0).sum())
    out["markets_with_two_or_more_winners"] = int((winners >= 2).sum())

    # --- binary Yes/No markets with exactly one winner
    yes = preds[(preds["outcome"] == "Yes") & (preds["n_outcomes"] == 2)][["market_id", "prediction_id", "is_winner"]]
    yes = yes.rename(columns={"prediction_id": "token", "is_winner": "yes_wins"})
    no = preds[(preds["outcome"] == "No") & (preds["n_outcomes"] == 2)][["market_id", "is_winner"]]
    no = no.rename(columns={"is_winner": "no_wins"})
    binary = yes.merge(no, on="market_id")
    binary = binary[binary["yes_wins"] ^ binary["no_wins"]]
    binary = binary.merge(markets, on="market_id", how="left")
    out["binary_markets"] = len(binary)
    out["binary_yes_rate"] = float(binary["yes_wins"].mean())

    # --- the label against markets.outcome_yes
    known = binary[binary["outcome_yes"].notna()]
    out["outcome_yes_filled_share_all_markets"] = float(markets["outcome_yes"].notna().mean())
    out["outcome_yes_disagreements"] = int((known["outcome_yes"].astype(bool) != known["yes_wins"]).sum())
    out["outcome_yes_compared"] = len(known)

    # --- close_time against expiration_time
    close, exp = binary["close_time"], binary["expiration_time"]
    both = close.notna() & exp.notna()
    out["close_after_expiration"] = int((both & (close > exp)).sum())
    out["close_before_expiration"] = int((both & (close < exp)).sum())

    # --- the final-bar leak: the last bar's close against the outcome
    last = bars.sort_values("timestamp").groupby("prediction_id").tail(1)[["prediction_id", "close"]]
    leak = binary.merge(last, left_on="token", right_on="prediction_id")
    out["final_bar_predicts_outcome"] = float(((leak["close"] > 0.5) == leak["yes_wins"]).mean())

    # --- the universe at t
    b = binary[binary["market_start_time"].notna() & binary["close_time"].notna()].copy()
    out["binary_with_start_and_close"] = len(b)
    b["t"] = b["market_start_time"].dt.floor("D") + pd.Timedelta(days=k)
    b = b[b["close_time"] > b["t"]]
    out["open_at_t"] = len(b)
    bt = bars.merge(b[["token", "t"]], left_on="prediction_id", right_on="token")
    bt = bt[bt["timestamp"] < bt["t"]].sort_values("timestamp")
    agg = bt.groupby("token").agg(trades=("trade_count", "sum"), last_close=("close", "last"))
    u = b.merge(agg, left_on="token", right_index=True)
    out["with_bars_before_t"] = len(u)
    u = u[u["trades"] >= min_trades].copy()
    out["universe"] = len(u)
    out["universe_events"] = int(u["event_id"].nunique())
    out["universe_yes_rate"] = float(u["yes_wins"].mean())
    contested = (u["last_close"] >= low) & (u["last_close"] < high)
    out["contested"] = int(contested.sum())
    out["near_decided"] = int((~contested).sum())
    out["universe_fee_flag"] = int(u["has_fee"].eq(True).sum())

    # --- expiration_time, in the universe
    ue = u["expiration_time"]
    out["expiration_null"] = int(ue.isna().sum())
    out["expiration_on_the_hour"] = int(((ue.dt.minute == 0) & (ue.dt.second == 0)).sum())
    out["expiration_median_days_after_open"] = float(((ue - u["market_start_time"]).dt.total_seconds() / 86400).median())

    # --- the split by t
    t_ns = u["t"].astype("int64").to_numpy()
    b1 = np.quantile(t_ns, split["train"])
    b2 = np.quantile(t_ns, split["train"] + split["validation"])
    when = pd.Series(np.where(t_ns < b1, "train", np.where(t_ns < b2, "validation", "test")), index=u.index)
    out["boundaries"] = [pd.Timestamp(int(x), tz="UTC").strftime("%Y-%m-%d") for x in (b1, b2)]
    out["split_sizes"] = {s: int((when == s).sum()) for s in ("train", "validation", "test")}
    out["split_events"] = {s: int(u.loc[when == s, "event_id"].nunique()) for s in ("train", "validation", "test")}
    t1, t2 = pd.Timestamp(int(b1), tz="UTC"), pd.Timestamp(int(b2), tz="UTC")
    out["purge_train"] = int(((when == "train") & (u["close_time"] > t1)).sum())
    out["purge_validation"] = int(((when == "validation") & (u["close_time"] > t2)).sum())
    out["events_straddling_a_boundary"] = int((pd.Series(when).groupby(u["event_id"]).nunique() > 1).sum())
    return out


def compare(actual: dict, expected: dict) -> list:
    """Plain-words differences between what was computed and what the design record says."""
    problems = []

    def walk(a, e, path):
        if isinstance(e, dict):
            for key, val in e.items():
                walk(a.get(key) if isinstance(a, dict) else None, val, f"{path}{key}.")
            return
        name = path[:-1]
        if isinstance(e, list):
            ok = list(a) == list(e) if a is not None else False
        elif isinstance(e, float):
            ok = a is not None and abs(a - e) <= 0.0005
        else:
            ok = a == e
        if not ok:
            problems.append(f"{name}: computed {a}, design record says {e}")

    walk(actual, expected, "")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit the real data against the design record")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    config = settings.load_config()
    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_DIR
    audit = compute_audit(data_dir, config)
    out_dir = settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    out_dir.mkdir(exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    problems = compare(audit, config.get("audit_expected", {}))
    if problems:
        print("\nDIFFERENCES FROM THE DESIGN RECORD:")
        for p in problems:
            print(" -", p)
        return 1
    print("\nEvery audited number matches the design record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
