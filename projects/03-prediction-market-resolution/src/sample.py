"""Made-up data in the exact shape of the real four tables.

Nothing here comes from Polymarket. Every title and question says SAMPLE. The numbers mean
nothing: this exists so the code can run, and be tested, with no download.
It includes the awkward cases the real data has (unresolved markets, 3-outcome markets,
team markets that are not Yes/No, missing close times, one-bar markets) and it mimics the
leak (the last bar of a resolved market sits at 0.999 or 0.001).
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from schema import arrow_schema
from settings import TABLES

_NS_PER_DAY = 86_400 * 10**9
_START = np.datetime64("2025-01-01T00:00:00", "ns").astype("int64")
_CATEGORIES = ["SAMPLE-politics", "SAMPLE-sports", "SAMPLE-crypto", None]


def _ts(values):
    """A list of nanosecond integers (or None) as a UTC timestamp column."""
    return pa.array(values, type=pa.int64()).cast(pa.timestamp("ns", tz="UTC"))


def _token(rng) -> str:
    return "".join(str(d) for d in rng.integers(0, 10, size=60))


def make_sample(n_events: int = 80, seed: int = 42) -> dict:
    """Four pyarrow tables, the same every time for the same seed."""
    rng = np.random.default_rng(seed)
    ev, mk, pr, oh = ({}, {}, {}, {})
    for d in (ev, mk, pr, oh):
        d["rows"] = []
    market_id = 1000

    for e in range(1, n_events + 1):
        event_id = f"SAMPLE-EV-{e:04d}"
        n_markets = int(rng.choice([1, 1, 1, 2, 3, 4]))
        event_start = _START + int(rng.integers(0, 420)) * _NS_PER_DAY
        ev["rows"].append(
            dict(
                event_id=event_id,
                event_ticker=f"SAMPLE-{e:04d}",
                series_id=None,
                title=f"SAMPLE event {e}",
                category=_CATEGORIES[int(rng.integers(0, 4))],
                category_original=None,
                event_start_time=event_start,
                mutually_exclusive=bool(n_markets > 1),
                volume=float(rng.uniform(0, 1e6)),
                open_interest=float(rng.uniform(0, 1e5)),
                n_questions=n_markets,
                multi_category_flag=False,
            )
        )
        for _ in range(n_markets):
            market_id += 1
            kind = rng.choice(["binary", "binary", "binary", "binary", "three", "team"])
            unresolved = bool(rng.random() < 0.12)
            start = event_start + int(rng.integers(0, 5)) * _NS_PER_DAY + int(rng.integers(0, 24)) * 3600 * 10**9
            life_days = float(rng.uniform(0.5, 30))
            close = start + int(life_days * _NS_PER_DAY)
            has_close = bool(rng.random() > 0.05)
            expiration = close + int(rng.integers(0, 3)) * 3600 * 10**9
            winner_yes = bool(rng.random() < 0.25)
            show_outcome_yes = bool(rng.random() < 0.3) and kind == "binary" and not unresolved
            mk["rows"].append(
                dict(
                    market_id=market_id,
                    question=f"SAMPLE question {market_id}",
                    category=_CATEGORIES[int(rng.integers(0, 4))],
                    category_original=None,
                    event_id=event_id,
                    outcome_yes=winner_yes if show_outcome_yes else None,
                    market_start_time=start,
                    close_time=close if has_close else None,
                    expiration_time=expiration,
                    can_close_early=bool(rng.random() < 0.3),
                    taker_base_fee=int(rng.choice([0, 0, 1000])),
                    has_fee=bool(rng.random() < 0.2),
                )
            )

            if kind == "three":
                names = ["SAMPLE A", "SAMPLE B", "SAMPLE C"]
            elif kind == "team":
                names = ["SAMPLE Team X", "SAMPLE Team Y"]
            else:
                names = ["Yes", "No"]
            tokens = [_token(rng) for _ in names]
            winner_idx = 0 if winner_yes else 1
            resolved_one_winner = (not unresolved) and kind != "three"
            for idx, name in enumerate(names):
                negative = tokens[1 - idx] if len(names) == 2 else None
                pr["rows"].append(
                    dict(
                        prediction_id=tokens[idx],
                        market_id=market_id,
                        outcome=name,
                        outcome_idx=idx,
                        n_outcomes=len(names),
                        negative_token=negative,
                        winner=(idx == winner_idx) if resolved_one_winner else None,
                    )
                )

            # Daily bars: some markets get a single bar, some skip days (stale prices).
            n_bars = 1 if rng.random() < 0.1 else int(rng.integers(2, max(3, int(life_days) + 2)))
            first_day = (start // _NS_PER_DAY) * _NS_PER_DAY
            day_offsets = np.sort(rng.choice(np.arange(0, n_bars + 3), size=n_bars, replace=False))
            price = float(rng.uniform(0.05, 0.6))
            for j, off in enumerate(day_offsets):
                price = float(np.clip(price + rng.normal(0, 0.05), 0.02, 0.98))
                last = j == len(day_offsets) - 1
                if last and resolved_one_winner and len(names) == 2 and kind == "binary":
                    price = 0.999 if winner_yes else 0.001  # mimics the leak in the real data
                bar_ts = first_day + int(off) * _NS_PER_DAY
                spread = float(abs(rng.normal(0, 0.02)))
                for idx, tok in enumerate(tokens):
                    if len(names) == 2:
                        p = price if idx == 0 else 1.0 - price
                    else:
                        p = float(rng.uniform(0.05, 0.9))
                    oh["rows"].append(
                        dict(
                            prediction_id=tok,
                            timestamp=bar_ts,
                            market_id=str(market_id),
                            outcome=names[idx],
                            open=min(max(p - spread / 2, 0.0), 1.0),
                            high=min(p + spread, 1.0),
                            low=max(p - spread, 0.0),
                            close=p,
                            volume=float(rng.uniform(0, 5000)),
                            trade_count=int(rng.integers(1, 60)),
                            open_interest=float(rng.uniform(0, 1e4)),
                        )
                    )

    out = {}
    for table, d in (("events", ev), ("markets", mk), ("predictions", pr), ("ohlcv_1d", oh)):
        schema = arrow_schema(table)
        columns = {}
        for field in schema:
            values = [row[field.name] for row in d["rows"]]
            if pa.types.is_timestamp(field.type):
                columns[field.name] = _ts(values)
            else:
                columns[field.name] = pa.array(values, type=field.type)
        out[table] = pa.table(columns, schema=schema)
    return out


def write_sample(folder, n_events: int = 80, seed: int = 42) -> Path:
    """Write the four made-up tables as parquet files into folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables = make_sample(n_events=n_events, seed=seed)
    for name in TABLES:
        pq.write_table(tables[name], folder / f"{name}.parquet")
    return folder
