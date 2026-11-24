import pyarrow.compute as pc

import schema
from sample import make_sample


def test_same_seed_same_data():
    a, b = make_sample(30, seed=1), make_sample(30, seed=1)
    for t in a:
        assert a[t].equals(b[t])


def test_different_seed_different_data():
    assert not make_sample(30, seed=1)["markets"].equals(make_sample(30, seed=2)["markets"])


def test_schemas_match_real_layout():
    for t, table in make_sample(30).items():
        assert table.schema.equals(schema.arrow_schema(t))


def test_joins_hold_together():
    d = make_sample(60)
    events = set(d["events"]["event_id"].to_pylist())
    markets = set(d["markets"]["market_id"].to_pylist())
    assert set(d["markets"]["event_id"].to_pylist()) <= events
    assert set(d["predictions"]["market_id"].to_pylist()) == markets
    tokens = set(d["predictions"]["prediction_id"].to_pylist())
    assert set(d["ohlcv_1d"]["prediction_id"].to_pylist()) <= tokens
    # ohlcv stores market_id as text, like the real file
    assert {str(m) for m in markets} >= set(d["ohlcv_1d"]["market_id"].to_pylist())


def test_resolved_yes_no_markets_have_exactly_one_winner():
    p = make_sample(120)["predictions"].to_pylist()
    by_market = {}
    for r in p:
        by_market.setdefault(r["market_id"], []).append(r)
    checked = 0
    for rows in by_market.values():
        if {r["outcome"] for r in rows} == {"Yes", "No"} and rows[0]["winner"] is not None:
            assert sum(1 for r in rows if r["winner"]) == 1
            checked += 1
    assert checked > 20


def test_awkward_cases_are_present():
    d = make_sample(200)
    p, m, o = d["predictions"], d["markets"], d["ohlcv_1d"]
    assert p["winner"].null_count > 0                      # unresolved markets
    assert pc.max(p["n_outcomes"]).as_py() == 3            # 3-outcome markets
    outcomes = set(p["outcome"].to_pylist())
    assert any(x.startswith("SAMPLE Team") for x in outcomes)  # 2-outcome, not Yes/No
    assert m["close_time"].null_count > 0
    assert m["outcome_yes"].null_count > 0 and m["outcome_yes"].null_count < len(m)
    assert pc.max(pc.utf8_length(p["prediction_id"])).as_py() >= 60
    # one-bar markets exist
    counts = {}
    for pid in o["prediction_id"].to_pylist():
        counts[pid] = counts.get(pid, 0) + 1
    assert 1 in counts.values()


def test_timestamps_are_midnight_utc_and_yes_no_prices_complement():
    o = make_sample(40)["ohlcv_1d"]
    assert pc.all(pc.equal(pc.hour(o["timestamp"]), 0)).as_py()
    rows = o.to_pylist()
    by_key = {}
    for r in rows:
        by_key.setdefault((r["market_id"], r["timestamp"]), {})[r["outcome"]] = r["close"]
    pairs = [v for v in by_key.values() if set(v) == {"Yes", "No"}]
    assert pairs and all(abs(v["Yes"] + v["No"] - 1.0) < 1e-9 for v in pairs)


def test_everything_says_sample():
    d = make_sample(20)
    assert all(t.startswith("SAMPLE") for t in d["events"]["title"].to_pylist())
    assert all(q.startswith("SAMPLE") for q in d["markets"]["question"].to_pylist())


def test_unresolved_yes_no_markets_exist():
    d = make_sample(200)
    rows = d["predictions"].to_pylist()
    by_market = {}
    for r in rows:
        by_market.setdefault(r["market_id"], []).append(r)
    unresolved = [v for v in by_market.values() if {r["outcome"] for r in v} == {"Yes", "No"} and v[0]["winner"] is None]
    assert len(unresolved) > 3


def test_last_bar_of_resolved_market_leaks_the_outcome():
    d = make_sample(200)
    won = {r["prediction_id"]: r["winner"] for r in d["predictions"].to_pylist() if r["outcome"] == "Yes" and r["winner"] is not None}
    last = {}
    for r in d["ohlcv_1d"].to_pylist():
        if r["prediction_id"] in won and (r["prediction_id"] not in last or r["timestamp"] > last[r["prediction_id"]]["timestamp"]):
            last[r["prediction_id"]] = r
    assert len(last) > 20
    for pid, r in last.items():
        assert (r["close"] == 0.999) if won[pid] else (r["close"] == 0.001)
