"""Hand-made tables with known answers. Each market below exists to test one rule."""

import numpy as np
import pandas as pd
import pytest

import build
import settings

D = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def _fixture():
    """Returns (markets, preds, bars). See the comment on each market."""
    markets, preds, bars = [], [], []

    def market(mid, event, start, close, yes_wins=True, kind="binary", outcome_yes=None, resolved=True, category="Politics"):
        markets.append(dict(
            market_id=mid, event_id=event, category=category, outcome_yes=outcome_yes,
            market_start_time=D(start) if start else pd.NaT, close_time=D(close) if close else pd.NaT,
            expiration_time=D("2026-03-01"), has_fee=False,
        ))
        names = {"binary": ["Yes", "No"], "three": ["A", "B", "C"], "team": ["Team X", "Team Y"]}[kind]
        for i, name in enumerate(names):
            win = None
            if resolved and kind != "three":
                win = (i == 0) == yes_wins
            preds.append(dict(prediction_id=f"{'Y' if i == 0 else 'N'}{mid}", market_id=mid, outcome=name,
                              n_outcomes=len(names), winner=win))

    def daily(mid, first_day, n_days, trades, open_first=0.25, close_of=lambda d: 0.30 + 0.01 * d, ramp=False):
        for d in range(n_days):
            day = D(first_day) + pd.Timedelta(days=d)
            bars.append(dict(prediction_id=f"Y{mid}", timestamp=day, open=open_first if d == 0 else close_of(d) - 0.01,
                             high=close_of(d) + 0.02, low=close_of(d) - 0.02, close=close_of(d), trade_count=trades + (d if ramp else 0)))

    # M1: the main case. Opens 2026-01-01 10:00, t = 2026-01-08. Bars Jan 1..Jan 12; only Jan 1..7 may be used.
    market(1, "E1", "2026-01-01 10:00", "2026-01-20", yes_wins=True)
    daily(1, "2026-01-01", 12, trades=5, ramp=True, close_of=lambda d: 0.30 + 0.01 * (d + 1) if d < 7 else 0.99)
    # Siblings of M1: M2 (3-outcome) and M3 (unresolved) opened before t; M4 opens after t.
    market(2, "E1", "2026-01-03", "2026-01-20", kind="three")
    market(3, "E1", "2026-01-05", "2026-01-20", resolved=False)
    market(4, "E1", "2026-01-09", "2026-01-20")
    # M14 opens exactly at t, so it still counts as a sibling (opened on or before t).
    market(14, "E1", "2026-01-08 00:00", "2026-01-20", kind="three")
    # M5: closes before t, so it is not open at t.
    market(5, "E5", "2026-01-01", "2026-01-05"); daily(5, "2026-01-01", 4, trades=10)
    # M6 has exactly 20 trades before t (in), M7 has 19 (out).
    market(6, "E6", "2026-01-01", "2026-01-20"); daily(6, "2026-01-01", 5, trades=4)
    # The No token of M6 also has bars. They must never be read (only the Yes token is used).
    bars.append(dict(prediction_id="N6", timestamp=D("2026-01-01"), open=0.7, high=0.8, low=0.6, close=0.75, trade_count=50))
    market(7, "E7", "2026-01-01", "2026-01-20"); daily(7, "2026-01-01", 1, trades=19)
    # M8 has no close time, M9 no start time.
    market(8, "E8", "2026-01-01", None); daily(8, "2026-01-01", 5, trades=10)
    market(9, "E9", None, "2026-01-20"); daily(9, "2026-01-01", 5, trades=10)
    # M10: two outcomes but not Yes/No.
    market(10, "E10", "2026-01-01", "2026-01-20", kind="team")
    # M11: the No side wins, so the label is 0.
    market(11, "E11", "2026-01-02", "2026-01-25", yes_wins=False); daily(11, "2026-01-02", 6, trades=6)
    # M12: no event id.
    market(12, None, "2026-01-01", "2026-01-20"); daily(12, "2026-01-01", 5, trades=10)
    # M15 closes exactly at t: not open at t, so it is out.
    market(15, "E15", "2026-01-01", "2026-01-08"); daily(15, "2026-01-01", 7, trades=10)
    # M13: bars only on or after t.
    market(13, "E13", "2026-01-01", "2026-01-20"); daily(13, "2026-01-08", 4, trades=50)

    return pd.DataFrame(markets), pd.DataFrame(preds), pd.DataFrame(bars)


@pytest.fixture(scope="module")
def built():
    config = settings.load_config()
    markets, preds, bars = _fixture()
    dataset, funnel = build.build_from_tables(markets, preds, bars, config)
    return config, dataset.set_index("market_id"), funnel


def test_who_is_in(built):
    _, ds, funnel = built
    assert sorted(ds.index) == [1, 6, 11, 12]
    assert funnel == {"binary_markets": 11, "with_start_and_close": 9, "open_at_t": 7, "with_bars_before_t": 5, "universe": 4}


def test_one_row_per_market_and_only_the_yes_token_is_read(built):
    _, ds, _ = built
    assert ds.index.is_unique
    assert ds.loc[6, "trades_before_t"] == 20 and ds.loc[6, "n_bars"] == 5


def test_t_is_open_date_plus_seven_days(built):
    _, ds, _ = built
    assert ds.loc[1, "t"] == D("2026-01-08")
    assert ds.loc[1, "open_time"] == D("2026-01-01 10:00")
    assert ds.loc[11, "t"] == D("2026-01-09")


def test_features_use_only_bars_before_t(built):
    _, ds, _ = built
    r = ds.loc[1]
    assert r["n_bars"] == 7
    assert r["last_close"] == pytest.approx(0.37)       # the Jan 7 bar, not the 0.99 bar dated on t
    assert r["first_open"] == pytest.approx(0.25)
    assert r["high"] == pytest.approx(0.39)
    assert r["low"] == pytest.approx(0.29)
    assert r["mean_close"] == pytest.approx(np.mean([0.31 + 0.01 * i for i in range(7)]))
    assert r["change_first_to_last"] == pytest.approx(0.12)
    assert r["trades_before_t"] == 56 and r["trades_last_bar"] == 11   # 5 + 6 + ... + 11
    assert r["days_since_last_bar"] == 1
    feature_values = ds.loc[1, build.feature_columns(settings.load_config())].tolist()
    assert 0.99 not in feature_values


def test_sibling_count_counts_only_markets_opened_by_t(built):
    _, ds, _ = built
    assert ds.loc[1, "sibling_count_at_t"] == 4   # M1, M2, M3 and M14 (opens exactly at t). M4 opens the day after.
    assert ds.loc[6, "sibling_count_at_t"] == 1
    assert ds.loc[12, "sibling_count_at_t"] == 1  # no event: its own event


def test_label_comes_from_the_winning_token(built):
    _, ds, _ = built
    assert ds.loc[1, "label"] == 1 and ds.loc[6, "label"] == 1 and ds.loc[11, "label"] == 0


def test_a_label_that_disagrees_with_outcome_yes_stops_the_build():
    config = settings.load_config()
    markets, preds, bars = _fixture()
    markets.loc[markets["market_id"] == 1, "outcome_yes"] = False   # M1's Yes token won
    with pytest.raises(ValueError, match="cross-check"):
        build.build_from_tables(markets, preds, bars, config)


def test_agreeing_outcome_yes_is_fine():
    config = settings.load_config()
    markets, preds, bars = _fixture()
    markets.loc[markets["market_id"] == 1, "outcome_yes"] = True
    build.build_from_tables(markets, preds, bars, config)


def test_event_key_for_a_market_with_no_event(built):
    _, ds, _ = built
    assert ds.loc[12, "event_key"] == "market:12" and ds.loc[1, "event_key"] == "E1"


def test_columns_are_exactly_meta_plus_features(built):
    config, ds, _ = built
    expected = [c for c in config["dataset"]["meta"] if c != "market_id"] + build.feature_columns(config)
    assert list(ds.columns) == expected


def test_excluded_fields_are_never_model_inputs(built):
    config, ds, _ = built
    inputs = set(build.feature_columns(config))
    for name in config["features"]["excluded"] + config["features"]["held_back"]:
        assert name.split(".")[-1] not in inputs
    assert "expiration_time" not in inputs and "close_time" not in inputs


def test_k_changes_t_and_the_bars_used():
    config = settings.load_config()
    markets, preds, bars = _fixture()
    ds, _ = build.build_from_tables(markets, preds, bars, config, k=3)
    r = ds.set_index("market_id").loc[12]   # 10 trades a day: three bars still reach 20 trades
    assert r["t"] == D("2026-01-04") and r["n_bars"] == 3


def test_rows_are_ordered_by_t_then_market(built):
    _, ds, _ = built
    t = ds["t"].tolist()
    assert t == sorted(t)


def test_build_is_repeatable():
    config = settings.load_config()
    a, _ = build.build_from_tables(*_fixture(), config)
    b, _ = build.build_from_tables(*_fixture(), config)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.skipif(not (settings.DATA_DIR / "ohlcv_1d.parquet").exists(), reason="real data not downloaded")
def test_real_funnel_matches_the_audit():
    config = settings.load_config()
    ds, funnel = build.build_dataset(settings.DATA_DIR, config)
    exp = config["audit_expected"]
    assert funnel == {
        "binary_markets": exp["binary_markets"], "with_start_and_close": exp["binary_markets"],
        "open_at_t": exp["open_at_t"], "with_bars_before_t": exp["with_bars_before_t"], "universe": exp["universe"],
    }
    assert ds["event_key"].nunique() >= exp["universe_events"]
    assert (ds["sibling_count_at_t"] >= 1).all()
