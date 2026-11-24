"""Stage 4 tests: the table, the config checks, and the two screens, on hand-made tables with known answers."""

import copy

import numpy as np
import pandas as pd
import pytest

import build
import leakage
import pipeline
import sample
import settings
import split

CONFIG = settings.load_config()
F = CONFIG["features"]
FEATURES = list(F["price_path"]) + list(F["activity"]) + list(F["context"])
COLUMNS = list(CONFIG["dataset"]["meta"]) + FEATURES


def frame(n, **over):
    """A train table with every dataset column. Anything not overridden is a harmless constant."""
    t = pd.Series(pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC").as_unit("ns"))
    data = dict(
        market_id=np.arange(1, n + 1), event_key=[f"E{i}" for i in range(n)], t=t,
        open_time=t - pd.Timedelta(days=7), close_time=t + pd.Timedelta(days=1), expiration_time=t + pd.Timedelta(days=2),
        label=np.zeros(n, dtype="int8"),
        last_close=0.5, first_open=0.5, high=0.6, low=0.4, mean_close=0.5, change_first_to_last=0.0,
        n_bars=3, days_since_last_bar=1, trades_before_t=30, trades_last_bar=10,
        category="A", sibling_count_at_t=1, has_fee=False,
    )
    data.update(over)
    return pd.DataFrame(data)[COLUMNS]


def by(rows, **match):
    hits = [r for r in rows if all(r[k] == v for k, v in match.items())]
    assert len(hits) == 1, (match, hits)
    return hits[0]


# ---- the table and the config checks

def test_the_table_covers_every_column_of_the_built_dataset(tmp_path):
    data = sample.write_sample(tmp_path, seed=CONFIG["seed"])
    dataset, _ = build.build_dataset(data, CONFIG, 7)
    assert leakage.check_table(CONFIG, list(dataset.columns)) == []


def test_the_table_says_what_we_know_and_what_we_only_assume():
    table = CONFIG["leakage_table"]
    assert all(isinstance(row["known_at_t"], str) for row in table.values())  # YAML would turn a bare yes into true
    of = lambda v: {n for n, r in table.items() if r["known_at_t"] == v}  # noqa: E731
    assert of("assumed") == {"category", "has_fee"}
    assert of("unverified") == {"expiration_time"}
    assert of("no") == {"label", "close_time", "events.volume", "events.open_interest", "events.n_questions"}
    assert {n for n, r in table.items() if r["role"] == "feature"} == set(FEATURES)


def test_check_table_finds_the_problems_it_should():
    ok = frame(5)
    assert leakage.check_table(CONFIG, list(ok.columns)) == []
    assert any("mystery" in p for p in leakage.check_table(CONFIG, list(ok.columns) + ["mystery"]))
    assert any("events.volume" in p for p in leakage.check_table(CONFIG, list(ok.columns) + ["events.volume"]))
    assert any("category" in p for p in leakage.check_table(CONFIG, [c for c in ok.columns if c != "category"]))


def test_an_excluded_field_may_sit_in_the_dataset_only_as_a_meta_column():
    c = copy.deepcopy(CONFIG)
    c["dataset"]["meta"] = [m for m in c["dataset"]["meta"] if m != "close_time"]
    problems = leakage.check_table(c, list(frame(5).columns))
    assert problems == ["close_time is excluded but sits in the dataset as a column"]


def _bad(path, value):
    c = copy.deepcopy(CONFIG)
    node = c
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return c


def _without(name):
    c = copy.deepcopy(CONFIG)
    del c["leakage_table"][name]
    return c


def _extra():
    c = copy.deepcopy(CONFIG)
    c["leakage_table"]["mystery"] = dict(role="meta", source="x", known_at_t="yes", reason="x")
    return c


@pytest.mark.parametrize(
    "bad",
    [
        _without("t"),
        _without("events.volume"),
        _extra(),
        _bad(("leakage_table", "last_close", "role"), "meta"),
        _bad(("leakage_table", "category", "known_at_t"), True),       # a bare yes in YAML
        _bad(("leakage_table", "last_close", "known_at_t"), "no"),      # a model input that is not known at t
        _bad(("leakage_table", "expiration_time", "known_at_t"), "yes"),  # held back means unverified
        _bad(("leakage_table", "events.volume", "known_at_t"), "yes"),  # excluded means no
        _bad(("leakage_table", "t", "known_at_t"), "maybe"),            # meta rows too
        _bad(("leakage_table", "t", "reason"), "  "),
        _bad(("leakage", "min_markets_in_segment"), 0),
        _bad(("leakage", "min_events_in_segment"), 0),
        _bad(("leakage", "clock_like_correlation"), 0),
        _bad(("leakage", "clock_like_correlation"), 1.5),
        _bad(("leakage", "sibling_bucket_edges"), [5, 1]),
        _bad(("leakage", "sibling_bucket_edges"), []),
        _bad(("leakage", "sibling_bucket_edges"), [0, 5]),
    ],
)
def test_bad_leakage_settings_are_refused(bad):
    with pytest.raises(ValueError):
        settings.validate_config(bad)


def test_values_on_the_edge_of_the_allowed_range_are_accepted():
    for path, value in [
        (("leakage", "min_markets_in_segment"), 1),
        (("leakage", "min_events_in_segment"), 1),
        (("leakage", "clock_like_correlation"), 1.0),
        (("leakage", "sibling_bucket_edges"), [1]),
    ]:
        settings.validate_config(_bad(path, value))


def test_a_row_with_a_missing_key_is_refused():
    c = copy.deepcopy(CONFIG)
    del c["leakage_table"]["t"]["reason"]
    with pytest.raises(ValueError, match="exactly"):
        settings.validate_config(c)


# ---- screen 1: groups where the outcome never varies

def test_sibling_group_labels():
    assert leakage.sibling_labels([1, 5, 15]) == ["1", "2-5", "6-15", "16+"]
    s = leakage.sibling_groups(pd.Series([1, 2, 5, 6, 15, 16, 100]), [1, 5, 15])
    assert list(s.astype(str)) == ["1", "2-5", "2-5", "6-15", "6-15", "16+", "16+"]


def _screen1_frame():
    # (category, markets, events, outcome): the first minimum is 50 markets, the second 10 events
    plan = [("A", 60, 12, "none"), ("B", 60, 12, "half"), ("C", 60, 3, "none"), ("D", 49, 12, "none"), ("E", 50, 10, "all"), (None, 50, 10, "half")]
    cats, events, labels = [], [], []
    for name, n, e, kind in plan:
        for i in range(n):
            cats.append(name)
            events.append(f"{name}-{i % e}")
            labels.append({"none": 0, "all": 1, "half": i % 2}[kind])
    n = len(cats)
    return frame(n, category=cats, event_key=events, label=np.array(labels, dtype="int8"))


def test_screen_1_flags_only_big_groups_with_one_outcome():
    rows = leakage.segment_rows(_screen1_frame(), CONFIG)
    a = by(rows, cut="category", value="A")
    assert (a["markets"], a["events"], a["yes_rate"], a["tested"], a["flagged"]) == (60, 12, 0.0, True, True)
    b = by(rows, cut="category", value="B")
    assert b["tested"] and not b["flagged"] and b["yes_rate"] == 0.5
    c = by(rows, cut="category", value="C")  # 60 markets but only 3 events: one big event, not tested
    assert (c["markets"], c["events"], c["tested"], c["flagged"]) == (60, 3, False, False)
    d = by(rows, cut="category", value="D")  # 49 markets: one short of the minimum
    assert (d["markets"], d["tested"], d["flagged"]) == (49, False, False)
    e = by(rows, cut="category", value="E")  # exactly 50 markets and 10 events: tested; all Yes
    assert (e["markets"], e["events"], e["yes_rate"], e["tested"], e["flagged"]) == (50, 10, 1.0, True, True)
    m = by(rows, cut="category", value="(missing)")
    assert m["tested"] and not m["flagged"] and m["markets"] == 50
    assert sorted(r["value"] for r in rows if r["flagged"]) == ["A", "E"]  # nothing flagged in the other cuts


def test_screen_1_cuts_by_sibling_group_has_fee_and_days_since_last_bar():
    n = 200
    group = np.arange(n) // 50
    df = frame(
        n, event_key=[f"G{g}-{i % 10}" for i, g in enumerate(group)],
        sibling_count_at_t=np.array([1, 3, 10, 20])[group], days_since_last_bar=np.array([1, 2, 3, 4])[group],
        has_fee=group >= 2,
    )
    rows = leakage.segment_rows(df, CONFIG)
    assert sorted(r["value"] for r in rows if r["cut"] == "sibling_count_group") == ["1", "16+", "2-5", "6-15"]
    assert sorted(r["value"] for r in rows if r["cut"] == "days_since_last_bar") == ["1", "2", "3", "4"]
    assert sorted(r["value"] for r in rows if r["cut"] == "has_fee") == ["False", "True"]
    assert all(r["flagged"] for r in rows if r["cut"] == "sibling_count_group")  # all No, 50 markets, 10 events each


def test_screen_1_minimums_come_from_the_config():
    c = copy.deepcopy(CONFIG)
    c["leakage"]["min_markets_in_segment"] = 60
    c["leakage"]["min_events_in_segment"] = 12
    rows = leakage.segment_rows(_screen1_frame(), c)
    assert by(rows, cut="category", value="A")["flagged"]
    assert not by(rows, cut="category", value="E")["tested"]  # 50 markets is now too few


# ---- screen 2: clock-like features

X_LOW = [2, 3, 4, 9, 5, 7, 10, 1, 6, 8]    # rank correlation with t: +0.393939393939
X_NEG = [9, 5, 8, 3, 7, 2, 6, 10, 4, 1]    # rank correlation with t: -0.393939393939
X_PAIRS = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]  # +0.939393939394


def _clock_frame():
    return frame(
        10, last_close=np.arange(1.0, 11.0), first_open=np.arange(10.0, 0.0, -1.0), low=X_LOW, mean_close=X_NEG,
        n_bars=X_PAIRS, has_fee=[False] * 5 + [True] * 5, trades_before_t=10.0 ** np.arange(10),
    )


def test_screen_2_rank_correlations_and_flags():
    rows = leakage.clock_like_rows(_clock_frame(), CONFIG)
    got = {r["feature"]: r for r in rows}
    assert got["last_close"]["rank_correlation_with_t"] == 1.0 and got["last_close"]["flagged"]
    assert got["first_open"]["rank_correlation_with_t"] == -1.0 and got["first_open"]["flagged"]  # down counts too
    assert got["n_bars"]["rank_correlation_with_t"] == pytest.approx(0.939393939394) and got["n_bars"]["flagged"]
    assert got["has_fee"]["rank_correlation_with_t"] == pytest.approx(0.870388279778) and got["has_fee"]["flagged"]
    assert got["low"]["rank_correlation_with_t"] == pytest.approx(0.393939393939) and not got["low"]["flagged"]
    assert got["trades_before_t"]["rank_correlation_with_t"] == 1.0  # ordered but curved: rank correlation, not Pearson
    assert got["mean_close"]["rank_correlation_with_t"] == pytest.approx(-0.393939393939) and not got["mean_close"]["flagged"]
    assert got["high"]["rank_correlation_with_t"] is None and "constant" in got["high"]["note"]
    assert got["category"]["rank_correlation_with_t"] is None and "not numeric" in got["category"]["note"]
    assert [r["feature"] for r in rows] == FEATURES  # every feature is accounted for, in the config order


def test_screen_2_threshold_is_at_least_and_works_both_ways():
    c = copy.deepcopy(CONFIG)
    c["leakage"]["clock_like_correlation"] = 1.0
    got = {r["feature"]: r for r in leakage.clock_like_rows(_clock_frame(), c)}
    assert got["last_close"]["flagged"] and got["first_open"]["flagged"]  # exactly at the line counts
    assert not got["n_bars"]["flagged"]
    c["leakage"]["clock_like_correlation"] = 0.393939393939
    got = {r["feature"]: r for r in leakage.clock_like_rows(_clock_frame(), c)}
    assert got["low"]["flagged"] and got["mean_close"]["flagged"]  # +0.39 and -0.39 sit exactly on the line


def test_constant_columns_raise_no_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rows = leakage.clock_like_rows(frame(10), CONFIG)  # every numeric feature is constant here
    assert all(r["rank_correlation_with_t"] is None for r in rows)


def test_booleans_with_gaps_are_read_as_numbers_and_text_is_not():
    got = leakage._as_number(pd.Series([True, None, False], dtype=object))
    assert got.iloc[0] == 1.0 and got.iloc[2] == 0.0 and np.isnan(got.iloc[1])
    assert leakage._as_number(pd.Series(["a", "b"], dtype=object)) is None


# ---- screen 3, the whole audit, and the train-only rule

def test_screen_3_is_cited_not_recomputed():
    report = leakage.audit(frame(10), CONFIG)
    assert report["final_bar_leak"]["value"] == CONFIG["audit_expected"]["final_bar_predicts_outcome"] == 0.9895
    assert "not recomputed" in report["final_bar_leak"]["note"]


def test_audit_needs_the_columns_the_screens_read():
    with pytest.raises(ValueError, match="category"):
        leakage.audit(frame(10).drop(columns=["category"]), CONFIG)


def test_summary_lines_show_flags_and_the_citation():
    text = "\n".join(leakage.summary_lines(leakage.audit(_screen1_frame(), CONFIG), CONFIG))
    assert "FLAG category = A: 60 markets, 12 events, 0% Yes" in text
    assert "FLAG category = E: 50 markets, 10 events, 100% Yes" in text
    assert "98.95% (cited from audit.py, not recomputed here)" in text
    assert "0 problems" in text


def _split_dir(tmp_path):
    data = sample.write_sample(tmp_path / "sample", seed=CONFIG["seed"])
    dataset, _ = build.build_dataset(data, CONFIG, 7)
    table, _ = split.assign(dataset, CONFIG)
    split.write_splits(table, tmp_path / "data", CONFIG, 7)
    return tmp_path / "data"


def test_main_reads_the_train_split_only(tmp_path, capsys, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)  # the report goes to tmp_path/outputs
    asked, real = [], split.read_split
    monkeypatch.setattr(split, "read_split", lambda *a, **k: (asked.append(a[2]), real(*a, **k))[1])
    assert leakage.main(["--data-dir", str(data_dir)]) == 0
    assert asked == ["train"]
    assert (tmp_path / "outputs" / "leakage_audit_k7.json").exists()
    out = capsys.readouterr().out
    assert "role" in out and "known at t" in out and "Validation and test were not read." in out


def test_main_exits_1_when_the_table_does_not_match_the_columns(tmp_path, capsys, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    path = split.split_path(data_dir, CONFIG, "train", 7)
    pd.read_parquet(path).assign(mystery=1).to_parquet(path, index=False)
    assert leakage.main(["--data-dir", str(data_dir)]) == 1
    assert "PROBLEM dataset column mystery has no row in leakage_table" in capsys.readouterr().out


def test_main_needs_the_split_files(tmp_path, capsys):
    assert leakage.main(["--data-dir", str(tmp_path)]) == 2
    assert "Run src/pipeline.py" in capsys.readouterr().out


def test_pipeline_prints_the_audit_on_made_up_data(capsys):
    assert pipeline.main(["--sample"]) == 0
    out = capsys.readouterr().out
    assert "Leakage audit (train only:" in out and "0 problems" in out
    assert "Stage 4:" in out
