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
        _bad(("leakage", "shift_number_difference"), 0),
        _bad(("leakage", "shift_number_difference"), True),
        _bad(("leakage", "shift_share_change"), 0),
        _bad(("leakage", "shift_share_change"), 1.5),
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
        (("leakage", "shift_share_change"), 1.0),
        (("leakage", "shift_number_difference"), 0.001),
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
    report = leakage.audit(frame(10), frame(10), CONFIG)
    assert report["final_bar_leak"]["value"] == CONFIG["audit_expected"]["final_bar_predicts_outcome"] == 0.9895
    assert "not recomputed" in report["final_bar_leak"]["note"]


def test_audit_needs_the_columns_the_screens_read():
    with pytest.raises(ValueError, match="category"):
        leakage.audit(frame(10).drop(columns=["category"]), frame(10), CONFIG)


def test_summary_lines_show_flags_and_the_citation():
    text = "\n".join(leakage.summary_lines(leakage.audit(_screen1_frame(), frame(10), CONFIG), CONFIG))
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


def test_main_reads_train_and_validation_but_never_test(tmp_path, capsys, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)  # the report goes to tmp_path/outputs
    asked, real = [], split.read_split
    monkeypatch.setattr(split, "read_split", lambda *a, **k: (asked.append(a[2]), real(*a, **k))[1])
    assert leakage.main(["--data-dir", str(data_dir)]) == 0
    assert asked == ["train", "validation"]  # never "test"
    assert (tmp_path / "outputs" / "leakage_audit_k7.json").exists()
    out = capsys.readouterr().out
    assert "role" in out and "known at t" in out and "Test was not read. Validation was read for its features only, not its outcomes." in out


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
    assert "Leakage audit (train only:" in out and "screen 4, train versus validation" in out and "0 problems" in out
    assert "Stage 7:" in out


def test_main_needs_the_validation_split_too(tmp_path, capsys, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    split.split_path(data_dir, CONFIG, "validation", 7).unlink()
    assert leakage.main(["--data-dir", str(data_dir)]) == 2
    assert "No validation split" in capsys.readouterr().out


# ---- screen 4: train versus validation, features only

def test_screen_4_numbers_are_measured_in_train_standard_deviations():
    train = frame(4, last_close=[0.0, 1.0, 0.0, 1.0])      # mean 0.5, sample sd 0.57735...
    sd = float(np.std([0, 1, 0, 1], ddof=1))
    same = leakage.shift_rows(train, frame(4, last_close=0.5), CONFIG)
    assert by(same, feature="last_close")["change"] == 0.0 and not by(same, feature="last_close")["flagged"]
    moved = leakage.shift_rows(train, frame(4, last_close=0.5 + 0.4 * sd), CONFIG)
    row = by(moved, feature="last_close")
    assert row["change"] == pytest.approx(0.4) and not row["flagged"]
    assert row["train"] == 0.5 and row["validation"] == pytest.approx(0.5 + 0.4 * sd)
    for sign in (+1, -1):                                   # exactly on the line counts, either direction
        edge = leakage.shift_rows(train, frame(4, last_close=0.5 + sign * 0.5 * sd), CONFIG)
        assert by(edge, feature="last_close")["flagged"]
        assert by(edge, feature="last_close")["change"] == pytest.approx(sign * 0.5)


def test_screen_4_a_number_that_is_constant_on_train():
    r = leakage.shift_rows(frame(4), frame(4, trades_last_bar=11), CONFIG)
    row = by(r, feature="trades_last_bar")
    assert row["flagged"] and row["change"] is None and "constant on train" in row["note"]
    assert not by(r, feature="high")["flagged"]            # constant on both sides and equal


def test_screen_4_yes_no_shares_are_compared_in_points():
    train = frame(20, has_fee=[True] + [False] * 19)            # 5%
    ok = leakage.shift_rows(train, frame(20, has_fee=[True] * 2 + [False] * 18), CONFIG)   # 10%: +5 points
    assert not by(ok, feature="has_fee")["flagged"]
    edge = leakage.shift_rows(train, frame(20, has_fee=[True] * 3 + [False] * 17), CONFIG)  # 15%: +10 points, on the line
    row = by(edge, feature="has_fee")
    assert row["flagged"] and row["change"] == pytest.approx(0.10) and row["train"] == 0.05 and row["validation"] == 0.15
    down = leakage.shift_rows(frame(20, has_fee=[True] * 6 + [False] * 14), frame(20, has_fee=[False] * 20), CONFIG)
    assert by(down, feature="has_fee")["flagged"] and by(down, feature="has_fee")["change"] == pytest.approx(-0.30)


def test_screen_4_the_late_fee_flag_shows_up_as_a_shift():
    train = frame(100, has_fee=[True] * 1 + [False] * 99)
    validation = frame(100, has_fee=[True] * 40 + [False] * 60)
    r = by(leakage.shift_rows(train, validation, CONFIG), feature="has_fee")
    assert r["flagged"] and r["change"] == pytest.approx(0.39)


def test_screen_4_category_uses_the_value_that_moved_most():
    train = frame(10, category=["A"] * 5 + ["B"] * 5)
    val = frame(10, category=["A"] * 2 + ["B"] * 5 + ["C"] * 3)
    row = by(leakage.shift_rows(train, val, CONFIG), feature="category")
    assert row["flagged"] and row["value"] in ("A", "C")
    assert row["change"] in (pytest.approx(-0.3), pytest.approx(0.3))
    assert row["note"] == "only in validation: C"
    edge = by(leakage.shift_rows(train, frame(10, category=["A"] * 6 + ["B"] * 4), CONFIG), feature="category")
    assert edge["flagged"] and abs(edge["change"]) == 0.1  # exactly 10 points counts, despite float error (0.6 - 0.5)
    under = by(leakage.shift_rows(frame(20, category=["A"] * 10 + ["B"] * 10), frame(20, category=["A"] * 11 + ["B"] * 9), CONFIG), feature="category")
    assert not under["flagged"] and abs(under["change"]) == 0.05
    small = by(leakage.shift_rows(train, frame(10, category=["A"] * 5 + ["B"] * 5), CONFIG), feature="category")
    assert not small["flagged"] and small["change"] == 0.0 and small["note"] is None


def test_screen_4_a_missing_category_is_a_value_of_its_own():
    train = frame(10, category=["A"] * 10)
    val = frame(10, category=["A"] * 5 + [None] * 5)
    row = by(leakage.shift_rows(train, val, CONFIG), feature="category")
    assert row["flagged"] and row["value"] in ("A", "(missing)") and "(missing)" in row["note"]


def test_screen_4_ignores_the_label_completely():
    train, val = frame(20, label=[0, 1] * 10), frame(20, last_close=0.9)
    a = leakage.audit(train, val.assign(label=[0, 1] * 10), CONFIG)
    b = leakage.audit(train, val.assign(label=[1] * 20), CONFIG)
    c = leakage.audit(train, val.drop(columns=["label"]), CONFIG)
    assert a["shift"] == b["shift"] == c["shift"]
    assert "yes_rate" not in str(a["shift"])


def test_screen_4_needs_markets_on_both_sides_and_the_same_columns():
    with pytest.raises(ValueError, match="at least one"):
        leakage.shift_rows(frame(5), frame(5).iloc[0:0], CONFIG)
    with pytest.raises(ValueError, match="lacks feature columns"):
        leakage.audit(frame(5), frame(5).drop(columns=["high"]), CONFIG)


def test_screen_4_thresholds_come_from_the_config():
    c = copy.deepcopy(CONFIG)
    c["leakage"]["shift_number_difference"] = 2.0
    c["leakage"]["shift_share_change"] = 0.5
    train = frame(4, last_close=[0.0, 1.0, 0.0, 1.0], has_fee=[True, False, False, False])
    val = frame(4, last_close=1.0, has_fee=[True, True, False, False])      # last_close +0.87 sd, has_fee +25 points
    assert by(leakage.shift_rows(train, val, CONFIG), feature="last_close")["flagged"]
    assert by(leakage.shift_rows(train, val, CONFIG), feature="has_fee")["flagged"]
    assert not by(leakage.shift_rows(train, val, c), feature="last_close")["flagged"]
    assert not by(leakage.shift_rows(train, val, c), feature="has_fee")["flagged"]


def test_screen_4_covers_every_feature_once():
    names = [r["feature"] for r in leakage.shift_rows(frame(5), frame(5), CONFIG)]
    assert sorted(names) == sorted(FEATURES)


def test_screen_4_summary_lines_name_the_flags():
    train = frame(100, has_fee=[True] + [False] * 99, last_close=np.linspace(0.2, 0.8, 100))
    val = frame(100, has_fee=[True] * 40 + [False] * 60, last_close=0.95)
    text = "\n".join(leakage.summary_lines(leakage.audit(train, val, CONFIG), CONFIG))
    assert "screen 4, train versus validation, features only (100 validation markets" in text
    assert "FLAG has_fee = True: 1.0% of train, 40.0% of validation (+39.0 points)" in text
    assert "FLAG last_close:" in text and "standard deviations" in text
    clean = "\n".join(leakage.summary_lines(leakage.audit(frame(10), frame(10), CONFIG), CONFIG))
    assert "0 of 13 features flagged" in clean


def test_screen_4_float_error_does_not_hide_a_move_of_exactly_half_a_deviation():
    train = frame(4, last_close=[1.0, 2.0, 3.0, 4.0])
    sd = float(np.std([1, 2, 3, 4], ddof=1))
    assert (2.5 + 0.5 * sd - 2.5) / sd < 0.5  # the plain division lands just under 0.5
    for sign in (+1, -1):
        row = by(leakage.shift_rows(train, frame(4, last_close=2.5 + sign * 0.5 * sd), CONFIG), feature="last_close")
        assert row["flagged"] and abs(row["change"]) == 0.5


def test_audit_hands_validation_to_screen_4_without_its_label(monkeypatch):
    seen = []
    real = leakage.shift_rows
    monkeypatch.setattr(leakage, "shift_rows", lambda tr, va, c: (seen.append(list(va.columns)), real(tr, va, c))[1])
    leakage.audit(frame(10), frame(10), CONFIG)
    assert len(seen) == 1 and "label" not in seen[0] and "category" in seen[0]
