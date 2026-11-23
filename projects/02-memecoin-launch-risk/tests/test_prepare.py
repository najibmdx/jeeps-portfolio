"""Tests for the prepare step, on the made-up sample and on small hand-made tables."""

import copy

import numpy as np
import pandas as pd
import pytest

import columns
import prepare
import sample
import settings


@pytest.fixture
def config():
    return settings.load_config()


@pytest.fixture
def made_up(config):
    rows, label_rows = sample.make_sample(n_coins=300, n_unlabelled=7, seed=5)
    return rows, pd.DataFrame(label_rows)


def coins(**columns_):
    """A tiny table of coins. Pass each column as a list."""
    return pd.DataFrame(columns_)


# ---------- labels ----------

def test_crash_is_min_ratio_below_the_line_and_exactly_on_the_line_is_not_a_crash(config):
    df = coins(min_ratio=[0.0, 0.29999, 0.3, 0.30001, 1.5])
    assert prepare.make_target(df, config)["crash"].tolist() == [1, 1, 0, 0, 0]


def test_the_files_own_label_column_is_ignored(config):
    df = coins(min_ratio=[0.1, 0.9], label=["low", "high"])      # a label that says the opposite
    flipped = coins(min_ratio=[0.1, 0.9], label=["high", "low"])
    assert prepare.make_target(df, config)["crash"].tolist() == [1, 0]
    assert prepare.make_target(flipped, config)["crash"].tolist() == [1, 0]


def test_a_missing_min_ratio_is_refused_not_silently_counted_as_no_crash(config):
    with pytest.raises(ValueError, match="missing values"):
        prepare.make_target(coins(min_ratio=[0.1, np.nan]), config)


def test_coins_without_a_result_are_set_aside_and_counted(made_up, config):
    rows, labels = made_up
    table = prepare.rows_to_table(rows)
    matched, counts = prepare.attach_labels(table, labels, config)
    assert counts["unlabelled"] == 7
    assert len(matched) == 293
    assert counts["label_without_feature"] == 0


def test_result_rows_with_no_coin_are_counted(made_up, config):
    rows, labels = made_up
    extra = pd.concat([labels, pd.DataFrame([{**labels.iloc[0].to_dict(), "mint_address": "NO_SUCH_COIN"}])])
    _, counts = prepare.attach_labels(prepare.rows_to_table(rows), extra, config)
    assert counts["label_without_feature"] == 1


def test_duplicate_ids_in_either_file_are_refused(made_up, config):
    rows, labels = made_up
    with pytest.raises(ValueError, match="duplicate"):
        prepare.attach_labels(prepare.rows_to_table(rows), pd.concat([labels, labels.iloc[:1]]), config)
    with pytest.raises(ValueError, match="duplicate"):
        prepare.rows_to_table(rows + [rows[0]])


def test_coins_with_different_columns_are_refused(made_up):
    rows, _ = made_up
    broken = copy.deepcopy(rows)
    del broken[10]["group2_demo_a"]
    with pytest.raises(ValueError, match="same columns"):
        prepare.rows_to_table(broken)


# ---------- scope ----------

def test_scope_filter_keeps_exactly_100_holders_and_exactly_60_seconds(config):
    df = coins(
        group3_holder_num=[99, 100, 100, 101, 100],
        group3_time_span_valid=[60, 59, 60, 60, 61],
    )
    kept = prepare.apply_scope(df, config)
    assert kept.index.tolist() == [2, 3, 4]       # both limits must hold; the limit itself is kept


# ---------- split ----------

def test_migration_time_is_creation_time_plus_seconds_to_migration(config):
    df = coins(mint_ts=[1000, 2000], group3_time_span=[50.0, 5.0])
    assert prepare.add_migration_time(df, config)["migration_time"].tolist() == [1050.0, 2005.0]


def test_a_coin_created_early_but_migrating_late_is_placed_by_its_migration(config):
    # coin 0 is created first but migrates last
    df = coins(mint_ts=list(range(10)), group3_time_span=[1000] + [1] * 9)
    df = prepare.add_migration_time(df, config)
    parts = prepare.split_by_time(df, config)
    assert 0 in parts["test"].index
    assert prepare.moved_by_creation_order(df, config) > 0
    for pile in parts.values():                 # inside a pile too, oldest migration first
        assert pile["migration_time"].is_monotonic_increasing
    assert parts["test"].index.tolist() == [9, 0]


def test_split_sizes_are_60_20_20_and_add_up(config):
    df = prepare.add_migration_time(coins(mint_ts=list(range(50)), group3_time_span=[1.0] * 50), config)
    parts = prepare.split_by_time(df, config)
    assert [len(parts[p]) for p in prepare.PARTS] == [30, 10, 10]


def test_piles_never_overlap_and_older_piles_are_older(made_up, config):
    rows, labels = made_up
    parts, _ = prepare.build_parts(rows, labels, config)
    ids = [set(parts[p]["mint_address"]) for p in prepare.PARTS]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])
    assert parts["train"]["migration_time"].max() <= parts["validation"]["migration_time"].min()
    assert parts["validation"]["migration_time"].max() <= parts["test"]["migration_time"].min()


def test_inside_each_pile_coins_run_oldest_to_newest_by_migration_time(made_up, config):
    rows, labels = made_up
    parts, _ = prepare.build_parts(rows, labels, config)
    for df in parts.values():
        assert df["migration_time"].is_monotonic_increasing


def test_the_split_is_repeatable(made_up, config):
    rows, labels = made_up
    a, _ = prepare.build_parts(rows, labels, config)
    b, _ = prepare.build_parts(rows, labels, config)
    for name in prepare.PARTS:
        assert a[name]["mint_address"].tolist() == b[name]["mint_address"].tolist()


def test_a_pile_that_would_be_empty_is_refused(config):
    df = prepare.add_migration_time(coins(mint_ts=[1, 2], group3_time_span=[1.0, 1.0]), config)
    with pytest.raises(ValueError, match="empty"):
        prepare.split_by_time(df, config)


def test_boundary_counts_use_the_label_window(config):
    # validation starts at t=10000. Window is 20 min = 1200 s.
    # train coins at 8700 (known: 1300 s before), 8900 (unknown: 1100 s), 9990 (unknown).
    # test starts at t=20000; validation coins at 10000 (known), 18700 (known: 1300 s before)
    # and 19500 (unknown: 500 s before).
    parts = {
        "train": coins(migration_time=[100, 8700, 8900, 9990]),
        "validation": coins(migration_time=[10000, 18700, 19500]),
        "test": coins(migration_time=[20000, 30000]),
    }
    assert prepare.boundary_counts(parts, 20) == [2, 1]


# ---------- the whole pipeline ----------

def test_report_counts_agree_with_each_other(made_up, config):
    rows, labels = made_up
    parts, report = prepare.build_parts(rows, labels, config)
    assert report["rows_feature"] == 300 and report["unlabelled"] == 7
    assert report["matched_before_filter"] == 293
    assert report["after_filter"] == sum(report["sizes"]) == sum(len(p) for p in parts.values())
    assert 0 < report["after_filter"] < 293        # the sample has coins on both sides of the filter
    assert report["crash_before_filter"] == int((labels["min_ratio"].iloc[:293] < 0.3).sum())


def test_every_kept_coin_passes_the_filter(made_up, config):
    rows, labels = made_up
    parts, _ = prepare.build_parts(rows, labels, config)
    for df in parts.values():
        assert (df["group3_holder_num"] >= 100).all()
        assert (df["group3_time_span_valid"] >= 60).all()


def test_comparison_flags_every_difference(config):
    verified = {"after_filter": 10, "sizes": [6, 2, 2]}
    assert all(ok for *_, ok in prepare.compare_with_verified({"after_filter": 10, "sizes": [6, 2, 2]}, verified))
    result = prepare.compare_with_verified({"after_filter": 11, "sizes": [6, 2, 2]}, verified)
    assert [r[3] for r in result] == [False, True]
    assert prepare.compare_with_verified({}, {"x": 1})[0][3] is False     # a missing number is a mismatch


def test_every_verified_fact_in_the_config_is_a_name_the_report_makes(made_up, config):
    rows, labels = made_up
    _, report = prepare.build_parts(rows, labels, config)
    assert set(config["verified_on_real_data"]) <= set(report)


# ---------- saving ----------

def test_saved_feature_files_hold_no_outcome_columns_and_reload_exactly(made_up, config, tmp_path):
    rows, labels = made_up
    parts, report = prepare.build_parts(rows, labels, config)
    prepare.save_parts(parts, rows, report, tmp_path, config)

    for name in prepare.PARTS:
        saved = prepare.load_features(tmp_path / f"{name}_features.csv")
        assert not set(config["columns"]["banned_as_features"]) & set(saved.columns)
        assert len(saved) == len(parts[name])
        assert saved["mint_address"].tolist() == parts[name]["mint_address"].tolist()
        for col in ("group3_time_span", "group2_demo_a", "migration_time"):
            np.testing.assert_array_equal(saved[col].to_numpy(), parts[name][col].to_numpy())
        assert saved["crash"].tolist() == parts[name]["crash"].tolist()
        columns.check_no_banned(columns.feature_columns(saved.columns, config), config)

    reference = pd.read_csv(tmp_path / "reference.csv")
    assert set(config["columns"]["banned_as_features"]) <= set(reference.columns)
    assert len(reference) == report["after_filter"]


def test_saved_series_line_up_with_their_coins(made_up, config, tmp_path):
    rows, labels = made_up
    parts, report = prepare.build_parts(rows, labels, config)
    prepare.save_parts(parts, rows, report, tmp_path, config)
    series = np.load(tmp_path / "test_series.npy", allow_pickle=False)
    test = parts["test"]
    assert series.shape == (len(test), 360, 5) and series.dtype == np.float32
    for position in (0, len(test) // 2, len(test) - 1):
        original = rows[int(test["row_id"].iloc[position])]
        assert original["mint_address"] == test["mint_address"].iloc[position]
        np.testing.assert_array_equal(series[position], original["ts"])


# ---------- the command ----------

def test_main_stops_and_saves_nothing_when_a_verified_number_differs(made_up, config, tmp_path, monkeypatch, capsys):
    rows, labels = made_up
    monkeypatch.setattr(prepare.get_data, "check_feature", lambda c, d: (True, "ok"))
    monkeypatch.setattr(prepare.get_data, "check_file", lambda *a: "ok")
    monkeypatch.setattr(prepare.safe_pickle, "load_file", lambda p: rows)
    monkeypatch.setattr(prepare.pd, "read_csv", lambda p: labels)
    # the made-up sample cannot match the real numbers, so main must refuse
    code = prepare.main([], config=config, data_dir=tmp_path)
    assert code == 1
    assert "STOPPED" in capsys.readouterr().out
    assert not (tmp_path / config["data"]["prepared_folder"]).exists()


def test_main_saves_when_every_number_matches(made_up, config, tmp_path, monkeypatch):
    rows, labels = made_up
    _, report = prepare.build_parts(rows, labels, config)
    matching = copy.deepcopy(config)
    matching["verified_on_real_data"] = {k: report[k] for k in config["verified_on_real_data"]}
    monkeypatch.setattr(prepare.get_data, "check_feature", lambda c, d: (True, "ok"))
    monkeypatch.setattr(prepare.get_data, "check_file", lambda *a: "ok")
    monkeypatch.setattr(prepare.safe_pickle, "load_file", lambda p: rows)
    monkeypatch.setattr(prepare.pd, "read_csv", lambda p: labels)
    assert prepare.main([], config=matching, data_dir=tmp_path) == 0
    assert (tmp_path / config["data"]["prepared_folder"] / "train_features.csv").exists()


def test_main_does_not_open_the_feature_file_if_its_hash_is_wrong(config, tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.get_data, "check_feature", lambda c, d: (False, "wrong hash"))
    def must_not_run(path):
        raise AssertionError("the file was opened although its hash was wrong")
    monkeypatch.setattr(prepare.safe_pickle, "load_file", must_not_run)
    assert prepare.main([], config=config, data_dir=tmp_path) == 1
