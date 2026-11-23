"""Tests for the leakage audit, on small made-up tables with known answers."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

import leakage_audit as la
import settings


@pytest.fixture
def config():
    return settings.load_config()


@pytest.fixture
def audit_settings(config):
    return config["audit"]


def make_train(n=3000, seed=0):
    """A made-up train pile: random columns plus the bookkeeping columns."""
    rng = np.random.default_rng(seed)
    crash = (rng.random(n) < 0.6).astype(int)
    time = np.sort(rng.integers(1_733_000_000, 1_738_000_000, n)).astype(float)
    return pd.DataFrame({
        "mint_address": [f"SAMPLE{i:05d}" for i in range(n)],
        "mint_ts": time - 100, "migration_time": time, "ts_len": 29, "crash": crash,
        "group1_noise": rng.random(n),
        "group2_noise": rng.random(n),
    })


def notes_for(names):
    return {n: {"group": n.split("_")[0], "meaning": "m", "known_at_migration": "code_read", "why": "w", "watch": ""}
            for n in names}


# ---------- AUC and rank correlation ----------

def test_auc_is_1_for_a_perfect_ranking_0_for_a_reversed_one_and_half_for_ties():
    y = [0, 0, 1, 1]
    assert la.auc_alone([1, 2, 3, 4], y) == 1.0
    assert la.auc_alone([4, 3, 2, 1], y) == 0.0
    assert la.auc_alone([5, 5, 5, 5], y) == 0.5


def test_auc_matches_a_hand_count():
    # positives score 3 and 1, negatives score 2 and 0: pairs won = (3>2, 3>0, 1>0) = 3 of 4
    assert la.auc_alone([3, 1, 2, 0], [1, 1, 0, 0]) == 0.75


def test_auc_is_undefined_with_only_one_class():
    assert np.isnan(la.auc_alone([1, 2, 3], [1, 1, 1]))


def test_rank_corr():
    assert la.rank_corr([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert la.rank_corr([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert la.rank_corr([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0


# ---------- bands ----------

def test_a_big_band_with_100_percent_crashes_is_extreme():
    values = pd.Series([0] * 60 + [1] * 60)
    y = pd.Series([1] * 60 + [0, 1] * 30)
    out = la.band_rates(values, y, bins=10, min_bin_size=30)
    assert out["extreme_bin"] is True and out["max_rate"] == 1.0


def test_a_big_band_with_0_percent_crashes_is_extreme():
    values = pd.Series([0] * 60 + [1] * 60)
    y = pd.Series([0] * 60 + [0, 1] * 30)
    assert la.band_rates(values, y, bins=10, min_bin_size=30)["extreme_bin"] is True


def test_a_tiny_band_with_100_percent_crashes_is_ignored():
    values = pd.Series([0] * 3 + [1] * 117)
    y = pd.Series([1] * 3 + [0, 1] * 58 + [0, 1])
    out = la.band_rates(values, y, bins=10, min_bin_size=30)
    assert out["extreme_bin"] is False        # the band of 3 coins is too small to trust


def test_few_distinct_values_each_get_their_own_band_and_many_get_cut_into_bands():
    few = la.band_rates(pd.Series(np.repeat([1, 2, 3], 50)), pd.Series(np.tile([0, 1], 75)), bins=10, min_bin_size=30)
    assert few["bands_used"] == 3
    many = la.band_rates(pd.Series(np.arange(1000.0)), pd.Series(np.tile([0, 1], 500)), bins=10, min_bin_size=30)
    assert many["bands_used"] == 10


# ---------- per-column flags ----------

def stats(values, y, time, audit_settings):
    return la.column_stats(pd.Series(values), pd.Series(y), pd.Series(time), audit_settings)


def test_a_column_that_gives_away_the_answer_is_flagged(audit_settings):
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.6).astype(int)
    out = stats(y * 10 + rng.random(2000), y, rng.random(2000), audit_settings)
    assert "strong_alone" in out["flags"] and "extreme_bin" in out["flags"]
    assert out["auc_alone"] > 0.99


def test_a_column_that_gives_away_the_opposite_answer_is_flagged_too(audit_settings):
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.6).astype(int)
    out = stats(-y * 10 + rng.random(2000), y, rng.random(2000), audit_settings)
    assert "strong_alone" in out["flags"] and out["auc_alone"] < 0.01


def test_a_random_column_is_not_flagged(audit_settings):
    rng = np.random.default_rng(2)
    out = stats(rng.random(3000), (rng.random(3000) < 0.6).astype(int), rng.random(3000), audit_settings)
    assert out["flags"] == ""
    assert 0.45 < out["auc_alone"] < 0.55


def test_a_mostly_zero_column_is_flagged(audit_settings):
    rng = np.random.default_rng(3)
    values = np.where(rng.random(2000) < 0.8, 0.0, rng.random(2000))
    out = stats(values, (rng.random(2000) < 0.6).astype(int), rng.random(2000), audit_settings)
    assert "mostly_zero" in out["flags"] and out["zero_share"] > 0.75


def test_a_column_with_half_zeros_exactly_is_not_flagged_but_just_over_is(audit_settings):
    y = np.tile([0, 1], 500)
    rng = np.random.default_rng(4)
    half = np.r_[np.zeros(500), rng.random(500) + 1]
    over = np.r_[np.zeros(501), rng.random(499) + 1]
    t = rng.random(1000)
    assert "mostly_zero" not in stats(half, y, t, audit_settings)["flags"]
    assert "mostly_zero" in stats(over, y, t, audit_settings)["flags"]


def test_a_constant_column_is_flagged_constant_not_mostly_zero(audit_settings):
    rng = np.random.default_rng(5)
    out = stats(np.zeros(1000), np.tile([0, 1], 500), rng.random(1000), audit_settings)
    assert out["flags"] == "constant"


def test_a_column_that_just_counts_time_is_flagged_as_a_clock(audit_settings):
    rng = np.random.default_rng(6)
    time = np.sort(rng.random(2000))
    out = stats(time + rng.random(2000) * 0.1, (rng.random(2000) < 0.6).astype(int), time, audit_settings)
    assert "clock_like" in out["flags"] and out["rank_corr_with_time"] > 0.9


def test_a_column_that_falls_as_time_passes_is_a_clock_too(audit_settings):
    rng = np.random.default_rng(7)
    time = np.sort(rng.random(2000))
    out = stats(-time + rng.random(2000) * 0.1, (rng.random(2000) < 0.6).astype(int), time, audit_settings)
    assert "clock_like" in out["flags"] and out["rank_corr_with_time"] < -0.9


# ---------- the audit of a whole pile ----------

def test_audit_gives_one_row_per_feature_column_in_order_and_never_audits_outcome_columns(config):
    train = make_train()
    train["min_ratio"] = 0.1          # a banned column that must be ignored
    train["return_ratio"] = 0.2
    table = la.audit_train(train, notes_for(["group1_noise", "group2_noise"]), config)
    assert table["column"].tolist() == ["group1_noise", "group2_noise"]
    assert not {"crash", "min_ratio", "return_ratio", "migration_time"} & set(table["column"])


def test_audit_refuses_when_a_column_has_no_note(config):
    with pytest.raises(ValueError, match="no note"):
        la.audit_train(make_train(), notes_for(["group1_noise"]), config)


def test_audit_refuses_when_a_note_has_no_column(config):
    with pytest.raises(ValueError, match="no column"):
        la.audit_train(make_train(), notes_for(["group1_noise", "group2_noise", "group3_ghost"]), config)


def test_audit_needs_the_crash_and_time_columns(config):
    with pytest.raises(ValueError, match="crash"):
        la.audit_train(make_train().drop(columns="crash"), notes_for(["group1_noise", "group2_noise"]), config)
    with pytest.raises(ValueError, match="migration_time"):
        la.audit_train(make_train().drop(columns="migration_time"), notes_for(["group1_noise", "group2_noise"]), config)


def test_notes_must_be_complete_and_use_known_values():
    good = notes_for(["group1_a"])
    la.check_notes(good)
    for field in la.NOTE_FIELDS:
        bad = copy.deepcopy(good)
        del bad["group1_a"][field]
        with pytest.raises(ValueError, match=field):
            la.check_notes(bad)
    bad = copy.deepcopy(good)
    bad["group1_a"]["known_at_migration"] = "yes"
    with pytest.raises(ValueError, match="known_at_migration"):
        la.check_notes(bad)
    bad = copy.deepcopy(good)
    bad["group1_a"]["meaning"] = "  "
    with pytest.raises(ValueError, match="meaning"):
        la.check_notes(bad)


# ---------- time drift and summary ----------

def test_crash_share_by_time_block_counts_every_coin_once(config):
    day = 86400
    train = pd.DataFrame({
        "migration_time": [0, 1, 2 * day, 8 * day, 9 * day, 15 * day] + [100.0],
        "crash": [1, 1, 0, 0, 1, 0] + [1],
    })
    out = la.crash_by_period(train, config)
    assert out["coins"].sum() == len(train)
    assert out["coins"].tolist() == [4, 2, 1]
    assert out["crash_share"].tolist() == [0.75, 0.5, 0.0]
    assert out["first_migration"].iloc[0] == "1970-01-01 00:00"


def test_summary_counts_the_flags(config):
    train = make_train()
    table = la.audit_train(train, notes_for(["group1_noise", "group2_noise"]), config)
    table.loc[0, "flags"] = "clock_like;mostly_zero"
    summary = la.summarise(table, train)
    assert summary["flag_counts"]["clock_like"] == 1 and summary["flag_counts"]["mostly_zero"] == 1
    assert summary["flagged_columns"]["clock_like"] == ["group1_noise"]
    assert summary["columns_audited"] == 2 and summary["train_coins"] == len(train)
    assert "AUC" in la.format_summary(summary, table)


# ---------- the command ----------

def write_train(tmp_path, config, train):
    folder = tmp_path / config["data"]["prepared_folder"]
    folder.mkdir()
    train.to_csv(folder / "train_features.csv", index=False)
    return folder


def test_main_audits_train_saves_evidence_and_opens_only_the_train_file(config, tmp_path, monkeypatch):
    write_train(tmp_path, config, make_train())
    opened = []
    real = la.prepare.load_features
    monkeypatch.setattr(la.prepare, "load_features", lambda p: (opened.append(str(p)), real(p))[1])
    out = tmp_path / "out"
    code = la.main([], config=config, data_dir=tmp_path, out_dir=out, notes=notes_for(["group1_noise", "group2_noise"]))
    assert code == 0
    assert len(opened) == 1 and opened[0].endswith("train_features.csv")
    saved = pd.read_csv(out / "leakage_audit.csv")
    assert saved["column"].tolist() == ["group1_noise", "group2_noise"]
    assert json.loads((out / "leakage_summary.json").read_text())["columns_audited"] == 2
    assert (out / "crash_by_period.csv").exists()


def test_main_stops_and_saves_nothing_when_notes_do_not_match(config, tmp_path, capsys):
    write_train(tmp_path, config, make_train())
    out = tmp_path / "out"
    code = la.main([], config=config, data_dir=tmp_path, out_dir=out, notes=notes_for(["group1_noise"]))
    assert code == 1 and "STOPPED" in capsys.readouterr().out
    assert not out.exists()


def test_main_says_what_to_do_when_the_train_file_is_missing(config, tmp_path, capsys):
    assert la.main([], config=config, data_dir=tmp_path, out_dir=tmp_path / "o", notes={}) == 1
    assert "prepare.py" in capsys.readouterr().out


# ---------- the real notes file and the settings ----------

def test_the_real_notes_file_covers_122_columns_in_four_groups():
    notes = la.load_notes()
    assert len(notes) == 122
    counts = pd.Series([n["group"] for n in notes.values()]).value_counts().to_dict()
    assert counts == {"group1": 6, "group2": 59, "group3": 22, "group4": 35}
    for name, note in notes.items():
        assert name.startswith(note["group"] + "_")


def test_no_banned_column_has_a_note():
    notes = la.load_notes()
    banned = set(settings.load_config()["columns"]["banned_as_features"]) | set(settings.load_config()["columns"]["not_tabular"])
    assert not banned & set(notes)


def test_the_two_columns_the_scope_filter_uses_are_in_the_notes(config):
    notes = la.load_notes()
    assert config["scope"]["holders_column"] in notes and config["scope"]["valid_swap_column"] in notes
    assert all(c in notes for c in config["split"]["order_columns"][1:])


def test_audit_settings_are_sane(audit_settings):
    assert audit_settings["bins"] >= 2 and audit_settings["min_bin_size"] >= 1
    assert 0 < audit_settings["zero_share_flag"] < 1
    assert 0.5 < audit_settings["alone_auc_flag"] < 1
    assert 0 < audit_settings["clock_corr_flag"] < 1
    assert audit_settings["period_days"] >= 1
