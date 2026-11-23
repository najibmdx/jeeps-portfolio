"""Tests for the rule about which columns may be features."""

import pytest

import columns
import settings


@pytest.fixture
def config():
    return settings.load_config()


def test_only_group_columns_are_features(config):
    names = ["group1_a", "group2_b", "mint_ts", "mint_address", "ts_len", "crash", "migration_time"]
    assert columns.feature_columns(names, config) == ["group1_a", "group2_b"]


def test_outcome_columns_are_never_features_even_if_they_look_like_group_columns(config):
    names = ["group3_x", "min_ratio", "return_ratio", "manipulated", "label"]
    assert columns.feature_columns(names, config) == ["group3_x"]
    sneaky = {"columns": {**config["columns"], "banned_as_features": config["columns"]["banned_as_features"] + ["group3_x"]}}
    assert columns.feature_columns(names, sneaky) == []


def test_order_of_columns_is_kept(config):
    names = ["group4_z", "group1_a", "group2_b"]
    assert columns.feature_columns(names, config) == names


@pytest.mark.parametrize("bad", ["min_ratio", "return_ratio", "manipulated", "label", "mint_ts", "ts", "mint_address", "ts_len"])
def test_check_no_banned_raises_for_each_forbidden_column(config, bad):
    with pytest.raises(ValueError, match=bad):
        columns.check_no_banned(["group1_a", bad], config)


def test_check_no_banned_accepts_clean_lists(config):
    columns.check_no_banned(["group1_a", "group2_b"], config)
