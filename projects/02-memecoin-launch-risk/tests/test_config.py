"""Tests that config.yaml holds the numbers Jeeps decided, and that bad settings are refused."""

import copy

import pytest

import settings


@pytest.fixture
def config():
    return settings.load_config()


def test_label_numbers_are_the_authors_numbers(config):
    assert config["label"]["crash_line"] == 0.3
    assert config["label"]["window_minutes"] == 20
    assert config["label"]["source_column"] == "min_ratio"


def test_scope_filter_is_the_authors_filter(config):
    assert config["scope"]["min_holders"] == 100
    assert config["scope"]["min_valid_swap_seconds"] == 60
    assert config["scope"]["holders_column"] == "group3_holder_num"
    assert config["scope"]["valid_swap_column"] == "group3_time_span_valid"


def test_split_is_60_20_20_ordered_by_migration_time(config):
    split = config["split"]
    assert (split["train"], split["validation"], split["test"]) == (0.6, 0.2, 0.2)
    assert split["order_columns"] == ["mint_ts", "group3_time_span"]  # migration time = sum


def test_outcome_columns_are_banned_as_features(config):
    banned = config["columns"]["banned_as_features"]
    for column in ("min_ratio", "return_ratio", "manipulated", "label"):
        assert column in banned


def test_data_hashes_match_the_files_we_inspected(config):
    assert config["data"]["feature"]["sha256"] == "1b27d71f911aa269b1953c834d67cc6d87d31dffc5e8f58f8847b6e560b41f71"
    assert config["data"]["feature"]["bytes"] == 517642519
    assert config["data"]["label"]["sha256"] == "dcfcce188fb352466b76ab8345f79633766c93e96d29729282ab114133fcc09a"
    assert config["data"]["label"]["bytes"] == 3861124


def test_licence_and_authors_are_credited(config):
    assert "CC BY-NC 4.0" in config["credit"]["licence"]
    assert "MELT" in config["credit"]["dataset"]


def test_split_that_does_not_add_up_to_one_is_refused(config):
    bad = copy.deepcopy(config)
    bad["split"]["test"] = 0.3
    with pytest.raises(ValueError, match="add up to 1"):
        settings.validate_config(bad)


def test_removing_an_outcome_column_from_the_ban_list_is_refused(config):
    bad = copy.deepcopy(config)
    bad["columns"]["banned_as_features"].remove("min_ratio")
    with pytest.raises(ValueError, match="min_ratio"):
        settings.validate_config(bad)


def test_a_badly_shaped_hash_is_refused(config):
    bad = copy.deepcopy(config)
    bad["data"]["label"]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="sha256"):
        settings.validate_config(bad)


def test_a_crash_line_outside_0_to_1_is_refused(config):
    bad = copy.deepcopy(config)
    bad["label"]["crash_line"] = 30
    with pytest.raises(ValueError, match="crash_line"):
        settings.validate_config(bad)
