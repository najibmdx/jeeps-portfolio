"""Tests that the experiment is driven by config.yaml and not by code."""

import columns
import settings
import split
import pandas as pd


def test_config_has_the_split_and_the_feature_groups():
    config = settings.load_config()
    assert config["split"]["train_end"] < config["split"]["valid_end"]
    assert set(config["features"]["groups"]) <= {"client", "contact", "history", "calendar", "economy"}


def test_feature_groups_from_the_config_give_valid_columns():
    config = settings.load_config()
    cols = columns.feature_columns(config["features"]["groups"])
    assert cols and columns.TARGET not in cols


def test_changing_the_cut_off_months_changes_the_sets():
    months = ["may", "jun", "dec", "mar", "apr", "may", "jun", "mar"]
    df = split.add_period(pd.DataFrame({"row_id": range(8), "month": months, "y": ["no"] * 8}))
    early = split.time_split(df, train_end=200812, valid_end=200903)
    late = split.time_split(df, train_end=200905, valid_end=200906)
    assert len(early["train"]) != len(late["train"])
