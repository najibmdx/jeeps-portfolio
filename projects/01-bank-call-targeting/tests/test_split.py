"""Tests for the split and the column list. They use small made-up tables, not the real data."""

import pandas as pd
import pytest

import columns
import split


def make_rows(months, yes_flags=None):
    """A tiny table in file order. months is a list of month names."""
    n = len(months)
    return pd.DataFrame({
        "row_id": range(n),
        "age": range(30, 30 + n),
        "month": months,
        "y": yes_flags or ["no"] * n,
    })


# A small "file": May-Jun 2008, Mar-Apr-May-Jun 2009, Mar 2010.
MONTHS = ["may", "may", "jun", "mar", "mar", "apr", "may", "jun", "mar"]


def test_drop_duplicates_keeps_first_and_ignores_row_id():
    df = pd.DataFrame({
        "row_id": [0, 1, 2],
        "age": [40, 40, 41],
        "month": ["may", "may", "may"],
        "y": ["no", "no", "no"],
    })
    out = split.drop_duplicates(df)
    assert list(out["row_id"]) == [0, 2]


def test_add_period_raises_the_year_when_months_wrap():
    out = split.add_period(make_rows(MONTHS))
    assert list(out["year"]) == [2008, 2008, 2008, 2009, 2009, 2009, 2009, 2009, 2010]
    assert out["period"].iloc[0] == 200805
    assert out["period"].iloc[-1] == 201003


def test_every_row_lands_in_exactly_one_set():
    df = split.add_period(make_rows(MONTHS))
    parts = split.time_split(df, train_end=200903, valid_end=200904)
    assert sum(len(p) for p in parts.values()) == len(df)
    ids = [i for p in parts.values() for i in p["row_id"]]
    assert sorted(ids) == sorted(df["row_id"])


def test_sets_follow_time_order():
    df = split.add_period(make_rows(MONTHS))
    parts = split.time_split(df, train_end=200903, valid_end=200904)
    assert parts["train"]["row_id"].max() < parts["validation"]["row_id"].min()
    assert parts["validation"]["row_id"].max() < parts["test"]["row_id"].min()


def test_no_month_is_shared_between_sets():
    df = split.add_period(make_rows(MONTHS))
    parts = split.time_split(df, train_end=200903, valid_end=200904)
    seen = [set(p["period"]) for p in parts.values()]
    assert seen[0].isdisjoint(seen[1])
    assert seen[1].isdisjoint(seen[2])
    assert seen[0].isdisjoint(seen[2])


def test_baseline_is_the_yes_rate_of_each_set():
    flags = ["no", "no", "yes", "no", "yes", "yes", "no", "yes", "yes"]
    df = split.add_period(make_rows(MONTHS, flags))
    parts = split.time_split(df, train_end=200903, valid_end=200904)
    table = split.baselines(parts)
    assert table.loc["train", "yes_rate"] == pytest.approx(2 / 5)       # rows 0-4: no no yes no yes
    assert table.loc["validation", "yes_rate"] == pytest.approx(1.0)    # row 5: yes
    assert table.loc["test", "yes_rate"] == pytest.approx(2 / 3)        # rows 6-8: no yes yes


def test_every_column_of_the_data_is_accounted_for():
    data_columns = [
        "row_id", "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous",
        "poutcome", "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m",
        "nr_employed", "y",
    ]
    assert sorted(columns.all_known_columns()) == sorted(data_columns)


def test_excluded_columns_are_never_features():
    everything = columns.feature_columns(("client", "contact", "history", "calendar", "economy"))
    for name in columns.EXCLUDED:
        assert name not in everything
    assert columns.TARGET not in everything
    assert columns.ROW_ID not in everything
