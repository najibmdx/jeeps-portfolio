"""Tests for the feature pipeline. They use a tiny made-up table, not the real data."""

import pandas as pd
import pytest

import columns
import features

FEATURES = ["age", "job", "contact", "pdays", "previous", "poutcome"]


def make_clients() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [30, 40, 50, 60],
        "job": ["admin.", "services", "unknown", "admin."],
        "contact": ["cellular", "telephone", "cellular", "cellular"],
        "pdays": [999, 3, 999, 10],
        "previous": [0, 1, 0, 2],
        "poutcome": ["nonexistent", "success", "nonexistent", "failure"],
        "y": ["no", "yes", "no", "yes"],
    })


def test_target_is_zero_or_one():
    assert list(features.make_target(make_clients())) == [0, 1, 0, 1]


def test_pdays_999_becomes_a_flag_and_a_zero():
    original = make_clients()
    out = features.add_pdays_flag(original)
    assert list(out["pdays_never"]) == [1, 0, 1, 0]
    assert list(out["pdays"]) == [0, 3, 0, 10]
    assert list(original["pdays"]) == [999, 3, 999, 10]  # the input is not changed


def test_unknown_is_kept_as_its_own_category():
    df = make_clients()
    pre = features.build_preprocessor(FEATURES).fit(df[FEATURES])
    names = pre.named_steps["columns"].get_feature_names_out()
    assert "words__job_unknown" in names


def test_target_is_not_in_the_features():
    df = make_clients()
    pre = features.build_preprocessor(FEATURES).fit(df[FEATURES])
    names = pre.named_steps["columns"].get_feature_names_out()
    assert not any(name.endswith("__y") or "__y_" in name for name in names)


def test_pipeline_learns_only_from_the_data_it_is_fitted_on():
    df = make_clients()
    train = df.iloc[:2]  # ages 30 and 40, mean 35
    pre = features.build_preprocessor(FEATURES, scale_numbers=True).fit(train[FEATURES])
    scaler = pre.named_steps["columns"].named_transformers_["numbers"]
    assert scaler.mean_[0] == pytest.approx(35.0)  # not the mean of all four rows (45)


def test_a_new_category_in_later_data_does_not_crash():
    df = make_clients()
    pre = features.build_preprocessor(FEATURES).fit(df[FEATURES])
    later = df.iloc[:1].copy()
    later["job"] = "student"  # never seen in training
    out = pre.transform(later[FEATURES])
    names = list(pre.named_steps["columns"].get_feature_names_out())
    job_columns = [i for i, name in enumerate(names) if name.startswith("words__job_")]
    assert out[0, job_columns].sum() == 0


def test_every_word_column_is_a_known_feature():
    everything = columns.feature_columns(("client", "contact", "history", "calendar", "economy"))
    assert set(columns.CATEGORICAL) <= set(everything)
