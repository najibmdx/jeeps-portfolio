"""Tests for the ablation study, on small made-up data."""

import numpy as np
import pandas as pd
import pytest

import ablation
from settings import load_config


def make_frame(n, seed):
    """Every column the project knows. job carries a little real signal; duration copies the outcome."""
    rng = np.random.default_rng(seed)
    job = rng.choice(["admin", "services", "student"], n)
    yes = rng.random(n) < np.where(job == "student", 0.35, 0.10)
    return pd.DataFrame({
        "row_id": np.arange(n),
        "age": rng.integers(18, 90, n),
        "job": job,
        "marital": rng.choice(["married", "single"], n),
        "education": rng.choice(["basic", "university"], n),
        "default": rng.choice(["no", "unknown"], n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "telephone"], n),
        "pdays": np.where(rng.random(n) < 0.9, 999, rng.integers(1, 20, n)),
        "previous": rng.integers(0, 3, n),
        "poutcome": rng.choice(["nonexistent", "failure"], n),
        "month": rng.choice(["may", "jun"], n),
        "day_of_week": rng.choice(["mon", "tue"], n),
        "emp_var_rate": rng.choice([-1.8, 1.1], n),
        "cons_price_idx": rng.random(n),
        "cons_conf_idx": rng.random(n),
        "euribor3m": rng.random(n),
        "nr_employed": rng.choice([5000.0, 5100.0], n),
        "duration": np.where(yes, 400, 60) + rng.integers(0, 30, n),   # copies the outcome on purpose
        "campaign": rng.integers(1, 10, n),
        "y": np.where(yes, "yes", "no"),
    })


def small_config():
    return {
        "seed": 1,
        "models": {"logistic_regression": {"max_iter": 500}},
        "evaluation": {"top_fraction": 0.10, "bootstrap_draws": 30},
        "ablations": [
            {"name": "baseline", "groups": ["client", "contact", "history"], "extra_columns": []},
            {"name": "plus_duration", "groups": ["client", "contact", "history"], "extra_columns": ["duration"]},
        ],
    }


def sets_without_test():
    return {"train": make_frame(1500, 0), "validation": make_frame(600, 1), "test": None}


def test_extra_columns_are_added_after_the_groups():
    setup = {"name": "x", "groups": ["client"], "extra_columns": ["duration"]}
    cols = ablation.feature_list(setup)
    assert cols[-1] == "duration"
    assert cols[:-1] == ["age", "job", "marital", "education", "default", "housing", "loan"]


def test_an_unknown_extra_column_is_refused():
    with pytest.raises(ValueError):
        ablation.feature_list({"name": "x", "groups": ["client"], "extra_columns": ["not_a_column"]})


def test_the_target_cannot_be_added_as_a_feature():
    with pytest.raises(ValueError):
        ablation.feature_list({"name": "x", "groups": ["client"], "extra_columns": ["y"]})


def test_only_train_and_validation_are_read():
    result = ablation.run(small_config(), sets_without_test())   # test is None: would crash if touched
    assert result["reference"] == "baseline"


def test_the_baseline_is_not_compared_with_itself():
    result = ablation.run(small_config(), sets_without_test())
    change = result["models"]["logistic_regression"]["change_vs_baseline"]
    assert set(change) == {"plus_duration"}


def test_adding_a_column_that_copies_the_outcome_clearly_helps():
    result = ablation.run(small_config(), sets_without_test())
    d = result["models"]["logistic_regression"]["change_vs_baseline"]["plus_duration"]
    assert d["low"] > 0
    assert d["verdict"] == "better in nearly every re-drawn exam"


def test_the_same_settings_give_the_same_result():
    first = ablation.run(small_config(), sets_without_test())
    second = ablation.run(small_config(), sets_without_test())
    assert first == second


def test_the_real_config_starts_with_the_model_s_own_columns():
    config = load_config()
    assert config["ablations"][0]["groups"] == config["features"]["groups"]
    assert config["ablations"][0]["extra_columns"] == []
    for setup in config["ablations"]:
        ablation.feature_list(setup)   # raises if any group or column is unknown
