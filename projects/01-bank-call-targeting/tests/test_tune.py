"""Tests for time-aware tuning, on small made-up data."""

import numpy as np
import pytest

import models
import tune
from settings import load_config
from test_ablation import make_frame
from test_cutoff import NoTest

PERIODS = [200805, 200806, 200807, 200808, 200810, 200811, 200812, 200903]   # 8 months, like gaps in the real data


def make_timed(n, seed):
    """Made-up training data in time order, with a period column."""
    df = make_frame(n, seed)
    rng = np.random.default_rng(seed)
    df["period"] = np.sort(rng.choice(PERIODS, n))
    return df


def small_config():
    return {
        "seed": 1,
        "features": {"groups": ["client", "contact", "history"]},
        "models": {"logistic_regression": {"max_iter": 500}},
        "evaluation": {"top_fraction": 0.10, "bootstrap_draws": 20},
        "tuning": {"min_train_months": 5, "grids": {"logistic_regression": {"C": [0.01, 1.0]}}},
    }


def test_a_learning_row_is_always_older_than_every_exam_row():
    df = make_timed(2400, 0)
    for learn, exam in tune.month_folds(df, 5):
        assert df["period"].iloc[learn].max() < df["period"].iloc[exam].min()


def test_each_exam_is_one_whole_month_and_no_month_is_shared():
    df = make_timed(2400, 0)
    for learn, exam in tune.month_folds(df, 5):
        exam_months = set(df["period"].iloc[exam])
        assert len(exam_months) == 1
        assert not exam_months & set(df["period"].iloc[learn])


def test_the_window_grows_and_there_is_one_fold_per_remaining_month():
    df = make_timed(2400, 0)
    folds = tune.month_folds(df, 5)
    assert len(folds) == len(PERIODS) - 5
    sizes = [len(learn) for learn, _ in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_too_few_months_is_refused():
    with pytest.raises(ValueError):
        tune.month_folds(make_timed(2400, 0), min_train_months=len(PERIODS))


def test_the_grid_expands_to_every_combination_in_a_fixed_order():
    grid = {"a": [1, 2], "b": [10, 20]}
    assert tune.expand_grid(grid) == [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}]


def test_tuning_never_reads_the_test_set():
    sets = NoTest(train=make_timed(2400, 0), validation=make_frame(600, 1), test=make_frame(600, 2))
    result = tune.run(small_config(), sets)             # raises if "test" is ever read
    assert set(result["models"]) == {"logistic_regression"}


def test_the_chosen_settings_have_the_highest_mean_practice_score():
    sets = {"train": make_timed(2400, 0), "validation": make_frame(600, 1), "test": None}
    r = tune.run(small_config(), sets)["models"]["logistic_regression"]
    best_mean = r["tuned_cv"]["cv_mean"]
    assert best_mean == max(row["cv_mean"] for row in r["all_tried"])
    assert 0 <= r["folds_improved"] <= r["n_folds"]


def test_the_same_settings_give_the_same_result():
    sets = {"train": make_timed(2400, 0), "validation": make_frame(600, 1), "test": None}
    assert tune.run(small_config(), sets) == tune.run(small_config(), sets)


def test_every_setting_named_in_the_real_config_is_accepted_by_its_model():
    config = load_config()
    cols = ["age", "job"]
    for name, grid in config["tuning"]["grids"].items():
        for combo in tune.expand_grid(grid):
            models.build_model(name, cols, {**config["models"][name], **combo}, config["seed"])   # raises on a typo
    assert set(config["tuning"]["grids"]) == set(config["models"])


def test_trying_only_the_default_settings_improves_nothing():
    config = small_config()
    config["tuning"]["grids"] = {"logistic_regression": {"C": [1.0]}}   # 1.0 is already the default
    sets = {"train": make_timed(2400, 0), "validation": make_frame(600, 1), "test": None}
    r = tune.run(config, sets)["models"]["logistic_regression"]
    assert r["folds_improved"] == 0
    assert r["validation_change"]["median"] == 0.0
