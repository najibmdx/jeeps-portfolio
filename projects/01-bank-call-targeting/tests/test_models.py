"""Tests for the models and the training run, on small made-up data."""

import numpy as np
import pandas as pd
import pytest

import columns
import models
import train

FEATURES = columns.feature_columns(("client", "contact", "history"))


def make_clients(n: int, seed: int) -> pd.DataFrame:
    """Made-up clients with every feature column the default groups need."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(20, 70, n),
        "job": rng.choice(["admin.", "services", "unknown"], n),
        "marital": rng.choice(["married", "single"], n),
        "education": rng.choice(["basic.4y", "university.degree"], n),
        "default": rng.choice(["no", "unknown"], n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "telephone"], n),
        "pdays": rng.choice([999, 3, 10], n),
        "previous": rng.integers(0, 3, n),
        "poutcome": rng.choice(["nonexistent", "success", "failure"], n),
        "y": rng.choice(["no", "yes"], n, p=[0.8, 0.2]),
    })


SMALL_CONFIG = {
    "seed": 1,
    "features": {"groups": ["client", "contact", "history"]},
    "evaluation": {"top_fraction": 0.10},
    "models": {
        "logistic_regression": {"max_iter": 200},
        "random_forest": {"n_estimators": 10, "min_samples_leaf": 2},
        "gradient_boosting": {"max_iter": 10},
    },
}


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_each_model_gives_a_probability_per_client(name):
    df = make_clients(80, seed=0)
    model = models.build_model(name, FEATURES, SMALL_CONFIG["models"][name], seed=1)
    model.fit(df[FEATURES], (df["y"] == "yes").astype(int))
    probs = model.predict_proba(df[FEATURES])
    assert probs.shape == (80, 2)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_an_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        models.build_model("magic_model", FEATURES, {}, seed=1)


def test_the_training_run_never_touches_the_test_set():
    sets = {"train": make_clients(120, 0), "validation": make_clients(40, 1), "test": None}
    metrics = train.run(SMALL_CONFIG, sets)   # would crash if it tried to read the test set
    assert set(metrics["models"]) == set(models.MODEL_NAMES)


def test_the_same_seed_gives_the_same_numbers():
    sets = {"train": make_clients(120, 0), "validation": make_clients(40, 1), "test": None}
    first = train.run(SMALL_CONFIG, sets)
    second = train.run(SMALL_CONFIG, sets)
    assert first == second
