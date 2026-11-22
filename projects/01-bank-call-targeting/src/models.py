"""The three models, each wrapped with the feature pipeline.

Every model is a Pipeline: feature pipeline first, then the model. So fitting
a model on the training set also fits the feature steps on the training set only.
"""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import features

MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")


def build_model(name: str, feature_cols: list[str], params: dict, seed: int) -> Pipeline:
    """Return an unfitted pipeline for the named model."""
    if name == "logistic_regression":
        estimator = LogisticRegression(random_state=seed, **params)
        scale = True  # logistic regression needs numbers on a similar scale
    elif name == "random_forest":
        estimator = RandomForestClassifier(random_state=seed, **params)
        scale = False
    elif name == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(random_state=seed, **params)
        scale = False
    else:
        raise ValueError(f"unknown model: {name}. Choose from {MODEL_NAMES}")

    return Pipeline([
        ("features", features.build_preprocessor(feature_cols, scale_numbers=scale)),
        ("model", estimator),
    ])
