"""The three scorers. Each gives every coin a number between 0 and 1: how likely it is to crash.

  logistic_regression  a simple scoring formula (numbers are put on a common scale first,
                       using the training pile only, because the scaler lives inside the pipeline)
  random_forest        many decision trees that vote
  gradient_boosting    trees added one by one, each fixing the mistakes of the ones before
"""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAMES = ("logistic_regression", "random_forest", "gradient_boosting")


def build_model(name: str, params: dict, seed: int):
    """An untrained scorer with the given settings. The seed makes every run repeat exactly."""
    if name == "logistic_regression":
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(random_state=seed, **params))])
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"unknown model {name!r}; choose from {NAMES}")
