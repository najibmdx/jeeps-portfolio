"""Turn the table into numbers that a model can read.

Steps, in order:
1. pdays: the value 999 means "never contacted before". We make a yes/no
   flag (pdays_never) and set the number to 0, so the flag carries the meaning.
2. Words (job, education, ...) become one 0/1 column per value.
   "unknown" is kept as a value of its own.
3. Optionally rescale the numbers (logistic regression needs this).

No peeking: the pipeline is fitted on the training set only. The validation
and test sets are only ever transformed with what the training set taught it.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import columns

PDAYS_NEVER = 999


def make_target(df: pd.DataFrame) -> pd.Series:
    """1 if the client said yes, otherwise 0."""
    return (df[columns.TARGET] == "yes").astype(int)


def add_pdays_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add pdays_never and replace 999 in pdays by 0. Returns a copy."""
    out = df.copy()
    if "pdays" not in out.columns:
        return out
    never = out["pdays"] == PDAYS_NEVER
    out["pdays_never"] = never.astype(int)
    out.loc[never, "pdays"] = 0
    return out


def build_preprocessor(feature_cols: list[str], scale_numbers: bool = False) -> Pipeline:
    """Build the (still unfitted) pipeline for the given feature columns."""
    word_cols = [c for c in feature_cols if c in columns.CATEGORICAL]
    number_cols = [c for c in feature_cols if c not in columns.CATEGORICAL]
    if "pdays" in feature_cols:
        number_cols.append("pdays_never")

    parts = []
    if word_cols:
        parts.append(("words", OneHotEncoder(handle_unknown="ignore", sparse_output=False), word_cols))
    if number_cols:
        parts.append(("numbers", StandardScaler() if scale_numbers else "passthrough", number_cols))

    return Pipeline([
        ("pdays", FunctionTransformer(add_pdays_flag)),
        ("columns", ColumnTransformer(parts)),
    ])


def main() -> None:
    import split
    from settings import load_config

    config = load_config()
    feature_cols = columns.feature_columns(config["features"]["groups"])
    sets = split.prepare(config)

    pre = build_preprocessor(feature_cols)
    pre.fit(sets["train"][feature_cols])  # learn from the training set only

    for name, part in sets.items():
        X = pre.transform(part[feature_cols])
        print(f"{name:10} rows {X.shape[0]:6}  columns {X.shape[1]}  missing values {int(pd.isna(X).sum())}")

    names = pre.named_steps["columns"].get_feature_names_out()
    print("first columns:", ", ".join(names[:6]))


if __name__ == "__main__":
    main()
