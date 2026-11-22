"""Which columns the model may use, and why.

The one rule: a column is allowed only if its value is known at the moment
the prediction is made, which is BEFORE the phone call.

Every column of the table is listed exactly once, either as a feature group,
as excluded, or as the target / row id. A test checks that none is forgotten.
"""

import pandas as pd

TARGET = "y"
ROW_ID = "row_id"

# Allowed from the start.
CORE_GROUPS = {
    "client": ["age", "job", "marital", "education", "default", "housing", "loan"],
    "contact": ["contact"],
    "history": ["pdays", "previous", "poutcome"],
}

# Known before the call, but they act as a clock (see eda.ipynb sections 3 and 6).
# Not in the first model; added back one group at a time in the ablation study.
ABLATION_GROUPS = {
    "calendar": ["month", "day_of_week"],
    "economy": ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"],
}

# Feature columns that hold words. Every other feature column holds a number.
CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "poutcome", "month", "day_of_week",
]

# Never used as features.
EXCLUDED = {
    "duration": "Length of the call: only known after the call ends, and strongly tied to the outcome.",
    "campaign": "Number of contacts in this campaign: may include the call being predicted, "
    "so its timing is unclear. Excluded in the first model, tested later as an ablation.",
}

GROUP_REASONS = {
    "client": "Facts about the client, known before dialling.",
    "contact": "How the client will be reached (cellular or telephone), chosen before dialling.",
    "history": "Result of earlier campaigns, known before dialling.",
    "calendar": "Known when dialling, but unclear what exactly they record, and the month works as a clock.",
    "economy": "Known when dialling, but these values move with the calendar, so they also work as a clock.",
}


def feature_columns(groups=("client", "contact", "history")) -> list[str]:
    """Return the feature columns for the chosen groups, in a fixed order."""
    available = {**CORE_GROUPS, **ABLATION_GROUPS}
    unknown = [g for g in groups if g not in available]
    if unknown:
        raise ValueError(f"unknown column groups: {unknown}")
    return [col for g in groups for col in available[g]]


def all_known_columns() -> list[str]:
    """Every column this module accounts for."""
    grouped = [c for cols in {**CORE_GROUPS, **ABLATION_GROUPS}.values() for c in cols]
    return [ROW_ID, *grouped, *EXCLUDED, TARGET]


def audit_table() -> pd.DataFrame:
    """One row per column: where it sits and why."""
    rows = [{"column": ROW_ID, "decision": "id only", "reason": "Row position, used to keep the date order."}]
    for name, cols in CORE_GROUPS.items():
        rows += [{"column": c, "decision": f"feature ({name})", "reason": GROUP_REASONS[name]} for c in cols]
    for name, cols in ABLATION_GROUPS.items():
        rows += [{"column": c, "decision": f"ablation ({name})", "reason": GROUP_REASONS[name]} for c in cols]
    rows += [{"column": c, "decision": "excluded", "reason": why} for c, why in EXCLUDED.items()]
    rows.append({"column": TARGET, "decision": "target", "reason": "Did the client say yes."})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(audit_table().to_string(index=False))
