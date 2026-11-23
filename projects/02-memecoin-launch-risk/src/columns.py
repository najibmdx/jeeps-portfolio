"""Which columns may be used as features.

Rule: a column is allowed only if it is known at the moment of migration. The tabular
feature columns all start with "group". The outcome columns (measured AFTER migration)
and the bookkeeping columns are never features, whatever they are called.
"""


def feature_columns(all_columns, config: dict) -> list:
    """The allowed feature columns, in their original order."""
    prefix = config["columns"]["feature_prefix"]
    banned = set(config["columns"]["banned_as_features"])
    bookkeeping = set(config["columns"]["not_tabular"])
    return [
        c for c in all_columns
        if c.startswith(prefix) and c not in banned and c not in bookkeeping
    ]


def check_no_banned(columns, config: dict) -> None:
    """Raise if any outcome or bookkeeping column is in a feature list."""
    forbidden = set(config["columns"]["banned_as_features"]) | set(config["columns"]["not_tabular"])
    found = sorted(forbidden & set(columns))
    if found:
        raise ValueError(f"these columns must never be features: {found}")
