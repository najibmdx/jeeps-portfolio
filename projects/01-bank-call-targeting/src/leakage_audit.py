"""Leakage audit: for every column, how often did clients say yes in each value or group?

Why: a column that is only known after the call (or that copies the outcome) shows
up as a segment where almost everyone said yes, or almost no one did.

How:
- Reads the TRAINING set only. The audit decides which columns the model may use,
  so validation and test rows must not influence it.
- Words: one segment per value ("unknown" is a value of its own).
- Numbers with 12 or fewer different values: one segment per value.
- Numbers with more values: cut into equal-sized groups (n_bins).
- pdays: the code 999 ("never contacted") is its own segment.
- A segment is FLAGGED when it has at least min_rows clients AND its yes-rate is
  flag_ratio times above or below the training yes-rate, or is exactly 0% or 100%.

Limit: a flag is a question to ask, not proof of leakage. A column can be flagged
because it is genuinely useful (for example, poutcome = success).
"""

from pathlib import Path

import pandas as pd

import columns
import features
import split
from settings import load_config

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

SENTINELS = {"pdays": 999}   # a code that means "no value", given its own segment
MAX_VALUES_AS_IS = 12        # number columns up to this many values keep one segment per value


def label_segments(values: pd.Series, n_bins: int, sentinel=None) -> pd.Series:
    """Give every row the name of its segment (a value, a bin, or the sentinel code)."""
    labels = pd.Series(index=values.index, dtype=object)
    rest = values
    if sentinel is not None:
        is_code = values == sentinel
        labels[is_code] = f"{sentinel} (code)"
        rest = values[~is_code]

    numeric = pd.api.types.is_numeric_dtype(rest)
    if not numeric or rest.nunique() <= MAX_VALUES_AS_IS:
        labels[rest.index] = rest.astype(str)
    else:
        labels[rest.index] = pd.qcut(rest, q=n_bins, duplicates="drop").astype(str)
    return labels


def audit_column(df: pd.DataFrame, col: str, base_rate: float, min_rows: int, flag_ratio: float, n_bins: int) -> pd.DataFrame:
    """One row per segment of one column: rows, yes-rate, and whether it is flagged."""
    values = df[col]
    numeric = pd.api.types.is_numeric_dtype(values)
    frame = pd.DataFrame({
        "segment": label_segments(values, n_bins, SENTINELS.get(col)),
        "yes": features.make_target(df),
    })
    if numeric:
        frame["value"] = values

    grouped = frame.groupby("segment", sort=False)
    out = grouped["yes"].agg(rows="size", yes_rate="mean").reset_index()
    out["_order"] = grouped["value"].mean().values if numeric else out["segment"]
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    out["times_base_rate"] = out["yes_rate"] / base_rate
    extreme = (
        (out["yes_rate"] >= flag_ratio * base_rate)
        | (out["yes_rate"] <= base_rate / flag_ratio)
        | out["yes_rate"].isin([0.0, 1.0])
    )
    out["flagged"] = (out["rows"] >= min_rows) & extreme
    out.insert(0, "column", col)
    return out


def audit(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Audit every column the project accounts for, except the row id and the target."""
    base_rate = float(features.make_target(df).mean())
    decisions = columns.audit_table().set_index("column")["decision"]
    to_audit = [c for c in columns.all_known_columns() if c not in (columns.ROW_ID, columns.TARGET)]

    parts = [
        audit_column(df, c, base_rate, settings["min_rows"], settings["flag_ratio"], settings["n_bins"])
        for c in to_audit
    ]
    table = pd.concat(parts, ignore_index=True)
    table.insert(1, "decision", table["column"].map(decisions))
    table["yes_rate"] = table["yes_rate"].round(4)
    table["times_base_rate"] = table["times_base_rate"].round(2)
    return table


def run(config: dict, sets: dict) -> pd.DataFrame:
    """Audit the training set. Only sets["train"] is read."""
    return audit(sets["train"], config["leakage_audit"])


def main() -> None:
    config = load_config()
    sets = split.prepare(config)
    table = run(config, sets)
    base_rate = float(features.make_target(sets["train"]).mean())

    print(f"training yes-rate (blind guess): {base_rate:.3f}   rows: {len(sets['train']):,}")
    print(f"flag: at least {config['leakage_audit']['min_rows']} rows and yes-rate "
          f"{config['leakage_audit']['flag_ratio']}x above or below {base_rate:.3f} (assumption)\n")

    print(f"{'column':16} {'decision':22} {'segments':>8} {'flagged':>8}  highest yes-rate segment")
    for col, part in table.groupby("column", sort=False):
        judged = part[part["rows"] >= config["leakage_audit"]["min_rows"]]   # tiny segments would mislead
        top = judged.loc[judged["yes_rate"].idxmax()] if len(judged) else part.loc[part["yes_rate"].idxmax()]
        print(f"{col:16} {part['decision'].iloc[0]:22} {len(part):8} {int(part['flagged'].sum()):8}  "
              f"{top['segment']} = {top['yes_rate']:.3f} ({top['rows']:,} rows)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "leakage_audit.csv"
    table.to_csv(path, index=False)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
