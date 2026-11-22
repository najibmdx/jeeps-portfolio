"""Clean the data and split it by time.

The file has no date column, but the rows are in date order and the month
name is given (eda.ipynb, section 3). We rebuild a year from the order, then
cut at the end of a month so that no month is shared between two sets.

Split chosen for project 01 (Option A):
    train      May 2008 to Apr 2009
    validation May 2009
    test       Jun 2009 to Nov 2010
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bank.db"
TABLE = "bank_marketing"

MONTH_NUMBER = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
FIRST_YEAR = 2008  # the authors state the file starts in May 2008

# Last month (as YYYYMM) that belongs to each set. Everything later is test.
TRAIN_END = 200904
VALID_END = 200905


def load_data(db_path: Path = DB_PATH, table: str = TABLE) -> pd.DataFrame:
    """Read the table back in its original row order."""
    with closing(sqlite3.connect(db_path)) as conn:
        return pd.read_sql(f"SELECT * FROM {table} ORDER BY row_id", conn)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated rows, keeping the first. row_id is ignored when comparing."""
    compare = [c for c in df.columns if c != "row_id"]
    return df[~df.duplicated(subset=compare, keep="first")].reset_index(drop=True)


def add_period(df: pd.DataFrame) -> pd.DataFrame:
    """Add year and period (YYYYMM). The year goes up each time the month wraps round."""
    month_no = df["month"].map(MONTH_NUMBER)
    wrapped = month_no < month_no.shift()
    out = df.copy()
    out["year"] = FIRST_YEAR + wrapped.cumsum()
    out["period"] = out["year"] * 100 + month_no
    return out


def time_split(df: pd.DataFrame, train_end: int = TRAIN_END, valid_end: int = VALID_END) -> dict:
    """Cut at month ends. df must already have a period column."""
    return {
        "train": df[df["period"] <= train_end],
        "validation": df[(df["period"] > train_end) & (df["period"] <= valid_end)],
        "test": df[df["period"] > valid_end],
    }


def baselines(splits: dict) -> pd.DataFrame:
    """Rows and yes-rate per set. The yes-rate is also the PR-AUC of a blind guesser."""
    rows = []
    for name, part in splits.items():
        rows.append({
            "set": name,
            "rows": len(part),
            "from": int(part["period"].min()),
            "to": int(part["period"].max()),
            "yes_rate": (part["y"] == "yes").mean(),
        })
    return pd.DataFrame(rows).set_index("set")


def prepare() -> dict:
    """Load, de-duplicate, date and split. Returns the three sets."""
    df = add_period(drop_duplicates(load_data()))
    return time_split(df)


def main() -> None:
    splits = prepare()
    table = baselines(splits)
    total = table["rows"].sum()
    table["share"] = table["rows"] / total
    print(table.to_string(formatters={"yes_rate": "{:.1%}".format, "share": "{:.1%}".format}))


if __name__ == "__main__":
    main()
