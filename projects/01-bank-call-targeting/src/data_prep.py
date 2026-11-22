"""Download the Bank Marketing data and load it into a SQLite database.

Run from the project folder:  python src/data_prep.py
"""
import io
import sqlite3
import urllib.request
import zipfile
from contextlib import closing
from pathlib import Path

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
INNER_ZIP = "bank-additional.zip"
CSV_MEMBER = "bank-additional/bank-additional-full.csv"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "bank-additional-full.csv"
DB_PATH = DATA_DIR / "bank.db"
TABLE = "bank_marketing"


def download_csv(csv_path: Path = CSV_PATH) -> Path:
    """Fetch the zip from UCI and save the one CSV we need. Skip if it exists."""
    if csv_path.exists():
        return csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP)))
    csv_path.write_bytes(inner.read(CSV_MEMBER))
    return csv_path


def load_to_sqlite(csv_path: Path, db_path: Path = DB_PATH, table: str = TABLE) -> int:
    """Write the CSV into a SQLite table and return the number of rows."""
    df = pd.read_csv(csv_path, sep=";")
    # Dots in column names (emp.var.rate) are awkward in SQL, so use underscores.
    df.columns = [name.replace(".", "_") for name in df.columns]
    with closing(sqlite3.connect(db_path)) as conn:
        # row_id keeps the file's original order. The rows are in date order,
        # and that order is our clock for the time-based split later.
        df.to_sql(table, conn, if_exists="replace", index=True, index_label="row_id")
    return len(df)


def main() -> None:
    csv_path = download_csv()
    n_rows = load_to_sqlite(csv_path)
    print(f"Loaded {n_rows} rows into {DB_PATH}")


if __name__ == "__main__":
    main()