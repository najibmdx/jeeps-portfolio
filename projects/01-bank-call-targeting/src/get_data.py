"""Build data/bank.db from the public UCI file, when it is missing.

The raw file is the UCI "Bank Marketing" data (Moro, Rita and Cortez, 2014),
file bank-additional-full.csv. The project reads a small database made from it,
with three changes and nothing else:
    1. a row_id column is added, counting from 0 in the order of the file
    2. dots in column names become underscores (emp.var.rate -> emp_var_rate)
    3. y stays as the words "yes" and "no"

Nothing is sorted, dropped or cleaned here. Duplicates are removed later, in split.py.
The test set is not touched: this file only prepares the raw table.

Run it:  python src/get_data.py
"""

import io
import sqlite3
import urllib.request
import zipfile
from contextlib import closing
from pathlib import Path

import pandas as pd

from columns import all_known_columns
from settings import load_config
from split import DB_PATH, TABLE

CSV_NAME = "bank-additional-full.csv"
TIMEOUT_SECONDS = 60


def download(url: str) -> bytes:
    """Fetch the zip file from the web."""
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def _read_member(archive: zipfile.ZipFile, wanted: str) -> bytes | None:
    """Return the bytes of the first file whose name ends with `wanted`, or None."""
    for name in archive.namelist():
        if name.endswith(wanted) and not name.startswith("__MACOSX"):
            return archive.read(name)
    return None


def find_csv(zip_bytes: bytes) -> bytes:
    """Find the csv inside the download. The UCI zip holds a second zip, so look inside both.

    Files are read into memory by name. Nothing is extracted to disk, so a strange
    file name inside the zip cannot write anywhere.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer:
        found = _read_member(outer, CSV_NAME)
        if found is not None:
            return found
        for name in outer.namelist():
            if name.endswith(".zip") and not name.startswith("__MACOSX"):
                with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                    found = _read_member(inner, CSV_NAME)
                    if found is not None:
                        return found
    raise FileNotFoundError(f"{CSV_NAME} was not found in the download.")


def to_table(csv_bytes: bytes) -> pd.DataFrame:
    """Turn the raw csv into the project's table (the three changes in the top note)."""
    df = pd.read_csv(io.BytesIO(csv_bytes), sep=";")
    df.columns = [c.replace(".", "_") for c in df.columns]
    df.insert(0, "row_id", range(len(df)))
    return df


def check_table(df: pd.DataFrame, expected_rows: int, expected_yes: int) -> None:
    """Stop early if the download is not the file this project was built on."""
    expected_columns = set(all_known_columns())
    if set(df.columns) != expected_columns:
        raise ValueError(
            f"Columns differ from the project's list. Missing: {sorted(expected_columns - set(df.columns))}; "
            f"extra: {sorted(set(df.columns) - expected_columns)}."
        )
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}.")
    yes = int((df["y"] == "yes").sum())
    if yes != expected_yes:
        raise ValueError(f"Expected {expected_yes} 'yes' rows, got {yes}.")


def _sqlite_type(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    return "TEXT"


def write_db(df: pd.DataFrame, db_path: Path, table: str = TABLE) -> None:
    """Write the table. It is built in a side file and renamed, so a failed run leaves no half file."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    side = db_path.with_name(db_path.name + ".part")
    side.unlink(missing_ok=True)
    columns = ", ".join(f'"{c}" {_sqlite_type(t)}' for c, t in df.dtypes.items())
    marks = ", ".join("?" for _ in df.columns)
    try:
        with closing(sqlite3.connect(side)) as conn:
            conn.execute(f'CREATE TABLE "{table}" ({columns})')
            conn.executemany(f'INSERT INTO "{table}" VALUES ({marks})', df.astype(object).itertuples(index=False, name=None))
            conn.commit()
        side.replace(db_path)
    finally:
        side.unlink(missing_ok=True)


def ensure_data(config: dict, db_path: Path = DB_PATH, fetch=download) -> bool:
    """Make sure the database exists. Return True if it was built now, False if it was already there."""
    if Path(db_path).exists():
        return False
    settings = config["data"]
    df = to_table(find_csv(fetch(settings["url"])))
    check_table(df, settings["expected_rows"], settings["expected_yes"])
    write_db(df, db_path)
    return True


def main() -> None:
    built = ensure_data(load_config())
    if built:
        print(f"Built {DB_PATH} from the UCI file.")
    else:
        print(f"{DB_PATH} already exists; nothing to do.")


if __name__ == "__main__":
    main()
