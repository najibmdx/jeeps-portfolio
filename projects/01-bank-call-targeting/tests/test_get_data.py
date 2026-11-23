"""Tests for building the database from the UCI download, on small made-up data. No internet is used."""

import io
import sqlite3
import zipfile
from contextlib import closing

import pandas as pd
import pytest

import get_data
import split
from settings import load_config
from test_ablation import make_frame

N = 200


def raw_csv_bytes(n=N):
    """A file shaped like the UCI one: semicolons, dotted names, no row_id."""
    df = make_frame(n, 3).drop(columns=["row_id"])
    dotted = {"emp_var_rate": "emp.var.rate", "cons_price_idx": "cons.price.idx",
              "cons_conf_idx": "cons.conf.idx", "nr_employed": "nr.employed"}
    df = df.rename(columns=dotted)
    return df.to_csv(sep=";", index=False).encode("utf-8")


def nested_zip(csv_bytes, junk=True):
    """The UCI shape: an outer zip that holds a second zip that holds the csv."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("bank-additional/bank-additional-full.csv", csv_bytes)
        z.writestr("bank-additional/bank-additional.csv", b"not the file we want")
        if junk:
            z.writestr("__MACOSX/bank-additional/._bank-additional-full.csv", b"junk")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("bank-additional.zip", inner.getvalue())
        z.writestr("bank.zip", b"another file")
    return outer.getvalue()


def config_for(df):
    return {"data": {"url": "unused", "expected_rows": len(df), "expected_yes": int((df["y"] == "yes").sum())}}


def test_the_csv_is_found_inside_the_zip_inside_the_zip():
    csv = raw_csv_bytes()
    assert get_data.find_csv(nested_zip(csv)) == csv


def test_the_csv_is_found_when_the_zip_is_flat():
    csv = raw_csv_bytes()
    flat = io.BytesIO()
    with zipfile.ZipFile(flat, "w") as z:
        z.writestr("bank-additional-full.csv", csv)
    assert get_data.find_csv(flat.getvalue()) == csv


def test_a_zip_without_the_csv_is_refused():
    flat = io.BytesIO()
    with zipfile.ZipFile(flat, "w") as z:
        z.writestr("other.csv", b"a;b\n1;2\n")
    with pytest.raises(FileNotFoundError):
        get_data.find_csv(flat.getvalue())


def test_the_table_has_the_three_changes_and_nothing_else():
    csv = raw_csv_bytes()
    table = get_data.to_table(csv)
    original = make_frame(N, 3)
    assert list(table["row_id"]) == list(range(N))          # counts from 0, in file order
    assert "emp_var_rate" in table.columns and "emp.var.rate" not in table.columns
    assert set(table["y"]) == {"yes", "no"}                 # y stays as words
    pd.testing.assert_frame_equal(table[original.columns.drop("row_id")].reset_index(drop=True),
                                  original.drop(columns=["row_id"]), check_dtype=False)


def test_the_built_database_can_be_read_by_the_project_loader(tmp_path):
    db = tmp_path / "data" / "bank.db"
    original = make_frame(N, 3)
    built = get_data.ensure_data(config_for(original), db_path=db, fetch=lambda url: nested_zip(raw_csv_bytes()))
    assert built is True
    loaded = split.load_data(db_path=db)
    assert len(loaded) == N
    pd.testing.assert_frame_equal(loaded[original.columns].reset_index(drop=True), original, check_dtype=False)


def test_column_types_match_the_original_database(tmp_path):
    db = tmp_path / "bank.db"
    original = make_frame(N, 3)
    get_data.ensure_data(config_for(original), db_path=db, fetch=lambda url: nested_zip(raw_csv_bytes()))
    with closing(sqlite3.connect(db)) as conn:
        types = dict(conn.execute(f"SELECT name, type FROM pragma_table_info('{split.TABLE}')").fetchall())
    assert types["row_id"] == "INTEGER" and types["age"] == "INTEGER" and types["pdays"] == "INTEGER"
    assert types["euribor3m"] == "REAL" and types["nr_employed"] == "REAL"
    assert types["y"] == "TEXT" and types["job"] == "TEXT"


def test_an_existing_database_is_left_alone_and_nothing_is_downloaded(tmp_path):
    db = tmp_path / "bank.db"
    db.write_bytes(b"my own file")

    def must_not_be_called(url):
        raise AssertionError("downloaded although the database exists")

    built = get_data.ensure_data({"data": {}}, db_path=db, fetch=must_not_be_called)
    assert built is False
    assert db.read_bytes() == b"my own file"


def test_the_wrong_row_count_is_refused_and_no_file_is_left(tmp_path):
    db = tmp_path / "bank.db"
    original = make_frame(N, 3)
    cfg = config_for(original)
    cfg["data"]["expected_rows"] = N + 1
    with pytest.raises(ValueError, match="rows"):
        get_data.ensure_data(cfg, db_path=db, fetch=lambda url: nested_zip(raw_csv_bytes()))
    assert not db.exists()
    assert not list(tmp_path.glob("*.part"))


def test_the_wrong_yes_count_is_refused(tmp_path):
    original = make_frame(N, 3)
    cfg = config_for(original)
    cfg["data"]["expected_yes"] += 1
    with pytest.raises(ValueError, match="yes"):
        get_data.ensure_data(cfg, db_path=tmp_path / "bank.db", fetch=lambda url: nested_zip(raw_csv_bytes()))


def test_a_file_with_the_wrong_columns_is_refused(tmp_path):
    csv = raw_csv_bytes().replace(b"housing", b"housing_x", 1)
    original = make_frame(N, 3)
    with pytest.raises(ValueError, match="Columns"):
        get_data.ensure_data(config_for(original), db_path=tmp_path / "bank.db", fetch=lambda url: nested_zip(csv))


def test_a_failed_write_leaves_no_half_built_file(tmp_path):
    db = tmp_path / "bank.db"
    bad = get_data.to_table(raw_csv_bytes())
    bad["age"] = [object()] * len(bad)   # a value sqlite cannot store
    with pytest.raises(Exception):
        get_data.write_db(bad, db)
    assert not db.exists()
    assert not list(tmp_path.glob("*.part"))


def test_the_settings_file_has_the_data_block():
    data = load_config()["data"]
    assert data["url"].startswith("https://")
    assert data["expected_rows"] == 41188 and data["expected_yes"] == 4640
