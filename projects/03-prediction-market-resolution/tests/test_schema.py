import pyarrow as pa
import pyarrow.parquet as pq

import schema


def test_expected_schema_has_no_problems():
    for table in schema.EXPECTED:
        assert schema.check_schema(table, schema.arrow_schema(table)) == []


def test_missing_wrong_and_extra_columns_are_named():
    s = schema.arrow_schema("events")
    broken = pa.schema([f for f in s if f.name != "title"] + [pa.field("surprise", pa.int64())])
    broken = broken.set(0, pa.field("event_id", pa.int64()))
    problems = schema.check_schema("events", broken)
    assert any("'title' is missing" in p for p in problems)
    assert any("'event_id' has type int64" in p for p in problems)
    assert any("unexpected column 'surprise'" in p for p in problems)


def test_unreadable_file_is_a_plain_problem(tmp_path):
    bad = tmp_path / "events.parquet"
    bad.write_bytes(b"not a parquet file")
    problems = schema.check_file("events", bad)
    assert len(problems) == 1 and "could not be read" in problems[0]


def test_check_file_on_good_file(tmp_path):
    path = tmp_path / "markets.parquet"
    pq.write_table(schema.arrow_schema("markets").empty_table(), path)
    assert schema.check_file("markets", path) == []
