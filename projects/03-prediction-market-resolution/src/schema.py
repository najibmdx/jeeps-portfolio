"""The column layout the four data tables must have.

Read from the real files on 24 Nov 2026 (parquet schemas), so a future change in the dataset
shows up as a plain-words problem instead of a confusing error later.
"""

import pyarrow as pa
import pyarrow.parquet as pq

TS = "timestamp[ns, tz=UTC]"

# (column name, arrow type as text) in file order.
EXPECTED = {
    "events": [
        ("event_id", "large_string"),
        ("event_ticker", "large_string"),
        ("series_id", "large_string"),
        ("title", "large_string"),
        ("category", "large_string"),
        ("category_original", "large_string"),
        ("event_start_time", TS),
        ("mutually_exclusive", "bool"),
        ("volume", "double"),
        ("open_interest", "double"),
        ("n_questions", "uint32"),
        ("multi_category_flag", "bool"),
    ],
    "markets": [
        ("market_id", "int64"),
        ("question", "large_string"),
        ("category", "large_string"),
        ("category_original", "large_string"),
        ("event_id", "large_string"),
        ("outcome_yes", "bool"),
        ("market_start_time", TS),
        ("close_time", TS),
        ("expiration_time", TS),
        ("can_close_early", "bool"),
        ("taker_base_fee", "int64"),
        ("has_fee", "bool"),
    ],
    "predictions": [
        ("prediction_id", "large_string"),
        ("market_id", "int64"),
        ("outcome", "large_string"),
        ("outcome_idx", "int64"),
        ("n_outcomes", "int64"),
        ("negative_token", "large_string"),
        ("winner", "bool"),
    ],
    "ohlcv_1d": [
        ("prediction_id", "large_string"),
        ("timestamp", TS),
        ("market_id", "large_string"),
        ("outcome", "large_string"),
        ("open", "double"),
        ("high", "double"),
        ("low", "double"),
        ("close", "double"),
        ("volume", "double"),
        ("trade_count", "int64"),
        ("open_interest", "double"),
    ],
}

_ARROW = {
    "large_string": pa.large_string(),
    "bool": pa.bool_(),
    "double": pa.float64(),
    "uint32": pa.uint32(),
    "int64": pa.int64(),
    TS: pa.timestamp("ns", tz="UTC"),
}


def arrow_schema(table: str) -> pa.Schema:
    """The exact arrow schema for one table."""
    return pa.schema([pa.field(name, _ARROW[kind]) for name, kind in EXPECTED[table]])


def check_schema(table: str, schema: pa.Schema) -> list:
    """Problems found, as plain sentences. An empty list means the layout is exactly as expected."""
    found = {field.name: str(field.type) for field in schema}
    expected = dict(EXPECTED[table])
    problems = []
    for name, kind in expected.items():
        if name not in found:
            problems.append(f"column '{name}' is missing")
        elif found[name] != kind:
            problems.append(f"column '{name}' has type {found[name]}, expected {kind}")
    for name in found:
        if name not in expected:
            problems.append(f"unexpected column '{name}'")
    return problems


def check_file(table: str, path) -> list:
    """Read only the header of a parquet file and check its layout."""
    try:
        schema = pq.read_schema(path)
    except Exception as error:
        return [f"could not be read as a parquet file ({type(error).__name__})"]
    return check_schema(table, schema)
