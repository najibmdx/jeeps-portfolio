"""Tests for the leakage audit, on small made-up data."""

import numpy as np
import pandas as pd

import columns
import leakage_audit

SETTINGS = {"min_rows": 50, "flag_ratio": 3, "n_bins": 5}


def make_frame(n=4000, seed=0):
    """A table with every column the project knows about. Nothing here predicts y."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "row_id": np.arange(n),
        "age": rng.integers(18, 90, n),
        "job": rng.choice(["admin", "services", "unknown"], n),
        "marital": rng.choice(["married", "single"], n),
        "education": rng.choice(["basic", "university"], n),
        "default": rng.choice(["no", "unknown"], n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "telephone"], n),
        "pdays": np.where(rng.random(n) < 0.9, 999, rng.integers(1, 20, n)),
        "previous": rng.integers(0, 3, n),
        "poutcome": rng.choice(["nonexistent", "failure"], n),
        "month": rng.choice(["may", "jun"], n),
        "day_of_week": rng.choice(["mon", "tue"], n),
        "emp_var_rate": rng.choice([-1.8, 1.1], n),
        "cons_price_idx": rng.random(n),
        "cons_conf_idx": rng.random(n),
        "euribor3m": rng.random(n),
        "nr_employed": rng.choice([5000.0, 5100.0], n),
        "duration": rng.integers(0, 600, n),
        "campaign": rng.integers(1, 10, n),
        "y": np.where(rng.random(n) < 0.2, "yes", "no"),
    })


def test_a_column_that_copies_the_outcome_is_flagged():
    df = make_frame()
    df["duration"] = np.where(df["y"] == "yes", 500, 50)
    table = leakage_audit.audit(df, SETTINGS)
    assert table[table["column"] == "duration"]["flagged"].any()


def test_a_column_unrelated_to_the_outcome_is_not_flagged():
    table = leakage_audit.audit(make_frame(), SETTINGS)
    assert not table[table["column"] == "job"]["flagged"].any()


def test_a_tiny_segment_is_never_flagged():
    df = make_frame()
    df.loc[:9, "job"] = "rare"          # 10 rows, all yes
    df.loc[:9, "y"] = "yes"
    job = lambda s: leakage_audit.audit(df, s).query("column == 'job' and segment == 'rare'")
    assert not job(SETTINGS)["flagged"].any()                       # 10 rows < min_rows 50
    assert job({**SETTINGS, "min_rows": 5})["flagged"].all()        # the rule is what hides it


def test_every_column_is_audited_except_row_id_and_target():
    table = leakage_audit.audit(make_frame(), SETTINGS)
    expected = set(columns.all_known_columns()) - {columns.ROW_ID, columns.TARGET}
    assert set(table["column"]) == expected


def test_the_pdays_code_999_gets_its_own_segment():
    table = leakage_audit.audit(make_frame(), SETTINGS)
    pdays = table[table["column"] == "pdays"]
    assert "999 (code)" in set(pdays["segment"])


def test_segments_cover_every_row_of_every_column():
    df = make_frame()
    table = leakage_audit.audit(df, SETTINGS)
    assert (table.groupby("column")["rows"].sum() == len(df)).all()


def test_the_audit_reads_only_the_training_set():
    config = {"leakage_audit": SETTINGS}
    sets = {"train": make_frame(), "validation": None, "test": None}   # would crash if touched
    table = leakage_audit.run(config, sets)
    assert len(table) > 0
