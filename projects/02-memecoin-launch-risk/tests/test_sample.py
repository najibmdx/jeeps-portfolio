"""Tests for the made-up sample used by tests and CI."""

import csv

import numpy as np

import safe_pickle
import sample
import settings


def test_the_same_seed_gives_the_same_sample():
    a, la = sample.make_sample(seed=1)
    b, lb = sample.make_sample(seed=1)
    assert la == lb
    for x, y in zip(a, b):
        assert list(x) == list(y)
        np.testing.assert_array_equal(x["ts"], y["ts"])


def test_layout_matches_the_authors_code():
    features, _ = sample.make_sample()
    coin = features[0]
    for key in ("mint_ts", "mint_address", "ts", "ts_len"):
        assert key in coin
    assert coin["ts"].shape == (360, 5) and coin["ts"].dtype == np.float32
    assert 1 <= coin["ts_len"] <= 360


def test_padding_after_ts_len_is_zero():
    features, _ = sample.make_sample()
    for coin in features:
        assert not coin["ts"][coin["ts_len"]:].any()


def test_addresses_are_unique_and_obviously_made_up():
    features, _ = sample.make_sample()
    addresses = [c["mint_address"] for c in features]
    assert len(set(addresses)) == len(addresses)
    assert all(a.startswith("SAMPLE") for a in addresses)


def test_some_coins_have_no_label_like_the_real_233():
    features, labels = sample.make_sample(n_coins=80, n_unlabelled=3)
    assert len(features) == 80 and len(labels) == 77
    assert {l["mint_address"] for l in labels} < {c["mint_address"] for c in features}


def test_the_sample_has_coins_on_both_sides_of_the_scope_filter():
    features, _ = sample.make_sample()
    scope = settings.load_config()["scope"]
    kept = [
        c for c in features
        if c[scope["holders_column"]] >= scope["min_holders"]
        and c[scope["valid_swap_column"]] >= scope["min_valid_swap_seconds"]
    ]
    assert 0 < len(kept) < len(features)


def test_the_sample_has_both_crash_and_no_crash_labels():
    _, labels = sample.make_sample()
    line = settings.load_config()["label"]["crash_line"]
    crashes = [l["min_ratio"] < line for l in labels]
    assert any(crashes) and not all(crashes)


def test_the_sample_never_holds_outcome_columns_among_the_features():
    features, _ = sample.make_sample()
    banned = settings.load_config()["columns"]["banned_as_features"]
    for coin in features:
        assert not set(banned) & set(coin)


def test_written_files_load_back_through_the_safe_loader(tmp_path):
    feature_path, label_path = sample.write_sample(tmp_path, n_coins=20, n_unlabelled=2)
    loaded = safe_pickle.load_file(feature_path)
    original, labels = sample.make_sample(n_coins=20, n_unlabelled=2)
    assert len(loaded) == 20
    for a, b in zip(loaded, original):
        assert list(a) == list(b)
        np.testing.assert_array_equal(a["ts"], b["ts"])
    with open(label_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == sample.LABEL_COLUMNS and len(rows) == 18
