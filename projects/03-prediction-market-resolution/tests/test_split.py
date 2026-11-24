"""Hand-made table with known answers. 20 markets, market i has t = 2026-01-i at midnight.

With 20 values the 60% quantile sits 11.4 steps from the first value: boundary 1 = Jan 12 09:36.
The 80% quantile sits 15.2 steps in: boundary 2 = Jan 16 04:48. So by t alone:
train = markets 1 to 12, validation = 13 to 16, test = 17 to 20 (12 / 4 / 4).

By default a market closes 2 hours after t. The special cases:
  E_P = {1, 2}      two markets of one event, both train
  E_S = {5, 14}     5 is train, 14 is validation by t: the event goes to train (earliest market), 14 moves
  E_V = {13, 18}    13 is validation, 18 is test by t: the event goes to validation, 18 moves
  E_T = {19, 20}    two markets of one event, both test
  market 4          has no event, so it is its own event (key "market:4")
  market 11         closes 1 minute after boundary 1: purged
  market 12         closes exactly at boundary 1: kept (the rule is strictly after)
  market 15         closes exactly at boundary 2: kept
  market 16         closes 1 minute after boundary 2: purged
  markets 14, 18    moved into an earlier split, so their t is after that split's end: purged
"""

import copy

import numpy as np
import pandas as pd
import pytest

import build
import settings
import split

D = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731

B1 = D("2026-01-12 09:36")
B2 = D("2026-01-16 04:48")


def _table():
    rows = []
    for i in range(1, 21):
        t = D(f"2026-01-{i:02d}")
        event = {1: "E_P", 2: "E_P", 5: "E_S", 14: "E_S", 13: "E_V", 18: "E_V", 19: "E_T", 20: "E_T"}.get(i, f"E{i}")
        if i == 4:
            event = "market:4"
        close = {11: B1 + pd.Timedelta(minutes=1), 12: B1, 15: B2, 16: B2 + pd.Timedelta(minutes=1)}.get(i, t + pd.Timedelta(hours=2))
        rows.append(dict(market_id=i, event_key=event, t=t, close_time=close, label=int(i % 3 == 0), last_close=0.1 * (i % 5)))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def config():
    return settings.load_config()


@pytest.fixture(scope="module")
def result(config):
    table, report = split.assign(_table(), config)
    return table.set_index("market_id"), report


def _ids(table, **conditions):
    mask = pd.Series(True, index=table.index)
    for col, val in conditions.items():
        mask &= table[col] == val
    return sorted(table.index[mask])


def test_boundaries(config):
    b1, b2 = split.boundaries(_table()["t"], config)
    assert (b1, b2) == (B1, B2)


def test_split_by_t_alone(result):
    table, report = result
    assert _ids(table, split_by_t="train") == list(range(1, 13))
    assert _ids(table, split_by_t="validation") == [13, 14, 15, 16]
    assert _ids(table, split_by_t="test") == [17, 18, 19, 20]
    assert report["by_t"]["sizes"] == {"train": 12, "validation": 4, "test": 4}
    assert report["boundaries"] == ["2026-01-12", "2026-01-16"]


def test_a_market_exactly_at_a_boundary_goes_to_the_later_split(config):
    # Real t values are midnights with many ties, so a boundary can land exactly on a t value.
    # 11 markets: the 60% point is value number 7 (Jan 7) and the 80% point is value number 9 (Jan 9).
    days = [1, 2, 3, 4, 5, 6, 7, 7, 9, 9, 10]
    rows = [
        dict(market_id=i + 1, event_key=f"E{i + 1}", t=D(f"2026-01-{d:02d}"), close_time=D(f"2026-01-{d:02d} 01:00"), label=0)
        for i, d in enumerate(days)
    ]
    table, report = split.assign(pd.DataFrame(rows), config)
    assert split.boundaries(pd.DataFrame(rows)["t"], config) == (D("2026-01-07"), D("2026-01-09"))
    # t on Jan 7 is not before boundary 1, so it is validation; t on Jan 9 is not before boundary 2, so it is test
    assert report["by_t"]["sizes"] == {"train": 6, "validation": 2, "test": 3}
    assert list(table["split_by_t"]) == ["train"] * 6 + ["validation"] * 2 + ["test"] * 3


def test_events_by_t_and_straddling(result):
    _, report = result
    # train: E_P, market:4, E_S and the eight single-market events 3, 6, 7, 8, 9, 10, 11, 12 -> 11 events
    assert report["by_t"]["events"] == {"train": 11, "validation": 4, "test": 3}
    assert report["by_t"]["events_straddling_a_boundary"] == 2  # E_S and E_V


def test_purge_counts_by_t_use_strictly_after(result):
    _, report = result
    # train: only market 11 (market 12 closes exactly at the boundary). validation: only market 16 (not 15)
    assert report["by_t"]["purge_train"] == 1
    assert report["by_t"]["purge_validation"] == 1


def test_events_are_kept_whole(result):
    table, report = result
    assert table.loc[14, "split"] == "train"        # moved to the split of market 5
    assert table.loc[18, "split"] == "validation"   # moved to the split of market 13
    assert table.loc[19, "split"] == table.loc[20, "split"] == "test"
    assert table.loc[1, "split"] == table.loc[2, "split"] == "train"
    assert report["events_whole"]["markets_moved"] == 2
    assert report["events_whole"]["sizes"] == {"train": 13, "validation": 4, "test": 3}
    assert report["events_whole"]["events"] == {"train": 11, "validation": 3, "test": 2}
    # no event sits in two splits
    assert (table.groupby("event_key")["split"].nunique() == 1).all()


def test_a_market_without_an_event_is_its_own_event(result):
    table, _ = result
    assert table.loc[4, "event_key"] == "market:4"
    assert (table["event_key"] == "market:4").sum() == 1


def test_purged_markets(result):
    table, report = result
    assert _ids(table, purged=True) == [11, 14, 16, 18]
    assert report["purged"] == {"train": 2, "validation": 2}
    # a market closing exactly at a boundary stays
    assert not table.loc[12, "purged"] and not table.loc[15, "purged"]


def test_no_purged_market_remains_in_the_written_parts(result, config, tmp_path):
    table, _ = result
    paths = split.write_splits(table.reset_index(), tmp_path, config, 7)
    written = {p: pd.read_parquet(path) for p, path in paths.items()}
    assert sorted(written["train"]["market_id"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    assert sorted(written["validation"]["market_id"]) == [13, 15]
    assert sorted(written["test"]["market_id"]) == [17, 19, 20]
    for p in ("train", "validation"):
        assert not set(written[p]["market_id"]) & {11, 14, 16, 18}


def test_written_parts_keep_the_dataset_columns_and_drop_the_helpers(result, config, tmp_path):
    table, _ = result
    paths = split.write_splits(table.reset_index(), tmp_path, config, 7)
    cols = list(pd.read_parquet(paths["train"]).columns)
    assert cols == ["market_id", "event_key", "t", "close_time", "label", "last_close"]
    assert all(h not in cols for h in split.HELPERS)


def test_file_names_hold_the_part_and_k(config, tmp_path):
    assert split.split_path(tmp_path, config, "train", 7).name == "train_k7.parquet"
    assert split.split_path(tmp_path, config, "test", 3).name == "test_k3.parquet"


def test_final_sizes_and_events(result):
    _, report = result
    assert report["final"]["sizes"] == {"train": 11, "validation": 2, "test": 3}
    assert report["final"]["events"] == {"train": 10, "validation": 2, "test": 2}


def test_yes_rates_use_train_and_validation_only(result):
    _, report = result
    # label is 1 for markets 3, 6, 9, 12, 15, 18...; kept train markets 1 to 10 and 12 have 3, 6, 9, 12 -> 4 of 11
    assert report["final"]["yes_rate"]["train"] == pytest.approx(4 / 11)
    assert report["final"]["yes_rate"]["validation"] == pytest.approx(0.5)  # markets 13 and 15
    assert "test" not in report["final"]["yes_rate"]


def test_the_test_split_cannot_be_read_back(result, config, tmp_path):
    table, _ = result
    split.write_splits(table.reset_index(), tmp_path, config, 7)
    assert len(split.read_split(tmp_path, config, "train", 7)) == 11
    assert len(split.read_split(tmp_path, config, "validation", 7)) == 2
    with pytest.raises(PermissionError):
        split.read_split(tmp_path, config, "test", 7)


def test_row_order_does_not_change_the_answer(config):
    base, _ = split.assign(_table(), config)
    shuffled = _table().sample(frac=1.0, random_state=1)
    other, _ = split.assign(shuffled, config)
    a = base.set_index("market_id").sort_index()[["split_by_t", "split", "purged"]]
    b = other.set_index("market_id").sort_index()[["split_by_t", "split", "purged"]]
    pd.testing.assert_frame_equal(a, b)


def test_too_few_markets_stops_with_a_plain_message(config):
    with pytest.raises(ValueError, match="at least 3"):
        split.assign(_table().head(2), config)


def test_config_must_keep_the_split_rules():
    config = settings.load_config()
    for key in ("purge_unknown_labels", "events_whole"):
        bad = copy.deepcopy(config)
        bad["split"][key] = False
        with pytest.raises(ValueError, match=key):
            settings.validate_config(bad)
    bad = copy.deepcopy(config)
    bad["dataset"]["split_file"] = "split.parquet"
    with pytest.raises(ValueError, match="split_file"):
        settings.validate_config(bad)


def test_split_cli_needs_the_dataset(tmp_path, capsys):
    assert split.main(["--data-dir", str(tmp_path)]) == 2
    assert "Run src/build.py first" in capsys.readouterr().out


def test_split_cli_writes_the_parts_and_a_report(tmp_path, capsys, monkeypatch):
    config = settings.load_config()
    path = build.dataset_path(tmp_path, config, 7)
    path.parent.mkdir(parents=True)
    _table().to_parquet(path, index=False)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)  # the report goes to tmp_path/outputs, not into the project
    assert split.main(["--data-dir", str(tmp_path)]) == 0
    assert (tmp_path / "derived" / "train_k7.parquet").exists()
    assert (tmp_path / "derived" / "test_k7.parquet").exists()
    assert (tmp_path / "outputs" / "split_report_k7.json").exists()
    assert "final:" in capsys.readouterr().out


# ---- on the real data: skipped when the four files are not on this machine

REAL = all((settings.DATA_DIR / f"{t}.parquet").exists() for t in settings.TABLES)


@pytest.mark.skipif(not REAL, reason="real data not downloaded")
def test_real_data_reproduces_the_audit_numbers_and_keeps_events_whole():
    config = settings.load_config()
    dataset, _ = build.build_dataset(settings.DATA_DIR, config)
    table, report = split.assign(dataset, config)
    want = config["audit_expected"]
    assert report["boundaries"] == want["boundaries"]
    assert report["by_t"]["sizes"] == want["split_sizes"]
    assert report["by_t"]["purge_train"] == want["purge_train"]
    assert report["by_t"]["purge_validation"] == want["purge_validation"]
    assert report["by_t"]["events_straddling_a_boundary"] == want["events_straddling_a_boundary"]
    # audit.py counts events that have an id; a market with no event id is its own event here
    with_id = ~table["event_key"].str.startswith("market:")
    got = {p: int(table.loc[(table["split_by_t"] == p) & with_id, "event_key"].nunique()) for p in split.PARTS}
    assert got == want["split_events"]
    # the rules that matter for leakage
    assert (table.groupby("event_key")["split"].nunique() == 1).all()
    kept = table[~table["purged"]]
    b1, b2 = split.boundaries(table["t"], config)
    assert (kept.loc[kept["split"] == "train", "close_time"] <= b1).all()
    assert (kept.loc[kept["split"] == "validation", "close_time"] <= b2).all()
    assert len(kept) + int(table["purged"].sum()) == len(table)
