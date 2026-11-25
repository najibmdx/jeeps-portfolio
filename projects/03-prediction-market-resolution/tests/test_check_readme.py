"""Tests for src/check_readme.py, on made-up results (the real ones are not committed)."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import check_readme as cr

PROJECT = Path(__file__).resolve().parent.parent


# ---------- made-up results, in the shape of the real files ----------

def D(point, low, high):
    return {"point": point, "low": low, "high": high, "verdict": cr.rule_verdict({"low": low, "high": high})}


def both(point, low, high):
    return {"log_loss": D(point, low, high), "brier": D(point / 3, low / 3, high / 3)}


def scores(lls):
    return {k: {"log_loss": v, "brier": round(v / 3, 6)} for k, v in lls.items()}


def wobble(lls):
    return {k: {"log_loss": {"low": v - 0.01, "high": v + 0.01}, "brier": {"low": v / 3 - 0.003, "high": v / 3 + 0.003}}
            for k, v in lls.items()}


def a_slice(lls, diffs, markets, events, yes, **extra):
    return {"markets": markets, "events": events, "yes_rate": yes, "scores": scores(lls), "wobble": wobble(lls),
            "difference_vs_raw_price": diffs, **extra}


# log loss of each forecast, all markets and contested
ALL = {"raw_price": 0.1500, "price_only_correction": 0.1510, "base_rate": 0.4500, "frozen_model": 0.1495}
CON = {"raw_price": 0.5000, "price_only_correction": 0.5010, "base_rate": 0.7000, "frozen_model": 0.5020}
DIFF_ALL = {"price_only_correction": both(0.0010, 0.0005, 0.0015), "base_rate": both(0.3, 0.29, 0.31),
            "frozen_model": both(-0.0005, -0.0012, 0.0002)}
DIFF_CON = {"price_only_correction": both(0.0010, -0.0005, 0.0025), "base_rate": both(0.2, 0.18, 0.22),
            "frozen_model": both(0.0020, -0.0010, 0.0050)}
BANDS = ["0 to 0.05", "0.05 to 0.1", "0.1 to 0.2", "0.2 to 0.3", "0.3 to 0.5", "0.5 to 0.7", "0.7 to 0.95", "0.95 and above"]


def rename(d, old, new):
    return {(new if k == old else k): v for k, v in d.items()}


def calibration():
    rows = [{"band": b, "markets": 100 * (i + 1), "mean_forecast": 0.1 * (i + 1) / 2, "yes_rate": 0.1 * (i + 1) / 2 + 0.01}
            for i, b in enumerate(BANDS)]
    rows[0] = {"band": BANDS[0], "markets": 0, "mean_forecast": None, "yes_rate": None}
    return rows


def final_window(windows_scale):
    slices = {}
    for name, lls, diffs, markets, events in (("all", ALL, DIFF_ALL, 900, 130), ("contested", CON, DIFF_CON, 240, 100)):
        slices[name] = a_slice(lls, diffs, markets, events, 0.18,
                               calibration={"frozen_model": calibration(), "raw_price": calibration()},
                               frozen_minus_price_only_correction=D(0.0002 * windows_scale, -0.0004, 0.0009))
    return {"events": 130, "slices": slices}


def make_final():
    wc = lambda n: {"markets": n, "t_first": "2025-01-01T00:00:00+00:00", "t_last": "2025-02-01T00:00:00+00:00",
                    "days_t_to_close": {"p50": 0.8641, "p90": 7.0, "p99": 26.0}, "share_closing_within_1_day": 0.625,
                    "yes_rate": 0.1818, "share_contested": 0.2698}
    return {
        "opened_at": "2026-11-25T02:30:42.482135+00:00", "test_markets": 900, "test_events": 130,
        "headline": {"reading": "No measurable difference from the market price in this test window.",
                     "validation_and_test": "agree"},
        "test": final_window(1), "validation": final_window(2),
        "window_check": {"train": wc(2000), "validation": wc(700), "test": wc(900)},
    }


def make_baselines():
    keys = ("raw_price", "price_only_correction", "base_rate")
    pick = lambda d: {k: d[k] for k in keys}
    dpick = lambda d: {k: d[k] for k in ("price_only_correction", "base_rate")}
    return {"slices": {"all": a_slice(pick(ALL), dpick(DIFF_ALL), 700, 120, 0.17),
                       "contested": a_slice(pick(CON), dpick(DIFF_CON), 200, 90, 0.36)}}


LADDER = ["raw_price", "base_rate", "lr_price_only", "lr_all_features", "random_forest", "gradient_boosting"]
LAD_ALL = {"raw_price": 0.1500, "base_rate": 0.4500, "lr_price_only": 0.1510, "lr_all_features": 0.1495,
           "random_forest": 0.1600, "gradient_boosting": 0.1570}
LAD_CON = {"raw_price": 0.5000, "base_rate": 0.7000, "lr_price_only": 0.5010, "lr_all_features": 0.5020,
           "random_forest": 0.5200, "gradient_boosting": 0.5100}
LAD_DIFF_ALL = {"base_rate": both(0.3, 0.29, 0.31), "lr_price_only": both(0.0010, 0.0005, 0.0015),
                "lr_all_features": both(-0.0005, -0.0012, 0.0002), "random_forest": both(0.0100, 0.0040, 0.0160),
                "gradient_boosting": both(0.0070, 0.0020, 0.0120)}
LAD_DIFF_CON = {"base_rate": both(0.2, 0.18, 0.22), "lr_price_only": both(0.0010, -0.0005, 0.0025),
                "lr_all_features": both(0.0020, -0.0010, 0.0050), "random_forest": both(0.0200, 0.0100, 0.0300),
                "gradient_boosting": both(0.0100, 0.0030, 0.0170)}


def make_models():
    train = {k: v + 0.08 for k, v in LAD_ALL.items()}
    return {"order": LADDER,
            "train": {"all": {"scores": scores(train)}},
            "validation": {"all": a_slice(LAD_ALL, LAD_DIFF_ALL, 700, 120, 0.17),
                           "contested": a_slice(LAD_CON, LAD_DIFF_CON, 200, 90, 0.36)}}


def make_tuning():
    def setting(i, params, v_all, v_con, default=False):
        return {"index": i, "params": params, "is_library_default": default,
                "validation": {"all": v_all, "contested": v_con}}

    models = {
        "lr_price_only": {"settings": [setting(0, {}, 0.1510, 0.5010, True)], "best_index": 0, "best_params": {},
                          "library_default_validation_log_loss": {"all": 0.1510, "contested": 0.5010}},
        "lr_all_features": {"settings": [setting(0, {"C": 0.1}, 0.1500, 0.5030), setting(1, {"C": 1.0}, 0.1495, 0.5020, True)],
                            "best_index": 1, "best_params": {"C": 1.0},
                            "library_default_validation_log_loss": {"all": 0.1495, "contested": 0.5020}},
        "random_forest": {"settings": [setting(0, {"min_samples_leaf": 50}, 0.1550, 0.5100),
                                       setting(1, {"min_samples_leaf": 1}, 0.1600, 0.5200, True)],
                          "best_index": 0, "best_params": {"min_samples_leaf": 50},
                          "library_default_validation_log_loss": {"all": 0.1600, "contested": 0.5200}},
        "gradient_boosting": {"settings": [setting(0, {"learning_rate": 0.1, "min_samples_leaf": 400}, 0.1560, 0.5110),
                                           setting(1, {"learning_rate": 0.1, "min_samples_leaf": 20}, 0.1570, 0.5100, True)],
                              "best_index": 0, "best_params": {"learning_rate": 0.1, "min_samples_leaf": 400},
                              "library_default_validation_log_loss": {"all": 0.1570, "contested": 0.5100}},
    }
    choice = {"winner": "lr_all_features", "params": {"C": 1.0}, "validation": {"all": 0.1495, "contested": 0.5020},
              "runner_up": "lr_price_only",
              "margin_runner_up_minus_winner": {"all": D(0.0015, -0.0006, 0.0030), "contested": D(-0.001, -0.003, 0.001)},
              "winner_minus_raw_price": {"all": D(-0.0005, -0.0012, 0.0002), "contested": D(0.0020, -0.0010, 0.0050)}}
    return {"order": LADDER, "models": models, "choice": choice}


ABL = ["raw_price", "lr_price_only", "price_path_only", "price_path_plus_activity", "all_columns"]
ABL_ALL = {"raw_price": 0.1500, "lr_price_only": 0.1510, "price_path_only": 0.1502, "price_path_plus_activity": 0.1512,
           "all_columns": 0.1495}
ABL_CON = {"raw_price": 0.5000, "lr_price_only": 0.5010, "price_path_only": 0.5025, "price_path_plus_activity": 0.5040,
           "all_columns": 0.5020}


def pair(a, b, point, low, high):
    return {"a": a, "b": b, **D(point, low, high)}


def make_ablation():
    diff_all = {"lr_price_only": both(0.0010, 0.0005, 0.0015), "price_path_only": both(0.0002, -0.0003, 0.0007),
                "price_path_plus_activity": both(0.0012, -0.0001, 0.0026), "all_columns": both(-0.0005, -0.0012, 0.0002)}
    diff_con = {"lr_price_only": both(0.0010, -0.0005, 0.0025), "price_path_only": both(0.0025, -0.0001, 0.0050),
                "price_path_plus_activity": both(0.0040, 0.0010, 0.0070), "all_columns": both(0.0020, -0.0010, 0.0050)}
    pairs_all = [pair("price_path_only", "lr_price_only", -0.0008, -0.0020, 0.0004),
                 pair("price_path_plus_activity", "price_path_only", 0.0010, 0.0003, 0.0017),   # worse
                 pair("all_columns", "price_path_only", -0.0007, -0.0010, -0.0004)]             # better
    pairs_con = [pair("price_path_only", "lr_price_only", 0.0015, -0.0010, 0.0040),
                 pair("price_path_plus_activity", "price_path_only", 0.0015, -0.0002, 0.0032)]
    return {"order": ABL,
            "rows": {"price_path_only": list("abcdef"), "price_path_plus_activity": list("abcdefghij"),
                     "all_columns": list("abcdefghijkl")},
            "validation": {"all": a_slice(ABL_ALL, diff_all, 700, 120, 0.17),
                           "contested": a_slice(ABL_CON, diff_con, 200, 90, 0.36)},
            "pairs": {"all": pairs_all, "contested": pairs_con}}


def make_split():
    return {"final": {"sizes": {"train": 2000, "validation": 700, "test": 900},
                      "events": {"train": 800, "validation": 120, "test": 130}}}


def make_results():
    return {"split": make_split(), "baselines": make_baselines(), "models": make_models(),
            "tuning": make_tuning(), "ablation": make_ablation(), "final": make_final()}


README_TEXT = f"# Made-up README\n\nSome words.\n\n{cr.RESULTS_BEGIN}\n{cr.RESULTS_END}\n\nMore words after the block.\n"


def write_project(root, results=None, readme=README_TEXT, eol="\n"):
    """A project folder with outputs/ and README.md. results: dict of the make_* dicts, or None entries to leave out."""
    results = make_results() if results is None else results
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    for key, data in results.items():
        if data is not None:
            (root / "outputs" / cr.FILES[key].format(k=cr.K)).write_text(json.dumps(data, indent=2), encoding="utf-8")
    (root / "README.md").write_bytes(readme.replace("\n", eol).encode("utf-8"))
    return root


@pytest.fixture
def project(tmp_path):
    return write_project(tmp_path)


@pytest.fixture
def built(project):
    assert cr.main(["--update"], project=project) == 0
    return project


def readme_of(project):
    return (project / "README.md").read_bytes().decode("utf-8")


# ---------- number formats and the reading rule ----------

def test_number_formats_by_hand():
    assert cr.f4(0.15310823846758975) == "0.1531"
    assert cr.s4(0.0002689330953757896) == "+0.0003"
    assert cr.s4(-0.0012527952042305447) == "-0.0013"
    assert cr.s4(0.002095729477397575) == "+0.0021"
    assert cr.s4(2.3e-05) == "0.0000"        # rounds to zero: no sign
    assert cr.s4(-4e-05) == "0.0000"         # no negative zero
    assert cr.s4(0.00005) == "+0.0001"       # one step above rounding to zero
    assert cr.pct1(0.39892036657321) == "39.9%"
    assert cr.f3(0.147433) == "0.147"


def test_diff_cell_and_range():
    d = {"point": 0.0002689, "low": -0.0012528, "high": 0.0020957}
    assert cr.diff_cell(d) == "+0.0003 (-0.0013 to +0.0021)"
    assert cr.rng({"low": 0.14261836, "high": 0.16507178}) == "0.1426 to 0.1651"


def test_reading_rule_matches_the_project_rule():
    import baselines
    cases = [(0.1, 0.2), (-0.2, -0.1), (-0.1, 0.1), (0.0, 0.2), (-0.2, 0.0), (1e-9, 1.0)]
    for low, high in cases:
        assert cr.rule_verdict({"low": low, "high": high}) == baselines.verdict(low, high)
    # lower is better: a range wholly above zero is worse, wholly below is better
    assert cr.rule_verdict({"low": 0.1, "high": 0.2}) == "worse"
    assert cr.rule_verdict({"low": -0.2, "high": -0.1}) == "better"


# ---------- the tables, by hand ----------

def test_baseline_rows_by_hand():
    wanted = dict(cr.expected_strings(make_results())[0])
    assert wanted["baselines: all price_only_correction"] == \
        "| All markets | Price-only correction | 0.1510 | 0.1410 to 0.1610 | 0.0503 | +0.0010 (+0.0005 to +0.0015) | worse |"
    assert wanted["baselines: all raw_price"] == "| All markets | Raw market price | 0.1500 | 0.1400 to 0.1600 | 0.0500 | - | - |"
    assert wanted["baselines: contested price_only_correction"].endswith("| +0.0010 (-0.0005 to +0.0025) | cannot be told apart |")


def test_tuning_choice_and_ablation_rows_by_hand():
    wanted = dict(cr.expected_strings(make_results())[0])
    assert wanted["tuning: random_forest"] == "| Random forest | 2 | min_samples_leaf=50 | 0.1550 | 0.1600 | 0.5100 |"
    assert wanted["tuning: lr_price_only"] == "| Price-only correction | 1 | none (nothing to tune) | 0.1510 | 0.1510 | 0.5010 |"
    assert wanted["choice: margin"] == "| Runner-up minus winner, all markets | +0.0015 (-0.0006 to +0.0030), cannot be told apart |"
    assert wanted["ablation: price_path_only"].startswith(
        "| Price path only | 6 | 0.1502 | +0.0002 (-0.0003 to +0.0007) | cannot be told apart | 0.5025 |")
    assert wanted["ablation: all_columns"].startswith("| All columns (the frozen model) | 12 | 0.1495 |")
    assert wanted["ablation: raw_price"] == "| Raw market price | - | 0.1500 | - | - | 0.5000 | - | - |"


def test_final_rows_by_hand():
    wanted = dict(cr.expected_strings(make_results())[0])
    assert wanted["headline: contested raw market price"] == \
        "| Contested (the headline slice): frozen model minus raw market price | +0.0020 (-0.0010 to +0.0050), cannot be told apart | +0.0020 (-0.0010 to +0.0050), cannot be told apart |"
    assert wanted["headline: all price-only correction"] == \
        "| All markets: frozen model minus price-only correction | +0.0002 (-0.0004 to +0.0009), cannot be told apart | +0.0004 (-0.0004 to +0.0009), cannot be told apart |"
    assert wanted["test: all frozen_model"] == \
        "| All markets | Frozen model (logistic regression, all inputs, C=1.0) | 0.1495 | 0.1395 to 0.1595 | 0.0498 | -0.0005 (-0.0012 to +0.0002) | cannot be told apart |"
    assert wanted["calibration: 0 to 0.05"] == "| 0 to 0.05 | 0 | - | - |"
    assert wanted["calibration: 0.1 to 0.2"] == "| 0.1 to 0.2 | 300 | 0.150 | 0.160 |"
    assert wanted["opened"] == "The test split was opened once, at 2026-11-25 02:30 UTC, on 900 markets in 130 events."
    assert wanted["reading"] == ("Reading, fixed before the run: **No measurable difference from the market price in this test "
                                 "window.** Validation and test agree.")
    assert wanted["window: test"] == "| Test | 900 | 2025-01-01 to 2025-02-01 | 0.86 | 62.5% | 18.2% | 27.0% |"
    assert wanted["split: train"] == "| Train | 2,000 | 800 |"


def test_pair_counts_by_hand():
    pairs, flagged = cr.pair_counts(make_ablation())
    assert len(pairs) == 5 and len(flagged) == 2
    assert cr.pairs_sentence(make_ablation()) == \
        "Pairwise comparisons in the ablation: 5 (both slices); flagged as different: 2 (A better: 1, A worse: 1)."
    wanted = dict(cr.expected_strings(make_results())[0])
    assert wanted["flagged pair: all price_path_plus_activity vs price_path_only"] == \
        "| All markets | Price path + activity | Price path only | +0.0010 (+0.0003 to +0.0017) | worse |"
    assert wanted["flagged pair: all all_columns vs price_path_only"].endswith("| -0.0007 (-0.0010 to -0.0004) | better |")


# ---------- update, then check ----------

def test_update_then_check_passes(built):
    assert cr.main([], project=built) == 0
    text = readme_of(built)
    assert text.startswith("# Made-up README") and text.rstrip().endswith("More words after the block.")
    assert "**Headline: frozen model against the market price**" in text


def test_update_twice_gives_identical_bytes(built):
    first = (built / "README.md").read_bytes()
    assert cr.main(["--update"], project=built) == 0
    assert (built / "README.md").read_bytes() == first


def test_update_replaces_old_block_content(built):
    path = built / "README.md"
    path.write_bytes(path.read_bytes().replace(b"0.1510", b"9.9999"))
    assert cr.main([], project=built) == 1
    assert cr.main(["--update"], project=built) == 0
    assert cr.main([], project=built) == 0


def test_windows_line_endings_are_kept_and_checked(tmp_path):
    root = write_project(tmp_path, eol="\r\n")
    assert cr.main(["--update"], project=root) == 0
    raw = (root / "README.md").read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    assert cr.main([], project=root) == 0


def test_print_tables_matches_the_block(built):
    block = cr.block_of(readme_of(built)).strip()
    assert block == cr.results_markdown(make_results()).strip()


# ---------- mutations: the check must fail ----------

def mutate_readme(project, old, new):
    path = project / "README.md"
    text = path.read_bytes().decode("utf-8")
    assert old in text
    path.write_bytes(text.replace(old, new, 1).encode("utf-8"))


def test_a_changed_digit_is_caught_and_the_row_is_named(built, capsys):
    mutate_readme(built, "| 0.1495 | 0.1395 to 0.1595 |", "| 0.1496 | 0.1395 to 0.1595 |")
    assert cr.main([], project=built) == 1
    assert "test: all frozen_model" in capsys.readouterr().out


def test_a_changed_reading_word_is_caught(built):
    mutate_readme(built, "+0.0010 (+0.0005 to +0.0015) | worse |", "+0.0010 (+0.0005 to +0.0015) | cannot be told apart |")
    assert cr.main([], project=built) == 1


def test_a_row_under_the_wrong_table_is_caught(built):
    """The same text in another table must not satisfy the check: each row has to stand under its own title."""
    lines = readme_of(built).split("\n")
    row = "| All markets | Price-only correction | 0.1510 | 0.1410 to 0.1610 | 0.0503 | +0.0010 (+0.0005 to +0.0015) | worse |"
    assert lines.count(row) == 2          # the made-up baselines and the made-up test table happen to share this row
    first = lines.index(row)
    del lines[first]                      # remove it from the baselines table only
    (built / "README.md").write_bytes("\n".join(lines).encode("utf-8"))
    assert cr.main([], project=built) == 1


def test_a_row_with_something_added_is_caught(built):
    """A row must match as a whole line: an extra cell or trailing text after the expected row is a difference."""
    mutate_readme(built, "| 0.1 to 0.2 | 300 | 0.150 | 0.160 |", "| 0.1 to 0.2 | 300 | 0.150 | 0.160 | extra |")
    assert cr.main([], project=built) == 1


def test_a_changed_range_end_is_caught(built):
    mutate_readme(built, "-0.0012 to +0.0002", "-0.0012 to +0.0003")
    assert cr.main([], project=built) == 1


def test_a_missing_row_is_caught(built):
    lines = readme_of(built).split("\n")
    kept = [ln for ln in lines if not ln.startswith("| Random forest | 2 |")]
    assert len(kept) == len(lines) - 1
    (built / "README.md").write_bytes("\n".join(kept).encode("utf-8"))
    assert cr.main([], project=built) == 1


def test_numbers_swapped_between_two_rows_are_caught(built):
    text = readme_of(built)
    a = "| Raw market price | 0.1500 |"
    b = "| Price-only correction | 0.1510 |"
    assert a in text and b in text
    swapped = text.replace(a, "@@").replace(b, a.replace("Raw market price", "Price-only correction")).replace("@@", b.replace("Price-only correction", "Raw market price"))
    assert swapped != text
    (built / "README.md").write_bytes(swapped.encode("utf-8"))
    assert cr.main([], project=built) == 1


def test_a_changed_sentence_is_caught(built):
    mutate_readme(built, "flagged as different: 2", "flagged as different: 1")
    assert cr.main([], project=built) == 1
    mutate_readme(built, "flagged as different: 1", "flagged as different: 2")
    mutate_readme(built, "Validation and test agree.", "Validation and test disagree.")
    assert cr.main([], project=built) == 1


def test_a_changed_results_file_is_caught(built):
    path = built / "outputs" / "final_test_k7.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["test"]["slices"]["all"]["scores"]["raw_price"]["log_loss"] = 0.1600
    path.write_text(json.dumps(data), encoding="utf-8")
    assert cr.main([], project=built) == 1


def test_missing_markers_are_caught(built):
    text = readme_of(built).replace(cr.RESULTS_END, "")
    (built / "README.md").write_bytes(text.encode("utf-8"))
    assert cr.main([], project=built) == 1
    with pytest.raises(ValueError):
        cr.update_readme_text("no markers here", make_results())


# ---------- a saved verdict must agree with its own range ----------

def flip_a_verdict(project, file_key, edit):
    path = project / "outputs" / cr.FILES[file_key].format(k=cr.K)
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_verdict_that_disagrees_with_its_range_fails_the_check(built, capsys):
    flip_a_verdict(built, "final", lambda d: d["test"]["slices"]["contested"]["difference_vs_raw_price"]["frozen_model"]["log_loss"].update(verdict="better"))
    assert cr.main([], project=built) == 1
    assert "saved verdict 'better' but its range reads 'cannot be told apart'" in capsys.readouterr().out


def test_update_refuses_a_verdict_that_disagrees_with_its_range(project):
    flip_a_verdict(project, "baselines", lambda d: d["slices"]["all"]["difference_vs_raw_price"]["base_rate"]["log_loss"].update(verdict="cannot be told apart"))
    before = (project / "README.md").read_bytes()
    assert cr.main(["--update"], project=project) == 1
    assert (project / "README.md").read_bytes() == before


def test_all_made_up_results_are_consistent():
    assert cr.verdict_problems(make_results()) == []


# ---------- missing files ----------

def test_no_result_files_gives_exit_2(tmp_path):
    root = write_project(tmp_path, results={k: None for k in cr.FILES})
    assert cr.main([], project=root) == 2
    assert cr.main(["--update"], project=root) == 2


def test_update_refuses_when_any_file_is_missing_and_leaves_the_readme_alone(tmp_path):
    results = make_results()
    results["tuning"] = None
    root = write_project(tmp_path, results=results)
    before = (root / "README.md").read_bytes()
    assert cr.main(["--update"], project=root) == 2
    assert (root / "README.md").read_bytes() == before


def test_check_runs_on_the_files_that_exist_and_skips_the_rest_with_a_note(built, capsys):
    (built / "outputs" / "tuning_k7.json").unlink()
    assert cr.main([], project=built) == 0
    out = capsys.readouterr().out
    assert "note: tuning: tuning table not checked" in out and "4 of 6" not in out and "5 of 6 files present" in out


def test_only_the_final_test_file_is_enough_for_a_partial_check(built):
    for key in ("split", "baselines", "models", "tuning", "ablation"):
        (built / "outputs" / cr.FILES[key].format(k=cr.K)).unlink()
    assert cr.main([], project=built) == 0
    mutate_readme(built, "| 0 to 0.05 | 0 | - | - |", "| 0 to 0.05 | 1 | - | - |")
    assert cr.main([], project=built) == 1


# ---------- the real README and the real committed files ----------

def test_the_real_readme_has_the_results_markers_and_no_hand_edits_outside_them():
    text = (PROJECT / "README.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert text.count(cr.RESULTS_BEGIN) == 1 and text.count(cr.RESULTS_END) == 1
    assert cr.block_of(text) is not None
    assert text.index(cr.RESULTS_BEGIN) < text.index(cr.RESULTS_END)


def test_the_real_readme_matches_whatever_real_results_are_present(capsys):
    """On the author's machine all six files are there; in CI only final_test_k7.json is committed."""
    present = [k for k, p in cr.FILES.items() if (PROJECT / "outputs" / p.format(k=cr.K)).exists()]
    if not present:
        pytest.skip("no saved results in outputs/")
    code = cr.main([])
    assert code == 0, capsys.readouterr().out


def test_the_committed_final_test_report_is_the_one_time_run():
    path = PROJECT / "outputs" / f"final_test_k{cr.K}.json"
    if not path.exists():
        pytest.skip("final_test_k7.json is not in outputs/")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["model"] == "lr_all_features" and report["params"] == {"C": 1.0}
    assert report["test_markets"] == 8991
    assert report["headline"]["reading"] in report["rules"]["readings"].values()
    assert "opened once" in " ".join(report["notes"])
    assert set(report["test"]["slices"]["all"]["scores"]) == {"raw_price", "price_only_correction", "base_rate", "frozen_model"}


# ---------- .gitignore: only the one-time test report is committed from outputs/ ----------

def git_ignored(relative):
    try:
        r = subprocess.run(["git", "check-ignore", "-q", relative], cwd=PROJECT, capture_output=True)
    except FileNotFoundError:
        pytest.skip("git is not installed")
    if r.returncode == 128:
        pytest.skip("not inside a git checkout")
    return r.returncode == 0


def test_only_the_final_test_report_is_kept_out_of_the_ignore_list():
    assert not git_ignored("outputs/final_test_k7.json")
    for name in ("outputs/tuning_k7.json", "outputs/baselines_k7.json", "outputs/models_k7.json",
                 "outputs/ablation_k7.json", "outputs/split_report_k7.json", "outputs/final_test_k3.json",
                 "outputs/anything_else.json", "data/test_opened.lock", "data/derived/test.parquet"):
        assert git_ignored(name), name
