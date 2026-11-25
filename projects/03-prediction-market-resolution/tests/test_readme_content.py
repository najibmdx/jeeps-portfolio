from pathlib import Path

import pytest
import yaml

import check_readme

PROJECT = Path(__file__).resolve().parent.parent
README = (PROJECT / "README.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
CONFIG = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))


def section(title):
    """The text of one '## title' section, up to the next '## '."""
    start = README.index(f"\n## {title}\n") + 1
    end = README.find("\n## ", start + 1)
    return README[start:end if end != -1 else len(README)]


def test_no_claim_sentence():
    assert README.startswith("# Project 03: Prediction Market Resolution\n\nLearning project. No claim about trading use, returns or profit.")
    assert README.count("No claim about trading use, returns or profit.") >= 3   # top, Results, Limits


def test_data_credit():
    assert "Akey" in README and "CC-BY-4.0" in README


def test_no_promises():
    low = README.lower()
    for phrase in ("profitable", "guaranteed", "make money", "beat the market", "alpha"):
        assert phrase not in low


def test_no_trading_or_money_claims_outside_the_no_claim_sentences():
    low = README.replace("No claim about trading use, returns or profit.", "").lower()
    for phrase in ("profit", "edge over", "outperform", "trading strategy", "you can trade", "winning"):
        assert phrase not in low, phrase


def test_readme_status_is_stage_10():
    status = next(line for line in README.split("\n") if line.startswith("**Status:"))
    assert status.startswith("**Status: Stage 10.")
    assert "one-time test has been run" in status and "Nothing was retuned or refitted afterwards" in status
    assert "Stage 9" not in status and "no final results to report" not in README


def test_every_script_is_documented():
    for command in ("python src/pipeline.py --sample", "python src/pipeline.py", "python src/download.py", "python src/audit.py",
                    "python src/leakage.py", "python src/baselines.py", "python src/models.py", "python src/tune.py",
                    "python src/freeze.py", "python src/ablate.py", "python src/final_test.py --confirm-open-test",
                    "python src/check_readme.py", "python src/check_readme.py --update"):
        assert command in README, command
    assert "a second run is refused" in README and "cannot change the frozen choice" in README
    assert "frozen_choice.yaml" in README and "untuned model ladder" in README


def test_the_run_of_the_test_is_stated_once_and_with_its_date():
    assert "It was run once, on 25 Nov 2026" in README
    assert "`frozen_choice.yaml` still says the test split was not opened" in README   # the file is as it was committed, not edited after


def test_the_result_in_short_matches_the_reading_fixed_in_the_config():
    short = section("Result in short")
    assert "cannot be told apart from the market's own price" in short
    readings = [v for k, v in CONFIG.items() if isinstance(v, dict) and "readings" in v][0]["readings"]
    assert readings["cannot be told apart"] == "No measurable difference from the market price in this test window."
    assert readings["cannot be told apart"] in README
    low, high = CONFIG["universe"]["contested_low"], CONFIG["universe"]["contested_high"]
    assert f"between {low} and {high}" in short


def test_decision_time_in_the_text_matches_the_config():
    assert CONFIG["decision_time"]["days_after_open"] == 7
    assert "first seven days" in README and "seven days after a market opens" in README


def test_the_sections_are_there_in_order():
    heads = [line for line in README.split("\n") if line.startswith("## ")]
    wanted = ["## The question", "## Result in short", "## Data", "## How to run", "## Results",
              "## Numbers that could mislead", "## Limits"]
    assert heads == wanted


def test_numbers_that_could_mislead_covers_the_agreed_points():
    text = section("Numbers that could mislead")
    for needle in ("Many comparisons", "chance", "One window", "sampling noise only", "The windows differ",
                   "quick-closing", "not \"equal\"", "Rounding", "flattering", "Calibration has no ranges", "Fee markets"):
        assert needle in text, needle
    assert text.count("\n- ") + (1 if text.lstrip("\n").startswith("- ") else 0) >= 10


def test_limits_cover_the_agreed_points():
    text = section("Limits")
    for needle in ("No text, no order book", "Four models", "Fees, spreads, costs and execution are not modelled",
                   "opened once", "not refitted on train plus validation", "third-party dataset"):
        assert needle in text, needle


def test_the_hand_written_sections_hold_no_result_tables_or_typed_results():
    """Numbers come from the saved results only: outside the results block there is no table at all."""
    outside = README.replace(check_readme.block_of(README), "")
    assert "\n|" not in outside


def test_the_results_block_is_generated_not_typed():
    block = check_readme.block_of(README)
    assert block is not None and block.strip().startswith("**Markets and events per split**")
    assert check_readme.RESULTS_BEGIN in README and "do not edit by hand" in check_readme.RESULTS_BEGIN
    for title in ("Baselines, validation", "One-time test: all four forecasts", "Headline: frozen model against the market price",
                  "Calibration of the frozen model on the test split, all markets",
                  "Column groups left out of the frozen model (validation log loss)", "The three windows side by side"):
        assert f"**{title}**" in block, title


def test_the_agreed_reading_is_in_the_readme_in_the_generated_form():
    assert ("Reading, fixed before the run: **No measurable difference from the market price in this test window.** "
            "Validation and test agree.") in README
    assert "The test split was opened once, at 2025" not in README        # a date from a made-up test must never leak in
    assert "The test split was opened once, at 2026-11-25 02:30 UTC, on 8,991 markets in 1,336 events." in README


def test_the_test_report_is_the_only_committed_result_file_and_the_readme_says_so():
    assert "the only file in `outputs/` that is committed" in README
    assert "`outputs/` is not committed, apart from `final_test_k7.json`" in README
