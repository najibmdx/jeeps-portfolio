"""Tests for the one-time test run, on small made-up data."""

import numpy as np
import pytest

import final
import models
from settings import load_config
from test_ablation import make_frame


def small_config():
    return {
        "seed": 1,
        "features": {"groups": ["client", "contact", "history"]},
        "models": {"logistic_regression": {"max_iter": 500}},
        "evaluation": {"top_fraction": 0.10, "bootstrap_draws": 20, "cutoff_share_step": 0.05},
        "scenarios": {"cheap": {"cost_per_call": 0.03}, "dear": {"cost_per_call": 0.2}},
        "final": {"model": "logistic_regression", "params": {"C": 0.01}},
    }


def sets_with(test_seed):
    return {"train": make_frame(1500, 0), "validation": make_frame(600, 1), "test": make_frame(600, test_seed)}


def test_the_final_settings_are_the_base_settings_plus_the_chosen_ones():
    name, params = final.final_model(small_config())
    assert name == "logistic_regression"
    assert params == {"max_iter": 500, "C": 0.01}


def test_the_cut_off_shares_are_chosen_on_validation_not_on_test():
    a = final.run(small_config(), sets_with(test_seed=2))
    b = final.run(small_config(), sets_with(test_seed=3))     # a different test set
    for scenario in a["scenarios"]:
        assert a["scenarios"][scenario]["validation"] == b["scenarios"][scenario]["validation"]
        assert a["scenarios"][scenario]["test"]["share"] == b["scenarios"][scenario]["validation"]["share"]
    assert a["test_pr_auc"] != b["test_pr_auc"]               # the test set only grades the choice


def test_the_test_result_by_hand_on_a_tiny_example():
    y = np.array([1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.1, 0.7])                   # ranking: yes, no, yes, no
    out = final.at_share(y, scores, share=0.75, cost_per_call=0.25)
    assert out["calls"] == 3
    assert out["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["recall"] == 1.0
    assert out["net_per_100_clients"] == 31.25                # (2 yes - 0.25 x 3 calls) / 4 clients x 100


def test_a_share_of_zero_phones_nobody():
    out = final.at_share(np.array([1, 0, 0, 1]), np.array([0.9, 0.8, 0.1, 0.7]), share=0.0, cost_per_call=0.25)
    assert out["calls"] == 0 and out["precision"] is None and out["net_per_100_clients"] == 0.0


def test_the_test_set_cannot_be_used_a_second_time(tmp_path):
    path = tmp_path / final.RESULT_FILE
    final.check_not_run_before(path)                          # first time: fine
    path.write_text("{}")
    with pytest.raises(RuntimeError):
        final.check_not_run_before(path)                      # second time: refused


def test_the_result_reports_the_yes_rate_of_each_period_next_to_the_score():
    r = final.run(small_config(), sets_with(test_seed=2))
    assert set(r["yes_rate"]) == {"train", "validation", "test"}
    assert r["test_times_blind_guess"] == pytest.approx(r["test_pr_auc"] / r["yes_rate"]["test"], abs=0.01)
    assert r["always_no"]["recall"] == 0.0


def test_the_real_config_names_a_model_and_settings_that_the_model_accepts():
    config = load_config()
    name, params = final.final_model(config)
    assert name in config["models"]
    models.build_model(name, ["age", "job"], params, config["seed"])   # raises on a wrong setting name
