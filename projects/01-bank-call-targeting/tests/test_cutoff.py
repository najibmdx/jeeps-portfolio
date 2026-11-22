"""Tests for the cost-scenario cut-off, on small made-up data."""

import numpy as np
import pytest

import cutoff
import evaluate
from settings import load_config
from test_ablation import make_frame


def test_the_rounding_of_the_top_share_is_not_fooled_by_floating_point():
    assert evaluate.top_k_from_fraction(100, 0.07) == 7   # 100 * 0.07 is 7.000000000000001 in Python


def test_net_value_by_hand_on_a_tiny_example():
    y = np.array([1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.1, 0.7])               # ranking: yes, no, yes, no
    out = cutoff.best_share(y, scores, cost_per_call=0.25, step=0.25)
    # net after 0,1,2,3,4 calls = 0, 0.75, 0.5, 1.25, 1.0  ->  best is 3 calls (share 0.75)
    assert out["calls"] == 3 and out["share"] == 0.75
    assert out["net_per_100_clients"] == 31.25
    assert out["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["recall"] == 1.0
    assert out["call_everyone_net_per_100_clients"] == 25.0


def test_a_perfect_ranking_phones_exactly_the_yes_clients():
    y = np.array([1] * 20 + [0] * 80)
    out = cutoff.best_share(y, scores=y.astype(float), cost_per_call=0.5, step=0.01)
    assert out["calls"] == 20 and out["share"] == 0.2


def test_a_call_that_costs_a_whole_subscription_is_never_worth_it():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    out = cutoff.best_share(y, y + rng.random(300), cost_per_call=1.0, step=0.01)
    assert out["calls"] == 0 and out["share"] == 0.0


def test_dearer_calls_never_mean_more_calls():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    scores = y * 0.5 + rng.random(2000)
    shares = [cutoff.best_share(y, scores, c, 0.01)["share"] for c in (0.03, 0.09, 0.15, 0.3)]
    assert shares == sorted(shares, reverse=True)


class NoTest(dict):
    """A sets dictionary that fails loudly if anyone asks for the test set."""

    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("the cut-off must never read the test set")
        return super().__getitem__(key)


def small_config():
    return {
        "seed": 1,
        "features": {"groups": ["client", "contact", "history"]},
        "models": {"logistic_regression": {"max_iter": 500}},
        "evaluation": {"top_fraction": 0.10, "cutoff_share_step": 0.05},
        "scenarios": {"cheap": {"cost_per_call": 0.03}, "dear": {"cost_per_call": 0.2}},
    }


def test_the_cut_off_is_chosen_without_the_test_set():
    sets = NoTest(train=make_frame(1500, 0), validation=make_frame(600, 1), test=make_frame(600, 2))
    out = cutoff.run(small_config(), sets)               # raises if "test" is ever read
    assert set(out) == {"cheap", "dear"}
    assert set(out["cheap"]["models"]) == {"logistic_regression"}


def test_the_real_config_has_three_labelled_cost_scenarios():
    config = load_config()
    costs = [s["cost_per_call"] for s in config["scenarios"].values()]
    assert len(costs) >= 3
    assert all(0 < c < 1 for c in costs)                 # a call costs less than one subscription
    assert costs == sorted(costs)                        # listed from cheap to dear
