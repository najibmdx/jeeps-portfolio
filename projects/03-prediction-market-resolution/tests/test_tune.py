"""Stage 7 tests: the grids, the choice rule (known answers on hand-made tables), that nothing but train is fitted on and
nothing but validation is scored, ties, the margin, the report, the command line and the pipeline."""

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import baselines
import build
import models
import pipeline
import sample
import settings
import split
import tune

CONFIG = settings.load_config()
EPS = CONFIG["metrics"]["price_clip"]
LADDER = CONFIG["models"]["ladder"]
TUNABLE = ["lr_all_features", "random_forest", "gradient_boosting"]


def frame(n, seed=0, **over):
    """A table with every column the models read. The outcome follows the price, so the price is informative."""
    rng = np.random.default_rng(seed)
    price = rng.uniform(0.05, 0.95, n)
    data = dict(
        event_key=[f"E{seed}-{i // 2}" for i in range(n)], label=(rng.random(n) < price).astype("int8"),
        last_close=price, first_open=price, high=np.minimum(price + 0.1, 1.0), low=np.maximum(price - 0.1, 0.0), mean_close=price,
        change_first_to_last=0.0, n_bars=3, days_since_last_bar=1,
        trades_before_t=rng.integers(20, 500, n), trades_last_bar=rng.integers(1, 100, n),
        category=rng.choice(["A", "B", "C"], n), sibling_count_at_t=rng.integers(1, 20, n), has_fee=False,
    )
    data.update(over)
    return pd.DataFrame(data)


def bce(p, yes_share):
    """The log loss of forecasting p for every market when yes_share of them resolve Yes. Written out here, not taken from the code under test."""
    return -(yes_share * math.log(p) + (1 - yes_share) * math.log(1 - p))


# ---- the hand-made tables. 8 contested markets (price 0.5) and 2 near-decided (price 0.01) in validation; 16 and 4 in train.
# Validation: 3 of the 8 contested resolve Yes, 1 of the 2 near-decided: 4 of 10 (40%). Train: 8 of 16 and 1 of 4: 9 of 20 (45%).

def table(prices, labels):
    n = len(prices)
    return frame(n, last_close=prices, first_open=prices, mean_close=prices, high=prices, low=prices, label=np.array(labels, dtype="int8"),
                 event_key=[f"H{len(prices)}-{i // 2}" for i in range(n)])


VAL = table([0.5] * 8 + [0.01] * 2, [1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
TRAIN = table([0.5] * 16 + [0.01] * 4, [1] * 8 + [0] * 8 + [1, 0, 0, 0])
VAL_ALL, VAL_CON, TRAIN_ALL, TRAIN_CON = 0.4, 0.375, 0.45, 0.5

# What every setting forecasts, in grid order (see tune.grid_settings). Each model forecasts one number for every market.
P = {
    "lr_all_features": [0.30, 0.34, 0.38, 0.46, 0.55, 0.65],      # C 0.001 ... 100. best is the third; the default (C=1) is the fourth
    "random_forest": [0.40, 0.40, 0.40, 0.45, 0.30, 0.20],        # leaf 200 ... 1. the first three tie exactly
    "gradient_boosting": [0.50, 0.50, 0.45, 0.41, 0.41, 0.30],    # (0.03,400) (0.03,100) (0.03,20) (0.1,400) (0.1,100) (0.1,20). the fourth and fifth tie
}


class Constant:
    """A stand-in model that forecasts the same probability for every market and remembers what it was fitted on."""

    def __init__(self, p, log):
        self.p, self.log = p, log

    def fit(self, X, y):
        self.log.append((len(X), [int(v) for v in y]))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


class Log(list):
    """The fits of the stand-in models. `install` swaps in a different table of forecasts."""


@pytest.fixture
def fakes(monkeypatch):
    """Swap the real models for Constant ones and the price correction for 'the price is right'. Returns the log of fits."""
    log = Log()

    def install(table_p=P, config=CONFIG):
        def build_model(name, cfg, columns, params=None):
            return Constant(table_p[name][tune.grid_settings(config, name).index(dict(params))], log)
        monkeypatch.setattr(models, "build_model", build_model)

    monkeypatch.setattr(baselines, "fit_correction", lambda price, y, eps: {"intercept": 0.0, "slope": 1.0})
    install()
    log.install = install
    return log


# ---- the grids

def test_the_grids_are_the_ones_agreed_with_jeeps():
    t = CONFIG["models"]["tuning"]
    assert set(t) == set(TUNABLE)
    assert sorted(t["lr_all_features"]["C"]) == [0.001, 0.01, 0.1, 1, 10, 100]
    assert sorted(t["random_forest"]["min_samples_leaf"]) == [1, 5, 20, 50, 100, 200]
    assert sorted(t["gradient_boosting"]["learning_rate"]) == [0.03, 0.1] and sorted(t["gradient_boosting"]["min_samples_leaf"]) == [20, 100, 400]
    assert CONFIG["models"]["forest_trees"] == 300 and CONFIG["models"]["selection"] == "lowest_validation_log_loss"
    assert CONFIG["models"]["frozen_file"] == "frozen_choice.yaml"
    for name in TUNABLE:
        assert len(tune.grid_settings(CONFIG, name)) == 6      # the same budget for every model


def test_grid_settings_are_listed_simplest_first_with_the_right_types():
    assert tune.grid_settings(CONFIG, "lr_price_only") == [{}]
    assert tune.grid_settings(CONFIG, "lr_all_features") == [{"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)]
    assert tune.grid_settings(CONFIG, "random_forest") == [{"min_samples_leaf": v} for v in (200, 100, 50, 20, 5, 1)]
    assert tune.grid_settings(CONFIG, "gradient_boosting") == [
        {"learning_rate": 0.03, "min_samples_leaf": 400}, {"learning_rate": 0.03, "min_samples_leaf": 100}, {"learning_rate": 0.03, "min_samples_leaf": 20},
        {"learning_rate": 0.1, "min_samples_leaf": 400}, {"learning_rate": 0.1, "min_samples_leaf": 100}, {"learning_rate": 0.1, "min_samples_leaf": 20}]
    for s in tune.grid_settings(CONFIG, "lr_all_features"):
        assert type(s["C"]) is float
    for s in tune.grid_settings(CONFIG, "random_forest"):
        assert type(s["min_samples_leaf"]) is int


def test_grid_settings_are_cast_to_their_types_whatever_the_yaml_holds():
    c = copy.deepcopy(CONFIG)
    c["models"]["tuning"]["lr_all_features"]["C"] = [1, 10]
    c["models"]["tuning"]["gradient_boosting"] = {"learning_rate": [1], "min_samples_leaf": [20]}
    assert [type(x["C"]) for x in tune.grid_settings(c, "lr_all_features")] == [float, float]
    assert [type(x["learning_rate"]) for x in tune.grid_settings(c, "gradient_boosting")] == [float]
    c["models"]["tuning"]["random_forest"]["min_samples_leaf"] = [5.0, 1.0]
    assert [type(x["min_samples_leaf"]) for x in tune.grid_settings(c, "random_forest")] == [int, int]


def test_every_grid_holds_the_library_default_and_the_defaults_are_the_real_library_defaults():
    assert LogisticRegression().get_params()["C"] == settings.LIBRARY_DEFAULTS["lr_all_features"]["C"]
    assert RandomForestClassifier().get_params()["min_samples_leaf"] == settings.LIBRARY_DEFAULTS["random_forest"]["min_samples_leaf"]
    gb = HistGradientBoostingClassifier().get_params()
    assert {"learning_rate": gb["learning_rate"], "min_samples_leaf": gb["min_samples_leaf"]} == settings.LIBRARY_DEFAULTS["gradient_boosting"]
    for name in TUNABLE:
        defaults = [s for s in tune.grid_settings(CONFIG, name) if tune.is_library_default(name, s)]
        assert defaults == [settings.LIBRARY_DEFAULTS[name]]
    assert tune.is_library_default("lr_price_only", {}) and not tune.is_library_default("random_forest", {"min_samples_leaf": 5})


def test_describe_names_the_setting():
    assert tune.describe({}) == "no setting to tune"
    assert tune.describe({"C": 100.0}) == "C=100" and tune.describe({"C": 0.001}) == "C=0.001"
    assert tune.describe({"learning_rate": 0.03, "min_samples_leaf": 400}) == "learning_rate=0.03, min_samples_leaf=400"


def test_a_tuned_setting_reaches_the_model_and_the_untuned_model_is_unchanged():
    columns, _ = models.input_columns(CONFIG)
    lr = models.build_model("lr_all_features", CONFIG, columns, {"C": 0.1}).named_steps["model"]
    assert lr.C == 0.1 and lr.max_iter == 1000 and lr.random_state == 42
    rf = models.build_model("random_forest", CONFIG, columns, {"min_samples_leaf": 50}).named_steps["model"]
    assert rf.min_samples_leaf == 50 and rf.n_estimators == 300 and rf.random_state == 42
    gb = models.build_model("gradient_boosting", CONFIG, columns, {"learning_rate": 0.03, "min_samples_leaf": 400}).named_steps["model"]
    assert gb.learning_rate == 0.03 and gb.min_samples_leaf == 400 and gb.random_state == 42
    for name in TUNABLE:
        plain = models.build_model(name, CONFIG, columns).named_steps["model"].get_params()
        default = models.build_model(name, CONFIG, columns, settings.LIBRARY_DEFAULTS[name]).named_steps["model"].get_params()
        assert plain == default


def test_a_setting_that_cannot_be_tuned_is_refused():
    columns, _ = models.input_columns(CONFIG)
    with pytest.raises(ValueError, match=r"random_forest cannot tune \['C'\]"):
        models.build_model("random_forest", CONFIG, columns, {"C": 1.0})
    with pytest.raises(ValueError, match="cannot tune"):
        models.build_model("lr_price_only", CONFIG, columns, {"C": 1.0})
    with pytest.raises(ValueError, match="no model is built"):
        models.build_model("lr_price_only", CONFIG, columns, {})


# ---- the settings are checked

def _bad(path, value, base=CONFIG):
    c = copy.deepcopy(base)
    node = c
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return c


def _without(name):
    c = copy.deepcopy(CONFIG)
    del c["models"]["tuning"][name]
    return c


@pytest.mark.parametrize(
    "bad",
    [
        _without("random_forest"),
        _without("lr_all_features"),
        _bad(("models", "tuning", "lr_price_only"), {"C": [1.0, 2.0]}),
        _bad(("models", "tuning"), ["C"]),
        _bad(("models", "tuning", "lr_all_features"), {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]}),
        _bad(("models", "tuning", "lr_all_features"), {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], "extra": [1]}),
        _bad(("models", "tuning", "lr_all_features", "C"), []),
        _bad(("models", "tuning", "lr_all_features", "C"), "0.1"),
        _bad(("models", "tuning", "lr_all_features", "C"), [0.001, 0.01, 0.1, 1.0, 10.0, "100"]),
        _bad(("models", "tuning", "lr_all_features", "C"), [0.001, 0.01, 0.1, 1.0, 10.0, True]),
        _bad(("models", "tuning", "lr_all_features", "C"), [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]),
        _bad(("models", "tuning", "lr_all_features", "C"), [-1.0, 0.01, 0.1, 1.0, 10.0, 100.0]),
        _bad(("models", "tuning", "lr_all_features", "C"), [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]),               # listed the wrong way round
        _bad(("models", "tuning", "lr_all_features", "C"), [0.001, 0.01, 0.1, 1.0, 1.0, 10.0]),                 # a value twice
        _bad(("models", "tuning", "lr_all_features", "C"), [0.001, 0.01, 0.1, 0.5, 10.0, 100.0]),               # no library default
        _bad(("models", "tuning", "random_forest", "min_samples_leaf"), [1, 5, 20, 50, 100, 200]),              # ascending: the wrong way round for leaves
        _bad(("models", "tuning", "random_forest", "min_samples_leaf"), [200, 100, 50, 20, 5, 0]),
        _bad(("models", "tuning", "random_forest", "min_samples_leaf"), [200, 100, 50, 20, 5.5, 1]),
        _bad(("models", "tuning", "random_forest", "min_samples_leaf"), [200, 100, 50, 20, 5, 2]),              # no library default (1)
        _bad(("models", "tuning", "random_forest", "min_samples_leaf"), [200, 100, 50, 20, 1]),                  # 5 settings, the others have 6
        _bad(("models", "tuning", "gradient_boosting", "learning_rate"), [0.1, 0.03]),                           # descending
        _bad(("models", "tuning", "gradient_boosting", "learning_rate"), [0.03, 0.1, 1.5]),
        _bad(("models", "tuning", "gradient_boosting", "learning_rate"), [0.0, 0.1]),
        _bad(("models", "tuning", "gradient_boosting", "learning_rate"), [0.03, 0.05]),                          # no library default (0.1)
        _bad(("models", "tuning", "gradient_boosting", "min_samples_leaf"), [20, 100, 400]),
        _bad(("models", "tuning", "gradient_boosting", "min_samples_leaf"), [400, 100]),                         # no default 20, and 4 settings
    ],
)
def test_bad_tuning_settings_are_refused(bad):
    with pytest.raises(ValueError):
        settings.validate_config(bad)


@pytest.mark.parametrize(
    "bad,message",
    [
        (_without("random_forest"), "exactly these models"),
        (_bad(("models", "tuning", "lr_all_features"), {"alpha": [1.0, 2.0]}), "exactly these settings"),
        (_bad(("models", "tuning", "lr_all_features", "C"), []), "non-empty list"),
        (_bad(("models", "tuning", "lr_all_features", "C"), 1.0), "non-empty list"),
        (_bad(("models", "tuning", "lr_all_features", "C"), [0.001, 0.01, 0.1, 1.0, 10.0, "100"]), "numbers only"),
        (_bad(("models", "tuning", "lr_all_features", "C"), [0.001, 0.01, 0.1, 1.0, 10.0, True]), "numbers only"),
        (_bad(("models", "tuning", "lr_all_features", "C"), [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]), "numbers above 0"),
        (_bad(("models", "tuning", "lr_all_features", "C"), [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]), "listed ascending"),
        (_bad(("models", "tuning", "lr_all_features", "C"), [0.001, 0.01, 0.1, 1.0, 1.0, 10.0]), "each value once"),
        (_bad(("models", "tuning", "lr_all_features", "C"), [0.001, 0.01, 0.1, 0.5, 10.0, 100.0]), "library default 1.0"),
        (_bad(("models", "tuning", "random_forest", "min_samples_leaf"), [1, 5, 20, 50, 100, 200]), "listed descending"),
        (_bad(("models", "tuning", "random_forest", "min_samples_leaf"), [200, 100, 50, 20, 5, 0]), "whole numbers, at least 1"),
        (_bad(("models", "tuning", "random_forest", "min_samples_leaf"), [200, 100, 50, 20, 5.5, 1]), "whole numbers, at least 1"),
        (_bad(("models", "tuning", "random_forest", "min_samples_leaf"), [200, 100, 50, 20, 1]), "same number of settings"),
        (_bad(("models", "tuning", "gradient_boosting", "learning_rate"), [0.03, 0.1, 1.5]), "above 0 and at most 1"),
        (_bad(("models", "tuning", "gradient_boosting", "learning_rate"), [0.0, 0.1]), "above 0 and at most 1"),
        (_bad(("models", "tuning", "gradient_boosting", "learning_rate"), [0.1, 0.03]), "listed ascending"),
        (_bad(("models", "tuning", "gradient_boosting", "min_samples_leaf"), [400, 100]), "library default 20"),
    ],
)
def test_bad_tuning_settings_say_what_is_wrong(bad, message):
    with pytest.raises(ValueError, match=message):
        settings.validate_config(bad)


def test_one_setting_per_model_is_refused_because_there_is_nothing_to_tune():
    c = copy.deepcopy(CONFIG)
    c["models"]["tuning"] = {"lr_all_features": {"C": [1.0]}, "random_forest": {"min_samples_leaf": [1]}, "gradient_boosting": {"learning_rate": [0.1], "min_samples_leaf": [20]}}
    with pytest.raises(ValueError, match="at least 2 settings"):
        settings.validate_config(c)


def test_tuning_settings_on_the_edge_are_accepted():
    c = copy.deepcopy(CONFIG)
    c["models"]["tuning"] = {"lr_all_features": {"C": [1.0, 2.0]}, "random_forest": {"min_samples_leaf": [2, 1]},
                             "gradient_boosting": {"learning_rate": [0.1, 1.0], "min_samples_leaf": [20]}}
    settings.validate_config(c)
    c["models"]["ladder"] = ["lr_price_only", "random_forest"]          # a ladder without two of the models needs only its own grid
    c["models"]["tuning"] = {"random_forest": {"min_samples_leaf": [5, 1]}}
    settings.validate_config(c)
    c["models"]["tuning"] = {"random_forest": {"min_samples_leaf": [5, 1]}, "lr_all_features": {"C": [1.0, 2.0]}}
    with pytest.raises(ValueError, match="exactly these models"):
        settings.validate_config(c)


def test_the_learning_rate_of_exactly_one_is_accepted():
    c = copy.deepcopy(CONFIG)
    c["models"]["tuning"]["gradient_boosting"] = {"learning_rate": [0.1, 1.0], "min_samples_leaf": [400, 100, 20]}
    settings.validate_config(c)


# ---- the choice rule

def test_pick_first_lowest_compares_at_four_decimals_and_takes_the_earliest_of_equals():
    assert tune.pick_first_lowest([0.5, 0.3, 0.2, 0.2]) == 2
    assert tune.pick_first_lowest([0.5, 0.30004, 0.3, 0.29996]) == 1       # 0.30004, 0.3 and 0.29996 are all 0.3000 at 4 decimals
    assert tune.pick_first_lowest([0.5, 0.30004, 0.3, 0.29994]) == 3       # 0.29994 is 0.2999: lower
    assert tune.pick_first_lowest([0.12346, 0.12344]) == 1                 # 0.1235 against 0.1234
    assert tune.pick_first_lowest([0.5]) == 0
    assert tune.pick_first_lowest([0.123456, 0.123449], decimals=6) == 1
    with pytest.raises(ValueError, match="missing"):
        tune.pick_first_lowest([0.5, float("nan")])
    with pytest.raises(ValueError, match="missing"):
        tune.pick_first_lowest([])


def test_known_answer_every_setting_is_scored_on_both_slices_next_to_its_train_score(fakes):
    r = tune.run(TRAIN, VAL, CONFIG)
    assert list(r["models"]) == LADDER
    for name in TUNABLE:
        rows = r["models"][name]["settings"]
        assert [row["index"] for row in rows] == list(range(6)) and [row["params"] for row in rows] == tune.grid_settings(CONFIG, name)
        for row, p in zip(rows, P[name]):
            assert row["validation"]["all"] == pytest.approx(bce(p, VAL_ALL), abs=1e-12)
            assert row["validation"]["contested"] == pytest.approx(bce(p, VAL_CON), abs=1e-12)
            assert row["train"]["all"] == pytest.approx(bce(p, TRAIN_ALL), abs=1e-12)
            assert row["train"]["contested"] == pytest.approx(bce(p, TRAIN_CON), abs=1e-12)
            assert row["gap"]["all"] == pytest.approx(bce(p, VAL_ALL) - bce(p, TRAIN_ALL), abs=1e-12)
            assert row["gap"]["contested"] == pytest.approx(bce(p, VAL_CON) - bce(p, TRAIN_CON), abs=1e-12)
        assert [row["index"] for row in rows if row["is_library_default"]] == [{"lr_all_features": 3, "random_forest": 5, "gradient_boosting": 5}[name]]
    only = r["models"]["lr_price_only"]["settings"]
    assert len(only) == 1 and only[0]["params"] == {} and only[0]["is_library_default"] and not r["models"]["lr_price_only"]["tunable"]
    # the price-only rung is the raw price here (the correction was set to "the price is right"): 8 markets at 0.5 and 2 at 0.01
    assert only[0]["validation"]["all"] == pytest.approx((8 * math.log(2) - math.log(0.01) - math.log(0.99)) / 10)
    assert only[0]["validation"]["contested"] == pytest.approx(math.log(2))
    assert only[0]["train"]["all"] == pytest.approx((16 * math.log(2) - math.log(0.01) - 3 * math.log(0.99)) / 20)
    assert only[0]["train"]["contested"] == pytest.approx(math.log(2))


def test_known_answer_the_best_setting_of_each_model_with_ties_going_to_the_simpler_setting(fakes):
    r = tune.run(TRAIN, VAL, CONFIG)
    assert r["models"]["lr_all_features"]["best_index"] == 2 and r["models"]["lr_all_features"]["best_params"] == {"C": 0.1}
    assert r["models"]["random_forest"]["best_index"] == 0 and r["models"]["random_forest"]["best_params"] == {"min_samples_leaf": 200}       # settings 0, 1, 2 tie
    assert r["models"]["gradient_boosting"]["best_index"] == 3 and r["models"]["gradient_boosting"]["best_params"] == {"learning_rate": 0.1, "min_samples_leaf": 400}   # 3 and 4 tie
    assert r["models"]["lr_price_only"]["best_index"] == 0
    # best minus the library default, on validation, all markets: never above 0, because the default is in every grid
    assert r["models"]["lr_all_features"]["best_minus_library_default_all"] == pytest.approx(bce(0.38, 0.4) - bce(0.46, 0.4))
    assert r["models"]["random_forest"]["best_minus_library_default_all"] == pytest.approx(bce(0.40, 0.4) - bce(0.20, 0.4))
    assert r["models"]["gradient_boosting"]["best_minus_library_default_all"] == pytest.approx(bce(0.41, 0.4) - bce(0.30, 0.4))
    assert r["models"]["random_forest"]["library_default_validation_log_loss"]["all"] == pytest.approx(bce(0.20, 0.4))
    assert [b["model"] for b in r["best_per_model"]] == LADDER


def test_known_answer_the_winner_the_runner_up_and_the_margin(fakes):
    r = tune.run(TRAIN, VAL, CONFIG)
    c = r["choice"]
    assert c["winner"] == "random_forest" and c["params"] == {"min_samples_leaf": 200}
    assert c["runner_up"] == "gradient_boosting" and c["runner_up_params"] == {"learning_rate": 0.1, "min_samples_leaf": 400}
    assert c["validation"]["all"] == pytest.approx(0.67301, abs=1e-5) and c["runner_up_validation"]["all"] == pytest.approx(0.67322, abs=1e-5)
    assert c["tie_at_4_decimals"] is False
    m = c["margin_runner_up_minus_winner"]
    assert m["all"]["point"] == pytest.approx(bce(0.41, 0.4) - bce(0.40, 0.4)) and m["all"]["point"] > 0
    assert m["contested"]["point"] == pytest.approx(bce(0.41, 0.375) - bce(0.40, 0.375))
    for s in ("all", "contested"):
        assert m[s]["low"] <= m[s]["high"] and m[s]["verdict"] in ("worse", "better", "cannot be told apart") and m[s]["reading"]
    assert c["winner_minus_raw_price"]["all"]["point"] == pytest.approx(bce(0.4, 0.4) - (8 * math.log(2) - math.log(0.01) - math.log(0.99)) / 10)
    assert c["lowest_contested_model"] == "lr_all_features"       # 0.38 sits closest to the contested Yes share of 0.375
    assert c["train"]["all"] == pytest.approx(bce(0.4, 0.45))


def test_the_raw_price_and_the_base_rate_are_scored_next_to_the_ladder_and_never_win(fakes):
    r = tune.run(TRAIN, VAL, CONFIG)
    assert r["order"] == ["raw_price", "base_rate"] + LADDER
    s = r["validation"]["all"]["scores"]
    assert s["base_rate"]["log_loss"] == pytest.approx(bce(0.45, 0.4))                 # the Yes share of train
    assert s["raw_price"]["log_loss"] == pytest.approx((8 * math.log(2) - math.log(0.01) - math.log(0.99)) / 10)
    assert s["random_forest"]["log_loss"] == pytest.approx(bce(0.40, 0.4)) and s["lr_all_features"]["log_loss"] == pytest.approx(bce(0.38, 0.4))
    assert s["gradient_boosting"]["log_loss"] == pytest.approx(bce(0.41, 0.4))
    assert r["validation"]["contested"]["scores"]["raw_price"]["log_loss"] == pytest.approx(math.log(2))
    assert r["validation"]["all"]["difference_vs_raw_price"]["random_forest"]["log_loss"]["point"] == pytest.approx(s["random_forest"]["log_loss"] - s["raw_price"]["log_loss"])


def test_a_raw_price_that_beats_every_model_is_reported_as_it_is_and_still_does_not_win(fakes, monkeypatch):
    monkeypatch.setattr(baselines, "fit_correction", lambda price, y, eps: {"intercept": 1.0, "slope": 1.0})     # the price-only rung is now worse than the price
    fakes.install({n: [0.9] * 6 for n in TUNABLE})
    r = tune.run(TRAIN, VAL, CONFIG)
    raw = r["validation"]["all"]["scores"]["raw_price"]["log_loss"]
    assert raw < r["choice"]["validation"]["all"] and r["choice"]["winner"] in LADDER
    assert r["choice"]["winner_minus_raw_price"]["all"]["point"] > 0
    assert "winner minus raw_price, all markets: +" in "\n".join(tune.summary_lines(r))


def test_a_tie_between_models_goes_to_the_earlier_model_in_the_ladder_even_when_the_later_one_is_lower_before_rounding(fakes):
    # lr_all_features forecasts 0.4003 everywhere, the forest 0.4 (the best a constant can do). They differ by about 2e-7: equal at 4 decimals.
    fakes.install({"lr_all_features": [0.4003] * 6, "random_forest": [0.4] * 6, "gradient_boosting": [0.5] * 6})
    r = tune.run(TRAIN, VAL, CONFIG)
    c = r["choice"]
    assert c["winner"] == "lr_all_features" and c["runner_up"] == "random_forest" and c["tie_at_4_decimals"] is True
    assert c["validation"]["all"] > c["runner_up_validation"]["all"]
    assert -1e-6 < c["margin_runner_up_minus_winner"]["all"]["point"] < 0
    assert "equal at 4 decimals; the earlier model in the ladder won the tie" in "\n".join(tune.summary_lines(r))
    # the settings of one model tie the same way: lr_all_features' six settings are equal, so the first one listed wins
    assert r["models"]["lr_all_features"]["best_index"] == 0


def test_a_difference_at_the_fourth_decimal_is_not_a_tie(fakes):
    fakes.install({"lr_all_features": [0.43] * 6, "random_forest": [0.40] * 6, "gradient_boosting": [0.5] * 6})     # about 0.6749 against 0.6730
    r = tune.run(TRAIN, VAL, CONFIG)
    assert r["choice"]["winner"] == "random_forest" and r["choice"]["tie_at_4_decimals"] is False


def test_the_wobble_does_not_decide_a_choice_it_cannot_separate(fakes):
    r = tune.run(TRAIN, VAL, CONFIG)       # 5 events: the range of the margin is wide, and the point estimate still decides
    m = r["choice"]["margin_runner_up_minus_winner"]["all"]
    assert r["choice"]["winner"] == "random_forest" and m["low"] < 0 < m["high"] and m["verdict"] == "cannot be told apart"
    assert "cannot be told apart" in m["reading"]


def test_the_scores_and_the_margin_get_the_configured_draws_seed_and_interval(fakes, monkeypatch):
    c = copy.deepcopy(CONFIG)
    c["seed"], c["evaluation"]["bootstrap_draws"], c["evaluation"]["interval"] = 7, 40, 0.8
    seen = {"weights": [], "score": [], "pair": []}
    real_w, real_s, real_p = baselines.resample_weights, baselines.score_slice, baselines.paired_difference
    monkeypatch.setattr(baselines, "resample_weights", lambda n, d, seed: (seen["weights"].append((n, d, seed)), real_w(n, d, seed))[1])
    monkeypatch.setattr(baselines, "score_slice", lambda *a, **k: (seen["score"].append((a[3].shape, a[5], a[6], k["names"])), real_s(*a, **k))[1])
    monkeypatch.setattr(baselines, "paired_difference", lambda *a, **k: (seen["pair"].append((a[4].shape, a[6], a[7])), real_p(*a, **k))[1])
    tune.run(TRAIN, VAL, c)
    assert seen["weights"] == [(5, 40, 7)]                                      # 5 events in validation, the configured draws and seed
    names = ["raw_price", "base_rate"] + LADDER
    assert seen["score"] == [((40, 5), 5, 0.8, names)] * 2                      # once per slice, 40 draws of 5 events, interval 0.8
    assert seen["pair"] == [((40, 5), 5, 0.8)] * 2


def test_the_margin_wobble_uses_the_configured_seed_draws_and_interval(fakes):
    c = copy.deepcopy(CONFIG)
    c["seed"], c["evaluation"]["bootstrap_draws"], c["evaluation"]["interval"] = 7, 40, 0.8
    r = tune.run(TRAIN, VAL, c)
    codes, uniques = pd.factorize(VAL["event_key"])
    weights = baselines.resample_weights(len(uniques), 40, 7)
    mask = np.ones(len(VAL), dtype=bool)
    expect = baselines.paired_difference(VAL, np.full(10, 0.41), np.full(10, 0.40), mask, weights, codes, len(uniques), 0.8)
    got = r["choice"]["margin_runner_up_minus_winner"]["all"]
    assert (got["point"], got["low"], got["high"]) == pytest.approx((expect["point"], expect["low"], expect["high"])) and got["verdict"] == expect["verdict"]
    assert r["bootstrap_draws"] == 40 and r["interval"] == 0.8 and r["seed"] == 7


# ---- only train is fitted on, only validation is scored, nothing is refitted

def test_every_setting_is_fitted_once_on_train_alone(fakes):
    tune.run(TRAIN, VAL, CONFIG)
    assert len(fakes) == 18                                                # 6 settings for each of the 3 tunable models. lr_price_only is not fitted by a model
    for n_rows, labels in fakes:
        assert n_rows == 20 and labels == [int(v) for v in TRAIN["label"]]  # train rows and train labels, never the 10 validation rows


def test_validation_labels_never_reach_the_fit_and_do_not_change_any_forecast(fakes):
    a = tune.run(TRAIN, VAL, CONFIG)
    flipped = VAL.assign(label=1 - VAL["label"])
    b = tune.run(TRAIN, flipped, CONFIG)
    # the forecasts are the same, so the fits saw the same things; only the scores differ
    assert all(labels == [int(v) for v in TRAIN["label"]] for _, labels in fakes)
    assert a["validation"]["all"]["scores"]["random_forest"]["log_loss"] != b["validation"]["all"]["scores"]["random_forest"]["log_loss"]


def test_with_real_models_the_validation_labels_cannot_change_a_fitted_model(monkeypatch):
    train, val = frame(300), frame(100, seed=1)
    seen = []
    real = models.build_model

    def spy(name, config, columns, params=None):
        model = real(name, config, columns, params)
        fit = model.fit
        model.fit = lambda X, y: (seen.append((len(X), int(np.sum(y)))), fit(X, y))[1]
        return model
    monkeypatch.setattr(models, "build_model", spy)
    tune.run(train, val, CONFIG)
    assert len(seen) == 18 and set(seen) == {(300, int(train["label"].sum()))}


def test_the_chosen_model_is_not_refitted_on_train_plus_validation(fakes):
    r = tune.run(TRAIN, VAL, CONFIG)
    assert r["rule"]["refit_on_train_plus_validation"] is False
    assert all(n_rows == len(TRAIN) for n_rows, _ in fakes) and "No refit on train + validation" in " ".join(r["notes"])


def test_the_run_is_repeatable_and_the_report_is_plain_json(fakes):
    a = tune.run(TRAIN, VAL, CONFIG)
    assert a == tune.run(TRAIN, VAL, CONFIG)
    assert json.loads(json.dumps(a)) == a


def test_run_needs_columns_markets_and_at_least_two_models(fakes):
    with pytest.raises(ValueError, match="lacks the columns"):
        tune.run(TRAIN.drop(columns=["last_close"]), VAL, CONFIG)
    with pytest.raises(ValueError, match="lacks the columns"):
        tune.run(TRAIN, VAL.drop(columns=["event_key"]), CONFIG)
    with pytest.raises(ValueError, match="no markets"):
        tune.run(TRAIN, VAL.iloc[0:0], CONFIG)
    with pytest.raises(ValueError, match="no markets"):
        tune.run(TRAIN.iloc[0:0], VAL, CONFIG)
    one = copy.deepcopy(CONFIG)
    one["models"]["ladder"], one["models"]["tuning"] = ["random_forest"], {"random_forest": CONFIG["models"]["tuning"]["random_forest"]}
    with pytest.raises(ValueError, match="at least two models"):
        tune.run(TRAIN, VAL, one)


def test_a_train_table_with_one_outcome_is_refused():
    with pytest.raises(ValueError, match="both outcomes"):
        tune.run(frame(100, label=np.ones(100, dtype="int8")), frame(50, seed=1), CONFIG)


def test_a_setting_is_chosen_on_all_markets_not_on_the_contested_slice(fakes):
    # 0.41 is closest to the Yes share of all markets (0.40), 0.37 closest to the contested share (0.375)
    fakes.install({"lr_all_features": [0.37, 0.41, 0.60, 0.60, 0.60, 0.60], "random_forest": [0.5] * 6, "gradient_boosting": [0.5] * 6})
    r = tune.run(TRAIN, VAL, CONFIG)
    rows = r["models"]["lr_all_features"]["settings"]
    assert rows[0]["validation"]["contested"] < rows[1]["validation"]["contested"] - 1e-3      # the contested slice would prefer the first
    assert r["models"]["lr_all_features"]["best_index"] == 1 and r["choice"]["winner"] == "lr_all_features" and r["choice"]["params"] == {"C": 0.01}


def test_the_contested_line_appears_only_when_another_model_is_lowest_there(fakes):
    r = tune.run(TRAIN, VAL, CONFIG)
    assert r["choice"]["winner"] == "random_forest" and r["choice"]["lowest_contested_model"] == "lr_all_features"
    assert "On the contested slice alone lr_all_features has the lowest log loss; the rule decides on all markets." in "\n".join(tune.summary_lines(r))
    fakes.install({"lr_all_features": [0.40] * 6, "random_forest": [0.5] * 6, "gradient_boosting": [0.5] * 6})
    r = tune.run(TRAIN, VAL, CONFIG)
    assert r["choice"]["winner"] == r["choice"]["lowest_contested_model"] == "lr_all_features"
    assert "On the contested slice alone" not in "\n".join(tune.summary_lines(r))


def test_an_empty_contested_slice_is_reported_not_crashed(fakes):
    val = table([0.01] * 6, [0, 0, 1, 0, 0, 0])      # all near-decided
    r = tune.run(TRAIN, val, CONFIG)
    assert r["validation"]["contested"]["markets"] == 0
    row = r["models"]["random_forest"]["settings"][0]
    assert row["validation"]["contested"] is None and row["gap"]["contested"] is None and row["gap"]["all"] is not None
    assert r["choice"]["margin_runner_up_minus_winner"].keys() == {"all"} and r["choice"]["lowest_contested_model"] is None
    text = "\n".join(tune.summary_lines(r))
    assert "contested: no markets in validation" in text and "n/a" in text


def test_a_ladder_of_two_models_with_one_grid_works(fakes):
    c = copy.deepcopy(CONFIG)
    c["models"]["ladder"], c["models"]["tuning"] = ["lr_price_only", "random_forest"], {"random_forest": CONFIG["models"]["tuning"]["random_forest"]}
    r = tune.run(TRAIN, VAL, c)
    assert r["order"] == ["raw_price", "base_rate", "lr_price_only", "random_forest"] and set(r["grids"]) == {"random_forest"}
    assert r["choice"]["winner"] == "random_forest" and r["choice"]["runner_up"] == "lr_price_only"


# ---- with the real models

@pytest.fixture(scope="module")
def real():
    train, val = frame(300), frame(100, seed=1)
    return train, val, tune.run(train, val, CONFIG)


def test_the_library_default_setting_reproduces_the_stage_6_numbers(real):
    train, val, r = real
    s6 = models.run(train, val, CONFIG)
    for name in LADDER:
        default = next(row for row in r["models"][name]["settings"] if row["is_library_default"])
        assert default["validation"]["all"] == pytest.approx(s6["validation"]["all"]["scores"][name]["log_loss"], abs=1e-9)
        assert default["validation"]["contested"] == pytest.approx(s6["validation"]["contested"]["scores"][name]["log_loss"], abs=1e-9)
        assert default["train"]["all"] == pytest.approx(s6["train"]["all"]["scores"][name]["log_loss"], abs=1e-9)
        assert default["gap"]["all"] == pytest.approx(s6["validation_minus_train_log_loss"]["all"][name], abs=1e-9)


def test_tuning_cannot_come_out_worse_than_untuned_on_validation(real):
    _, _, r = real
    for name in TUNABLE:
        m = r["models"][name]
        assert m["best_minus_library_default_all"] <= 0
        assert m["settings"][m["best_index"]]["validation"]["all"] == min(row["validation"]["all"] for row in m["settings"])


def test_the_winner_is_the_lowest_validation_log_loss_of_the_bests_and_is_on_the_ladder(real):
    _, _, r = real
    c = r["choice"]
    best = {b["model"]: b["validation"]["all"] for b in r["best_per_model"]}
    assert c["winner"] in LADDER and c["validation"]["all"] == best[c["winner"]] <= min(round(v, 4) for v in best.values()) + 1e-4
    assert c["runner_up"] != c["winner"] and c["runner_up_validation"]["all"] >= c["validation"]["all"] - 1e-4
    assert r["validation"]["all"]["scores"][c["winner"]]["log_loss"] == pytest.approx(c["validation"]["all"])
    assert c["params"] in tune.grid_settings(CONFIG, c["winner"])


def test_the_report_names_what_it_did(real):
    train, val, r = real
    assert r["train_markets"] == 300 and r["validation_markets"] == 100 and r["inputs"] == models.input_columns(CONFIG)[0]
    assert list(r["dropped_inputs"]) == ["has_fee"] and r["seed"] == 42 and r["forest_trees"] == 300
    assert set(r["grids"]) == set(TUNABLE) and r["rule"]["wobble"].startswith("shown next to the margin, never used to decide")
    assert any("slightly flattering" in n for n in r["notes"]) and any("test split was not read" in n for n in r["notes"])


def test_summary_lines_show_every_setting_the_best_the_default_and_the_choice(real):
    _, _, r = real
    text = "\n".join(tune.summary_lines(r))
    for name in LADDER:
        assert name in text
    assert text.count("<- best") == len(LADDER) and text.count("    * ") == 3
    for name in TUNABLE:
        for row in r["models"][name]["settings"]:
            assert f"{row['validation']['all']:.4f}" in text and f"{row['gap']['all']:+.4f}" in text
    assert "Choice (validation log loss, all markets)" in text and f"winner:    {r['choice']['winner']}" in text
    assert "slightly flattering" in text and "The range never decides." in text and "Light tuning." in text
    assert text.count("(* = library default, the Stage 6 model)") == 3 and "no setting to tune" in text
    assert text.count("best minus library default") == 3                  # the three tunable models, not the price-only rung


# ---- the command line and the pipeline

def _split_dir(tmp_path):
    data = sample.write_sample(tmp_path / "sample", seed=CONFIG["seed"])
    dataset, _ = build.build_dataset(data, CONFIG, 7)
    table_, _ = split.assign(dataset, CONFIG)
    split.write_splits(table_, tmp_path / "data", CONFIG, 7)
    return tmp_path / "data"


def test_main_reads_train_and_validation_but_never_test_and_writes_the_report(tmp_path, capsys, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    asked, real_read = [], split.read_split
    monkeypatch.setattr(split, "read_split", lambda *a, **k: (asked.append(a[2]), real_read(*a, **k))[1])
    assert tune.main(["--data-dir", str(data_dir)]) == 0
    assert asked == ["train", "validation"]
    report = json.loads((tmp_path / "outputs" / "tuning_k7.json").read_text(encoding="utf-8"))
    assert report["choice"]["winner"] in LADDER and not (tmp_path / "frozen_choice.yaml").exists()
    out = capsys.readouterr().out
    assert "Light tuning." in out and "Choice (validation log loss, all markets)" in out and "Nothing is frozen yet" in out and "Test was not read." in out


def test_main_needs_no_test_split_file(tmp_path, capsys, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    split.split_path(data_dir, CONFIG, "test", 7).unlink()
    assert tune.main(["--data-dir", str(data_dir)]) == 0
    assert "Light tuning." in capsys.readouterr().out


def test_main_needs_both_split_files(tmp_path, capsys):
    assert tune.main(["--data-dir", str(tmp_path)]) == 2
    assert "No train split" in capsys.readouterr().out
    data_dir = _split_dir(tmp_path)
    split.split_path(data_dir, CONFIG, "validation", 7).unlink()
    assert tune.main(["--data-dir", str(data_dir)]) == 2
    assert "No validation split" in capsys.readouterr().out


def test_main_uses_the_k_it_is_given(tmp_path, capsys):
    data_dir = _split_dir(tmp_path)
    assert tune.main(["--data-dir", str(data_dir), "--k", "3"]) == 2
    assert "No train split" in capsys.readouterr().out


def test_main_reports_a_problem_in_plain_words(tmp_path, capsys, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    path = split.split_path(data_dir, CONFIG, "train", 7)
    pd.read_parquet(path).assign(label=1).to_parquet(path, index=False)
    assert tune.main(["--data-dir", str(data_dir)]) == 1
    assert "PROBLEM" in capsys.readouterr().out
    assert not (tmp_path / "outputs" / "tuning_k7.json").exists()


def test_pipeline_hands_the_tuning_train_and_validation_rows_only_and_freezes_nothing(monkeypatch, capsys, tmp_path):
    seen = {}
    real_run = tune.run
    monkeypatch.setattr(tune, "run", lambda tr, va, c: (seen.update(train=tr, validation=va), real_run(tr, va, c))[1])
    frozen_path = settings.PROJECT_DIR / "frozen_choice.yaml"
    before = sorted(p.name for p in settings.PROJECT_DIR.iterdir())
    frozen_before = frozen_path.read_bytes() if frozen_path.exists() else None
    assert pipeline.main(["--sample"]) == 0
    # The real frozen file is committed, so it may exist; the run must neither create nor change it.
    assert sorted(p.name for p in settings.PROJECT_DIR.iterdir()) == before
    assert (frozen_path.read_bytes() if frozen_path.exists() else None) == frozen_before
    data = sample.write_sample(tmp_path, seed=CONFIG["seed"])
    dataset, _ = build.build_dataset(data, CONFIG, 7)
    table_, _ = split.assign(dataset, CONFIG)
    expect = lambda part: set(table_[(table_["split"] == part) & ~table_["purged"]]["market_id"])  # noqa: E731
    assert set(seen["train"]["market_id"]) == expect("train") and set(seen["validation"]["market_id"]) == expect("validation")
    assert not set(seen["validation"]["market_id"]) & expect("test")
    out = capsys.readouterr().out
    assert "Light tuning." in out and "Stage 7:" in out and "Nothing is frozen by this run." in out and "The test split was not opened by this run." in out


def test_the_pipeline_stops_with_a_plain_message_when_the_tuning_cannot_run(monkeypatch, capsys):
    def boom(*a):
        raise ValueError("no spread in the prices")
    monkeypatch.setattr(tune, "run", boom)
    assert pipeline.main(["--sample"]) == 1
    assert "PROBLEM tuning: no spread in the prices" in capsys.readouterr().out
