import copy

import pytest

import settings


@pytest.fixture
def config():
    return settings.load_config()


def test_decisions_are_locked(config):
    assert config["decision_time"]["days_after_open"] == 7
    assert config["decision_time"]["robustness_days_after_open"] == 3
    assert config["universe"]["min_trades_before_t"] == 20
    assert (config["universe"]["contested_low"], config["universe"]["contested_high"]) == (0.05, 0.95)
    s = config["split"]
    assert (s["train"], s["validation"], s["test"]) == (0.6, 0.2, 0.2)
    assert config["seed"] == 42
    assert config["metrics"]["decides"] == "log_loss"


def test_outcomes_are_text_not_booleans(config):
    # YAML reads a bare No as false. The quotes in config.yaml prevent that.
    assert config["question"]["outcomes"] == ["Yes", "No"]


def test_file_sizes_and_urls(config):
    sizes = {k: v["bytes"] for k, v in config["data"]["files"].items()}
    assert sizes == {"events": 7581274, "markets": 22591844, "predictions": 119721790, "ohlcv_1d": 161540522}
    for entry in config["data"]["files"].values():
        assert entry["url"].startswith("https://huggingface.co/datasets/vgregoire/polymarket-users/resolve/main/")
        assert entry["sha256"] is None  # recorded in Stage 1


def test_leaking_fields_stay_excluded(config):
    for name in ("close_time", "events.volume", "events.open_interest", "events.n_questions"):
        assert name in config["features"]["excluded"]


def _bad(config, path, value):
    c = copy.deepcopy(config)
    node = c
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return c


@pytest.mark.parametrize(
    "path,value",
    [
        (("split", "train"), 0.7),
        (("split", "test"), 0.0),
        (("split", "order_by"), "close_time"),
        (("decision_time", "days_after_open"), 0),
        (("decision_time", "days_after_open"), 3),
        (("decision_time", "bars_strictly_before_t"), False),
        (("universe", "contested_low"), 0.97),
        (("universe", "min_trades_before_t"), 0),
        (("seed"), "x"),
        (("metrics", "decides"), "brier"),
        (("models", "selection"), "best_test"),
        (("question", "n_outcomes"), 3),
    ],
)
def test_impossible_settings_are_refused(config, path, value):
    path = path if isinstance(path, tuple) else (path,)
    with pytest.raises(ValueError):
        settings.validate_config(_bad(config, path, value))


def test_removing_a_leak_guard_is_refused(config):
    c = copy.deepcopy(config)
    c["features"]["excluded"].remove("close_time")
    with pytest.raises(ValueError):
        settings.validate_config(c)


def test_excluded_field_as_feature_is_refused(config):
    c = copy.deepcopy(config)
    c["features"]["context"].append("close_time")
    with pytest.raises(ValueError):
        settings.validate_config(c)


def test_bad_fingerprint_is_refused(config):
    c = copy.deepcopy(config)
    c["data"]["files"]["events"]["sha256"] = "abc"
    with pytest.raises(ValueError):
        settings.validate_config(c)
