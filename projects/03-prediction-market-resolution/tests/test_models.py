"""Stage 6 tests: which columns go in, how they are prepared, that the models learn what is there to learn,
that validation never leaks into the fit, and the train-versus-validation report."""

import copy
import math

import numpy as np
import pandas as pd
import pytest

import baselines
import build
import models
import pipeline
import sample
import settings
import split

CONFIG = settings.load_config()
EPS = CONFIG["metrics"]["price_clip"]
INPUTS = ["last_close", "first_open", "high", "low", "mean_close", "change_first_to_last", "n_bars", "days_since_last_bar",
          "trades_before_t", "trades_last_bar", "category", "sibling_count_at_t"]


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


def with_dropped(dropped):
    c = copy.deepcopy(CONFIG)
    c["models"]["dropped_features"] = dropped
    return c


# ---- which columns go in, and how they are prepared

def test_has_fee_is_not_a_model_input_and_the_rest_are_in_config_order():
    columns, dropped = models.input_columns(CONFIG)
    assert columns == INPUTS and list(dropped) == ["has_fee"] and "6 of 23,897" in dropped["has_fee"]
    columns, dropped = models.input_columns(with_dropped({}))
    assert columns == INPUTS + ["has_fee"] and dropped == {}


def test_yes_no_columns_become_numbers_and_a_missing_category_is_its_own_category():
    X = models.model_inputs(frame(4, has_fee=[True, False, True, False], category=["A", None, "B", np.nan]), INPUTS + ["has_fee"])
    assert list(X["has_fee"]) == [1.0, 0.0, 1.0, 0.0] and X["has_fee"].dtype == "float64"
    assert list(X["category"]) == ["A", "(missing)", "B", "(missing)"]


def test_a_missing_number_is_refused_not_filled():
    with pytest.raises(ValueError, match="missing values.*trades_before_t"):
        models.model_inputs(frame(4, trades_before_t=[10.0, np.nan, 30.0, 40.0]), INPUTS)
    with pytest.raises(ValueError, match="lacks model input columns"):
        models.model_inputs(frame(4).drop(columns=["n_bars"]), INPUTS)


def test_log_odds_of_prices():
    out = models._log_odds([0.5, 0.9, 0.0, 1.0], EPS)
    assert out == pytest.approx([0.0, math.log(9), math.log(0.001 / 0.999), math.log(0.999 / 0.001)])


def test_the_logistic_regression_sees_price_levels_as_standardised_log_odds_then_numbers_then_categories():
    train = frame(4, last_close=[0.1, 0.2, 0.3, 0.9], trades_before_t=[10, 20, 30, 100], category=["A", "B", "A", "B"])
    columns, _ = models.input_columns(CONFIG)
    pipe = models.build_model("lr_all_features", CONFIG, columns)
    pre = pipe.named_steps["prepare"].fit(models.model_inputs(train, columns))
    names = list(pre.get_feature_names_out())
    assert names[:5] == ["price_levels__last_close", "price_levels__first_open", "price_levels__high", "price_levels__low", "price_levels__mean_close"]
    assert names[-2:] == ["category__category_A", "category__category_B"] and "number__has_fee" not in names
    price = np.array([0.1, 0.2, 0.3, 0.9])
    z = np.log(price / (1 - price))
    out = pre.transform(models.model_inputs(train, columns))
    assert out[:, 0] == pytest.approx((z - z.mean()) / z.std())
    trades = np.array([10.0, 20.0, 30.0, 100.0])
    assert out[:, names.index("number__trades_before_t")] == pytest.approx((trades - trades.mean()) / trades.std())


def test_the_forests_see_raw_numbers_and_a_category_they_never_saw_becomes_zeros():
    train = frame(6, category=["A", "B", "A", "B", "A", "B"])
    columns, _ = models.input_columns(CONFIG)
    pre = models.build_model("random_forest", CONFIG, columns).named_steps["prepare"].fit(models.model_inputs(train, columns))
    out = pre.transform(models.model_inputs(frame(3, seed=5, category=["A", "Z", "B"]), columns))
    names = list(pre.get_feature_names_out())
    assert out[:, names.index("number__last_close")] == pytest.approx(frame(3, seed=5)["last_close"].to_numpy())
    cat = out[:, [names.index("category__category_A"), names.index("category__category_B")]]
    assert cat.tolist() == [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]     # Z is unseen: all zeros


def test_the_models_are_built_with_the_untuned_settings_from_config():
    columns, _ = models.input_columns(CONFIG)
    rf = models.build_model("random_forest", CONFIG, columns).named_steps["model"]
    assert rf.n_estimators == 300 and rf.random_state == 42
    lr = models.build_model("lr_all_features", CONFIG, columns).named_steps["model"]
    assert lr.max_iter == 1000 and lr.C == 1.0 and lr.random_state == 42
    gb = models.build_model("gradient_boosting", CONFIG, columns).named_steps["model"]
    assert gb.random_state == 42 and gb.learning_rate == 0.1
    with pytest.raises(ValueError, match="no model is built"):
        models.build_model("lr_price_only", CONFIG, columns)


# ---- fitting and forecasting

def test_every_ladder_step_is_fitted_and_has_fee_is_not_read():
    fitted, columns, correction = models.fit_models(frame(300), CONFIG)
    assert list(fitted) == CONFIG["models"]["ladder"] and columns == INPUTS
    assert fitted["lr_price_only"] == correction
    for name in ("lr_all_features", "random_forest", "gradient_boosting"):
        assert "has_fee" not in list(fitted[name].named_steps["prepare"].feature_names_in_)


def test_forecasts_are_clipped_probabilities_for_every_model():
    train, val = frame(300), frame(100, seed=1)
    fitted, columns, _ = models.fit_models(train, CONFIG)
    fc = models.forecast_all(fitted, columns, float(train["label"].mean()), val, CONFIG)
    assert set(fc) == {"raw_price", "base_rate"} | set(CONFIG["models"]["ladder"])
    for p in fc.values():
        assert len(p) == 100 and p.min() >= EPS and p.max() <= 1 - EPS


def test_the_fit_is_repeatable():
    a = models.run(frame(300), frame(100, seed=1), CONFIG)
    assert a == models.run(frame(300), frame(100, seed=1), CONFIG)


def test_validation_labels_never_reach_the_fit_or_the_forecasts():
    train, val = frame(300), frame(100, seed=1)
    fitted, columns, _ = models.fit_models(train, CONFIG)
    base = float(train["label"].mean())
    a = models.forecast_all(fitted, columns, base, val, CONFIG)
    b = models.forecast_all(fitted, columns, base, val.assign(label=1 - val["label"]), CONFIG)
    c = models.forecast_all(fitted, columns, base, val.drop(columns=["label"]), CONFIG)
    for name in a:
        assert np.array_equal(a[name], b[name]) and np.array_equal(a[name], c[name])


def test_a_forecast_for_a_market_does_not_depend_on_the_other_validation_markets():
    train, val = frame(300), frame(100, seed=1)
    fitted, columns, _ = models.fit_models(train, CONFIG)
    base = float(train["label"].mean())
    whole = models.forecast_all(fitted, columns, base, val, CONFIG)
    part = models.forecast_all(fitted, columns, base, val.iloc[10:20], CONFIG)
    for name in whole:
        assert whole[name][10:20] == pytest.approx(part[name], rel=1e-9, abs=1e-12)   # not bit-for-bit: the maths library rounds differently by batch size


def test_the_models_need_both_outcomes_in_train():
    with pytest.raises(ValueError, match="both outcomes"):
        models.fit_models(frame(100, label=np.ones(100, dtype="int8")), CONFIG)


def test_a_category_only_in_validation_does_not_crash_and_is_reported():
    val = frame(60, seed=1, category=["A", "B", "C", "Z"] * 15)
    r = models.run(frame(300, category=["A", "B", "C"] * 100), val, CONFIG)
    assert r["category_levels_only_in_validation"] == ["Z"] and r["category_levels_in_train"] == ["A", "B", "C"]
    assert "categories in validation but not in train" in "\n".join(models.summary_lines(r))


# ---- do the models learn what is there to learn, and does the report show overfitting

def test_the_models_find_a_pattern_the_price_does_not_hold():
    # The outcome is decided by trades_last_bar; the price carries nothing. The price-only rung cannot see it.
    def make(n, seed):
        f = frame(n, seed=seed)
        f["label"] = (f["trades_last_bar"] > 50).astype("int8")
        f["last_close"] = f["first_open"] = f["mean_close"] = 0.3
        f["last_close"] = np.random.default_rng(seed + 100).uniform(0.2, 0.4, n)
        return f
    r = models.run(make(600, 0), make(300, 1), CONFIG)
    ll = {n: r["validation"]["all"]["scores"][n]["log_loss"] for n in r["order"]}
    assert ll["lr_price_only"] > 0.6 and ll["base_rate"] > 0.6
    for name in ("random_forest", "gradient_boosting", "lr_all_features"):
        assert ll[name] < ll["base_rate"] - 0.2, (name, ll)


def test_a_forest_on_pure_noise_shows_a_large_overfit_gap():
    train, val = frame(600, seed=0), frame(300, seed=1)
    train["label"] = np.random.default_rng(7).integers(0, 2, 600).astype("int8")
    val["label"] = np.random.default_rng(8).integers(0, 2, 300).astype("int8")
    r = models.run(train, val, CONFIG)
    gap = r["validation_minus_train_log_loss"]["all"]
    assert r["train"]["all"]["scores"]["random_forest"]["log_loss"] < 0.3 and gap["random_forest"] > 0.5
    assert gap["base_rate"] < 0.2 and gap["random_forest"] > gap["lr_all_features"]


# ---- the whole run

def test_run_report_has_train_and_validation_side_by_side():
    train, val = frame(300), frame(100, seed=1)
    r = models.run(train, val, CONFIG)
    assert r["train_markets"] == 300 and r["validation_markets"] == 100 and r["order"] == ["raw_price", "base_rate"] + CONFIG["models"]["ladder"]
    assert r["inputs"] == INPUTS and list(r["dropped_inputs"]) == ["has_fee"] and r["base_rate_train"] == float(train["label"].mean())
    assert r["settings"] == {"seed": 42, "forest_trees": 300, "logistic_max_iter": 1000, "other": "library defaults"}
    for s in ("all", "contested"):
        v, t = r["validation"][s], r["train"][s]
        assert set(v["scores"]) == set(t["scores"]) == set(r["order"])
        assert set(v["difference_vs_raw_price"]) == set(r["order"]) - {"raw_price"}
        for n in r["order"]:
            assert r["validation_minus_train_log_loss"][s][n] == pytest.approx(v["scores"][n]["log_loss"] - t["scores"][n]["log_loss"])
            w = v["wobble"][n]["log_loss"]
            assert w["low"] <= w["high"]
    y = val["label"].to_numpy()
    raw = baselines.clip_probability(val["last_close"].to_numpy(), EPS)
    assert r["validation"]["all"]["scores"]["raw_price"]["log_loss"] == pytest.approx(baselines.log_loss(y, raw))
    ty = train["label"].to_numpy()
    assert r["train"]["all"]["scores"]["raw_price"]["log_loss"] == pytest.approx(baselines.log_loss(ty, baselines.clip_probability(train["last_close"].to_numpy(), EPS)))


def test_the_price_only_rung_equals_the_stage_5_correction():
    train, val = frame(300), frame(100, seed=1)
    r, b = models.run(train, val, CONFIG), baselines.run(train, val, CONFIG)
    for s in ("all", "contested"):
        assert r["validation"][s]["scores"]["lr_price_only"] == b["slices"][s]["scores"]["price_only_correction"]
    assert r["correction"] == b["correction"]


def test_the_ladder_in_config_is_respected():
    c = copy.deepcopy(CONFIG)
    c["models"]["ladder"] = ["lr_price_only", "random_forest"]
    r = models.run(frame(300), frame(100, seed=1), c)
    assert r["order"] == ["raw_price", "base_rate", "lr_price_only", "random_forest"]


def test_an_empty_slice_is_reported_not_crashed():
    val = frame(40, seed=1, last_close=0.01, first_open=0.01, mean_close=0.01, high=0.02, low=0.0)   # all near-decided
    r = models.run(frame(300), val, CONFIG)
    assert r["validation"]["contested"]["markets"] == 0 and r["validation_minus_train_log_loss"].keys() == {"all"}
    assert "contested: no markets in train or validation" in "\n".join(models.summary_lines(r))


def test_run_needs_columns_and_markets():
    with pytest.raises(ValueError, match="lacks the columns"):
        models.run(frame(100).drop(columns=["last_close"]), frame(50, seed=1), CONFIG)
    with pytest.raises(ValueError, match="no markets"):
        models.run(frame(100), frame(50, seed=1).iloc[0:0], CONFIG)


def test_summary_lines_name_the_numbers():
    r = models.run(frame(300), frame(100, seed=1), CONFIG)
    text = "\n".join(models.summary_lines(r))
    assert "Model ladder, untuned (library defaults, seed 42). Fitted on train (300 markets)" in text and "validation (100 markets" in text
    assert "inputs (12): last_close" in text and "not a model input: has_fee" in text
    assert "overfit gap" in text and "log loss, random_forest minus raw_price" in text
    for n in r["order"]:
        assert n in text


# ---- the command line, the pipeline and the settings

def _split_dir(tmp_path):
    data = sample.write_sample(tmp_path / "sample", seed=CONFIG["seed"])
    dataset, _ = build.build_dataset(data, CONFIG, 7)
    table_, _ = split.assign(dataset, CONFIG)
    split.write_splits(table_, tmp_path / "data", CONFIG, 7)
    return tmp_path / "data"


def test_main_reads_train_and_validation_but_never_test(tmp_path, capsys, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    asked, real = [], split.read_split
    monkeypatch.setattr(split, "read_split", lambda *a, **k: (asked.append(a[2]), real(*a, **k))[1])
    assert models.main(["--data-dir", str(data_dir)]) == 0
    assert asked == ["train", "validation"]
    assert (tmp_path / "outputs" / "models_k7.json").exists()
    out = capsys.readouterr().out
    assert "Model ladder, untuned" in out and "Test was not read." in out


def test_main_needs_both_split_files(tmp_path, capsys):
    assert models.main(["--data-dir", str(tmp_path)]) == 2
    assert "No train split" in capsys.readouterr().out
    data_dir = _split_dir(tmp_path)
    split.split_path(data_dir, CONFIG, "validation", 7).unlink()
    assert models.main(["--data-dir", str(data_dir)]) == 2
    assert "No validation split" in capsys.readouterr().out


def test_main_reports_a_problem_in_plain_words(tmp_path, capsys, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    path = split.split_path(data_dir, CONFIG, "train", 7)
    pd.read_parquet(path).assign(label=1).to_parquet(path, index=False)
    assert models.main(["--data-dir", str(data_dir)]) == 1
    assert "PROBLEM" in capsys.readouterr().out


def test_pipeline_hands_the_models_train_and_validation_rows_only(monkeypatch, capsys, tmp_path):
    seen = {}
    real = models.run
    monkeypatch.setattr(models, "run", lambda tr, va, c: (seen.update(train=tr, validation=va), real(tr, va, c))[1])
    assert pipeline.main(["--sample"]) == 0
    data = sample.write_sample(tmp_path, seed=CONFIG["seed"])
    dataset, _ = build.build_dataset(data, CONFIG, 7)
    table_, _ = split.assign(dataset, CONFIG)
    expect = lambda part: set(table_[(table_["split"] == part) & ~table_["purged"]]["market_id"])  # noqa: E731
    assert set(seen["train"]["market_id"]) == expect("train")
    assert set(seen["validation"]["market_id"]) == expect("validation")
    assert not set(seen["validation"]["market_id"]) & expect("test")
    out = capsys.readouterr().out
    assert "Model ladder, untuned" in out and "Stage 7:" in out and "The test split was not opened by this run." in out


def test_the_pipeline_stops_with_a_plain_message_when_the_models_cannot_run(monkeypatch, capsys):
    def boom(*a):
        raise ValueError("no spread in the prices")
    monkeypatch.setattr(models, "run", boom)
    assert pipeline.main(["--sample"]) == 1
    assert "PROBLEM models: no spread in the prices" in capsys.readouterr().out


def _bad(path, value):
    c = copy.deepcopy(CONFIG)
    node = c
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return c


@pytest.mark.parametrize(
    "bad",
    [
        _bad(("models", "forest_trees"), 0),
        _bad(("models", "forest_trees"), 2.5),
        _bad(("models", "forest_trees"), True),
        _bad(("models", "logistic_max_iter"), 0),
        _bad(("models", "dropped_features"), {"has_fee": "  "}),
        _bad(("models", "dropped_features"), {"mystery": "no such column"}),
        _bad(("models", "dropped_features"), {"close_time": "an excluded field is not a feature"}),
        _bad(("models", "dropped_features"), ["has_fee"]),
        _bad(("models", "dropped_features"), {c: "x" for c in INPUTS + ["has_fee"]}),
        _bad(("models", "log_odds_columns"), ["last_close", "last_close"]),
        _bad(("models", "log_odds_columns"), ["mystery"]),
        _bad(("models", "log_odds_columns"), "last_close"),
    ],
)
def test_bad_model_settings_are_refused(bad):
    with pytest.raises(ValueError):
        settings.validate_config(bad)


def test_values_on_the_edge_of_the_allowed_range_are_accepted():
    for path, value in [(("models", "forest_trees"), 1), (("models", "logistic_max_iter"), 1), (("models", "dropped_features"), {}),
                        (("models", "log_odds_columns"), []), (("models", "dropped_features"), {c: "x" for c in INPUTS})]:
        settings.validate_config(_bad(path, value))


# ---- closing gaps found by the break-it run

def test_forecasts_clip_the_raw_price_and_the_base_rate_and_the_price_only_rung_reads_last_close():
    train = frame(300)
    fitted, columns, _ = models.fit_models(train, CONFIG)
    val = frame(4, seed=1, last_close=[0.0, 1.0, 0.3, 0.6], first_open=[0.9, 0.1, 0.8, 0.2])
    fc = models.forecast_all(fitted, columns, 0.0, val, CONFIG)
    assert fc["raw_price"].tolist() == [EPS, 1 - EPS, 0.3, 0.6]
    assert fc["base_rate"].tolist() == [EPS] * 4
    assert fc["lr_price_only"] == pytest.approx(baselines.apply_correction(np.array([0.0, 1.0, 0.3, 0.6]), fitted["lr_price_only"], EPS))
    assert fc["base_rate"].max() <= 1 - EPS and models.forecast_all(fitted, columns, 1.0, val, CONFIG)["base_rate"][0] == 1 - EPS


def test_the_wobble_uses_the_configured_seed_draws_and_interval():
    c = copy.deepcopy(CONFIG)
    c["seed"], c["evaluation"]["bootstrap_draws"], c["evaluation"]["interval"] = 7, 40, 0.8
    train, val = frame(300), frame(120, seed=1)
    r, b = models.run(train, val, c), baselines.run(train, val, c)
    for s in ("all", "contested"):
        assert r["validation"][s]["wobble"]["raw_price"] == b["slices"][s]["wobble"]["raw_price"]
        assert r["validation"][s]["difference_vs_raw_price"]["base_rate"] == b["slices"][s]["difference_vs_raw_price"]["base_rate"]
    assert r["bootstrap_draws"] == 40 and r["interval"] == 0.8


def test_train_scores_use_train_labels_and_brier_is_brier_and_an_empty_train_slice_is_skipped():
    train, val = frame(300), frame(100, seed=1)
    r = models.run(train, val, CONFIG)
    fitted, columns, _ = models.fit_models(train, CONFIG)
    fc = models.forecast_all(fitted, columns, float(train["label"].mean()), train, CONFIG)
    y = train["label"].to_numpy()
    for n in r["order"]:
        assert r["train"]["all"]["scores"][n]["brier"] == pytest.approx(baselines.brier(y, fc[n]))
        assert r["train"]["all"]["scores"][n]["log_loss"] == pytest.approx(baselines.log_loss(y, fc[n]))
    low = np.random.default_rng(3).uniform(0.005, 0.04, 300)
    near = frame(300, last_close=low, first_open=low, mean_close=low, high=0.05, low=0.0)   # no contested train markets
    r2 = models.run(near, val, CONFIG)
    assert r2["train"]["contested"] == {"markets": 0, "scores": {}} and "contested" not in r2["validation_minus_train_log_loss"]
    assert "contested: no markets in train or validation" in "\n".join(models.summary_lines(r2))


def test_summary_shows_the_gap_and_the_verdict_for_each_model():
    r = models.run(frame(300), frame(100, seed=1), CONFIG)
    text = "\n".join(models.summary_lines(r))
    for n in r["order"]:
        assert f"{r['validation_minus_train_log_loss']['all'][n]:+.4f}" in text
        if n != "raw_price":
            v = r["validation"]["all"]["difference_vs_raw_price"][n]["log_loss"]["verdict"]
            assert f"log loss, {n} minus raw_price" in text and v in text


def test_main_uses_the_k_it_is_given(tmp_path, capsys):
    data_dir = _split_dir(tmp_path)
    assert models.main(["--data-dir", str(data_dir), "--k", "3"]) == 2
    assert "No train split" in capsys.readouterr().out


def test_the_scoring_helper_needs_raw_price_in_its_names():
    val = frame(20)
    with pytest.raises(ValueError):
        baselines.score_slice(val, {"base_rate": np.full(20, 0.3)}, np.ones(20, dtype=bool), np.ones((3, 5)), np.arange(20) // 4, 5, 0.9, names=["base_rate"])


def test_a_log_odds_list_that_is_not_a_list_is_refused():
    with pytest.raises(ValueError):
        settings.validate_config(_bad(("models", "log_odds_columns"), {"last_close": 1}))
