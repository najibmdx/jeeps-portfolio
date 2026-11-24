"""Stage 5 tests: scores, the price correction, slices, the calibration table and the event wobble, on hand-made tables."""

import copy
import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import baselines
import build
import pipeline
import sample
import settings
import split

CONFIG = settings.load_config()
EPS = CONFIG["metrics"]["price_clip"]


def table(prices, labels, events=None):
    n = len(prices)
    return pd.DataFrame({
        "last_close": np.asarray(prices, dtype="float64"), "label": np.asarray(labels, dtype="int8"),
        "event_key": events if events is not None else [f"E{i}" for i in range(n)],
    })


# ---- scores

def test_log_loss_and_brier_on_two_markets():
    y, p = [1, 0], [0.8, 0.1]
    assert baselines.log_loss(y, p) == pytest.approx((-math.log(0.8) - math.log(0.9)) / 2)
    assert baselines.brier(y, p) == pytest.approx((0.2 ** 2 + 0.1 ** 2) / 2)


def test_log_loss_refuses_probabilities_of_exactly_0_or_1():
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="strictly between"):
            baselines.log_loss([1], [bad])


def test_clip_makes_a_confident_miss_cost_about_6_9_not_infinity():
    fc = baselines.clip_probability([0.0, 1.0, 0.5], EPS)
    assert list(fc) == [0.001, 0.999, 0.5]
    assert baselines.log_loss([1], fc[:1]) == pytest.approx(-math.log(0.001))
    assert baselines.log_loss([1], fc[:1]) == pytest.approx(6.907755, abs=1e-5)


# ---- the correction

def levels(*spec):
    """spec = (price, markets, yes) per level."""
    prices = [p for p, n, _ in spec for _ in range(n)]
    labels = [1 if i < yes else 0 for p, n, yes in spec for i in range(n)]
    return np.array(prices), np.array(labels)


def test_a_correct_price_gives_intercept_0_and_slope_1():
    price, y = levels((0.1, 10, 1), (0.5, 10, 5), (0.9, 10, 9))
    c = baselines.fit_correction(price, y, EPS)
    assert c["intercept"] == pytest.approx(0.0, abs=1e-8) and c["slope"] == pytest.approx(1.0, abs=1e-8)


def test_prices_that_are_too_extreme_get_a_slope_below_1():
    price, y = levels((0.1, 10, 3), (0.9, 10, 7))   # priced 10% and 90%, really 30% and 70%
    c = baselines.fit_correction(price, y, EPS)
    expected = math.log(0.7 / 0.3) / math.log(0.9 / 0.1)
    assert c["intercept"] == pytest.approx(0.0, abs=1e-8) and c["slope"] == pytest.approx(expected, abs=1e-8)
    assert c["slope"] < 1


def test_a_price_that_is_too_high_everywhere_gets_a_negative_intercept():
    price, y = levels((0.2, 10, 1), (0.5, 10, 3), (0.8, 10, 7))
    c = baselines.fit_correction(price, y, EPS)
    assert c["intercept"] < 0
    p = expit(c["intercept"] + c["slope"] * np.log(price / (1 - price)))   # the fit solves the two score equations
    z = np.log(price / (1 - price))
    assert np.sum(y - p) == pytest.approx(0, abs=1e-7) and np.sum(z * (y - p)) == pytest.approx(0, abs=1e-7)


def test_the_fit_still_settles_when_prices_sit_on_the_clip():
    price, y = levels((0.0, 6, 0), (0.1, 4, 4), (0.3, 4, 1), (0.5, 4, 1), (0.7, 4, 3), (1.0, 6, 6))
    c = baselines.fit_correction(price, y, EPS)
    assert np.isfinite(c["intercept"]) and np.isfinite(c["slope"])


def test_the_correction_is_refused_when_it_cannot_be_fitted():
    with pytest.raises(ValueError, match="both outcomes"):
        baselines.fit_correction([0.2, 0.4], [1, 1], EPS)
    with pytest.raises(ValueError, match="not all the same"):
        baselines.fit_correction([0.3, 0.3, 0.3, 0.3], [0, 1, 0, 1], EPS)
    with pytest.raises(ValueError, match="perfectly|settle"):
        baselines.fit_correction([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], EPS)


def test_apply_correction_with_identity_gives_back_the_clipped_price():
    out = baselines.apply_correction([0.0, 0.25, 1.0], {"intercept": 0.0, "slope": 1.0}, EPS)
    assert out == pytest.approx([0.001, 0.25, 0.999])


def test_forecasts_use_train_for_fitting_and_validation_for_prices():
    train = table(*levels((0.1, 10, 3), (0.9, 10, 7)))
    val = table([0.1, 0.9, 0.0], [0, 1, 0])
    fc, fitted = baselines.forecasts(train, val, CONFIG)
    assert fitted["base_rate_train"] == 0.5 and list(fc["base_rate"]) == [0.5, 0.5, 0.5]
    assert list(fc["raw_price"]) == [0.1, 0.9, 0.001]
    assert fc["price_only_correction"][0] == pytest.approx(0.3, abs=1e-6) and fc["price_only_correction"][1] == pytest.approx(0.7, abs=1e-6)


# ---- slices and the calibration table

def test_contested_includes_0_05_and_excludes_0_95():
    v = table([0.0499, 0.05, 0.5, 0.9499, 0.95, 1.0], [0] * 6)
    masks = baselines.slice_masks(v, CONFIG)
    assert list(masks["contested"]) == [False, True, True, True, False, False]
    assert masks["all"].all()


def test_calibration_bands_and_edges():
    edges = CONFIG["metrics"]["calibration_bands"]
    assert baselines.band_labels(edges)[0] == "0 to 0.05" and baselines.band_labels(edges)[-1] == "0.95 and above"
    assert len(baselines.band_labels(edges)) == len(edges) - 1
    p = [0.0, 0.049, 0.05, 0.1, 0.19, 0.5, 0.95, 1.0]
    y = [0, 0, 1, 1, 0, 1, 1, 0]
    rows = {r["band"]: r for r in baselines.calibration_rows(p, y, edges)}
    assert rows["0 to 0.05"]["markets"] == 2 and rows["0 to 0.05"]["yes_rate"] == 0.0
    assert rows["0.05 to 0.1"]["markets"] == 1 and rows["0.05 to 0.1"]["yes_rate"] == 1.0      # 0.05 belongs to the band that starts at 0.05
    assert rows["0.1 to 0.2"]["markets"] == 2 and rows["0.1 to 0.2"]["yes_rate"] == 0.5          # 0.1 starts its band
    assert rows["0.1 to 0.2"]["mean_forecast"] == pytest.approx(0.145)
    assert rows["0.5 to 0.7"]["markets"] == 1
    assert rows["0.95 and above"]["markets"] == 2 and rows["0.95 and above"]["yes_rate"] == 0.5  # the last band includes 1
    assert rows["0.2 to 0.3"] == {"band": "0.2 to 0.3", "markets": 0, "mean_forecast": None, "yes_rate": None}
    assert sum(r["markets"] for r in rows.values()) == len(p)


# ---- the wobble

def test_resample_weights_draw_whole_events_and_are_repeatable():
    w = baselines.resample_weights(7, 50, seed=1)
    assert w.shape == (50, 7) and (w.sum(axis=1) == 7).all() and (w == w.round()).all()
    assert np.array_equal(w, baselines.resample_weights(7, 50, seed=1))
    assert not np.array_equal(w, baselines.resample_weights(7, 50, seed=2))


def test_weighted_scores_by_hand():
    w = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    sums, counts = np.array([3.0, 1.0]), np.array([3.0, 1.0])   # event A: 3 markets with loss 1 each; event B: 1 market, loss 1
    assert list(baselines.weighted_scores(w, sums, counts)) == [1.0, 1.0, 1.0]
    sums = np.array([3.0, 0.0])                                    # event B has loss 0
    assert list(baselines.weighted_scores(w, sums, counts)) == [1.0, 0.75, 0.0]


def test_the_wobble_resamples_events_not_markets():
    # Event A has 3 markets, all wrong; event B has 1 market, right. Re-drawn sets can only hold whole events,
    # so the Brier score is 1 (A,A), 0.75 (A,B) or 0 (B,B). Resampling single markets would give other values.
    val = table([0.0, 0.0, 0.0, 1.0], [1, 1, 1, 1], ["A", "A", "A", "B"])
    train = table(*levels((0.1, 10, 3), (0.9, 10, 7)))
    config = copy.deepcopy(CONFIG)
    config["evaluation"]["bootstrap_draws"] = 400
    r = baselines.run(train, val, config)
    w = r["slices"]["all"]["wobble"]["raw_price"]["brier"]
    assert r["slices"]["all"]["scores"]["raw_price"]["brier"] == pytest.approx((3 * 0.999 ** 2 + 0.001 ** 2) / 4)
    low, high = 0.001 ** 2, 0.999 ** 2
    assert w["low"] == pytest.approx(low) and w["high"] == pytest.approx(high)   # the extremes are (B,B) and (A,A)


def test_one_event_gives_no_wobble():
    val = table([0.2, 0.4, 0.6], [0, 1, 0], ["A", "A", "A"])
    train = table(*levels((0.1, 10, 3), (0.9, 10, 7)))
    r = baselines.run(train, val, CONFIG)
    for b in baselines.BASELINES:
        w = r["slices"]["all"]["wobble"][b]["log_loss"]
        assert w["low"] == pytest.approx(w["high"]) == pytest.approx(r["slices"]["all"]["scores"][b]["log_loss"])


def test_verdict_rules():
    assert baselines.verdict(-0.3, -0.1) == "better"
    assert baselines.verdict(0.1, 0.3) == "worse"
    assert baselines.verdict(-0.1, 0.1) == "cannot be told apart"
    assert baselines.verdict(0.0, 0.2) == "cannot be told apart"      # touching zero is not clear
    assert baselines.verdict(-0.2, 0.0) == "cannot be told apart"


# ---- the whole run

def _train():
    return table(*levels((0.1, 40, 4), (0.3, 40, 12), (0.6, 40, 24), (0.9, 40, 36)),
                 events=[f"T{i // 2}" for i in range(160)])


def _validation():
    prices = [0.02, 0.04, 0.08, 0.15, 0.25, 0.45, 0.7, 0.97]
    labels = [0, 0, 0, 0, 1, 0, 1, 1]
    return table(prices * 5, labels * 5, [f"V{i // 2}" for i in range(40)])


def test_run_scores_all_three_baselines_on_both_slices():
    r = baselines.run(_train(), _validation(), CONFIG)
    assert r["validation_markets"] == 40 and r["validation_events"] == 20 and r["train_markets"] == 160
    assert r["slices"]["all"]["markets"] == 40 and r["slices"]["contested"]["markets"] == 25    # 0.02, 0.04 and 0.97 are near-decided
    assert r["slices"]["all"]["yes_rate"] == pytest.approx(3 / 8)
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1] * 5)
    raw = baselines.clip_probability(np.array([0.02, 0.04, 0.08, 0.15, 0.25, 0.45, 0.7, 0.97] * 5), EPS)
    assert r["slices"]["all"]["scores"]["raw_price"]["log_loss"] == pytest.approx(baselines.log_loss(y, raw))
    base = r["base_rate_train"]
    assert base == pytest.approx(76 / 160)
    assert r["slices"]["all"]["scores"]["base_rate"]["log_loss"] == pytest.approx(baselines.log_loss(y, np.full(40, base)))
    for s in r["slices"].values():
        for b in baselines.BASELINES:
            w = s["wobble"][b]["log_loss"]
            assert w["low"] <= s["scores"][b]["log_loss"] * 1.5 and w["low"] <= w["high"]
        assert set(s["difference_vs_raw_price"]) == {"price_only_correction", "base_rate"}
        d = s["difference_vs_raw_price"]["base_rate"]["log_loss"]
        assert d["point"] == pytest.approx(s["scores"]["base_rate"]["log_loss"] - s["scores"]["raw_price"]["log_loss"])
        assert d["verdict"] in ("better", "worse", "cannot be told apart")
    assert sum(x["markets"] for x in r["slices"]["all"]["calibration"]["raw_price"]) == 40
    assert sum(x["markets"] for x in r["slices"]["contested"]["calibration"]["raw_price"]) == 25


def test_the_run_is_repeatable_and_the_seed_matters():
    a = baselines.run(_train(), _validation(), CONFIG)
    assert a == baselines.run(_train(), _validation(), CONFIG)
    c = copy.deepcopy(CONFIG)
    c["seed"] = 7
    assert a["slices"]["all"]["wobble"] != baselines.run(_train(), _validation(), c)["slices"]["all"]["wobble"]


def test_an_empty_slice_is_reported_not_crashed():
    val = table([0.01, 0.02, 0.99], [0, 0, 1], ["A", "B", "C"])    # all near-decided
    r = baselines.run(_train(), val, CONFIG)
    assert r["slices"]["contested"] == {"markets": 0, "events": 0, "yes_rate": None, "note": "no markets in this slice",
                                        "calibration": r["slices"]["contested"]["calibration"]}
    assert "contested: no markets" in "\n".join(baselines.summary_lines(r))


def test_run_needs_columns_and_markets():
    with pytest.raises(ValueError, match="lacks the columns"):
        baselines.run(_train().drop(columns=["last_close"]), _validation(), CONFIG)
    with pytest.raises(ValueError, match="no markets"):
        baselines.run(_train(), _validation().iloc[0:0], CONFIG)


def test_summary_lines_name_the_numbers():
    text = "\n".join(baselines.summary_lines(baselines.run(_train(), _validation(), CONFIG)))
    assert "Baselines on validation (40 markets, 20 events)" in text and "fitted on train (160 markets)" in text
    assert "base rate from train: 47.5%" in text
    assert "all (40 markets, 20 events, 37.5% Yes)" in text and "contested (25 markets" in text
    assert "price_only_correction minus raw_price" in text and "middle 90% of 200 re-drawn sets of whole events" in text
    assert "0.95 and above" in text


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
    assert baselines.main(["--data-dir", str(data_dir)]) == 0
    assert asked == ["train", "validation"]
    assert (tmp_path / "outputs" / "baselines_k7.json").exists()
    out = capsys.readouterr().out
    assert "Baselines on validation" in out and "Test was not read." in out


def test_main_needs_both_split_files(tmp_path, capsys):
    assert baselines.main(["--data-dir", str(tmp_path)]) == 2
    assert "No train split" in capsys.readouterr().out
    data_dir = _split_dir(tmp_path)
    split.split_path(data_dir, CONFIG, "validation", 7).unlink()
    assert baselines.main(["--data-dir", str(data_dir)]) == 2
    assert "No validation split" in capsys.readouterr().out


def test_main_reports_a_problem_in_plain_words(tmp_path, capsys, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    path = split.split_path(data_dir, CONFIG, "train", 7)
    pd.read_parquet(path).assign(label=1).to_parquet(path, index=False)    # one outcome only
    assert baselines.main(["--data-dir", str(data_dir)]) == 1
    assert "PROBLEM the correction needs train markets with both outcomes" in capsys.readouterr().out


def test_pipeline_runs_the_baselines_on_made_up_data(capsys):
    assert pipeline.main(["--sample"]) == 0
    out = capsys.readouterr().out
    assert "Baselines on validation (" in out and "Stage 7:" in out and "The test split was not opened by this run." in out


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
        _bad(("metrics", "price_clip"), 0),
        _bad(("metrics", "price_clip"), 0.5),
        _bad(("metrics", "price_clip"), True),
        _bad(("metrics", "calibration_bands"), [0, 1]),
        _bad(("metrics", "calibration_bands"), [0.1, 0.5, 1]),
        _bad(("metrics", "calibration_bands"), [0, 0.5, 0.9]),
        _bad(("metrics", "calibration_bands"), [0, 0.6, 0.5, 1]),
        _bad(("metrics", "calibration_bands"), [0, 0.5, 0.5, 1]),
        _bad(("evaluation", "bootstrap_draws"), 0),
        _bad(("evaluation", "bootstrap_draws"), 2.5),
        _bad(("evaluation", "interval"), 0),
        _bad(("evaluation", "interval"), 1),
    ],
)
def test_bad_baseline_settings_are_refused(bad):
    with pytest.raises(ValueError):
        settings.validate_config(bad)


def test_values_on_the_edge_of_the_allowed_range_are_accepted():
    for path, value in [(("metrics", "price_clip"), 0.4999), (("metrics", "calibration_bands"), [0, 0.5, 1]),
                        (("evaluation", "bootstrap_draws"), 1), (("evaluation", "interval"), 0.999)]:
        settings.validate_config(_bad(path, value))


# ---- gaps found by the break-it run

def test_apply_correction_uses_both_numbers_and_clips_the_result():
    assert baselines.apply_correction([0.5], {"intercept": 1.0, "slope": 1.0}, EPS)[0] == pytest.approx(expit(1.0))
    assert baselines.apply_correction([0.5], {"intercept": 0.0, "slope": 3.0}, EPS)[0] == pytest.approx(0.5)
    assert baselines.apply_correction([0.5], {"intercept": -1.0, "slope": 2.0}, EPS)[0] == pytest.approx(expit(-1.0))
    out = baselines.apply_correction([0.001, 0.999], {"intercept": 0.0, "slope": 5.0}, EPS)
    assert out[0] == 0.001 and out[1] == 0.999      # a huge slope would give 1e-15 and 1 - 1e-15 without the clip


def test_a_fit_that_runs_out_of_steps_is_refused():
    price, y = levels((0.1, 10, 3), (0.9, 10, 7))
    with pytest.raises(ValueError, match="did not settle"):
        baselines.fit_correction(price, y, EPS, max_iter=1)


def test_the_range_is_the_middle_share_of_the_draws():
    draws = np.arange(101, dtype="float64")
    assert baselines._range(draws, 0.90) == pytest.approx({"low": 5.0, "high": 95.0})
    assert baselines._range(draws, 0.50) == pytest.approx({"low": 25.0, "high": 75.0})


def test_differences_are_paired_draw_by_draw():
    val = table([0.2, 0.4, 0.6, 0.1], [0, 1, 0, 0], ["A", "B", "C", "D"])
    fc = {b: baselines.clip_probability(val["last_close"].to_numpy(), EPS) for b in baselines.BASELINES}
    codes, uniques = pd.factorize(val["event_key"])
    w = baselines.resample_weights(len(uniques), 100, seed=3)
    s = baselines.score_slice(val, fc, np.ones(4, dtype=bool), w, codes, len(uniques), 0.9)
    d = s["difference_vs_raw_price"]["price_only_correction"]["log_loss"]
    assert d["low"] == d["high"] == d["point"] == 0.0 and d["verdict"] == "cannot be told apart"   # identical forecasts: zero in every draw
    assert s["wobble"]["raw_price"]["log_loss"]["low"] < s["wobble"]["raw_price"]["log_loss"]["high"]


def test_a_baseline_that_is_clearly_worse_reads_worse():
    train = table(*levels((0.1, 10, 3), (0.9, 10, 7)))
    val = table([0.01] * 10 + [0.99] * 10, [0] * 10 + [1] * 10, [f"V{i}" for i in range(20)])   # the price is right every time
    r = baselines.run(train, val, CONFIG)
    d = r["slices"]["all"]["difference_vs_raw_price"]["base_rate"]
    assert d["log_loss"]["point"] > 0 and d["log_loss"]["low"] > 0 and d["log_loss"]["verdict"] == "worse"
    assert d["brier"]["verdict"] == "worse"


def test_the_pipeline_hands_the_baselines_train_and_validation_rows_only(monkeypatch, capsys):
    seen = {}
    real = baselines.run
    monkeypatch.setattr(baselines, "run", lambda tr, va, c: (seen.update(train=tr, validation=va), real(tr, va, c))[1])
    assert pipeline.main(["--sample"]) == 0
    data = sample.write_sample(__import__("tempfile").mkdtemp(), seed=CONFIG["seed"])
    dataset, _ = build.build_dataset(data, CONFIG, 7)
    table_, _ = split.assign(dataset, CONFIG)
    expect = lambda part: set(table_[(table_["split"] == part) & ~table_["purged"]]["market_id"])  # noqa: E731
    assert set(seen["train"]["market_id"]) == expect("train")
    assert set(seen["validation"]["market_id"]) == expect("validation")
    assert not set(seen["validation"]["market_id"]) & expect("test")


def test_the_pipeline_stops_with_a_plain_message_when_the_baselines_cannot_run(monkeypatch, capsys):
    def boom(*a):
        raise ValueError("no spread in the prices")
    monkeypatch.setattr(baselines, "run", boom)
    assert pipeline.main(["--sample"]) == 1
    assert "PROBLEM baselines: no spread in the prices" in capsys.readouterr().out


# ---- Stage 7: the paired difference between two forecasts

def _pair_table():
    """4 markets, 2 events of 2. Forecast A is informative, forecast B says 0.5 everywhere. Three hand-made re-draws of the 2 events."""
    val = pd.DataFrame({"label": [1, 0, 1, 0], "event_key": ["a", "a", "b", "b"]})
    codes, uniques = pd.factorize(val["event_key"])
    weights = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    return val, codes, len(uniques), weights, np.array([0.8, 0.3, 0.6, 0.1]), np.full(4, 0.5)


def test_paired_difference_log_loss_known_answer():
    val, codes, n, weights, pa, pb = _pair_table()
    mask = np.ones(4, dtype=bool)
    # Losses of A: -ln 0.8, -ln 0.7 | -ln 0.6, -ln 0.9, so the event sums are 0.579818 and 0.616186. B costs ln 2 = 0.693147 per market.
    # Re-draws (a, b), (a, a), (b, b) give A scores 0.299001, 0.289909, 0.308093. Interval 1 gives the whole spread.
    r = baselines.paired_difference(val, pa, pb, mask, weights, codes, n, 1.0)
    assert r["point"] == pytest.approx(-0.394146, abs=1e-5)
    assert r["low"] == pytest.approx(-0.403238, abs=1e-5) and r["high"] == pytest.approx(-0.385054, abs=1e-5)
    assert r["verdict"] == "better"
    swapped = baselines.paired_difference(val, pb, pa, mask, weights, codes, n, 1.0)
    assert (swapped["point"], swapped["low"], swapped["high"]) == pytest.approx((0.394146, 0.385054, 0.403238), abs=1e-5)
    assert swapped["verdict"] == "worse"
    same = baselines.paired_difference(val, pa, pa, mask, weights, codes, n, 1.0)
    assert (same["point"], same["low"], same["high"]) == (0.0, 0.0, 0.0) and same["verdict"] == "cannot be told apart"


def test_paired_difference_brier_known_answer():
    val, codes, n, weights, pa, pb = _pair_table()
    r = baselines.paired_difference(val, pa, pb, np.ones(4, dtype=bool), weights, codes, n, 1.0, metric="brier")
    # A: squared errors 0.04, 0.09, 0.16, 0.01, mean 0.075. B: 0.25 everywhere. The re-draws of A score 0.075, 0.065, 0.085.
    assert r["point"] == pytest.approx(-0.175) and r["low"] == pytest.approx(-0.185) and r["high"] == pytest.approx(-0.165) and r["verdict"] == "better"


def test_paired_difference_uses_the_slice_only_and_skips_a_re_draw_that_misses_the_slice():
    val, codes, n, weights, pa, pb = _pair_table()
    mask = np.array([True, True, False, False])         # event a only; the third re-draw has no event a and is left out
    r = baselines.paired_difference(val, pa, pb, mask, weights, codes, n, 1.0)
    expect = (-math.log(0.8) - math.log(0.7)) / 2 - math.log(2)
    assert r["point"] == pytest.approx(expect) and r["low"] == pytest.approx(expect) and r["high"] == pytest.approx(expect)


def test_paired_difference_range_is_the_configured_middle_share():
    val, codes, n, _, pa, pb = _pair_table()
    weights = baselines.resample_weights(n, 400, 3)
    all_ = np.ones(4, dtype=bool)
    full = baselines.paired_difference(val, pa, pb, all_, weights, codes, n, 1.0)
    mid = baselines.paired_difference(val, pa, pb, all_, weights, codes, n, 0.5)
    assert full["low"] < mid["low"] <= mid["high"] < full["high"] and mid["point"] == full["point"]


def test_paired_difference_refuses_an_empty_slice_and_an_unknown_score():
    val, codes, n, weights, pa, pb = _pair_table()
    with pytest.raises(ValueError, match="no markets"):
        baselines.paired_difference(val, pa, pb, np.zeros(4, dtype=bool), weights, codes, n, 0.9)
    with pytest.raises(ValueError, match="metric"):
        baselines.paired_difference(val, pa, pb, np.ones(4, dtype=bool), weights, codes, n, 0.9, metric="accuracy")


def test_paired_difference_against_the_raw_price_agrees_with_score_slice():
    rng = np.random.default_rng(5)
    n = 60
    price = rng.uniform(0.05, 0.95, n)
    val = table(price, (rng.random(n) < price).astype(int), [f"E{i // 3}" for i in range(n)])
    other = baselines.clip_probability(np.clip(price + rng.normal(0, 0.1, n), 0.01, 0.99), EPS)
    raw = baselines.clip_probability(price, EPS)
    codes, uniques = pd.factorize(val["event_key"])
    weights = baselines.resample_weights(len(uniques), 50, 1)
    mask = baselines.slice_masks(val, CONFIG)["contested"]
    s = baselines.score_slice(val, {"raw_price": raw, "other": other}, mask, weights, codes, len(uniques), 0.9, names=["raw_price", "other"])
    d = baselines.paired_difference(val, other, raw, mask, weights, codes, len(uniques), 0.9)
    expect = s["difference_vs_raw_price"]["other"]["log_loss"]
    assert (d["point"], d["low"], d["high"], d["verdict"]) == (expect["point"], expect["low"], expect["high"], expect["verdict"])
