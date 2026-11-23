"""Tests for Stage 4: piles loader, metrics, models, training, wobble."""

import json

import numpy as np
import pandas as pd
import pytest

import metrics
import models
import piles
import settings
import train
import uncertainty


@pytest.fixture
def config():
    return settings.load_config()


def make_pile(n, seed, signal=True):
    """Made-up pile: group2_a carries signal, group3_b is noise, group1_c / group4_d must stay unused."""
    rng = np.random.default_rng(seed)
    crash = rng.integers(0, 2, n)
    a = rng.normal(size=n) + (1.5 * crash if signal else 0)
    return pd.DataFrame({
        "mint_address": [f"M{i}" for i in range(n)], "mint_ts": np.arange(n), "migration_time": np.arange(n),
        "ts_len": 360, "crash": crash,
        "group1_c": rng.normal(size=n), "group2_a": a, "group3_b": rng.normal(size=n),
        "group4_d": rng.normal(size=n),
    })


@pytest.fixture
def data_dir(tmp_path, config):
    folder = tmp_path / config["data"]["prepared_folder"]
    folder.mkdir()
    for name, seed in (("train", 1), ("validation", 2), ("test", 3)):
        make_pile(400, seed).to_csv(folder / f"{name}_features.csv", index=False)
    return tmp_path


# ---- config ----
def test_config_values(config):
    assert config["features"]["groups"] == ["group2", "group3"]
    assert config["evaluation"]["top_fraction"] == 0.10
    assert config["evaluation"]["bootstrap_draws"] == 200
    assert config["evaluation"]["reference_model"] in config["models"]
    assert set(config["models"]) == set(models.NAMES)


# ---- feature selection ----
def test_select_features_only_chosen_groups(config):
    cols = list(make_pile(10, 0).columns)
    assert piles.select_features(cols, config) == ["group2_a", "group3_b"]


def test_select_features_errors_when_group_has_no_columns(config):
    with pytest.raises(ValueError, match="group3"):
        piles.select_features(["group2_a", "crash", "mint_ts"], config)


def test_select_features_never_returns_banned(config):
    cfg = {**config, "columns": {**config["columns"], "banned_as_features": ["group2_a"]}}
    with pytest.raises(ValueError, match="group2"):      # its only column is banned, so the group is empty
        piles.select_features(list(make_pile(10, 0).columns), cfg)


# ---- piles loader ----
def test_load_piles_returns_train_and_validation(config, data_dir):
    got = piles.load_piles(data_dir, config)
    assert set(got) == {"train", "validation"}


def test_load_piles_refuses_test(config, data_dir):
    with pytest.raises(ValueError, match="test"):
        piles.load_piles(data_dir, config, parts=("train", "test"))


def test_load_piles_never_opens_test_file(config, data_dir, monkeypatch):
    import prepare
    opened = []
    real = prepare.load_features
    monkeypatch.setattr(prepare, "load_features", lambda p: (opened.append(str(p)), real(p))[1])
    piles.load_piles(data_dir, config)
    assert opened and not any(o.endswith("test_features.csv") for o in opened)


def test_load_piles_missing_file(config, tmp_path):
    (tmp_path / config["data"]["prepared_folder"]).mkdir()
    with pytest.raises(FileNotFoundError, match="prepare.py"):
        piles.load_piles(tmp_path, config)


def test_load_piles_rejects_bad_crash_column(config, data_dir):
    path = data_dir / config["data"]["prepared_folder"] / "train_features.csv"
    df = pd.read_csv(path)
    df.loc[0, "crash"] = 2
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="crash"):
        piles.load_piles(data_dir, config)


# ---- metrics: hand-counted ----
def test_precision_at_fraction_hand_count():
    y = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    assert metrics.precision_at_fraction(y, s, 0.2) == 0.5      # top 2: 1,0
    assert metrics.precision_at_fraction(y, s, 0.3) == pytest.approx(2 / 3)  # top 3: 1,0,1


def test_precision_tie_break_is_row_order():
    y = [1, 0, 0, 0]
    assert metrics.precision_at_fraction(y, [0.5, 0.5, 0.5, 0.5], 0.25) == 1.0   # first row wins
    assert metrics.precision_at_fraction([0, 1, 0, 0], [0.5] * 4, 0.25) == 0.0


def test_precision_takes_at_least_one_coin():
    assert metrics.precision_at_fraction([1, 0, 0], [0.9, 0.1, 0.0], 0.01) == 1.0


def test_pr_auc_perfect_and_constant():
    y = np.array([1, 1, 0, 0, 0, 0])
    assert metrics.pr_auc(y, [0.9, 0.8, 0.3, 0.2, 0.1, 0.0]) == 1.0
    assert metrics.pr_auc(y, [0.5] * 6) == pytest.approx(2 / 6)   # no information = prevalence


def test_side_metrics_baselines_and_survivor_flip():
    y = np.array([1, 1, 1, 0])           # 75% crash, 25% survivor
    perfect = np.array([0.9, 0.8, 0.7, 0.1])
    m = metrics.side_metrics(y, perfect, 0.25)
    assert m["crash"]["baseline"] == 0.75 and m["survivor"]["baseline"] == 0.25
    assert m["crash"]["pr_auc"] == 1.0 and m["survivor"]["pr_auc"] == 1.0
    assert m["survivor"]["precision_top"] == 1.0 and m["survivor"]["lift_top"] == 4.0
    worst = metrics.side_metrics(y, 1 - perfect, 0.25)
    assert worst["survivor"]["precision_top"] == 0.0


def test_side_metrics_constant_scores_equal_baseline():
    y = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    m = metrics.side_metrics(y, np.full(8, 0.3), 0.5)
    for side in ("crash", "survivor"):
        assert m[side]["pr_auc"] == m[side]["baseline"]
        assert m[side]["pr_auc_times_baseline"] == 1.0


def test_side_metrics_needs_both_sides():
    with pytest.raises(ValueError):
        metrics.side_metrics(np.ones(5), np.random.rand(5), 0.1)


# ---- models ----
def test_build_model_each_name(config):
    for name in models.NAMES:
        assert hasattr(models.build_model(name, config["models"][name], 42), "fit")


def test_build_model_unknown():
    with pytest.raises(ValueError, match="unknown"):
        models.build_model("magic", {}, 1)


def test_scaler_is_fit_on_train_only(config):
    m = models.build_model("logistic_regression", config["models"]["logistic_regression"], 42)
    train_df = make_pile(200, 1)[["group2_a", "group3_b"]]
    other = make_pile(200, 2)[["group2_a", "group3_b"]] + 100
    m.fit(train_df, make_pile(200, 1)["crash"])
    np.testing.assert_allclose(m.named_steps["scale"].mean_, train_df.mean().to_numpy())
    m.predict_proba(other)       # predicting must not change the scaler
    np.testing.assert_allclose(m.named_steps["scale"].mean_, train_df.mean().to_numpy())


# ---- training ----
def test_fit_models_uses_only_chosen_columns(config, data_dir):
    got = train.fit_models(config, piles.load_piles(data_dir, config))
    assert got["columns"] == ["group2_a", "group3_b"]
    assert set(got["scores"]) == set(models.NAMES)
    for s in got["scores"].values():
        assert len(s["validation"]) == 400 and ((s["validation"] >= 0) & (s["validation"] <= 1)).all()


def test_models_beat_lazy_guess_on_signal(config, data_dir):
    result = train.run(config, piles.load_piles(data_dir, config))
    for m in result["models"].values():
        assert m["validation"]["crash"]["pr_auc_times_baseline"] > 1.2


def test_run_is_repeatable(config, data_dir):
    a = train.run(config, piles.load_piles(data_dir, config))
    b = train.run(config, piles.load_piles(data_dir, config))
    assert a == b


def test_main_saves_metrics_and_leaves_test_alone(config, data_dir, tmp_path, capsys):
    out = tmp_path / "out"
    assert train.main(config=config, data_dir=data_dir, out_dir=out) == 0
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["n_features"] == 2 and "test" not in saved["baselines"]
    assert "test_features" not in capsys.readouterr().out


def test_main_stops_without_data(config, tmp_path):
    (tmp_path / config["data"]["prepared_folder"]).mkdir()
    assert train.main(config=config, data_dir=tmp_path, out_dir=tmp_path / "o") == 1
    assert not (tmp_path / "o").exists()


# ---- wobble ----
def test_bootstrap_same_exam_for_all_and_repeatable():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = y + rng.normal(0, 0.8, 300)
    scores = {"ref": good, "same": good.copy(), "noise": rng.random(300)}
    a = uncertainty.bootstrap(y, scores, 50, 7, "ref")
    assert a == uncertainty.bootstrap(y, scores, 50, 7, "ref")
    assert a["difference_crash"]["same"] == {"low": 0.0, "median": 0.0, "high": 0.0}
    assert a["difference_crash"]["noise"]["high"] < 0


def test_bootstrap_range_ordered_and_covers_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = {"ref": y + rng.normal(0, 1, 400)}
    r = uncertainty.bootstrap(y, s, 100, 3)
    c = r["pr_auc_crash"]["ref"]
    assert c["low"] <= c["median"] <= c["high"]
    assert c["low"] < metrics.pr_auc(y, s["ref"]) < c["high"]


def test_verdict_words():
    assert "better" in uncertainty.verdict({"low": 0.01, "median": 0.02, "high": 0.03})
    assert "worse" in uncertainty.verdict({"low": -0.03, "median": -0.02, "high": -0.01})
    assert "cannot" in uncertainty.verdict({"low": -0.01, "median": 0.0, "high": 0.02})


def test_uncertainty_main_saves(config, data_dir, tmp_path):
    cfg = {**config, "evaluation": {**config["evaluation"], "bootstrap_draws": 10}}
    assert uncertainty.main(config=cfg, data_dir=data_dir, out_dir=tmp_path / "o") == 0
    assert (tmp_path / "o" / "wobble.json").exists()


def test_tie_break_is_row_order_on_a_big_pile():
    # 1000 tied scores: only a stable sort reliably picks the first 100 rows (all crashes here)
    y = np.array([1] * 100 + [0] * 900)
    assert metrics.precision_at_fraction(y, np.full(1000, 0.5), 0.10) == 1.0


def test_models_learn_from_train_not_validation(config, data_dir):
    # validation has the OPPOSITE relationship to train; a model trained on train must do worse than the lazy guess there
    path = data_dir / config["data"]["prepared_folder"] / "validation_features.csv"
    df = pd.read_csv(path)
    df["crash"] = 1 - df["crash"]
    df.to_csv(path, index=False)
    result = train.run(config, piles.load_piles(data_dir, config))
    for name, m in result["models"].items():
        assert m["train"]["crash"]["pr_auc_times_baseline"] > 1.2, name
        assert m["validation"]["crash"]["pr_auc_times_baseline"] < 1.0, name


def test_tie_break_matches_row_order_reference_with_many_ties():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 5000)
    s = rng.integers(0, 3, 5000) / 2          # only three distinct scores: huge ties
    order = sorted(range(5000), key=lambda i: (-s[i], i))[:500]
    assert metrics.precision_at_fraction(y, s, 0.10) == y[order].mean()
