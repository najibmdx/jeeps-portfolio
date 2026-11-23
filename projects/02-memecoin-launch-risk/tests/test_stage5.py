"""Tests for Stage 5: time-ordered tuning, tuned run, and column-group tests."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

import ablation
import piles
import settings
import tune
import tuning


@pytest.fixture
def config():
    cfg = settings.load_config()
    # tiny grids so the tests stay fast; the real grids are checked separately
    cfg["tuning"]["grids"] = {"logistic_regression": {"C": [0.1, 1.0]},
                              "random_forest": {"max_depth": [3, None]},
                              "gradient_boosting": {"learning_rate": [0.05, 0.1]}}
    cfg["models"]["random_forest"]["n_estimators"] = 20
    cfg["models"]["gradient_boosting"]["max_iter"] = 20
    cfg["evaluation"]["bootstrap_draws"] = 10
    return cfg


def make_pile(n, seed, flip=False):
    rng = np.random.default_rng(seed)
    crash = rng.integers(0, 2, n)
    sig = 1.5 * crash
    return pd.DataFrame({
        "mint_address": [f"M{i}" for i in range(n)], "mint_ts": np.arange(n),
        "migration_time": np.arange(n) * 100.0, "ts_len": 360, "crash": 1 - crash if flip else crash,
        "group1_migrate_year": 2025 + sig, "group1_migrate_month": 1 + sig, "group1_other": rng.normal(size=n),
        "group2_a": rng.normal(size=n) + sig, "group3_b": rng.normal(size=n),
        "group4_d": rng.normal(size=n),
    })


@pytest.fixture
def data_dir(tmp_path, config):
    folder = tmp_path / config["data"]["prepared_folder"]
    folder.mkdir()
    for name, seed in (("train", 1), ("validation", 2), ("test", 3)):
        make_pile(600, seed).to_csv(folder / f"{name}_features.csv", index=False)
    return tmp_path


# ---- config ----
def test_real_config_values():
    c = settings.load_config()
    assert c["tuning"]["n_blocks"] == 4
    assert set(c["tuning"]["grids"]) == set(c["models"])
    assert c["ablation"]["never_use"] == ["group1_migrate_year", "group1_migrate_month"]
    assert list(c["ablation"]["sets"])[0] == "reference" and c["ablation"]["sets"]["reference"] == []
    assert c["ablation"]["model"] in c["models"]


# ---- folds ----
def test_folds_grade_only_on_later_coins():
    t = np.arange(1000) * 10.0
    folds = tuning.make_folds(t, 4, 0)
    assert len(folds) == 3
    for train_idx, grade_idx in folds:
        assert t[train_idx].max() < t[grade_idx].min()
        assert len(set(train_idx) & set(grade_idx)) == 0


def test_folds_are_expanding_blocks():
    folds = tuning.make_folds(np.arange(1000.0), 4, 0)
    assert [len(f[0]) for f in folds] == [250, 500, 750]
    assert [len(f[1]) for f in folds] == [250, 250, 250]


def test_folds_drop_training_coins_whose_label_was_not_known_yet():
    t = np.arange(1000) * 100.0          # one coin every 100 s; a coin exactly 1200 s before is kept, the 11 after it go
    plain = tuning.make_folds(t, 4, 0)
    purged = tuning.make_folds(t, 4, 1200)
    for (p_train, _), (q_train, grade) in zip(plain, purged):
        start = t[grade[0]]
        assert t[q_train].max() <= start - 1200
        assert len(p_train) - len(q_train) == 11


def test_folds_refuse_unsorted_pile():
    with pytest.raises(ValueError, match="sorted"):
        tuning.make_folds(np.array([3.0, 1.0, 2.0, 4.0]), 2, 0)


def test_time_ordered_sorts_stably():
    df = pd.DataFrame({"migration_time": [2, 1, 1, 3], "k": list("abcd")})
    assert list(tuning.time_ordered(df)["k"]) == ["b", "c", "a", "d"]


def test_grid_points_all_combinations_in_order():
    pts = tuning.grid_points({"a": [1, 2], "b": [None, 5]})
    assert pts == [{"a": 1, "b": None}, {"a": 1, "b": 5}, {"a": 2, "b": None}, {"a": 2, "b": 5}]


# ---- scoring ----
def test_fold_score_perfect_and_lazy():
    y = np.array([1, 1, 1, 0])
    perfect = tuning.fold_score(y, [0.9, 0.8, 0.7, 0.1])
    assert perfect["crash"] == pytest.approx(1 / 0.75) and perfect["survivor"] == pytest.approx(4.0)
    assert perfect["average"] == pytest.approx((1 / 0.75 + 4.0) / 2)
    lazy = tuning.fold_score(y, np.full(4, 0.5))
    assert lazy["crash"] == pytest.approx(1.0) and lazy["survivor"] == pytest.approx(1.0)


def test_fold_score_needs_both_sides():
    with pytest.raises(ValueError):
        tuning.fold_score(np.ones(4), np.random.rand(4))


def test_fold_score_is_not_rounded():
    y = np.array([1, 0, 1, 0, 0, 1, 1])
    s = np.array([0.91, 0.52, 0.33, 0.78, 0.12, 0.66, 0.45])
    got = tuning.fold_score(y, s)["crash"]
    assert got != round(got, 2)


# ---- tune_model ----
def test_tune_model_picks_best_and_earliest_on_ties(config):
    p = tuning.time_ordered(make_pile(600, 1))
    cols = ["group2_a", "group3_b"]
    folds = tuning.make_folds(p["migration_time"], 4, 0)
    out = tuning.tune_model("logistic_regression", config["models"]["logistic_regression"], {"C": [1.0, 1.0]},
                            p[cols], p["crash"], folds, 42)
    assert out["best"] == {"C": 1.0}
    assert out["trials"][0]["score"] == out["trials"][1]["score"] == out["best_score"]
    out2 = tuning.tune_model("logistic_regression", config["models"]["logistic_regression"], {"C": [1e-9, 1.0]},
                             p[cols], p["crash"], folds, 42)
    assert out2["best"] == {"C": 1.0}                 # a crippled setting must lose
    assert max(t["score"] for t in out2["trials"]) == out2["best_score"]


def test_tune_model_is_repeatable(config):
    p = tuning.time_ordered(make_pile(600, 1))
    cols = ["group2_a", "group3_b"]
    folds = tuning.make_folds(p["migration_time"], 4, 0)
    a = tuning.tune_model("random_forest", config["models"]["random_forest"], {"max_depth": [3, None]}, p[cols], p["crash"], folds, 42)
    b = tuning.tune_model("random_forest", config["models"]["random_forest"], {"max_depth": [3, None]}, p[cols], p["crash"], folds, 42)
    assert a == b


def test_tuning_never_sees_validation(config, data_dir, monkeypatch):
    seen = {}
    real = tuning.tune_model

    def spy(name, base, grid, X, y, folds, seed):
        seen["rows"] = len(X)
        return real(name, base, grid, X, y, folds, seed)

    monkeypatch.setattr(tuning, "tune_model", spy)
    tune.run_tuning(config, piles.load_piles(data_dir, config)["train"])
    assert seen["rows"] == 600       # the train pile only


# ---- tune main ----
def test_tune_main_saves_everything(config, data_dir, tmp_path):
    out, params = tmp_path / "out", tmp_path / "tuned.yaml"
    assert tune.main(config=config, data_dir=data_dir, out_dir=out, params_path=params) == 0
    for f in ("tuning.json", "tuned_metrics.json", "tuned_wobble.json"):
        assert (out / f).exists()
    saved = yaml.safe_load(params.read_text())
    assert set(saved["models"]) == set(config["models"])
    assert saved["feature_groups"] == ["group2", "group3"]
    assert "test" not in json.loads((out / "tuned_metrics.json").read_text())["baselines"]


def test_saved_params_are_base_plus_best(config, data_dir, tmp_path):
    tuned = tune.run_tuning(config, piles.load_piles(data_dir, config)["train"])
    path = tmp_path / "p.yaml"
    tune.save_params(tuned, config, path)
    saved = yaml.safe_load(path.read_text())["models"]
    assert saved["random_forest"]["n_estimators"] == 20            # base setting kept
    assert saved["random_forest"]["max_depth"] == tuned["random_forest"]["best"]["max_depth"]


def test_tune_main_stops_without_data(config, tmp_path):
    (tmp_path / config["data"]["prepared_folder"]).mkdir()
    assert tune.main(config=config, data_dir=tmp_path, out_dir=tmp_path / "o", params_path=tmp_path / "p.yaml") == 1
    assert not (tmp_path / "p.yaml").exists()


# ---- feature selection with groups and drop ----
def test_select_features_groups_and_drop(config):
    cols = list(make_pile(10, 0).columns)
    got = piles.select_features(cols, config, groups=["group2", "group3", "group1"],
                                drop=["group1_migrate_year", "group1_migrate_month"])
    assert got == ["group1_other", "group2_a", "group3_b"]


def test_select_features_drop_typo_raises(config):
    with pytest.raises(ValueError, match="do not exist"):
        piles.select_features(list(make_pile(10, 0).columns), config, groups=["group2"], drop=["group1_migrate_yaer"])


# ---- ablation ----
def test_column_sets_never_include_calendar_columns_or_banned(config):
    sets = ablation.column_sets(make_pile(10, 0).columns, config)
    assert list(sets)[0] == "reference" and sets["reference"] == ["group2_a", "group3_b"]
    for cols in sets.values():
        assert "group1_migrate_year" not in cols and "group1_migrate_month" not in cols
    assert "group1_other" in sets["plus_group1"] and "group4_d" in sets["plus_group4"]
    assert "group4_d" in sets["plus_group1_and_group4"] and "group1_other" in sets["plus_group1_and_group4"]


def test_flags_only_clear_gains():
    w = {"difference_crash": {"a": {"low": 0.01, "median": 0.02, "high": 0.03}, "b": {"low": -0.01, "median": 0.0, "high": 0.01}},
         "difference_survivor": {"a": {"low": -0.01, "median": 0, "high": 0.01}, "b": {"low": 0.002, "median": 0.01, "high": 0.02}}}
    f = ablation.flags(w)
    assert set(f) == {"a", "b"} and "crash" in f["a"] and "survivor" in f["b"] and "leakage" in f["a"]
    w["difference_survivor"]["b"]["low"] = -0.001
    assert set(ablation.flags(w)) == {"a"}


def test_ablation_run_and_main(config, data_dir, tmp_path):
    out, params = tmp_path / "out", tmp_path / "tuned.yaml"
    assert tune.main(config=config, data_dir=data_dir, out_dir=out, params_path=params) == 0
    assert ablation.main(config=config, data_dir=data_dir, out_dir=out, params_path=params) == 0
    saved = json.loads((out / "ablation.json").read_text())
    assert list(saved["sets"]) == list(config["ablation"]["sets"])
    assert saved["sets"]["reference"]["n_columns"] == 2


def test_ablation_stops_without_tuned_params(config, data_dir, tmp_path):
    assert ablation.main(config=config, data_dir=data_dir, out_dir=tmp_path / "o", params_path=tmp_path / "nope.yaml") == 1


def test_ablation_detects_a_helping_group(config, data_dir, tmp_path):
    # make group4 carry the answer; it must be flagged as helping
    for name in ("train", "validation"):
        path = data_dir / config["data"]["prepared_folder"] / f"{name}_features.csv"
        df = pd.read_csv(path)
        df["group4_d"] = df["crash"] * 3.0 + np.random.default_rng(0).normal(size=len(df)) * 0.1
        df["group2_a"] = np.random.default_rng(1).normal(size=len(df))       # reference now has no signal
        df.to_csv(path, index=False)
    config["evaluation"]["bootstrap_draws"] = 40
    params = tmp_path / "t.yaml"
    assert tune.main(config=config, data_dir=data_dir, out_dir=tmp_path / "o", params_path=params) == 0
    result = ablation.run(config, piles.load_piles(data_dir, config), ablation.load_tuned_params(params))
    assert "plus_group4" in result["flags"]


def test_tune_model_ties_keep_the_earliest_grid_point(config):
    # both max_iter values converge to the same model, so the scores are identical: the first one must win
    p = tuning.time_ordered(make_pile(600, 1))
    cols = ["group2_a", "group3_b"]
    folds = tuning.make_folds(p["migration_time"], 4, 0)
    out = tuning.tune_model("logistic_regression", config["models"]["logistic_regression"], {"max_iter": [2000, 3000]},
                            p[cols], p["crash"], folds, 42)
    assert out["trials"][0]["score"] == out["trials"][1]["score"]
    assert out["best"] == {"max_iter": 2000}


def test_tuned_models_config_uses_the_best_settings(config):
    tuned = {"logistic_regression": {"best": {"C": 7.0}}, "random_forest": {"best": {"max_depth": 5}},
             "gradient_boosting": {"best": {"learning_rate": 0.2}}}
    merged = tune.tuned_models_config(config, tuned)["models"]
    assert merged["logistic_regression"]["C"] == 7.0 and merged["logistic_regression"]["max_iter"] == 2000
    assert merged["random_forest"]["max_depth"] == 5 and merged["random_forest"]["n_estimators"] == 20
    assert merged["gradient_boosting"]["learning_rate"] == 0.2
    assert "C" not in config["models"]["logistic_regression"]        # the original config is untouched
