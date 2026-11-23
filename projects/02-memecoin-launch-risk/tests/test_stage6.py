"""Tests for Stage 6: cut-offs and the two mistakes, the frozen choice, and the test pile that opens once."""

import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import decision
import final_test
import holdout
import piles
import select_final
import settings
import train
import uncertainty

SRC = Path(settings.PROJECT_DIR) / "src"


@pytest.fixture
def config():
    cfg = settings.load_config()
    cfg["models"]["random_forest"]["n_estimators"] = 20
    cfg["models"]["gradient_boosting"]["max_iter"] = 20
    cfg["evaluation"]["bootstrap_draws"] = 10
    return cfg


def make_pile(n, seed):
    rng = np.random.default_rng(seed)
    crash = rng.integers(0, 2, n)
    sig = 1.0 * crash
    return pd.DataFrame({
        "mint_address": [f"M{i}" for i in range(n)], "mint_ts": np.arange(n),
        "migration_time": np.arange(n) * 100.0, "ts_len": 360, "crash": crash,
        "group1_migrate_year": 2025 + sig, "group2_a": rng.normal(size=n) + sig,
        "group2_b": rng.normal(size=n), "group3_b": rng.normal(size=n) + 0.5 * sig,
        "group4_d": rng.normal(size=n),
    })


@pytest.fixture
def workspace(tmp_path, config):
    """data dir with three piles, a tuned_params file, and paths for outputs and the frozen file."""
    folder = tmp_path / config["data"]["prepared_folder"]
    folder.mkdir()
    for name, seed in (("train", 1), ("validation", 2), ("test", 3)):
        make_pile(600, seed).to_csv(folder / f"{name}_features.csv", index=False)
    params = tmp_path / "tuned_params.yaml"
    params.write_text(yaml.safe_dump({"note": "test", "feature_groups": config["features"]["groups"],
                                      "models": config["models"]}, sort_keys=False), encoding="utf-8")
    return {"data_dir": tmp_path, "params_path": params, "frozen_path": tmp_path / "frozen_choice.yaml",
            "out_dir": tmp_path / "out"}


def run_select(config, ws):
    return select_final.main([], config=config, data_dir=ws["data_dir"], out_dir=ws["out_dir"],
                             params_path=ws["params_path"], frozen_path=ws["frozen_path"])


def run_final(config, ws, *flags):
    return final_test.main(list(flags), config=config, data_dir=ws["data_dir"], out_dir=ws["out_dir"],
                           params_path=ws["params_path"], frozen_path=ws["frozen_path"])


# ---------------------------------------------------------------- config
def test_real_config_final_section():
    c = settings.load_config()
    f = c["final"]
    assert f["selection"] == "average_pr_auc"
    assert set(f["scenarios"]) == {"equal", "crasher_costs_3x", "skipped_good_costs_3x"}
    assert f["scenarios"]["equal"] == {"holding_crasher": 1, "skipped_good": 1}
    assert f["scenarios"]["crasher_costs_3x"] == {"holding_crasher": 3, "skipped_good": 1}
    assert f["scenarios"]["skipped_good_costs_3x"] == {"holding_crasher": 1, "skipped_good": 3}
    assert set(f["candidates"]) == set(c["models"])


def test_config_file_has_no_duplicate_top_level_keys():
    """Two `tuning:` blocks once sat in the file and the later one silently won."""
    keys = re.findall(r"^([A-Za-z_]+):", (settings.PROJECT_DIR / "config.yaml").read_text(encoding="utf-8"), flags=re.M)
    assert len(keys) == len(set(keys)), sorted({k for k in keys if keys.count(k) > 1})


@pytest.mark.parametrize("change", [
    lambda f: f.update(selection="best_vibes"),
    lambda f: f.update(candidates=["not_a_model"]),
    lambda f: f.update(candidates=[]),
    lambda f: f.update(scenarios={}),
    lambda f: f["scenarios"]["equal"].update(skipped_good=0),
    lambda f: f["scenarios"]["equal"].pop("skipped_good"),
    lambda f: f.update(grid_shares=[0.5, 1.5]),
])
def test_bad_final_settings_are_refused(change):
    cfg = copy.deepcopy(settings.load_config())
    change(cfg["final"])
    with pytest.raises(ValueError):
        settings.validate_config(cfg)


def test_real_tuned_params_fit_the_real_config():
    cfg = settings.load_config()
    tuned = select_final.tuned_config(cfg, settings.PROJECT_DIR / cfg["tuning"]["tuned_params_file"])
    assert set(tuned["models"]) == set(cfg["models"])


def test_tuned_params_for_other_groups_are_refused(config, workspace):
    workspace["params_path"].write_text(yaml.safe_dump({"feature_groups": ["group2"], "models": config["models"]}))
    with pytest.raises(ValueError, match="feature groups"):
        select_final.tuned_config(config, workspace["params_path"])


# ---------------------------------------------------------------- decision: counting
Y = np.array([1, 1, 0, 0, 1, 0])
S = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])


def test_counts_by_hand():
    c = decision.counts_at(Y, S, 0.5)          # flags the coins scoring 0.9, 0.8, 0.7
    assert c == {"n": 6, "crashers": 3, "survivors": 3, "flagged": 3, "flagged_crash": 2,
                 "skipped_good": 1, "held_crasher": 1, "kept_survivor": 2}


def test_rates_by_hand_on_both_sides():
    r = decision.rates(decision.counts_at(Y, S, 0.5))
    assert r["flag_share"] == 0.5
    assert r["crash_precision"] == round(2 / 3, 4) and r["crash_recall"] == round(2 / 3, 4)
    assert r["survivor_precision"] == round(2 / 3, 4) and r["survivor_recall"] == round(2 / 3, 4)


def test_flag_is_at_or_above_the_cutoff():
    assert decision.counts_at(Y, S, 0.7)["flagged"] == 3      # the coin scoring exactly 0.7 IS flagged
    assert decision.counts_at(Y, S, 0.7000001)["flagged"] == 2


def test_the_two_lazy_options():
    nothing, everything = decision.flag_nothing(Y), decision.flag_everything(Y)
    assert (nothing["flagged"], nothing["held_crasher"], nothing["skipped_good"]) == (0, 3, 0)
    assert (everything["flagged"], everything["held_crasher"], everything["skipped_good"]) == (6, 0, 3)


def test_rates_are_none_when_nothing_to_divide_by():
    r = decision.rates(decision.flag_nothing(Y))
    assert r["crash_precision"] is None and r["crash_recall"] == 0.0
    assert r["survivor_precision"] == 0.5 and r["survivor_recall"] == 1.0
    r = decision.rates(decision.flag_everything(Y))
    assert r["survivor_precision"] is None and r["crash_recall"] == 1.0


def test_cost_weights_each_mistake():
    c = decision.counts_at(Y, S, 0.5)          # one of each mistake
    assert decision.cost(c, {"holding_crasher": 1, "skipped_good": 1}) == 2
    assert decision.cost(c, {"holding_crasher": 3, "skipped_good": 1}) == 4
    assert decision.cost(c, {"holding_crasher": 1, "skipped_good": 3}) == 4
    assert decision.cost_per_coin(c, {"holding_crasher": 3, "skipped_good": 1}) == pytest.approx(4 / 6)
    with pytest.raises(ValueError):
        decision.cost(c, {"holding_crasher": 1})


def test_best_lazy_cost_is_the_cheaper_option():
    sc = {"holding_crasher": 1, "skipped_good": 1}
    assert decision.best_lazy_cost(Y, sc) == 3
    assert decision.best_lazy_cost(np.array([1, 1, 1, 0]), sc) == 1       # flag everything skips 1 good coin
    assert decision.best_lazy_cost(np.array([1, 0, 0, 0]), sc) == 1       # flag nothing holds 1 crasher


# ---------------------------------------------------------------- decision: cut-off choice
def random_case(seed, n=300, levels=40):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = np.round(rng.random(n) * 0.6 + 0.3 * y, 2)           # rounding makes many ties on purpose
    return y, s


@pytest.mark.parametrize("seed", range(5))
def test_curve_agrees_with_counting_at_every_cutoff(seed):
    y, s = random_case(seed)
    c = decision.curve(y, s)
    for i, cutoff in enumerate(c["cutoff"]):
        counts = decision.counts_at(y, s, cutoff)
        assert counts["flagged"] == c["flagged"][i]
        assert counts["skipped_good"] == c["skipped_good"][i]
        assert counts["held_crasher"] == c["held_crasher"][i]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("scenario", [{"holding_crasher": 1, "skipped_good": 1},
                                      {"holding_crasher": 3, "skipped_good": 1},
                                      {"holding_crasher": 1, "skipped_good": 3}])
def test_best_cutoff_is_the_true_minimum(seed, scenario):
    y, s = random_case(seed)
    best = decision.best_cutoff(y, s, scenario)
    candidates = [np.inf] + sorted(set(s.tolist()))
    brute = min(decision.cost(decision.counts_at(y, s, c), scenario) for c in candidates)
    assert decision.cost(best["counts"], scenario) == brute
    assert best["counts"] == decision.counts_at(y, s, best["cutoff"])


def test_a_tie_goes_to_the_higher_cutoff():
    y, s = np.array([1, 0]), np.array([0.1, 0.9])      # every cut-off costs 1: flag nothing wins the tie
    best = decision.best_cutoff(y, s, {"holding_crasher": 1, "skipped_good": 1})
    assert best["cutoff"] == np.inf and best["counts"]["flagged"] == 0


def test_a_perfect_scorer_gets_zero_cost():
    y = np.array([1, 1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    best = decision.best_cutoff(y, s, {"holding_crasher": 1, "skipped_good": 1})
    assert best["cutoff"] == 0.7
    assert (best["counts"]["skipped_good"], best["counts"]["held_crasher"]) == (0, 0)


@pytest.mark.parametrize("seed", range(5))
def test_pricier_crashers_flag_more_and_pricier_good_coins_flag_fewer(seed):
    y, s = random_case(seed)
    flagged = {name: decision.best_cutoff(y, s, sc)["counts"]["flagged"] for name, sc in
               settings.load_config()["final"]["scenarios"].items()}
    assert flagged["skipped_good_costs_3x"] <= flagged["equal"] <= flagged["crasher_costs_3x"]


def test_curve_needs_both_sides():
    with pytest.raises(ValueError):
        decision.curve(np.ones(5, dtype=int), np.arange(5.0))


def test_cutoff_at_flag_share_flags_about_that_share():
    s = np.linspace(0, 1, 100)
    cut = decision.cutoff_at_flag_share(s, 0.3)
    assert decision.counts_at(np.zeros(100, dtype=int), s, cut)["flagged"] == 30


def test_describe_shows_both_mistakes():
    line = decision.describe("chosen", 0.5, decision.counts_at(Y, S, 0.5))
    assert "skipped good     1" in line and "held crasher     1" in line
    assert "(none)" in decision.describe("flag nothing", np.inf, decision.flag_nothing(Y))


# ---------------------------------------------------------------- the test pile opens once
def test_modelling_loader_still_refuses_the_test_pile(config, workspace):
    with pytest.raises(ValueError):
        piles.load_piles(workspace["data_dir"], config, parts=("train", "test"))


def test_holdout_refuses_without_the_confirm_flag(config, workspace):
    workspace["frozen_path"].write_text("x")
    with pytest.raises(PermissionError):
        holdout.open_test_pile(workspace["data_dir"], config, workspace["frozen_path"])
    assert not holdout.is_locked(workspace["data_dir"], config)


def test_holdout_refuses_without_a_frozen_file(config, workspace):
    with pytest.raises(FileNotFoundError, match="Run select_final.py"):
        holdout.open_test_pile(workspace["data_dir"], config, workspace["frozen_path"], confirm=True)
    assert not holdout.is_locked(workspace["data_dir"], config)


def test_holdout_opens_once_and_a_second_call_is_refused(config, workspace):
    workspace["frozen_path"].write_text("x")
    pile = holdout.open_test_pile(workspace["data_dir"], config, workspace["frozen_path"], confirm=True)
    assert len(pile) == 600 and holdout.is_locked(workspace["data_dir"], config)
    record = json.loads(holdout.lock_path(workspace["data_dir"], config).read_text())
    assert record["frozen_file"] == "frozen_choice.yaml" and len(record["frozen_sha256"]) == 64
    with pytest.raises(FileExistsError, match="already opened"):
        holdout.open_test_pile(workspace["data_dir"], config, workspace["frozen_path"], confirm=True)


def test_two_runs_racing_cannot_both_get_in(config, workspace, monkeypatch):
    """Even if both runs passed the exists() check at the same moment, creating the lock in 'x' mode lets only one in."""
    workspace["frozen_path"].write_text("x")
    holdout.open_test_pile(workspace["data_dir"], config, workspace["frozen_path"], confirm=True)
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: False if self.name == config["final"]["test_lock_file"] else real_exists(self))
    with pytest.raises(FileExistsError):
        holdout.open_test_pile(workspace["data_dir"], config, workspace["frozen_path"], confirm=True)


def test_missing_test_file_does_not_burn_the_lock(config, workspace):
    workspace["frozen_path"].write_text("x")
    (workspace["data_dir"] / config["data"]["prepared_folder"] / "test_features.csv").unlink()
    with pytest.raises(FileNotFoundError, match="prepare.py"):
        holdout.open_test_pile(workspace["data_dir"], config, workspace["frozen_path"], confirm=True)
    assert not holdout.is_locked(workspace["data_dir"], config)


def test_lock_is_written_before_the_file_is_read(config, workspace):
    workspace["frozen_path"].write_text("x")
    bad = workspace["data_dir"] / config["data"]["prepared_folder"] / "test_features.csv"
    bad.write_text("crash\n7\n")
    with pytest.raises(ValueError):
        holdout.open_test_pile(workspace["data_dir"], config, workspace["frozen_path"], confirm=True)
    assert holdout.is_locked(workspace["data_dir"], config)


# ---------------------------------------------------------------- choosing the model
def test_choose_model_takes_the_highest_average():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    good = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.25])
    bad = np.array([0.5, 0.4, 0.6, 0.5, 0.45, 0.55, 0.5, 0.4])
    name, table = select_final.choose_model(y, {"a": bad, "b": good}, ["a", "b"])
    assert name == "b" and table["b"]["average"] > table["a"]["average"]
    assert table["b"]["average"] == pytest.approx((table["b"]["crash_pr_auc"] + table["b"]["survivor_pr_auc"]) / 2)


def _find_case(pick_wrong_side_differs):
    """Seeded search for two score sets where the winner on one side alone is NOT the winner on the average."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    for _ in range(4000):
        a, b = rng.random(60), rng.random(60)
        _, t = select_final.choose_model(y, {"a": a, "b": b}, ["a", "b"])
        by_average = "a" if t["a"]["average"] > t["b"]["average"] else "b"
        for side in ("crash_pr_auc", "survivor_pr_auc"):
            by_side = "a" if t["a"][side] > t["b"][side] else "b"
            if pick_wrong_side_differs == side and by_side != by_average:
                return y, a, b, by_average
    raise AssertionError("no case found")


@pytest.mark.parametrize("side", ["crash_pr_auc", "survivor_pr_auc"])
def test_choose_model_uses_the_average_of_both_sides_not_one_side(side):
    y, a, b, expected = _find_case(side)
    assert select_final.choose_model(y, {"a": a, "b": b}, ["a", "b"])[0] == expected


def test_choose_model_tie_goes_to_the_earlier_candidate():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.8, 0.2])
    assert select_final.choose_model(y, {"a": s, "b": s.copy()}, ["a", "b"])[0] == "a"
    assert select_final.choose_model(y, {"a": s, "b": s.copy()}, ["b", "a"])[0] == "b"


# ---------------------------------------------------------------- select_final end to end
def test_select_final_never_needs_the_test_pile(config, workspace):
    (workspace["data_dir"] / config["data"]["prepared_folder"] / "test_features.csv").unlink()
    assert run_select(config, workspace) == 0
    assert not holdout.is_locked(workspace["data_dir"], config)


def test_select_final_writes_a_complete_frozen_file(config, workspace):
    assert run_select(config, workspace) == 0
    frozen = yaml.safe_load(workspace["frozen_path"].read_text())
    assert frozen["model"] in config["final"]["candidates"]
    assert frozen["model_params"] == config["models"][frozen["model"]]
    assert frozen["fit_on"].startswith("train only") and frozen["train_rows"] == 600
    assert len(frozen["columns_sha256"]) == 64 and frozen["seed"] == 42
    assert set(frozen["scenarios"]) == set(config["final"]["scenarios"])
    for sc in frozen["scenarios"].values():
        assert {"costs", "cutoff", "validation"} <= set(sc)
        assert sc["validation"]["counts"]["n"] == 600
    assert (workspace["out_dir"] / "select_final.json").exists()
    curve = pd.read_csv(workspace["out_dir"] / "validation_cutoff_curve.csv")
    assert {"cutoff", "skipped_good", "held_crasher", "cost_per_coin_equal"} <= set(curve.columns)
    assert curve["flagged"].iloc[0] == 0 and curve["flagged"].iloc[-1] == 600


def test_frozen_choice_matches_an_independent_recomputation(config, workspace):
    run_select(config, workspace)
    frozen = yaml.safe_load(workspace["frozen_path"].read_text())
    tuned = select_final.tuned_config(config, workspace["params_path"])
    fitted = train.fit_models(tuned, piles.load_piles(workspace["data_dir"], config))
    scores = fitted["scores"][frozen["model"]]["validation"]
    for name, sc in config["final"]["scenarios"].items():
        assert frozen["scenarios"][name]["cutoff"] == decision.best_cutoff(fitted["y_validation"], scores, sc)["cutoff"]
    table = frozen["validation"]["selection_table"]
    assert frozen["model"] == max(table, key=lambda n: table[n]["average"])


def test_select_final_is_repeatable(config, workspace):
    run_select(config, workspace)
    first = workspace["frozen_path"].read_bytes()
    run_select(config, workspace)
    assert workspace["frozen_path"].read_bytes() == first


def test_select_final_refuses_once_the_test_pile_was_opened(config, workspace):
    run_select(config, workspace)
    before = workspace["frozen_path"].read_bytes()
    assert run_final(config, workspace, "--confirm-open-test") == 0
    assert run_select(config, workspace) == 1
    assert workspace["frozen_path"].read_bytes() == before


def test_modelling_steps_run_without_a_test_file(config, workspace):
    (workspace["data_dir"] / config["data"]["prepared_folder"] / "test_features.csv").unlink()
    assert train.main([], config=config, data_dir=workspace["data_dir"], out_dir=workspace["out_dir"]) == 0
    assert uncertainty.main([], config=config, data_dir=workspace["data_dir"], out_dir=workspace["out_dir"]) == 0


def test_only_the_final_step_can_open_the_test_pile():
    """Static scan of the code: who may mention the test file or the opener."""
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "test_features" in text:
            assert path.name == "holdout.py", f"{path.name} names the test feature file"
        if "open_test_pile" in text:
            assert path.name in ("holdout.py", "final_test.py"), f"{path.name} calls the test-pile opener"
        if re.search(r"^import holdout|^from holdout", text, flags=re.M):
            assert path.name in ("select_final.py", "final_test.py"), f"{path.name} imports holdout"
    select_text = (SRC / "select_final.py").read_text(encoding="utf-8")
    assert "open_test_pile" not in select_text


# ---------------------------------------------------------------- final_test end to end
def test_without_the_confirm_flag_nothing_is_opened(config, workspace, capsys):
    run_select(config, workspace)
    assert run_final(config, workspace) == 2
    assert not holdout.is_locked(workspace["data_dir"], config)
    assert not (workspace["out_dir"] / "final_test.json").exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_final_test_needs_the_frozen_file_first(config, workspace):
    assert run_final(config, workspace, "--confirm-open-test") == 1
    assert not holdout.is_locked(workspace["data_dir"], config)


def test_final_test_runs_once_and_never_changes_the_frozen_file(config, workspace):
    run_select(config, workspace)
    frozen_bytes = workspace["frozen_path"].read_bytes()
    assert run_final(config, workspace, "--confirm-open-test") == 0
    assert holdout.is_locked(workspace["data_dir"], config)
    assert workspace["frozen_path"].read_bytes() == frozen_bytes
    result = json.loads((workspace["out_dir"] / "final_test.json").read_text())
    assert result["n_test"] == 600 and result["model"] == yaml.safe_load(frozen_bytes)["model"]
    assert set(result["scenarios"]) == set(config["final"]["scenarios"])


def test_second_run_is_refused_and_results_are_not_overwritten(config, workspace, capsys):
    run_select(config, workspace)
    run_final(config, workspace, "--confirm-open-test")
    first = (workspace["out_dir"] / "final_test.json").read_bytes()
    capsys.readouterr()
    assert run_final(config, workspace, "--confirm-open-test") == 1
    assert "already opened" in capsys.readouterr().out
    assert (workspace["out_dir"] / "final_test.json").read_bytes() == first


def test_frozen_cutoffs_and_train_only_fit_are_what_gets_graded(config, workspace):
    """Re-do the test grading independently: refit on TRAIN only, apply the frozen cut-offs."""
    run_select(config, workspace)
    frozen = yaml.safe_load(workspace["frozen_path"].read_text())
    run_final(config, workspace, "--confirm-open-test")
    result = json.loads((workspace["out_dir"] / "final_test.json").read_text())

    tuned = select_final.tuned_config(config, workspace["params_path"])
    train_pile = piles.load_piles(workspace["data_dir"], config, parts=("train",))["train"]
    cols = piles.select_features(train_pile.columns, config)
    model = final_test.fit_scores(config, tuned, train_pile, cols, [frozen["model"]])[frozen["model"]]
    test = pd.read_csv(workspace["data_dir"] / config["data"]["prepared_folder"] / "test_features.csv")
    scores = model.predict_proba(test[cols])[:, 1]
    for name, sc in frozen["scenarios"].items():
        assert result["scenarios"][name]["cutoff"] == sc["cutoff"]
        assert result["scenarios"][name]["counts"] == decision.counts_at(test["crash"].to_numpy(), scores, sc["cutoff"])


def test_a_changed_frozen_file_is_caught_before_the_test_pile_opens(config, workspace, capsys):
    run_select(config, workspace)
    frozen = yaml.safe_load(workspace["frozen_path"].read_text())
    frozen["model_params"]["tampered"] = True
    workspace["frozen_path"].write_text(yaml.safe_dump(frozen, sort_keys=False))
    assert run_final(config, workspace, "--confirm-open-test") == 1
    assert "does not match" in capsys.readouterr().out
    assert not holdout.is_locked(workspace["data_dir"], config)


def test_frozen_file_for_other_columns_is_caught(config, workspace, capsys):
    run_select(config, workspace)
    frozen = yaml.safe_load(workspace["frozen_path"].read_text())
    frozen["columns_sha256"] = "0" * 64
    workspace["frozen_path"].write_text(yaml.safe_dump(frozen, sort_keys=False))
    assert run_final(config, workspace, "--confirm-open-test") == 1
    assert "columns" in capsys.readouterr().out and not holdout.is_locked(workspace["data_dir"], config)


def test_changed_scenario_costs_are_caught(config, workspace):
    run_select(config, workspace)
    frozen = yaml.safe_load(workspace["frozen_path"].read_text())
    other = copy.deepcopy(config)
    other["final"]["scenarios"]["equal"] = {"holding_crasher": 2, "skipped_good": 1}
    with pytest.raises(ValueError, match="scenarios"):
        final_test.check_frozen(frozen, other, select_final.tuned_config(other, workspace["params_path"]),
                                piles.select_features(make_pile(5, 1).columns, other))


def test_evaluate_by_hand(config):
    y = np.array([1, 1, 0, 0, 1, 0])
    frozen = {"model": "m", "validation": {"side_metrics": {"crash": {"baseline": 0.7}, "survivor": {"baseline": 0.3}}},
              "scenarios": {"equal": {"costs": {"holding_crasher": 1, "skipped_good": 1}, "cutoff": 0.5,
                                      "validation": {"cost_per_coin": 0.1}}}}
    cfg = {**config, "evaluation": {**config["evaluation"], "bootstrap_draws": 20}}
    result = final_test.evaluate(cfg, frozen, y, {"m": S, "ref": 1 - S}, "ref")
    sc = result["scenarios"]["equal"]
    assert sc["counts"]["skipped_good"] == 1 and sc["counts"]["held_crasher"] == 1
    assert sc["cost_per_coin"] == round(2 / 6, 4)
    assert sc["cost_per_coin_flag_nothing"] == 0.5 and sc["cost_per_coin_flag_everything"] == 0.5
    assert result["ranking"]["m"]["crash"]["baseline"] == 0.5
    assert "difference_crash" in result["wobble"]


def test_saving_bootstrap_is_repeatable_and_a_perfect_scorer_always_saves(config):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    perfect = y * 0.8 + 0.1
    sc = {"holding_crasher": 1, "skipped_good": 1}
    a = final_test.bootstrap_saving(y, perfect, 0.5, sc, 30, 7)
    assert a == final_test.bootstrap_saving(y, perfect, 0.5, sc, 30, 7)
    assert a["share_of_draws_cheaper_than_lazy"] == 1.0 and a["low"] > 0


def test_a_useless_cutoff_shows_no_saving(config):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    noise = rng.random(400)
    out = final_test.bootstrap_saving(y, noise, 0.5, {"holding_crasher": 1, "skipped_good": 1}, 50, 3)
    assert out["median"] <= 0.02 and out["share_of_draws_cheaper_than_lazy"] < 0.5


def test_format_result_states_the_limits(config, workspace, capsys):
    run_select(config, workspace)
    run_final(config, workspace, "--confirm-open-test")
    text = capsys.readouterr().out
    assert "not a trade signal" in text and "INVENTED" in text and "sampling noise" in text
    assert "skipped good" in text and "held crasher" in text
