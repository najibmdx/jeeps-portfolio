"""Stage 7 tests: frozen_choice.yaml is written from the tuning report, holds no test number, is never overwritten,
is refused once the test has been opened, and reads back only if it still fits config.yaml."""

import copy
import json

import pytest
import yaml

import freeze
import models
import settings
import tune
from test_tune import CONFIG, TRAIN, VAL, bce, fakes  # noqa: F401  (fakes is a fixture)

TODAY = "2026-11-25"


@pytest.fixture
def project(tmp_path, monkeypatch, fakes):  # noqa: F811
    """A throw-away project folder with a tuning report made from the hand-made tables, and a data folder with no lock file."""
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "data").mkdir()
    report = tune.run(TRAIN, VAL, CONFIG)
    (tmp_path / "outputs" / "tuning_k7.json").write_text(json.dumps(report), encoding="utf-8")
    return tmp_path, report


def run_freeze(tmp_path):
    return freeze.main(["--data-dir", str(tmp_path / "data")], today=TODAY)


def test_the_frozen_file_holds_the_choice_and_nothing_from_the_test(project, capsys):
    tmp_path, report = project
    assert run_freeze(tmp_path) == 0
    text = (tmp_path / "frozen_choice.yaml").read_text(encoding="utf-8")
    frozen = yaml.safe_load(text)
    assert list(frozen) == list(freeze.KEYS)
    assert frozen["frozen_on"] == TODAY and frozen["k"] == 7
    assert frozen["model"] == "random_forest" and frozen["params"] == {"min_samples_leaf": 200}
    assert frozen["fixed_settings"] == {"seed": 42, "forest_trees": 300}
    assert frozen["fitted_on"].startswith("train only") and "no refit on train + validation" in frozen["fitted_on"]
    assert frozen["train_markets"] == 20 and frozen["validation_markets"] == 10 and frozen["inputs"] == models.input_columns(CONFIG)[0]
    assert frozen["validation"] == {"log_loss_all": pytest.approx(bce(0.4, 0.4)), "log_loss_contested": pytest.approx(bce(0.4, 0.375))}
    ru = frozen["runner_up"]
    assert ru["model"] == "gradient_boosting" and ru["params"] == {"learning_rate": 0.1, "min_samples_leaf": 400}
    assert ru["log_loss_all"] == pytest.approx(bce(0.41, 0.4)) and ru["log_loss_contested"] == pytest.approx(bce(0.41, 0.375))
    margin = frozen["margin_runner_up_minus_winner_all"]
    assert set(margin) == {"point", "low", "high", "verdict"} and margin["point"] == pytest.approx(bce(0.41, 0.4) - bce(0.4, 0.4))
    assert margin == {k: report["choice"]["margin_runner_up_minus_winner"]["all"][k] for k in margin}
    raw = frozen["winner_minus_raw_price_all"]
    assert set(raw) == {"point", "low", "high", "verdict"} and raw == {k: report["choice"]["winner_minus_raw_price"]["all"][k] for k in raw}
    assert raw != {k: report["choice"]["winner_minus_raw_price"]["contested"][k] for k in raw}      # the all-markets numbers, not the contested ones
    assert frozen["tie_at_4_decimals"] is False and frozen["tuning_report"] == "outputs/tuning_k7.json" and frozen["test_split"] == "not opened"
    assert "lowest validation log loss on all markets" in frozen["rule"] and "never used to decide" in frozen["rule"]
    assert text.startswith("# The model chosen on validation (Stage 7).") and "BEFORE the test split is opened" in text and "slightly flattering" in text
    out = capsys.readouterr().out
    assert "Written to frozen_choice.yaml" in out and "commit it BEFORE the test split is opened" in out and "model: random_forest" in out


def test_the_frozen_file_has_no_test_key_and_no_test_number(project):
    tmp_path, _ = project
    run_freeze(tmp_path)
    frozen = yaml.safe_load((tmp_path / "frozen_choice.yaml").read_text(encoding="utf-8"))
    assert [k for k in freeze._keys(frozen) if freeze.is_test_key(k)] == ["test_split"]
    assert "log_loss_contested" in list(freeze._keys(frozen))      # contains the letters "test" but is a validation score
    assert freeze.read_frozen(tmp_path / "frozen_choice.yaml", CONFIG) == frozen


def test_a_tie_at_four_decimals_is_recorded(project):
    tmp_path, _ = project
    _edit_report(tmp_path, lambda r: r["choice"].update(tie_at_4_decimals=True))
    assert run_freeze(tmp_path) == 0
    assert yaml.safe_load((tmp_path / "frozen_choice.yaml").read_text(encoding="utf-8"))["tie_at_4_decimals"] is True


def test_a_file_that_would_not_pass_its_own_checks_is_removed_and_reported(project, monkeypatch, capsys):
    tmp_path, _ = project
    real = freeze.make_frozen
    monkeypatch.setattr(freeze, "make_frozen", lambda *a: dict(real(*a), test_split="opened"))
    assert run_freeze(tmp_path) == 1
    assert not (tmp_path / "frozen_choice.yaml").exists()
    assert "does not pass its own checks, so it was removed" in capsys.readouterr().out


def test_the_date_is_written_as_text_and_comes_from_today_by_default(project, monkeypatch):
    tmp_path, _ = project
    assert freeze.main(["--data-dir", str(tmp_path / "data")]) == 0
    frozen = yaml.safe_load((tmp_path / "frozen_choice.yaml").read_text(encoding="utf-8"))
    import datetime
    assert isinstance(frozen["frozen_on"], str) and frozen["frozen_on"] == datetime.date.today().isoformat()


def test_a_frozen_choice_is_never_overwritten(project, capsys):
    tmp_path, _ = project
    path = tmp_path / "frozen_choice.yaml"
    path.write_text("model: lr_all_features\n", encoding="utf-8")
    assert run_freeze(tmp_path) == 1
    assert path.read_text(encoding="utf-8") == "model: lr_all_features\n"
    out = capsys.readouterr().out
    assert "already exists" in out and "deleting the file by hand" in out and "Nothing was written" in out


def test_nothing_is_frozen_once_the_test_split_has_been_opened(project, capsys):
    tmp_path, _ = project
    (tmp_path / "data" / CONFIG["models"]["test_lock_file"]).write_text("opened", encoding="utf-8")
    assert run_freeze(tmp_path) == 1
    assert not (tmp_path / "frozen_choice.yaml").exists()
    assert "test split has already been opened" in capsys.readouterr().out


def test_nothing_is_frozen_without_a_tuning_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    assert freeze.main(["--data-dir", str(tmp_path)], today=TODAY) == 2
    assert "Run src/tune.py first" in capsys.readouterr().out and not (tmp_path / "frozen_choice.yaml").exists()


def _edit_report(tmp_path, edit):
    path = tmp_path / "outputs" / "tuning_k7.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    edit(report)
    path.write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda r: r["choice"].update(winner="raw_price"), "not on the ladder"),                  # the raw price is a yardstick, never a candidate
        (lambda r: r["choice"].update(winner="mystery"), "not on the ladder"),
        (lambda r: r["choice"].update(params={"min_samples_leaf": 7}), "not in the grid of config.yaml; run src/tune.py again"),   # a stale report
        (lambda r: r.update(seed=1), "different seed from config.yaml"),
        (lambda r: r["choice"].pop("winner"), "winner"),
    ],
)
def test_a_report_that_does_not_fit_config_yaml_is_refused_and_nothing_is_written(project, capsys, edit, message):
    tmp_path, _ = project
    _edit_report(tmp_path, edit)
    assert run_freeze(tmp_path) == 1
    out = capsys.readouterr().out
    assert "PROBLEM" in out and message in out and "removed" not in out and not (tmp_path / "frozen_choice.yaml").exists()




@pytest.mark.parametrize("key,expected", [("test_split", True), ("test_log_loss", True), ("log_loss_test", True), ("Test", True), ("a_test_b", True), ("test", True),
                                          ("log_loss_contested", False), ("contested", False), ("latest", False), ("testing", False), ("attest", False), ("validation", False)])
def test_which_keys_name_the_test_split(key, expected):
    assert freeze.is_test_key(key) is expected


def test_the_fixed_settings_of_each_model():
    assert freeze.fixed_settings("lr_price_only", CONFIG) == {}
    assert freeze.fixed_settings("lr_all_features", CONFIG) == {"seed": 42, "logistic_max_iter": 1000}
    assert freeze.fixed_settings("random_forest", CONFIG) == {"seed": 42, "forest_trees": 300}
    assert freeze.fixed_settings("gradient_boosting", CONFIG) == {"seed": 42}


@pytest.mark.parametrize("name,params", [("lr_all_features", {"C": 0.1}), ("random_forest", {"min_samples_leaf": 5}),
                                         ("gradient_boosting", {"learning_rate": 0.03, "min_samples_leaf": 100}), ("lr_price_only", {})])
def test_every_model_can_be_frozen_and_read_back_with_its_settings(project, name, params):
    tmp_path, report = project
    report["choice"].update(winner=name, params=params)
    frozen = freeze.make_frozen(report, CONFIG, 7, TODAY)
    path = tmp_path / "frozen_choice.yaml"
    path.write_text(freeze.frozen_text(frozen), encoding="utf-8")
    back = freeze.read_frozen(path, CONFIG)
    assert back["model"] == name and back["params"] == params and back["fixed_settings"] == freeze.fixed_settings(name, CONFIG)


# ---- reading it back (Stage 9 learns the chosen model from here and nowhere else)

def _written(project):
    tmp_path, _ = project
    run_freeze(tmp_path)
    path = tmp_path / "frozen_choice.yaml"
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def _rewrite(path, frozen):
    path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")


def test_a_missing_file_is_refused_in_plain_words(tmp_path):
    with pytest.raises(ValueError, match=r"frozen_choice\.yaml is missing: the choice must be frozen"):
        freeze.read_frozen(tmp_path / "frozen_choice.yaml", CONFIG)


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda f: f.pop("model"), "exactly these entries"),
        (lambda f: f.update(extra=1), "exactly these entries"),
        (lambda f: f.update(test_log_loss=0.1), "exactly these entries"),
        (lambda f: f.update(test_split="opened"), "no test numbers"),
        (lambda f: f["validation"].update(test_log_loss=0.5), "no test numbers"),
        (lambda f: f.update(model="raw_price"), "not on the ladder"),
        (lambda f: f.update(model="base_rate"), "not on the ladder"),
        (lambda f: f.update(params={"min_samples_leaf": 7}), "not in the grid"),
        (lambda f: f.update(params={}), "not in the grid"),
        (lambda f: f.update(fixed_settings={"seed": 1, "forest_trees": 300}), "fixed settings differ"),
        (lambda f: f.update(fixed_settings={"seed": 42}), "fixed settings differ"),
        (lambda f: f.update(frozen_on="yesterday"), "YYYY-MM-DD"),
        (lambda f: f.update(frozen_on="2026-11-25 10:00"), "YYYY-MM-DD"),
    ],
)
def test_a_frozen_file_that_no_longer_fits_is_refused(project, edit, message):
    path, frozen = _written(project)
    edit(frozen)
    _rewrite(path, frozen)
    with pytest.raises(ValueError, match=message):
        freeze.read_frozen(path, CONFIG)


def test_a_file_that_is_not_a_mapping_is_refused(tmp_path):
    path = tmp_path / "frozen_choice.yaml"
    for text in ("- a\n- b\n", "just text\n", ""):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="exactly these entries"):
            freeze.read_frozen(path, CONFIG)


def test_a_change_in_config_yaml_after_freezing_is_caught(project):
    path, _ = _written(project)
    changed = copy.deepcopy(CONFIG)
    changed["models"]["forest_trees"] = 500
    with pytest.raises(ValueError, match="fixed settings differ"):
        freeze.read_frozen(path, changed)
    changed = copy.deepcopy(CONFIG)
    changed["models"]["tuning"]["random_forest"]["min_samples_leaf"] = [100, 50, 20, 5, 2, 1]
    with pytest.raises(ValueError, match="not in the grid"):
        freeze.read_frozen(path, changed)
    changed = copy.deepcopy(CONFIG)
    changed["models"]["ladder"] = ["lr_price_only", "lr_all_features"]
    changed["models"]["tuning"] = {"lr_all_features": CONFIG["models"]["tuning"]["lr_all_features"]}
    with pytest.raises(ValueError, match="not on the ladder"):
        freeze.read_frozen(path, changed)


def test_freezing_never_reads_the_data_folder_for_anything_but_the_lock_file(project):
    tmp_path, _ = project
    (tmp_path / "data" / "train_k7.parquet").write_text("not a parquet file", encoding="utf-8")
    assert run_freeze(tmp_path) == 0


def _report_for(report, winner):
    """The report with another model as the winner (its first grid setting), to test the fixed-settings check per model."""
    r = copy.deepcopy(report)
    r["choice"]["winner"] = winner
    r["choice"]["params"] = tune.grid_settings(CONFIG, winner)[0]
    return r


@pytest.mark.parametrize(
    "winner,key,value",
    [("random_forest", "forest_trees", 500), ("lr_all_features", "logistic_max_iter", 2000)],
)
def test_a_changed_fixed_setting_of_the_winner_is_refused(project, winner, key, value):
    _, report = project
    r = _report_for(report, winner)
    r[key] = value  # the report was made with another value than config.yaml now says
    with pytest.raises(ValueError, match=f"{key} = {value} but config.yaml now says {CONFIG['models'][key]}; run src/tune.py again"):
        freeze.make_frozen(r, CONFIG, 7, TODAY)


@pytest.mark.parametrize(
    "winner,irrelevant",
    [("random_forest", "logistic_max_iter"), ("lr_all_features", "forest_trees"), ("gradient_boosting", "forest_trees"), ("lr_price_only", "logistic_max_iter")],
)
def test_a_setting_the_winner_does_not_use_does_not_matter(project, winner, irrelevant):
    _, report = project
    r = _report_for(report, winner)
    r[irrelevant] = 12345
    assert freeze.make_frozen(r, CONFIG, 7, TODAY)["model"] == winner


@pytest.mark.parametrize("winner", ["random_forest", "lr_all_features", "gradient_boosting", "lr_price_only"])
def test_a_report_that_matches_the_config_freezes_for_every_model(project, winner):
    _, report = project
    assert freeze.make_frozen(_report_for(report, winner), CONFIG, 7, TODAY)["fixed_settings"] == freeze.fixed_settings(winner, CONFIG)


@pytest.mark.parametrize("winner,key", [("random_forest", "forest_trees"), ("lr_all_features", "logistic_max_iter")])
def test_a_report_without_the_fixed_setting_is_refused(project, winner, key):
    _, report = project
    r = _report_for(report, winner)
    r.pop(key)
    with pytest.raises(ValueError, match="run src/tune.py again"):
        freeze.make_frozen(r, CONFIG, 7, TODAY)
