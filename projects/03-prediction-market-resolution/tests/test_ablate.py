"""Stage 8 tests: the rows and the settings that check them, known answers on hand-made tables (stand-in models whose
forecast depends only on the columns they were given), that each row is fitted on train and on its own columns only,
the paired differences, the frozen-model reproduction check, and the command line."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

import ablate
import baselines
import freeze
import models
import settings
import split
import tune
from test_tune import CONFIG, EPS, TRAIN, VAL, VAL_ALL, VAL_CON, bce, frame, _split_dir  # noqa: F401

ROWS = CONFIG["ablation"]["rows"]
PAIRS = CONFIG["ablation"]["pairs"]
FROZEN = ("lr_all_features", {"C": 1.0})
# What the stand-in model forecasts for each row (one number for every market).
P = {"price_path_only": 0.40, "price_path_plus_activity": 0.35, "price_path_plus_category": 0.45,
     "price_path_plus_sibling_count": 0.30, "all_minus_sibling_count": 0.50, "all_columns": 0.41}


class Spy:
    """A stand-in model. Forecasts one number for every market, and remembers what it was built and fitted with."""

    def __init__(self, p, columns, params, log):
        self.p, self.columns, self.params, self.log = p, columns, params, log

    def fit(self, X, y):
        self.log.append({"columns": list(X.columns), "wanted": list(self.columns), "params": self.params, "n": len(X), "y": [int(v) for v in y]})
        return self

    def predict_proba(self, X):
        assert list(X.columns) == list(self.columns)  # the forecast reads the same columns as the fit
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


@pytest.fixture
def fakes(monkeypatch):
    log = []
    by_columns = {tuple(c): n for n, c in ROWS.items()}
    monkeypatch.setattr(baselines, "fit_correction", lambda price, y, eps: {"intercept": 0.0, "slope": 1.0})
    monkeypatch.setattr(models, "build_model", lambda name, cfg, columns, params=None: Spy(P[by_columns[tuple(columns)]], columns, params, log))
    return log


def loss(p, labels):
    """Log loss of a forecast list, written out here (not taken from the code under test)."""
    p, y = np.asarray(p, dtype=float), np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@pytest.fixture
def report(fakes):
    return ablate.run(TRAIN, VAL, CONFIG, *FROZEN)


# ---- the settings

def test_the_rows_are_the_ones_agreed_with_jeeps():
    assert list(ROWS) == ["price_path_only", "price_path_plus_activity", "price_path_plus_category", "price_path_plus_sibling_count", "all_minus_sibling_count", "all_columns"]
    g = CONFIG["ablation"]["groups"]
    assert {k: len(v) for k, v in g.items()} == {"price_path": 6, "activity": 4, "context": 2}
    assert ROWS["all_columns"] == models.input_columns(CONFIG)[0]
    assert set(ROWS["price_path_plus_sibling_count"]) - set(ROWS["price_path_only"]) == {"sibling_count_at_t"}
    assert set(ROWS["price_path_plus_category"]) - set(ROWS["price_path_only"]) == {"category"}
    assert set(ROWS["all_columns"]) - set(ROWS["all_minus_sibling_count"]) == {"sibling_count_at_t"}
    assert len(PAIRS) == 14 and ["all_columns", "all_minus_sibling_count"] in PAIRS and ["price_path_plus_sibling_count", "price_path_only"] in PAIRS
    assert all(set(r) <= set(models.input_columns(CONFIG)[0]) and "last_close" in r for r in ROWS.values())


def _bad(edit):
    c = copy.deepcopy(CONFIG)
    edit(c)
    return c


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda c: c["ablation"]["groups"]["activity"].append("last_close"), "sits in two groups"),
        (lambda c: c["ablation"]["groups"]["activity"].remove("n_bars"), "must cover exactly the model inputs"),
        (lambda c: c["ablation"]["groups"].update(extra=["has_fee"]), "must cover exactly the model inputs"),
        (lambda c: c["ablation"]["groups"].update(activity=[]), "non-empty list of columns"),
        (lambda c: c["ablation"]["rows"].pop("all_columns"), "must include all_columns"),
        (lambda c: c["ablation"]["rows"].update(raw_price=["last_close"]), "taken by a yardstick"),
        (lambda c: c["ablation"]["rows"].update(lr_price_only=["last_close"]), "taken by a yardstick"),
        (lambda c: c["ablation"]["rows"].update(price_path_only=[]), "non-empty list of model inputs"),
        (lambda c: c["ablation"]["rows"].update(price_path_only=["last_close", "last_close"]), "each once"),
        (lambda c: c["ablation"]["rows"].update(price_path_only=["last_close", "has_fee"]), "each once"),
        (lambda c: c["ablation"]["rows"].update(price_path_only=["n_bars"]), "at least one price column"),
        (lambda c: c["ablation"]["rows"]["all_columns"].reverse(), "in the order of config.yaml features"),
        (lambda c: c["ablation"]["rows"]["all_columns"].remove("category"), "must list every model input"),
        (lambda c: c["ablation"].update(pairs=[]), "non-empty list"),
        (lambda c: c["ablation"]["pairs"].append(["price_path_only", "mystery"]), "two different rows"),
        (lambda c: c["ablation"]["pairs"].append(["price_path_only", "price_path_only"]), "two different rows"),
        (lambda c: c["ablation"]["pairs"].append(["price_path_only", "lr_price_only"]), "listed twice"),
        (lambda c: c["ablation"].update(reproduce_tolerance=0), "small number above 0"),
        (lambda c: c["ablation"].update(reproduce_tolerance=0.01), "small number above 0"),
        (lambda c: c["ablation"].update(reproduce_tolerance=True), "small number above 0"),
        (lambda c: c["ablation"].update(reproduce_tolerance="1e-7"), "small number above 0"),
    ],
)
def test_bad_ablation_settings_are_refused_in_plain_words(edit, message):
    with pytest.raises(ValueError, match=message):
        settings.validate_config(_bad(edit))


def test_the_good_settings_pass_and_raw_price_may_be_a_pair_member():
    settings.validate_config(copy.deepcopy(CONFIG))
    c = _bad(lambda c: c["ablation"]["pairs"].append(["all_columns", "price_path_only"]))
    settings.validate_config(c)


# ---- known answers

def test_each_row_is_fitted_on_train_only_and_on_its_own_columns_with_the_frozen_settings(fakes):
    ablate.run(TRAIN, VAL, CONFIG, *FROZEN)
    assert [f["wanted"] for f in fakes] == list(ROWS.values())              # one fit per row, in order
    for f in fakes:
        assert f["columns"] == f["wanted"] and f["params"] == {"C": 1.0}
        assert f["n"] == len(TRAIN) and f["y"] == [int(v) for v in TRAIN["label"]]   # train rows and train labels, never validation


def test_every_row_is_scored_by_hand_on_both_slices(report):
    for name, p in P.items():
        for sname, share in (("all", VAL_ALL), ("contested", VAL_CON)):
            assert report["validation"][sname]["scores"][name]["log_loss"] == pytest.approx(bce(p, share), abs=1e-12)
    raw = [0.5] * 8 + [0.01] * 2
    assert report["validation"]["all"]["scores"]["raw_price"]["log_loss"] == pytest.approx(loss(raw, VAL["label"]), abs=1e-12)
    assert report["validation"]["all"]["scores"]["lr_price_only"]["log_loss"] == pytest.approx(loss(raw, VAL["label"]), abs=1e-12)  # the faked correction is the identity
    assert report["validation"]["all"]["markets"] == 10 and report["validation"]["contested"]["markets"] == 8


def test_every_pair_is_a_minus_b_by_hand_and_the_sign_says_who_is_lower(report):
    raw = [0.5] * 8 + [0.01] * 2
    labels = list(VAL["label"])
    forecast = {n: [p] * 10 for n, p in P.items()}
    forecast.update(lr_price_only=raw, raw_price=raw)
    for sname, idx in (("all", list(range(10))), ("contested", list(range(8)))):
        got = {(p["a"], p["b"]): p for p in report["pairs"][sname]}
        assert list(got) == [tuple(p) for p in PAIRS]                  # in the order of config.yaml
        for (a, b), d in got.items():
            want = loss([forecast[a][i] for i in idx], [labels[i] for i in idx]) - loss([forecast[b][i] for i in idx], [labels[i] for i in idx])
            assert d["point"] == pytest.approx(want, abs=1e-12)
            assert d["low"] <= d["high"] and d["verdict"] in ("better", "worse", "cannot be told apart")
    d = {(p["a"], p["b"]): p for p in report["pairs"]["all"]}
    assert d[("all_columns", "all_minus_sibling_count")]["point"] == pytest.approx(bce(0.41, 0.4) - bce(0.50, 0.4))
    assert d[("price_path_plus_sibling_count", "price_path_only")]["point"] == pytest.approx(bce(0.30, 0.4) - bce(0.40, 0.4))


def test_a_pair_between_two_equal_forecasts_is_zero_and_cannot_be_told_apart(fakes, monkeypatch):
    same = dict(P, price_path_plus_activity=0.40)
    by_columns = {tuple(c): n for n, c in ROWS.items()}
    monkeypatch.setattr(models, "build_model", lambda name, cfg, columns, params=None: Spy(same[by_columns[tuple(columns)]], columns, params, fakes))
    r = ablate.run(TRAIN, VAL, CONFIG, *FROZEN)
    d = {(p["a"], p["b"]): p for p in r["pairs"]["all"]}[("price_path_plus_activity", "price_path_only")]
    assert d["point"] == 0 and d["low"] == 0 and d["high"] == 0 and d["verdict"] == "cannot be told apart"


def test_the_wobble_uses_the_configured_draws_seed_interval_and_the_same_draws_for_every_pair(fakes, monkeypatch):
    seen = {"weights": [], "paired": []}
    real_w, real_p, real_s = baselines.resample_weights, baselines.paired_difference, baselines.score_slice
    monkeypatch.setattr(baselines, "resample_weights", lambda *a: (seen.update(args=a), real_w(*a))[1])
    monkeypatch.setattr(baselines, "paired_difference", lambda *a, **k: (seen["paired"].append(a), real_p(*a, **k))[1])
    monkeypatch.setattr(baselines, "score_slice", lambda *a, **k: (seen["weights"].append((a[3], a[6])), real_s(*a, **k))[1])
    ablate.run(TRAIN, VAL, CONFIG, *FROZEN)
    n_events = VAL["event_key"].nunique()
    assert seen["args"] == (n_events, CONFIG["evaluation"]["bootstrap_draws"], CONFIG["seed"])
    assert len(seen["paired"]) == 2 * len(PAIRS) and all(a[4] is seen["weights"][0][0] and a[6] == n_events and a[7] == CONFIG["evaluation"]["interval"] for a in seen["paired"])
    assert all(w is seen["weights"][0][0] and i == CONFIG["evaluation"]["interval"] for w, i in seen["weights"])


def test_validation_labels_never_change_what_is_fitted(fakes):
    ablate.run(TRAIN, VAL, CONFIG, *FROZEN)
    first = [dict(f) for f in fakes]
    fakes.clear()
    ablate.run(TRAIN, VAL.assign(label=1 - VAL["label"]), CONFIG, *FROZEN)
    assert fakes == first


def test_a_slice_with_no_markets_is_left_empty_not_invented(fakes):
    near_decided = frame(10, last_close=[0.01] * 10, label=np.array([1, 0] * 5, dtype="int8"), event_key=[f"N{i // 2}" for i in range(10)])
    r = ablate.run(TRAIN, near_decided, CONFIG, *FROZEN)
    assert r["validation"]["contested"]["markets"] == 0 and r["pairs"]["contested"] == [] and len(r["pairs"]["all"]) == len(PAIRS)
    assert "contested: no markets in validation" in "\n".join(ablate.summary_lines(r))


# ---- the frozen model is reproduced

def test_all_columns_must_reproduce_the_frozen_score(fakes):
    r = ablate.run(TRAIN, VAL, CONFIG, *FROZEN, frozen_log_loss_all=bce(0.41, VAL_ALL))
    assert r["reproduces_frozen"]["difference"] == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError, match="frozen model is not reproduced"):
        ablate.run(TRAIN, VAL, CONFIG, *FROZEN, frozen_log_loss_all=bce(0.41, VAL_ALL) + 0.001)


def test_the_reproduction_tolerance_is_the_configured_one(fakes):
    tol = CONFIG["ablation"]["reproduce_tolerance"]
    ablate.run(TRAIN, VAL, CONFIG, *FROZEN, frozen_log_loss_all=bce(0.41, VAL_ALL) + tol / 2)
    with pytest.raises(ValueError, match="not reproduced"):
        ablate.run(TRAIN, VAL, CONFIG, *FROZEN, frozen_log_loss_all=bce(0.41, VAL_ALL) + tol * 2)
    with pytest.raises(ValueError, match="not reproduced"):
        ablate.run(TRAIN, VAL, CONFIG, *FROZEN, frozen_log_loss_all=bce(0.41, VAL_ALL) - tol * 2)    # either direction


def test_no_reference_means_no_check(report):
    assert report["reproduces_frozen"] is None


@pytest.mark.parametrize("model", ["lr_price_only", "raw_price", "mystery"])
def test_a_model_with_nothing_to_ablate_is_refused(fakes, model):
    with pytest.raises(ValueError, match="nothing to ablate"):
        ablate.run(TRAIN, VAL, CONFIG, model, {})


def test_missing_columns_and_empty_tables_are_refused_in_plain_words(fakes):
    with pytest.raises(ValueError, match="the validation table lacks"):
        ablate.run(TRAIN, VAL.drop(columns=["event_key"]), CONFIG, *FROZEN)
    with pytest.raises(ValueError, match="the train table lacks"):
        ablate.run(TRAIN.drop(columns=["label"]), VAL, CONFIG, *FROZEN)
    with pytest.raises(ValueError, match="the validation table has no markets"):
        ablate.run(TRAIN, VAL.iloc[0:0], CONFIG, *FROZEN)
    with pytest.raises(ValueError, match="the train table has no markets"):
        ablate.run(TRAIN.iloc[0:0], VAL, CONFIG, *FROZEN)


# ---- the real models

@pytest.fixture(scope="module")
def real():
    train, val = frame(600, seed=0), frame(300, seed=1)
    return train, val


@pytest.mark.parametrize("model,params", [("lr_all_features", {"C": 1.0}), ("random_forest", {"min_samples_leaf": 50}),
                                          ("gradient_boosting", {"learning_rate": 0.1, "min_samples_leaf": 100})])
def test_every_row_runs_with_every_kind_of_model_including_rows_with_no_category(real, model, params):
    train, val = real
    r = ablate.run(train, val, CONFIG, model, params)
    assert r["order"] == ["raw_price", "lr_price_only"] + list(ROWS) and "category" not in ROWS["price_path_only"]
    for n in ROWS:
        assert 0 < r["validation"]["all"]["scores"][n]["log_loss"] < 3


def test_the_all_columns_row_is_exactly_the_stage_6_and_7_model(real):
    train, val = real
    fitted, columns, _ = models.fit_models(train, CONFIG)
    p = baselines.clip_probability(fitted["lr_all_features"].predict_proba(models.model_inputs(val, columns))[:, 1], EPS)
    r = ablate.run(train, val, CONFIG, "lr_all_features", {"C": 1.0}, frozen_log_loss_all=baselines.log_loss(val["label"].to_numpy(dtype=float), p))
    assert abs(r["reproduces_frozen"]["difference"]) < 1e-9


def test_a_column_outside_a_row_cannot_change_that_row(real):
    train, val = real
    a = ablate.run(train, val, CONFIG, "lr_all_features", {"C": 1.0})
    junk = lambda d: d.assign(sibling_count_at_t=np.random.default_rng(5).integers(1, 999, len(d)))  # noqa: E731
    b = ablate.run(junk(train), junk(val), CONFIG, "lr_all_features", {"C": 1.0})
    for n in ("price_path_only", "price_path_plus_activity", "price_path_plus_category", "all_minus_sibling_count"):
        assert a["validation"]["all"]["scores"][n]["log_loss"] == pytest.approx(b["validation"]["all"]["scores"][n]["log_loss"], abs=1e-12), n
    for n in ("price_path_plus_sibling_count", "all_columns"):
        assert a["validation"]["all"]["scores"][n]["log_loss"] != pytest.approx(b["validation"]["all"]["scores"][n]["log_loss"], abs=1e-9), n


# ---- the report and what is printed

def test_the_report_names_what_it_did(report):
    assert report["model"] == "lr_all_features" and report["params"] == {"C": 1.0} and report["train_markets"] == 20 and report["validation_markets"] == 10
    assert report["rows"] == ROWS and report["order"] == ["raw_price", "lr_price_only"] + list(ROWS)
    assert report["settings"] == {"seed": 42, "forest_trees": 300, "logistic_max_iter": 1000}
    assert any("Nothing was retuned" in n for n in report["notes"]) and any("test split was not read" in n for n in report["notes"])
    assert any("calendar" in n for n in report["notes"])
    json.dumps(report)                                                   # writable as it is


def test_summary_lines_show_every_row_pair_and_the_sibling_answer(fakes):
    r = ablate.run(TRAIN, VAL, CONFIG, *FROZEN, frozen_log_loss_all=bce(0.41, VAL_ALL))
    text = "\n".join(ablate.summary_lines(r))
    for n in list(ROWS) + ["raw_price", "lr_price_only"]:
        assert n in text
    for g in CONFIG["ablation"]["groups"]:
        assert f"{g}: " in text
    assert text.count("pairs, log loss, A minus B") == 2 and text.count(" minus ") >= 2 * len(PAIRS)
    assert f"{bce(0.40, VAL_ALL):.4f}" in text and f"{bce(0.41, VAL_CON):.4f}" in text
    assert "all_columns reproduces the frozen validation score" in text and "Sibling count, all markets:" in text
    assert "price_path_plus_sibling_count minus price_path_only" in text.split("Sibling count, all markets:")[1]
    assert "Nothing here changes the frozen choice" in text and "calendar" in text and "Nothing is retuned" in text


# ---- the command line

def _frozen_file(path, ll, model="lr_all_features", params=None):
    frozen = {k: None for k in freeze.KEYS}
    frozen.update(frozen_on="2026-11-25", k=7, model=model, params=params if params is not None else {"C": 1.0}, fixed_settings=freeze.fixed_settings(model, CONFIG),
                  fitted_on=freeze.FITTED_ON, train_markets=1, validation_markets=1, inputs=models.input_columns(CONFIG)[0],
                  validation={"log_loss_all": ll, "log_loss_contested": 0.5}, runner_up={"model": "lr_price_only"}, margin_runner_up_minus_winner_all={},
                  winner_minus_raw_price_all={}, tie_at_4_decimals=False, rule="r", tuning_report="outputs/tuning_k7.json", test_split=freeze.TEST_STATE)
    path.write_text(freeze.frozen_text(frozen), encoding="utf-8")


@pytest.fixture
def project(tmp_path, monkeypatch):
    data_dir = _split_dir(tmp_path)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    train, val = split.read_split(data_dir, CONFIG, "train", 7), split.read_split(data_dir, CONFIG, "validation", 7)
    ll = ablate.run(train, val, CONFIG, "lr_all_features", {"C": 0.1})["validation"]["all"]["scores"]["all_columns"]["log_loss"]  # not the library default
    _frozen_file(tmp_path / "frozen_choice.yaml", ll, params={"C": 0.1})
    return tmp_path, data_dir


def test_main_reads_train_and_validation_never_test_and_writes_the_report(project, capsys, monkeypatch):
    tmp_path, data_dir = project
    asked, real_read = [], split.read_split
    monkeypatch.setattr(split, "read_split", lambda *a, **k: (asked.append(a[2]), real_read(*a, **k))[1])
    assert ablate.main(["--data-dir", str(data_dir)]) == 0
    assert asked == ["train", "validation"]
    report = json.loads((tmp_path / "outputs" / "ablation_k7.json").read_text(encoding="utf-8"))
    assert report["model"] == "lr_all_features" and report["params"] == {"C": 0.1} and report["reproduces_frozen"]["difference"] == pytest.approx(0, abs=1e-9)
    out = capsys.readouterr().out
    assert "Column-group ablations on validation" in out and "Test was not read." in out and "reproduces the frozen validation score" in out
    assert (tmp_path / "frozen_choice.yaml").exists()


def test_main_needs_no_test_split_file(project, capsys):
    _, data_dir = project
    split.split_path(data_dir, CONFIG, "test", 7).unlink()
    assert ablate.main(["--data-dir", str(data_dir)]) == 0


def test_main_never_changes_the_frozen_file(project):
    tmp_path, data_dir = project
    before = (tmp_path / "frozen_choice.yaml").read_bytes()
    assert ablate.main(["--data-dir", str(data_dir)]) == 0
    assert (tmp_path / "frozen_choice.yaml").read_bytes() == before


def test_main_needs_both_split_files(tmp_path, capsys):
    assert ablate.main(["--data-dir", str(tmp_path)]) == 2
    assert "No train split" in capsys.readouterr().out
    data_dir = _split_dir(tmp_path)
    split.split_path(data_dir, CONFIG, "validation", 7).unlink()
    assert ablate.main(["--data-dir", str(data_dir)]) == 2
    assert "No validation split" in capsys.readouterr().out


def test_main_uses_the_k_it_is_given(tmp_path, capsys):
    data_dir = _split_dir(tmp_path)
    assert ablate.main(["--data-dir", str(data_dir), "--k", "3"]) == 2
    assert "No train split" in capsys.readouterr().out


def test_main_needs_the_frozen_file_and_reads_it_through_read_frozen(project, capsys):
    tmp_path, data_dir = project
    (tmp_path / "frozen_choice.yaml").unlink()
    assert ablate.main(["--data-dir", str(data_dir)]) == 2
    out = capsys.readouterr().out
    assert "PROBLEM" in out and "frozen_choice.yaml is missing" in out and not (tmp_path / "outputs" / "ablation_k7.json").exists()
    _frozen_file(tmp_path / "frozen_choice.yaml", 0.1, params={"C": 7.0})          # a setting that is not in the grid
    assert ablate.main(["--data-dir", str(data_dir)]) == 2
    assert "not in the grid" in capsys.readouterr().out


def test_main_stops_when_the_frozen_score_is_not_reproduced(project, capsys):
    tmp_path, data_dir = project
    _frozen_file(tmp_path / "frozen_choice.yaml", 0.1)
    assert ablate.main(["--data-dir", str(data_dir)]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM" in out and "frozen model is not reproduced" in out and not (tmp_path / "outputs" / "ablation_k7.json").exists()


def test_main_says_so_when_the_frozen_model_is_the_price_only_rung(project, capsys):
    tmp_path, data_dir = project
    _frozen_file(tmp_path / "frozen_choice.yaml", 0.1, model="lr_price_only", params={})
    assert ablate.main(["--data-dir", str(data_dir)]) == 1
    assert "nothing to ablate" in capsys.readouterr().out


# ---- found by the break-it run

def test_the_price_correction_is_fitted_on_train_prices_and_train_labels(monkeypatch, fakes):
    seen = []
    monkeypatch.setattr(baselines, "fit_correction", lambda price, y, eps: (seen.append((list(price), list(y), eps)), {"intercept": 0.0, "slope": 1.0})[1])
    ablate.run(TRAIN, VAL, CONFIG, *FROZEN)
    assert seen == [(list(TRAIN["last_close"]), list(TRAIN["label"]), EPS)]


def test_the_report_says_which_clip_and_interval_were_used(report):
    assert report["price_clip"] == EPS and report["interval"] == CONFIG["evaluation"]["interval"] and report["bootstrap_draws"] == CONFIG["evaluation"]["bootstrap_draws"]


def test_the_printed_table_shows_the_number_of_columns_of_each_row_and_the_interval_in_percent(fakes):
    text = "\n".join(ablate.summary_lines(ablate.run(TRAIN, VAL, CONFIG, *FROZEN)))
    for n, cols in ROWS.items():
        line = next(l for l in text.splitlines() if l.startswith(f"    {n} "))
        assert line.split()[1] == str(len(cols))
    assert "the middle 90% of 200 re-drawn sets" in text


def test_every_pair_line_ends_with_its_verdict_and_both_sibling_answers_are_printed_with_theirs(fakes):
    r = ablate.run(TRAIN, VAL, CONFIG, *FROZEN)
    text = "\n".join(ablate.summary_lines(r))
    for p in r["pairs"]["all"]:
        line = next(l for l in text.splitlines() if l.startswith(f"      {p['a']} minus {p['b']}:"))
        assert line.endswith(p["verdict"]) and f"{p['point']:+.4f}" in line
    sibling = text.split("Sibling count, all markets:")[1].splitlines()[0]
    assert "price_path_plus_sibling_count minus price_path_only" in sibling and "all_columns minus all_minus_sibling_count" in sibling
    for p in r["pairs"]["all"]:
        if (p["a"], p["b"]) in (("all_columns", "all_minus_sibling_count"), ("price_path_plus_sibling_count", "price_path_only")):
            assert f"{p['a']} minus {p['b']} {p['point']:+.4f} ({p['verdict']})" in sibling
