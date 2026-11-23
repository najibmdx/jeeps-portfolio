"""Tests for src/check_readme.py, on made-up results with numbers chosen so the expected text can be worked out by hand.

CI has no real outputs, so these tests build their own small outputs folder and README.
"""

import ast
import copy
import json

import pytest

import check_readme as cr
import settings

SRC = settings.PROJECT_DIR / "src"
MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]


# ---------- made-up results ----------

def side(base, pr, top_prec, lift):
    return {"baseline": base, "pr_auc": pr, "pr_auc_times_baseline": round(pr / base, 2),
            "precision_top": top_prec, "lift_top": lift}


def block(crash_pr, surv_pr, crash_base=0.7, top=(0.9, 0.5)):
    return {"crash": side(crash_base, crash_pr, top[0], 1.2), "survivor": side(round(1 - crash_base, 4), surv_pr, top[1], 1.7)}


def make_metrics():
    vals = {"logistic_regression": (0.8, 0.4), "random_forest": (0.82, 0.48), "gradient_boosting": (0.81, 0.47)}
    models = {}
    for name, (c, s) in vals.items():
        models[name] = {"train": block(c + 0.05, s + 0.1, 0.6), "validation": block(c, s)}
    return {"feature_groups": ["group2", "group3"], "n_features": 81, "top_fraction": 0.1, "models": models,
            "baselines": {"train": {"crash": 0.6, "survivor": 0.4}, "validation": {"crash": 0.7, "survivor": 0.3}}}


def make_tuned_metrics():
    out = copy.deepcopy(make_metrics())
    out["models"]["random_forest"]["validation"]["crash"]["pr_auc"] = 0.83
    out["models"]["gradient_boosting"]["validation"]["survivor"]["pr_auc"] = 0.49
    return out


def rng(low, median, high):
    return {"low": low, "median": median, "high": high}


def make_wobble():
    return {
        "n_draws": 200, "reference": "logistic_regression",
        "pr_auc_crash": {"logistic_regression": rng(0.79, 0.8, 0.81), "random_forest": rng(0.8, 0.82, 0.84),
                         "gradient_boosting": rng(0.79, 0.81, 0.83)},
        "pr_auc_survivor": {"logistic_regression": rng(0.38, 0.4, 0.42), "random_forest": rng(0.45, 0.48, 0.5),
                            "gradient_boosting": rng(0.44, 0.47, 0.5)},
        "difference_crash": {"random_forest": rng(0.005, 0.02, 0.03), "gradient_boosting": rng(-0.002, 0.01, 0.02)},
        "difference_survivor": {"random_forest": rng(0.03, 0.08, 0.1), "gradient_boosting": rng(0.0, 0.07, 0.09)},
    }


def make_tuning():
    out = {}
    for name, best in (("logistic_regression", {"C": 0.1}), ("random_forest", {"max_depth": None, "min_samples_leaf": 20}),
                       ("gradient_boosting", {"learning_rate": 0.03, "max_iter": 100})):
        out[name] = {"best": best, "best_score": 1.4,
                     "trials": [{"grid_point": best, "score": 1.4}, {"grid_point": {}, "score": 1.3}, {"grid_point": {}, "score": 1.35}]}
    return out


def make_ablation():
    def sv(c, s):
        return {"n_columns": 81, "validation": block(c, s)}
    return {"model": "gradient_boosting", "reference": "reference",
            "sets": {"reference": sv(0.81, 0.47), "plus_group1": {"n_columns": 90, "validation": block(0.82, 0.48)}},
            "wobble": {"n_draws": 200, "reference": "reference",
                       "difference_crash": {"plus_group1": rng(0.001, 0.01, 0.02)},
                       "difference_survivor": {"plus_group1": rng(-0.01, 0.006, 0.02)}},
            "flags": {"plus_group1": "helps, but treat as possible leakage"}}


def make_summary():
    return {"train_coins": 12981, "train_crash_share": 0.6, "columns_audited": 122, "columns_unverified_timing": 36,
            "migration_time_alone_auc": 0.5433,
            "flag_counts": {"extreme_bin": 0, "clock_like": 2},
            "flagged_columns": {"extreme_bin": [], "clock_like": ["group1_migrate_year", "group1_migrate_month"]}}


def counts(n, flagged, flagged_crash, crashers):
    return {"n": n, "crashers": crashers, "survivors": n - crashers, "flagged": flagged, "flagged_crash": flagged_crash,
            "skipped_good": flagged - flagged_crash, "held_crasher": crashers - flagged_crash,
            "kept_survivor": (n - crashers) - (flagged - flagged_crash)}


def make_select():
    scen = {}
    for key, (hc, sg, cutoff, flagged, fc) in {"equal": (1, 1, 0.4, 800, 700), "skipped_good_costs_3x": (1, 3, 0.6, 500, 450)}.items():
        c = counts(1000, flagged, fc, 700)
        scen[key] = {"costs": {"holding_crasher": hc, "skipped_good": sg}, "cutoff": cutoff,
                     "validation": {"counts": c, "cost_per_coin": 0.35, "cost_per_coin_flag_nothing": 0.7,
                                    "cost_per_coin_flag_everything": 0.3}}
    return {"model": "random_forest", "train_rows": 6000, "validation": {
        "rows": 1000, "crash_share": 0.7,
        "selection_table": {"logistic_regression": {"crash_pr_auc": 0.8, "survivor_pr_auc": 0.4, "average": 0.6},
                            "random_forest": {"crash_pr_auc": 0.83, "survivor_pr_auc": 0.48, "average": 0.655},
                            "gradient_boosting": {"crash_pr_auc": 0.81, "survivor_pr_auc": 0.49, "average": 0.65}}},
            "scenarios": scen}


def make_test():
    sc = {}
    for key, (cutoff, flagged, fc) in {"equal": (0.4, 520, 330), "skipped_good_costs_3x": (0.6, None, None)}.items():
        if flagged is None:                       # a cut-off that flags nothing: precision on the crash side is undefined
            c = counts(1000, 0, 0, 600)
        else:
            c = counts(1000, flagged, fc, 600)
        sc[key] = {"costs": {"holding_crasher": 1, "skipped_good": 1}, "cutoff": cutoff, "counts": c,
                   "rates": {"flag_share": round(c["flagged"] / c["n"], 4),
                             "crash_precision": None if c["flagged"] == 0 else round(c["flagged_crash"] / c["flagged"], 4),
                             "crash_recall": round(c["flagged_crash"] / c["crashers"], 4),
                             "survivor_precision": round(c["kept_survivor"] / (c["n"] - c["flagged"]), 4),
                             "survivor_recall": round(c["kept_survivor"] / c["survivors"], 4)},
                   "cost_per_coin": 0.46, "cost_per_coin_flag_nothing": 0.6, "cost_per_coin_flag_everything": 0.4,
                   "validation_cost_per_coin": 0.3,
                   "saving_per_coin_vs_cheaper_lazy_option": {"low": -0.03, "median": -0.06, "high": -0.01,
                                                              "share_of_draws_cheaper_than_lazy": 0.0},
                   }
    wob = make_wobble()
    wob["pr_auc_crash"] = {k: wob["pr_auc_crash"][k] for k in ("random_forest", "logistic_regression")}
    wob["pr_auc_survivor"] = {k: wob["pr_auc_survivor"][k] for k in ("random_forest", "logistic_regression")}
    wob["difference_crash"] = {"random_forest": wob["difference_crash"]["random_forest"]}
    wob["difference_survivor"] = {"random_forest": wob["difference_survivor"]["random_forest"]}
    return {"model": "random_forest", "reference": "logistic_regression", "n_test": 1000, "test_crash_share": 0.6,
            "ranking": {"random_forest": block(0.78, 0.52, crash_base=0.6), "logistic_regression": block(0.75, 0.5, crash_base=0.6)},
            "validation_for_comparison": block(0.83, 0.48), "wobble": wob,
            "flag_nothing": counts(1000, 0, 0, 600), "flag_everything": counts(1000, 1000, 600, 600), "scenarios": sc}


FILES = {"metrics.json": make_metrics, "select_final.json": make_select, "final_test.json": make_test,
         "tuning.json": make_tuning, "tuned_metrics.json": make_tuned_metrics, "wobble.json": make_wobble,
         "tuned_wobble.json": make_wobble, "ablation.json": make_ablation, "leakage_summary.json": make_summary}


VERIFIED = {"matched_before_filter": 1000, "unlabelled": 12, "after_filter": 500, "crash_share_after_filter_percent": 63.2,
            "migration_ranges": [["2024-12-09 22:33", "2025-01-24 14:23"], ["2025-01-24 14:24", "2025-02-06 07:13"],
                                 ["2025-02-06 07:13", "2025-02-28 23:43"]]}


@pytest.fixture
def project(tmp_path):
    """A made-up project folder: outputs/ plus a README that is exactly the printed tables."""
    out = tmp_path / "outputs"
    out.mkdir()
    for name, maker in FILES.items():
        (out / name).write_text(json.dumps(maker()), encoding="utf-8")
    import yaml
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"verified_on_real_data": VERIFIED}), encoding="utf-8")
    (tmp_path / "README.md").write_text("# readme\n\n" + cr.tables_markdown(out, {"verified_on_real_data": VERIFIED})
                                        + f"\n\nmargin {cr.f4(0.005)}\n", encoding="utf-8")
    return tmp_path


def rewrite(project, name, edit):
    path = project / "outputs" / name
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------- the happy path and exit codes ----------

def test_a_readme_built_from_the_results_passes(project, capsys):
    assert cr.main([], project=project) == 0
    assert "match the saved results" in capsys.readouterr().out


def test_no_results_gives_exit_code_2_not_a_crash(tmp_path, capsys):
    assert cr.main([], project=tmp_path) == 2
    assert "No results to check against" in capsys.readouterr().out


def test_a_missing_readme_gives_exit_code_2(project):
    (project / "README.md").unlink()
    assert cr.main([], project=project) == 2


def test_a_changed_number_gives_exit_code_1_and_names_the_row(project, capsys):
    text = (project / "README.md").read_text(encoding="utf-8")
    assert "| Random forest, untuned | 0.820 (0.800 to 0.840)" in text
    (project / "README.md").write_text(text.replace("| Random forest, untuned | 0.820 (0.800 to 0.840)", "| Random forest, untuned | 0.821 (0.800 to 0.840)", 1),
                                       encoding="utf-8")
    assert cr.main([], project=project) == 1
    assert "Wobble, untuned models (validation): random_forest" in capsys.readouterr().out


def test_a_number_in_the_wrong_row_is_caught(project):
    """Swap the two models' labels: every number is still in the README, but under the wrong model."""
    text = (project / "README.md").read_text(encoding="utf-8")
    assert "| Random forest, untuned | 0.820 (" in text and "| Gradient boosting, untuned | 0.810 (" in text
    swapped = (text.replace("| Random forest, untuned | 0.820 (", "| TMP | 0.820 (")
                   .replace("| Gradient boosting, untuned | 0.810 (", "| Random forest, untuned | 0.810 (")
                   .replace("| TMP | 0.820 (", "| Gradient boosting, untuned | 0.820 ("))
    (project / "README.md").write_text(swapped, encoding="utf-8")
    assert cr.main([], project=project) == 1


def test_every_expected_text_is_a_whole_table_row_with_its_label_first(project):
    wanted, _ = cr.expected_strings(project / "outputs", cr.load_config(project))
    rows = [text for label, text in wanted if text.startswith("|")]
    assert len(rows) >= 30
    for text in rows:
        assert text.endswith(" |") and text.count("|") >= 3


# ---------- hand-worked expectations ----------

def test_pile_rows_use_sizes_and_shares_from_the_right_files(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    assert wanted["piles: train"] == "| Train | 6,000 | 60.0% | 40.0% |"
    assert wanted["piles: validation"] == "| Validation | 1,000 | 70.0% | 30.0% |"
    assert wanted["piles: test"] == "| Test | 1,000 | 60.0% | 40.0% |"


def test_untuned_row_is_worked_out_by_hand(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    # random forest: validation crash 0.82 / 0.7 = 1.17x, survivor 0.48 / 0.3 = 1.60x; train (lazy guess 0.6 / 0.4) 0.87 / 0.6 = 1.45x and 0.58 / 0.4 = 1.45x
    assert wanted["untuned models: random_forest"] == "| Random forest | 0.820 (1.17x) | 0.480 (1.60x) | 0.870 (1.45x) | 0.580 (1.45x) | 0.900 | 0.500 |"
    assert wanted["untuned models: lazy guess"] == "| Lazy guess | 0.700 | 0.300 | 0.600 | 0.400 | - | - |"


def test_tuning_row_shows_best_worst_and_count_and_the_before_and_after(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    assert wanted["tuning: random_forest"] == ("| Random forest | {max_depth: None, min_samples_leaf: 20} | 1.400 (worst trial 1.300, 3 trials) "
                                               "| 0.820 to 0.830 | 0.480 to 0.480 |")
    assert wanted["tuning: gradient_boosting"].endswith("| 0.810 to 0.810 | 0.470 to 0.490 |")


def test_verdict_words_follow_the_range_not_the_median(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    row = wanted["Wobble, untuned models (validation): gradient_boosting"]
    assert "+0.010 (-0.002 to +0.020), cannot be told apart" in row          # range includes 0
    assert "+0.070 (+0.000 to +0.090), cannot be told apart" in row          # low edge exactly 0 still includes 0
    assert "+0.020 (+0.005 to +0.030), better" in wanted["Wobble, untuned models (validation): random_forest"]


def test_the_verdict_rule_edges():
    assert cr.verdict({"low": 0.0, "median": 0.01, "high": 0.02}) == "cannot be told apart"
    assert cr.verdict({"low": -0.02, "median": -0.01, "high": 0.0}) == "cannot be told apart"
    assert cr.verdict({"low": 0.001, "median": 0.01, "high": 0.02}) == "better"
    assert cr.verdict({"low": -0.03, "median": -0.02, "high": -0.001}) == "worse"


def test_the_reference_model_has_no_difference_cell(project):
    row = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])["Wobble, untuned models (validation): logistic_regression"]
    assert row.count("reference") == 2


def test_ablation_rows_and_the_caution_flag(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    assert wanted["ablation: reference"] == "| reference | 81 | 0.810 | - | 0.470 | - | - |"
    assert wanted["ablation: plus_group1"] == ("| plus_group1 | 90 | 0.820 | +0.010 (+0.001 to +0.020), better | 0.480 "
                                               "| +0.006 (-0.010 to +0.020), cannot be told apart | possible leakage or unverified timing |")


def test_leakage_rows(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    assert wanted["audit: columns audited"] == "| Columns audited | 122 | - |"
    assert wanted["audit: unverified timing"] == "| Columns with unverified timing | 36 | - |"
    assert wanted["audit: clock_like"] == "| Flagged clock_like | 2 | group1_migrate_year, group1_migrate_month |"
    assert wanted["audit: extreme_bin"] == "| Flagged extreme_bin | 0 | - |"
    assert wanted["audit: migration time alone"].startswith("| Migration time alone, ranking AUC (0.5 = no better than chance) | 0.5433 |")


def test_choice_margin_is_the_gap_between_the_top_two_averages(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    assert wanted["choice margin between the top two models"] == "0.0050"          # 0.655 - 0.650
    assert wanted["choice: random_forest"] == "| Random forest | 0.830 | 0.480 | 0.655 |"


def test_validation_cutoff_row_counts_the_two_mistakes_and_the_share_flagged(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    # 800 of 1000 flagged; skipped good = 800 - 700 = 100; held crashers = 700 - 700 = 0
    assert wanted["validation cut-offs: equal"] == ("| equal (holding a crasher 1, skipping a good coin 1) | 0.4000 | 80.0% | 100 | 0 "
                                                    "| 0.3500 | 0.3000 | 0.7000 |")


def test_test_cutoff_row_and_the_saving_text(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    row = wanted["test cut-offs: equal"]
    # 520 flagged, 330 of them crash: skipped good 190; crashers 600 so held 270
    assert row.startswith("| equal | 0.4000 | 52.0% | 190 | 270 | 0.4600 | 0.4000 | 0.6000 | 0.3000 |")
    assert row.endswith("-0.0600 (-0.0300 to -0.0100), cheaper in 0% |")


def test_a_cutoff_that_flags_nothing_shows_a_dash_not_a_crash(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    row = wanted["test precision and recall: skipped_good_costs_3x"]
    assert row.startswith("| skipped_good_costs_3x | - | 0.0% |")


def test_test_ranking_rows(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    assert wanted["test ranking: random_forest on validation"] == "| Random forest on validation | 0.830 (1.19x) | 0.480 (1.60x) |"
    assert wanted["test ranking: random_forest on test"] == "| Random forest on test | 0.780 (1.30x) | 0.520 (1.30x) |"
    assert wanted["test ranking: logistic_regression on test"].startswith("| Logistic regression on test (reference) |")
    assert wanted["test ranking: lazy guess"] == "| Lazy guess on test | 0.600 | 0.400 |"


# ---------- missing files ----------

def test_missing_files_are_skipped_with_notes_and_still_pass(project, capsys):
    for name in ("final_test.json", "ablation.json", "tuning.json"):
        (project / "outputs" / name).unlink()
    assert cr.main([], project=project) == 0
    out = capsys.readouterr().out
    assert "final_test.json missing" in out and "ablation.json missing" in out and "tuning.json or tuned_metrics.json missing" in out


def test_a_readme_without_the_test_tables_fails_once_the_test_file_exists(project, capsys):
    (project / "outputs" / "final_test.json").unlink()
    text = cr.tables_markdown(project / "outputs", cr.load_config(project)) + "\n\nmargin 0.0050\n"
    (project / "README.md").write_text(text, encoding="utf-8")
    assert cr.main([], project=project) == 0
    (project / "outputs" / "final_test.json").write_text(json.dumps(make_test()), encoding="utf-8")
    assert cr.main([], project=project) == 1


def test_a_missing_prose_number_is_caught(project, capsys):
    text = (project / "README.md").read_text(encoding="utf-8").replace("margin 0.0050", "margin 0.0060")
    (project / "README.md").write_text(text, encoding="utf-8")
    assert cr.main([], project=project) == 1
    assert "choice margin" in capsys.readouterr().out


def test_stale_results_are_caught_after_a_rerun(project):
    """The README is old; the outputs changed (a model got better). The check must fail."""
    rewrite(project, "metrics.json", lambda d: d["models"]["random_forest"]["validation"]["crash"].update(pr_auc=0.9, pr_auc_times_baseline=1.29))
    assert cr.main([], project=project) == 1


def test_print_tables_prints_without_checking(project, capsys):
    (project / "README.md").unlink()
    assert cr.main(["--print-tables"], project=project) == 0
    out = capsys.readouterr().out
    assert "**Untuned models**" in out and "| Random forest |" in out


def test_the_checker_never_imports_the_code_that_opens_the_test_pile():
    tree = ast.parse((SRC / "check_readme.py").read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"final_test", "holdout", "piles", "select_final"}


def test_no_two_expected_rows_are_identical_so_one_table_cannot_hide_a_stale_row_in_another(project):
    wanted, _ = cr.expected_strings(project / "outputs", cr.load_config(project))
    texts = [text for _label, text in wanted]
    assert len(texts) == len(set(texts))


def test_every_row_has_as_many_cells_as_its_header(project):
    tables, _ = cr.build_tables(project / "outputs", cr.load_config(project))
    assert len(tables) >= 13
    for title, header, rows in tables:
        for label, cells in rows:
            assert len(cells) == len(header), f"{title}: {label}"


def test_funnel_and_date_rows_come_from_the_verified_facts_in_config(project):
    wanted = dict(cr.expected_strings(project / "outputs", cr.load_config(project))[0])
    assert wanted["funnel: kept"] == "| Coins kept by the scope filter | 500 |"
    assert wanted["funnel: unlabelled"] == "| Coins set aside (no result in the label file) | 12 |"
    assert wanted["funnel: crash share"] == "| Crash share of the kept coins | 63.2% |"
    assert wanted["dates: validation"] == "| Validation pile: first and last migration (UTC) | 2025-01-24 14:24 to 2025-02-06 07:13 |"


def test_a_wrong_date_in_the_readme_is_caught(project):
    text = (project / "README.md").read_text(encoding="utf-8").replace("2025-02-06 07:13 to 2025-02-28 23:43", "2025-02-06 07:13 to 2025-02-28 23:44")
    (project / "README.md").write_text(text, encoding="utf-8")
    assert cr.main([], project=project) == 1


def test_without_a_config_file_the_funnel_is_skipped_with_a_note(project, capsys):
    (project / "config.yaml").unlink()
    (project / "README.md").write_text(cr.tables_markdown(project / "outputs") + "\n\nmargin 0.0050\n", encoding="utf-8")
    assert cr.main([], project=project) == 0
    assert "config.yaml (verified_on_real_data) not found" in capsys.readouterr().out
