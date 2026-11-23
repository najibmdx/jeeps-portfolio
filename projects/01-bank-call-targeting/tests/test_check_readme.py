"""Tests for the README number checker, on small made-up results."""

import csv
import json

import pytest

import check_readme

MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]


def rng(low, median, high):
    return {"low": low, "median": median, "high": high}


def write_outputs(folder, with_test=True):
    """Made-up results with the same shape as the real files."""
    folder.mkdir(parents=True, exist_ok=True)
    # each model gets slightly different numbers, so a swap between two models can be seen
    metrics = {"train_baseline": 0.0627, "validation_baseline": 0.0905,
               "always_no": {"pr_auc_blind_guess": 0.0905, "precision": 0.0, "recall": 0.0},
               "models": {n: {"train_pr_auc": 0.1187 + i / 100, "train_times_blind_guess": 1.89 + i / 10,
                              "validation_pr_auc": 0.1448 + i / 100, "validation_times_blind_guess": 1.6 + i / 10,
                              "validation_precision_at_k": 0.1672 + i / 100, "validation_recall_at_k": 0.1851 + i / 100,
                              "validation_lift_at_k": 1.85 + i / 10} for i, n in enumerate(MODELS)}}
    wobble = {"reference": "logistic_regression",
              "pr_auc": {n: rng(0.128 + i / 100, 0.146 + i / 100, 0.169 + i / 100) for i, n in enumerate(MODELS)},
              "lift_at_k": {n: rng(1.6 + i / 10, 1.86 + i / 10, 2.11 + i / 10) for i, n in enumerate(MODELS)},
              "pr_auc_minus_reference": {n: rng(-0.005, 0.006, 0.018) for n in MODELS[1:]}}
    verdict = "cannot be told apart (range includes 0)"
    better = "better in nearly every re-drawn exam"
    ablation = {"reference": "baseline", "models": {n: {
        "pr_auc": {"baseline": rng(0.128, 0.146 + i / 100, 0.169)},
        "change_vs_baseline": {"plus_campaign": {**rng(0.003, 0.006 + i / 100, 0.009), "verdict": better},
                               "plus_economy": {**rng(-0.009, -0.003, 0.003), "verdict": verdict}}} for i, n in enumerate(MODELS)}}
    tuning = {"models": {n: {
        "best_params": {"C": 0.01},
        "default_cv": {"cv_mean": 0.237 + i / 100, "cv_sd": 0.194}, "tuned_cv": {"cv_mean": 0.242 + i / 100, "cv_sd": 0.202},
        "validation_pr_auc": {"default": 0.1448, "tuned": 0.1508},
        "validation_change": {**rng(0.0023, 0.0057, 0.0103), "verdict": better},
        "folds_improved": 3, "n_folds": 4} for i, n in enumerate(MODELS)}}
    for name, data in [("metrics", metrics), ("wobble", wobble), ("ablation", ablation), ("tuning_results", tuning)]:
        (folder / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    with open(folder / "leakage_audit.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["column", "decision", "segment", "rows", "yes_rate", "times_base_rate", "flagged"])
        w.writerow(["month", "ablation (calendar)", "oct", 67, 0.6269, 10.0, True])
        w.writerow(["age", "feature (client)", "(17.999, 31.0]", 6168, 0.0809, 1.29, False])
    if with_test:
        test = {"model": "logistic_regression", "settings": {"max_iter": 1000, "C": 0.01},
                "yes_rate": {"train": 0.0627, "validation": 0.0905, "test": 0.445},
                "test_clients": 4962, "test_pr_auc": 0.6072,
                "test_pr_auc_range": {"low": 0.589, "median": 0.6071, "high": 0.6281},
                "always_no": {"accuracy": 0.555, "precision": 0.0, "recall": 0.0, "pr_auc_blind_guess": 0.445},
                "test_times_blind_guess": 1.36,
                "top_k": 497, "test_precision_at_k": 0.7062, "test_recall_at_k": 0.159, "test_lift_at_k": 1.59,
                "scenarios": {k: {"cost_per_call": c,
                                  "validation": {"share": sh, "net_per_100_clients": 6.08, "call_everyone_net_per_100_clients": 6.05},
                                  "test": {"net_per_100_clients": n, "call_everyone_net_per_100_clients": 41.5}}
                              for k, c, sh, n in [("calls_cheap", 0.03, 0.97, 40.9), ("calls_expensive", 0.15, 0.05, 2.92)]}}
        (folder / "test_results.json").write_text(json.dumps(test), encoding="utf-8")


def readme_for(folder):
    """A README that contains exactly the expected numbers."""
    wanted, _ = check_readme.expected_strings(folder)
    return "\n".join(text for _, text in wanted)


def test_a_readme_that_has_every_number_passes(tmp_path):
    write_outputs(tmp_path)
    wanted, notes = check_readme.expected_strings(tmp_path)
    assert wanted and not notes
    assert check_readme.find_missing(readme_for(tmp_path), wanted) == []


def test_a_changed_number_in_the_readme_is_caught(tmp_path):
    write_outputs(tmp_path)
    wanted, _ = check_readme.expected_strings(tmp_path)
    wrong = readme_for(tmp_path).replace("0.607 (0.589 to 0.628)", "0.670 (0.589 to 0.628)")
    missing = check_readme.find_missing(wrong, wanted)
    assert [label for label, _ in missing] == ["test table: PR-AUC"]


def test_a_changed_result_makes_the_old_readme_stale(tmp_path):
    write_outputs(tmp_path)
    old_readme = readme_for(tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    metrics["models"]["random_forest"]["validation_pr_auc"] = 0.2
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    wanted, _ = check_readme.expected_strings(tmp_path)
    assert [label for label, _ in check_readme.find_missing(old_readme, wanted)] == ["models table: random_forest"]


def swap_labels(text, a, b):
    return text.replace(a, "@@@").replace(b, a).replace("@@@", b)


def test_numbers_under_the_wrong_model_label_are_caught(tmp_path):
    """Codex review finding 1: swapping two row labels used to pass, because numbers were found anywhere."""
    write_outputs(tmp_path)
    wanted, _ = check_readme.expected_strings(tmp_path)
    swapped = swap_labels(readme_for(tmp_path), "| Random forest |", "| Gradient boosting |")
    labels = {label for label, _ in check_readme.find_missing(swapped, wanted)}
    assert {"models table: random_forest", "models table: gradient_boosting"} <= labels
    assert "wobble table: random_forest" in labels and "tuning table: gradient_boosting" in labels


def test_numbers_under_the_wrong_scenario_label_are_caught(tmp_path):
    write_outputs(tmp_path)
    wanted, _ = check_readme.expected_strings(tmp_path)
    swapped = swap_labels(readme_for(tmp_path), "| cheap (0.03) |", "| expensive (0.15) |")
    labels = {label for label, _ in check_readme.find_missing(swapped, wanted)}
    assert labels == {"scenario table: calls_cheap", "scenario table: calls_expensive"}


def test_a_stale_best_setting_is_caught_both_ways(tmp_path):
    """Codex review finding 2: the 'best settings tried' column used to be unchecked."""
    write_outputs(tmp_path)
    wanted, _ = check_readme.expected_strings(tmp_path)
    readme_edited = readme_for(tmp_path).replace("{C: 0.01}", "{C: 10}")           # README changed by hand
    assert any(label.startswith("tuning table") for label, _ in check_readme.find_missing(readme_edited, wanted))
    old_readme = readme_for(tmp_path)                                               # saved results changed, README old
    tuning = json.loads((tmp_path / "tuning_results.json").read_text())
    tuning["models"]["logistic_regression"]["best_params"] = {"C": 10}
    (tmp_path / "tuning_results.json").write_text(json.dumps(tuning))
    wanted, _ = check_readme.expected_strings(tmp_path)
    assert [label for label, _ in check_readme.find_missing(old_readme, wanted)] == ["tuning table: logistic_regression"]


def test_the_one_time_test_numbers_are_skipped_when_that_file_is_missing(tmp_path):
    write_outputs(tmp_path, with_test=False)
    wanted, notes = check_readme.expected_strings(tmp_path)
    assert not any(label.startswith("test table") for label, _ in wanted)
    assert any("test_results.json missing" in note for note in notes)


def test_main_says_there_is_nothing_to_check_when_there_are_no_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check_readme, "PROJECT", tmp_path)
    assert check_readme.main() == 2
    assert "run.sh" in capsys.readouterr().out


def test_main_returns_1_when_a_number_is_wrong_and_0_when_all_match(tmp_path, monkeypatch):
    write_outputs(tmp_path / "outputs")
    monkeypatch.setattr(check_readme, "PROJECT", tmp_path)
    (tmp_path / "README.md").write_text(readme_for(tmp_path / "outputs"), encoding="utf-8")
    assert check_readme.main() == 0
    (tmp_path / "README.md").write_text("no numbers here", encoding="utf-8")
    assert check_readme.main() == 1


def test_the_real_readme_has_the_sections_the_brief_asks_for():
    text = (check_readme.PROJECT / "README.md").read_text(encoding="utf-8")
    for needle in ["10.24432/C5K306", "Moro", "AI tools used", "Numbers that could mislead", "bash run.sh"]:
        assert needle in text
