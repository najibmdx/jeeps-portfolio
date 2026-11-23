"""Check that the numbers in README.md match the saved results in outputs/.

    python src/check_readme.py

Run `bash run.sh` first, so outputs/ holds the latest results. Every table row is checked as a
whole, starting with its label, so a number in the wrong row is caught. A number that appears only
in the prose is checked by itself. The numbers that come from the one-time test run (outputs/test_results.json) are checked only if that file exists, because
CI never runs the test set.

Exit code: 0 all numbers found, 1 a number is missing or different, 2 no results to check against.
"""

import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def s3(x):
    return f"{x:+.3f}"


def pct(x):
    return f"{x * 100:.2f}%"


def read_json(outputs: Path, name: str):
    path = outputs / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def span(d, f=f3):
    return f"{f(d['low'])} to {f(d['high'])}"


def sspan(d):
    return f"{s3(d['low'])} to {s3(d['high'])}"


NAMES = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting"}


def short_verdict(text: str) -> str:
    """The saved verdict sentence, shortened the way the README tables show it."""
    if "cannot" in text:
        return "cannot tell"
    return "better" if "better" in text else "worse"


def wobble_verdict(d) -> str:
    """Does the range of a difference include 0?"""
    if d["low"] <= 0 <= d["high"]:
        return "cannot be told apart"
    return "better" if d["low"] > 0 else "worse"


def expected_strings(outputs: Path):
    """Return ([(label, text that must appear in the README)], [notes about what was skipped]).

    Table numbers are checked as a whole table row, starting with the row's own label, so a
    number that sits in the wrong row (the wrong model, setup or scenario) is caught.
    Numbers that appear only in the prose are checked by themselves.
    """
    wanted, notes = [], []

    metrics = read_json(outputs, "metrics.json")
    for name, m in metrics["models"].items():
        wanted.append((f"models table: {name}",
                       f"| {NAMES[name]} | {f3(m['train_pr_auc'])} ({f2(m['train_times_blind_guess'])}x) | "
                       f"{f3(m['validation_pr_auc'])} ({f2(m['validation_times_blind_guess'])}x) | "
                       f"{f3(m['validation_precision_at_k'])} | {f3(m['validation_recall_at_k'])} | {f2(m['validation_lift_at_k'])} |"))
    a = metrics["always_no"]
    wanted.append(("models table: always no",
                   f"| Always \"no\" | - | {f4(a['pr_auc_blind_guess'])} (blind guess) | {f3(a['precision'])} | {f3(a['recall'])} | - |"))
    wanted += [
        ("blind guess, validation", f4(metrics["validation_baseline"])),
        ("blind guess, training", f4(metrics["train_baseline"])),
        ("train yes-rate", pct(metrics["train_baseline"])),
        ("validation yes-rate", pct(metrics["validation_baseline"])),
    ]

    wobble = read_json(outputs, "wobble.json")
    if wobble:
        for name in MODELS:
            diff = "-" if name == wobble["reference"] else f"{s3(wobble['pr_auc_minus_reference'][name]['median'])} ({sspan(wobble['pr_auc_minus_reference'][name])})"
            verdict = "reference" if name == wobble["reference"] else wobble_verdict(wobble["pr_auc_minus_reference"][name])
            wanted.append((f"wobble table: {name}",
                           f"| {NAMES[name]} | {f3(wobble['pr_auc'][name]['median'])} ({span(wobble['pr_auc'][name])}) | "
                           f"{f2(wobble['lift_at_k'][name]['median'])} ({span(wobble['lift_at_k'][name], f2)}) | {diff} | {verdict} |"))
    else:
        notes.append("wobble.json missing: wobble numbers not checked")

    ablation = read_json(outputs, "ablation.json")
    if ablation:
        for name in MODELS:
            base = ablation["models"][name]["pr_auc"][ablation["reference"]]["median"]
            wanted.append((f"ablation baseline {name}", f"{NAMES[name].lower()} {f3(base)}"))
        setups = list(ablation["models"][MODELS[0]]["change_vs_baseline"])
        for setup in setups:
            cells = []
            for name in MODELS:
                d = ablation["models"][name]["change_vs_baseline"][setup]
                cells.append(f"{s3(d['median'])} ({sspan(d)}), {short_verdict(d['verdict'])}")
            wanted.append((f"ablation table: {setup}", f"| {setup} | " + " | ".join(cells) + " |"))
    else:
        notes.append("ablation.json missing: ablation numbers not checked")

    tuning = read_json(outputs, "tuning_results.json")
    if tuning:
        for name, r in tuning["models"].items():
            best = str(r["best_params"]).replace("'", "")
            change = r["validation_change"]
            wanted.append((f"tuning table: {name}",
                           f"| {NAMES[name]} | {best} | {f3(r['default_cv']['cv_mean'])} +/- {f3(r['default_cv']['cv_sd'])} | "
                           f"{f3(r['tuned_cv']['cv_mean'])} +/- {f3(r['tuned_cv']['cv_sd'])} | {r['folds_improved']} of {r['n_folds']} | "
                           f"{f3(r['validation_pr_auc']['default'])} to {f3(r['validation_pr_auc']['tuned'])} | "
                           f"{s3(change['median'])} ({sspan(change)}), {short_verdict(change['verdict'])} |"))
    else:
        notes.append("tuning_results.json missing: tuning numbers not checked")

    audit_path = outputs / "leakage_audit.csv"
    if audit_path.exists():
        with open(audit_path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["flagged"] == "True":
                    wanted.append((f"audit table: {r['column']} {r['segment']}",
                                   f"| {r['column']} | {r['segment']} | {int(r['rows']):,} | {pct(float(r['yes_rate']))} | {f2(float(r['times_base_rate']))} |"))
    else:
        notes.append("leakage_audit.csv missing: audit numbers not checked")

    test = read_json(outputs, "test_results.json")
    if test:
        y = test["yes_rate"]
        a = test["always_no"]
        wanted += [
            ("test table: model", f"| Model | {test['model']}, C = {test['settings']['C']}, max_iter = {test['settings']['max_iter']}, fitted on training rows only |"),
            ("test table: clients", f"| Test clients | {test['test_clients']:,} |"),
            ("test table: yes-rates", f"| Yes-rate: train / validation / test | {pct(y['train'])} / {pct(y['validation'])} / {pct(y['test'])} |"),
            ("test table: PR-AUC", f"| Test PR-AUC (middle 90%) | {f3(test['test_pr_auc'])} ({span(test['test_pr_auc_range'])}) |"),
            ("test table: blind guess", f"| Blind guess PR-AUC on the same set | {f3(a['pr_auc_blind_guess'])} |"),
            ("test table: multiple", f"| Test PR-AUC as a multiple of a blind guess | {f2(test['test_times_blind_guess'])}x |"),
            ("test table: top 10%", f"| Top 10% ({test['top_k']} calls): precision / recall / lift | "
                                   f"{f3(test['test_precision_at_k'])} / {f3(test['test_recall_at_k'])} / {f2(test['test_lift_at_k'])} |"),
            ("test table: always no", f"| Always \"no\": accuracy / precision / recall | {f3(a['accuracy'])} / {f3(a['precision'])} / {f3(a['recall'])} |"),
        ]
        for key, sc in test["scenarios"].items():
            v, t = sc["validation"], sc["test"]
            wanted.append((f"scenario table: {key}",
                           f"| {key.replace('calls_', '')} ({sc['cost_per_call']}) | {v['share'] * 100:.0f}% | {f2(v['net_per_100_clients'])} | "
                           f"{f2(v['call_everyone_net_per_100_clients'])} | {f2(t['net_per_100_clients'])} | {f2(t['call_everyone_net_per_100_clients'])} |"))
    else:
        notes.append("test_results.json missing: one-time test numbers not checked (normal in CI)")
    return wanted, notes


def find_missing(readme_text: str, wanted):
    """Return the (label, text) pairs whose text is not in the README."""
    return [(label, text) for label, text in wanted if text not in readme_text]


def main() -> int:
    outputs = PROJECT / "outputs"
    if read_json(outputs, "metrics.json") is None:
        print("No results to check against. Run `bash run.sh` first.")
        return 2
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    wanted, notes = expected_strings(outputs)
    missing = find_missing(readme, wanted)
    for note in notes:
        print("note:", note)
    if missing:
        print(f"{len(missing)} of {len(wanted)} numbers are not in README.md as saved:")
        for label, text in missing:
            print(f"  {label}: expected {text!r}")
        return 1
    print(f"README numbers match the saved results ({len(wanted)} checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
