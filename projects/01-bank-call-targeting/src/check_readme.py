"""Check that the numbers in README.md match the saved results in outputs/.

    python src/check_readme.py

Run `bash run.sh` first, so outputs/ holds the latest results. The numbers that come from the
one-time test run (outputs/test_results.json) are checked only if that file exists, because
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


def expected_strings(outputs: Path):
    """Return ([(label, text that must appear in the README)], [notes about what was skipped])."""
    wanted, notes = [], []

    metrics = read_json(outputs, "metrics.json")
    for name, m in metrics["models"].items():
        wanted += [
            (f"{name} train", f"{f3(m['train_pr_auc'])} ({f2(m['train_times_blind_guess'])}x)"),
            (f"{name} validation", f"{f3(m['validation_pr_auc'])} ({f2(m['validation_times_blind_guess'])}x)"),
            (f"{name} precision", f"| {f3(m['validation_precision_at_k'])} | {f3(m['validation_recall_at_k'])} | {f2(m['validation_lift_at_k'])} |"),
        ]
    wanted += [
        ("blind guess, validation", f4(metrics["validation_baseline"])),
        ("blind guess, training", f4(metrics["train_baseline"])),
        ("train yes-rate", pct(metrics["train_baseline"])),
        ("validation yes-rate", pct(metrics["validation_baseline"])),
    ]

    wobble = read_json(outputs, "wobble.json")
    if wobble:
        for name in MODELS:
            wanted.append((f"wobble PR-AUC {name}", f"{f3(wobble['pr_auc'][name]['median'])} ({span(wobble['pr_auc'][name])})"))
            wanted.append((f"wobble lift {name}", f"{f2(wobble['lift_at_k'][name]['median'])} ({span(wobble['lift_at_k'][name], f2)})"))
        for name, d in wobble["pr_auc_minus_reference"].items():
            wanted.append((f"wobble difference {name}", f"{s3(d['median'])} ({sspan(d)})"))
    else:
        notes.append("wobble.json missing: wobble numbers not checked")

    ablation = read_json(outputs, "ablation.json")
    if ablation:
        for name in MODELS:
            m = ablation["models"][name]
            wanted.append((f"ablation baseline {name}", f"{NAMES[name].lower()} {f3(m['pr_auc'][ablation['reference']]['median'])}"))
            for setup, d in m["change_vs_baseline"].items():
                wanted.append((f"ablation {name} {setup}", f"{s3(d['median'])} ({sspan(d)})"))
    else:
        notes.append("ablation.json missing: ablation numbers not checked")

    tuning = read_json(outputs, "tuning_results.json")
    if tuning:
        for name, r in tuning["models"].items():
            wanted += [
                (f"tuning {name} default", f"{f3(r['default_cv']['cv_mean'])} +/- {f3(r['default_cv']['cv_sd'])}"),
                (f"tuning {name} tuned", f"{f3(r['tuned_cv']['cv_mean'])} +/- {f3(r['tuned_cv']['cv_sd'])}"),
                (f"tuning {name} validation", f"{f3(r['validation_pr_auc']['default'])} to {f3(r['validation_pr_auc']['tuned'])}"),
                (f"tuning {name} change", f"{s3(r['validation_change']['median'])} ({sspan(r['validation_change'])})"),
                (f"tuning {name} folds", f"{r['folds_improved']} of {r['n_folds']}"),
            ]
    else:
        notes.append("tuning_results.json missing: tuning numbers not checked")

    audit_path = outputs / "leakage_audit.csv"
    if audit_path.exists():
        with open(audit_path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["flagged"] == "True":
                    wanted.append((f"audit {r['column']} {r['segment']}",
                                   f"| {r['column']} | {r['segment']} | {int(r['rows']):,} | {pct(float(r['yes_rate']))} |"))
    else:
        notes.append("leakage_audit.csv missing: audit numbers not checked")

    test = read_json(outputs, "test_results.json")
    if test:
        wanted += [
            ("test clients", f"{test['test_clients']:,}"),
            ("test yes-rate", pct(test["yes_rate"]["test"])),
            ("test PR-AUC", f"{f3(test['test_pr_auc'])} ({span(test['test_pr_auc_range'])})"),
            ("test blind guess", f3(test["always_no"]["pr_auc_blind_guess"])),
            ("test multiple", f"{f2(test['test_times_blind_guess'])}x"),
            ("test top 10%", f"{f3(test['test_precision_at_k'])} / {f3(test['test_recall_at_k'])} / {f2(test['test_lift_at_k'])}"),
            ("test top 10% calls", f"({test['top_k']} calls)"),
        ]
        for key, s in test["scenarios"].items():
            wanted.append((f"scenario {key}",
                           f"| {s['validation']['share'] * 100:.0f}% | {f2(s['validation']['net_per_100_clients'])} | "
                           f"{f2(s['validation']['call_everyone_net_per_100_clients'])} | {f2(s['test']['net_per_100_clients'])} | "
                           f"{f2(s['test']['call_everyone_net_per_100_clients'])} |"))
    else:
        notes.append("test_results.json missing: one-time test numbers not checked (normal in CI)")
    return wanted, notes


NAMES = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting"}


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
