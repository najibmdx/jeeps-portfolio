"""Check that the numbers in README.md match the saved results in outputs/.

    python src/check_readme.py                run the check
    python src/check_readme.py --print-tables print every results table as Markdown, from outputs/

Run `python src/pipeline.py` first (and `python src/final_test.py --confirm-open-test` was run once
earlier), so outputs/ holds the latest results. Every table row is checked as a whole, starting with
the row's own label, so a number sitting in the wrong row is caught. A few numbers that appear only in
the prose are checked by themselves (see prose_numbers). The data-funnel and date tables come from
config.yaml (verified_on_real_data), which prepare.py reproduces on the real data before it saves anything.

Files that are missing are skipped with a note, because outputs/ is never committed and CI has none
of the real results. The check cannot run in CI; its own tests do (on made-up results).

Exit code: 0 all numbers found, 1 a number is missing or different, 2 no results to check against.
"""

import json
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent
MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]
NAMES = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
         "gradient_boosting": "Gradient boosting"}
SIDES = ("crash", "survivor")


# ---------- number formats (one place, so README and check always agree) ----------

def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def s3(x):
    return f"{x:+.3f}"


def s4(x):
    return f"{x:+.4f}"


def pct1(x):
    return f"{x * 100:.1f}%"


def pct0(x):
    return f"{x * 100:.0f}%"


def name_of(model):
    return NAMES.get(model, model)


def read_json(outputs: Path, name: str):
    path = outputs / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def span(d, f=f3):
    return f"{f(d['low'])} to {f(d['high'])}"


def sspan(d, f=s3):
    return f"{f(d['low'])} to {f(d['high'])}"


def verdict(d) -> str:
    """Does the range of a difference include 0? The same rule as src/uncertainty.py, in short words."""
    if d["low"] <= 0 <= d["high"]:
        return "cannot be told apart"
    return "better" if d["low"] > 0 else "worse"


def crash_survivor_cells(m, f=f3):
    """'0.823 (1.15x)' for each side of one side_metrics block."""
    return {s: f"{f(m[s]['pr_auc'])} ({f2(m[s]['pr_auc_times_baseline'])}x)" for s in SIDES}


def table_md(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def cell_or_dash(x, f=pct1):
    return "-" if x is None else f(x)


# ---------- the tables ----------

def piles_table(metrics, select, test):
    """Pile sizes and crash shares. Train size and validation size come from select_final.json."""
    rows = []
    if metrics is not None:
        base = metrics["baselines"]
        sizes = {"train": select["train_rows"] if select else None,
                 "validation": select["validation"]["rows"] if select else None}
        for pile in ("train", "validation"):
            if sizes[pile] is None:
                continue
            c = base[pile]["crash"]
            rows.append((f"piles: {pile}", [pile.capitalize(), f"{sizes[pile]:,}", pct1(c), pct1(1 - c)]))
    if test is not None:
        c = test["test_crash_share"]
        rows.append(("piles: test", ["Test", f"{test['n_test']:,}", pct1(c), pct1(1 - c)]))
    header = ["Pile", "Coins", "Crash share (lazy guess for the crash side)", "Survivor share (lazy guess for the survivor side)"]
    return "Piles", header, rows


def funnel_table(verified):
    """The data funnel and the pile dates, from the facts prepare.py checks against the real files."""
    v = verified
    rows = [
        ("funnel: matched", ["Coins with a result", f"{v['matched_before_filter']:,}"]),
        ("funnel: unlabelled", ["Coins set aside (no result in the label file)", f"{v['unlabelled']:,}"]),
        ("funnel: kept", ["Coins kept by the scope filter", f"{v['after_filter']:,}"]),
        ("funnel: crash share", ["Crash share of the kept coins", f"{v['crash_share_after_filter_percent']}%"]),
    ]
    for pile, (start, end) in zip(("Train", "Validation", "Test"), v["migration_ranges"]):
        rows.append((f"dates: {pile.lower()}", [f"{pile} pile: first and last migration (UTC)", f"{start} to {end}"]))
    return "Data funnel and pile dates", ["Measure", "Value"], rows


def untuned_table(metrics):
    rows = []
    for name, m in metrics["models"].items():
        v, t = crash_survivor_cells(m["validation"]), crash_survivor_cells(m["train"])
        rows.append((f"untuned models: {name}",
                     [name_of(name), v["crash"], v["survivor"], t["crash"], t["survivor"],
                      f3(m["validation"]["crash"]["precision_top"]), f3(m["validation"]["survivor"]["precision_top"])]))
    b = metrics["baselines"]
    rows.append(("untuned models: lazy guess",
                 ["Lazy guess", f3(b["validation"]["crash"]), f3(b["validation"]["survivor"]),
                  f3(b["train"]["crash"]), f3(b["train"]["survivor"]), "-", "-"]))
    header = ["Model", "Validation, crash side: PR-AUC (x lazy guess)", "Validation, survivor side: PR-AUC (x lazy guess)",
              "Train, crash side", "Train, survivor side", "Validation precision, top 10%, crash side",
              "Validation precision, top 10%, survivor side"]
    return "Untuned models", header, rows


def wobble_table(wobble, label, tag):
    """`tag` goes in the first cell of every row, so two wobble tables can never hold an identical row."""
    ref = wobble["reference"]
    rows = []
    for name in wobble["pr_auc_crash"]:
        cells = [f"{name_of(name)}, {tag}"]
        for side in SIDES:
            p = wobble[f"pr_auc_{side}"][name]
            cells.append(f"{f3(p['median'])} ({span(p)})")
            if name == ref:
                cells.append("reference")
            else:
                d = wobble[f"difference_{side}"][name]
                cells.append(f"{s3(d['median'])} ({sspan(d)}), {verdict(d)}")
        rows.append((f"{label}: {name}", cells))
    header = ["Model", "Crash PR-AUC, middle 90%", f"Crash: difference vs {name_of(ref).lower()}",
              "Survivor PR-AUC, middle 90%", f"Survivor: difference vs {name_of(ref).lower()}"]
    return label, header, rows


def tuning_table(tuning, metrics, tuned_metrics):
    rows = []
    for name, r in tuning.items():
        scores = [t["score"] for t in r["trials"]]
        before, after = metrics["models"][name]["validation"], tuned_metrics["models"][name]["validation"]
        best = str(r["best"]).replace("'", "")
        rows.append((f"tuning: {name}",
                     [name_of(name), best, f"{f3(r['best_score'])} (worst trial {f3(min(scores))}, {len(scores)} trials)",
                      f"{f3(before['crash']['pr_auc'])} to {f3(after['crash']['pr_auc'])}",
                      f"{f3(before['survivor']['pr_auc'])} to {f3(after['survivor']['pr_auc'])}"]))
    header = ["Model", "Best settings tried", "Time-ordered tuning score: best (x lazy guess, average of both sides)",
              "Validation crash PR-AUC: untuned to tuned", "Validation survivor PR-AUC: untuned to tuned"]
    return "Tuning, before and after", header, rows


def ablation_table(ablation):
    ref = ablation["reference"]
    sets, wobble, flags = ablation["sets"], ablation["wobble"], ablation.get("flags", {})
    rows = []
    for setup, s in sets.items():
        cells = [setup, str(s["n_columns"])]
        for side in SIDES:
            cells.append(f3(s["validation"][side]["pr_auc"]))
            if setup == ref:
                cells.append("-")
            else:
                d = wobble[f"difference_{side}"][setup]
                cells.append(f"{s3(d['median'])} ({sspan(d)}), {verdict(d)}")
        cells.append("possible leakage or unverified timing" if setup in flags else "-")
        rows.append((f"ablation: {setup}", cells))
    header = ["Setup", "Columns", "Crash PR-AUC", "Crash: change vs reference", "Survivor PR-AUC",
              "Survivor: change vs reference", "Caution"]
    return f"Column tests (model: {ablation['model']})", header, rows


def leakage_table(summary):
    rows = [
        ("audit: columns audited", ["Columns audited", f"{summary['columns_audited']:,}", "-"]),
        ("audit: unverified timing", ["Columns with unverified timing", f"{summary['columns_unverified_timing']:,}", "-"]),
        ("audit: migration time alone",
         ["Migration time alone, ranking AUC (0.5 = no better than chance)", f4(summary["migration_time_alone_auc"]), "-"]),
    ]
    for kind, count in summary["flag_counts"].items():
        names = ", ".join(summary["flagged_columns"].get(kind, [])) or "-"
        rows.append((f"audit: {kind}", [f"Flagged {kind}", str(count), names]))
    return "Leakage audit", ["Measure", "Value", "Columns"], rows


def choice_table(select):
    rows = []
    for name, r in select["validation"]["selection_table"].items():
        rows.append((f"choice: {name}", [name_of(name), f3(r["crash_pr_auc"]), f3(r["survivor_pr_auc"]), f3(r["average"])]))
    header = ["Tuned model", "Validation crash PR-AUC", "Validation survivor PR-AUC", "Average (the choice rule)"]
    return "Model choice on validation", header, rows


def margin_between_first_two(select):
    avgs = sorted((r["average"] for r in select["validation"]["selection_table"].values()), reverse=True)
    return avgs[0] - avgs[1]


def cutoff_table(select):
    rows = []
    for key, sc in select["scenarios"].items():
        v, c = sc["validation"], sc["validation"]["counts"]
        costs = sc["costs"]
        rows.append((f"validation cut-offs: {key}",
                     [f"{key} (holding a crasher {costs['holding_crasher']}, skipping a good coin {costs['skipped_good']})",
                      f4(sc["cutoff"]), pct1(c["flagged"] / c["n"]), f"{c['skipped_good']:,}", f"{c['held_crasher']:,}",
                      f4(v["cost_per_coin"]), f4(v["cost_per_coin_flag_everything"]), f4(v["cost_per_coin_flag_nothing"])]))
    header = ["Scenario (invented costs)", "Cut-off", "Share of coins flagged", "Skipped good coins", "Held crashers",
              "Cost per coin: this cut-off", "Cost per coin: flag everything", "Cost per coin: flag nothing"]
    return "Cut-offs chosen on validation", header, rows


def test_ranking_table(test):
    m, ref = test["model"], test["reference"]
    val = test["validation_for_comparison"]
    cell = lambda x: f"{f3(x['pr_auc'])} ({f2(x['pr_auc_times_baseline'])}x)"
    base = test["ranking"][m]
    rows = [(f"test ranking: {m} on validation", [f"{name_of(m)} on validation", cell(val["crash"]), cell(val["survivor"])])]
    for name, r in test["ranking"].items():
        label = f"{name_of(name)} on test" + ("" if name == m else " (reference)")
        rows.append((f"test ranking: {name} on test", [label, cell(r["crash"]), cell(r["survivor"])]))
    rows.append(("test ranking: lazy guess",
                 ["Lazy guess on test", f3(base["crash"]["baseline"]), f3(base["survivor"]["baseline"])]))
    return "Ranking quality on the test pile", ["Row", "Crash side: PR-AUC (x lazy guess)", "Survivor side: PR-AUC (x lazy guess)"], rows


def test_cutoff_table(test):
    rows = []
    for key, sc in test["scenarios"].items():
        c, s = sc["counts"], sc["saving_per_coin_vs_cheaper_lazy_option"]
        rows.append((f"test cut-offs: {key}",
                     [key, f4(sc["cutoff"]), pct1(sc["rates"]["flag_share"]), f"{c['skipped_good']:,}", f"{c['held_crasher']:,}",
                      f4(sc["cost_per_coin"]), f4(sc["cost_per_coin_flag_everything"]), f4(sc["cost_per_coin_flag_nothing"]),
                      f4(sc["validation_cost_per_coin"]),
                      f"{s4(s['median'])} ({sspan(s, s4)}), cheaper in {pct0(s['share_of_draws_cheaper_than_lazy'])}"]))
    header = ["Scenario (invented costs)", "Cut-off (frozen)", "Share flagged", "Skipped good coins", "Held crashers",
              "Cost per coin: frozen cut-off", "Cost per coin: flag everything", "Cost per coin: flag nothing",
              "Same cut-off on validation", "Saving per coin vs the cheaper lazy option (middle 90%)"]
    return "Frozen cut-offs on the test pile", header, rows


def test_rates_table(test):
    rows = []
    for key, sc in test["scenarios"].items():
        r = sc["rates"]
        rows.append((f"test precision and recall: {key}",
                     [key, cell_or_dash(r["crash_precision"]), cell_or_dash(r["crash_recall"]),
                      cell_or_dash(r["survivor_precision"]), cell_or_dash(r["survivor_recall"])]))
    header = ["Scenario", "Crash precision", "Crash recall", "Survivor precision", "Survivor recall"]
    return "Precision and recall at the frozen cut-offs on the test pile", header, rows


def build_tables(outputs: Path, config=None):
    """Return ([(title, header, [(label, [cells])])], [notes about what was skipped])."""
    tables, notes = [], []
    if config and config.get("verified_on_real_data"):
        tables.append(funnel_table(config["verified_on_real_data"]))
    else:
        notes.append("config.yaml (verified_on_real_data) not found: data funnel and dates not checked")
    metrics = read_json(outputs, "metrics.json")
    select = read_json(outputs, "select_final.json")
    test = read_json(outputs, "final_test.json")
    tuning = read_json(outputs, "tuning.json")
    tuned_metrics = read_json(outputs, "tuned_metrics.json")
    wobble = read_json(outputs, "wobble.json")
    tuned_wobble = read_json(outputs, "tuned_wobble.json")
    ablation = read_json(outputs, "ablation.json")
    summary = read_json(outputs, "leakage_summary.json")

    tables.append(piles_table(metrics, select, test))
    if select is None:
        notes.append("select_final.json missing: train and validation pile sizes, model choice and validation cut-offs not checked")
    if test is None:
        notes.append("final_test.json missing: the one-time test numbers are not checked (normal in CI)")
    tables.append(untuned_table(metrics))
    if wobble:
        tables.append(wobble_table(wobble, "Wobble, untuned models (validation)", "untuned"))
    else:
        notes.append("wobble.json missing: untuned wobble not checked")
    if tuning and tuned_metrics:
        tables.append(tuning_table(tuning, metrics, tuned_metrics))
    else:
        notes.append("tuning.json or tuned_metrics.json missing: tuning table not checked")
    if tuned_wobble:
        tables.append(wobble_table(tuned_wobble, "Wobble, tuned models (validation)", "tuned"))
    else:
        notes.append("tuned_wobble.json missing: tuned wobble not checked")
    if ablation:
        tables.append(ablation_table(ablation))
    else:
        notes.append("ablation.json missing: column tests not checked")
    if summary:
        tables.append(leakage_table(summary))
    else:
        notes.append("leakage_summary.json missing: leakage audit numbers not checked")
    if select:
        tables.append(choice_table(select))
        tables.append(cutoff_table(select))
    if test:
        tables.append(test_ranking_table(test))
        tables.append(wobble_table(test["wobble"], "Wobble on the test pile", "test pile"))
        tables.append(test_cutoff_table(test))
        tables.append(test_rates_table(test))
    return tables, notes


def prose_numbers(outputs: Path):
    """Numbers that sit in sentences, not in a table. Each must appear in the README by itself."""
    select = read_json(outputs, "select_final.json")
    if not select:
        return []
    return [("choice margin between the top two models", f4(margin_between_first_two(select)))]


def expected_strings(outputs: Path, config=None):
    """Return ([(label, text that must appear in the README)], [notes about what was skipped])."""
    tables, notes = build_tables(outputs, config)
    wanted = []
    for _title, _header, rows in tables:
        for label, cells in rows:
            wanted.append((label, "| " + " | ".join(str(c) for c in cells) + " |"))
    wanted += prose_numbers(outputs)
    return wanted, notes


def load_config(project: Path):
    path = project / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None


def tables_markdown(outputs: Path, config=None) -> str:
    tables, _ = build_tables(outputs, config)
    parts = []
    for title, header, rows in tables:
        parts.append(f"**{title}**\n\n" + table_md(header, [cells for _label, cells in rows]))
    return "\n\n".join(parts)


def find_missing(readme_text: str, wanted):
    """Return the (label, text) pairs whose text is not in the README."""
    return [(label, text) for label, text in wanted if text not in readme_text]


def main(argv=None, project=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    project = Path(project) if project else PROJECT
    outputs = project / "outputs"
    if read_json(outputs, "metrics.json") is None:
        print("No results to check against. Run `python src/pipeline.py` first.")
        return 2
    config = load_config(project)
    if "--print-tables" in argv:
        print(tables_markdown(outputs, config))
        return 0
    readme_path = project / "README.md"
    if not readme_path.exists():
        print("README.md not found.")
        return 2
    readme = readme_path.read_text(encoding="utf-8")
    wanted, notes = expected_strings(outputs, config)
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
