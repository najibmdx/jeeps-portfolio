"""Make the README results tables from the saved results, and check that the README still matches them.

    python src/check_readme.py                 check README.md against outputs/*.json
    python src/check_readme.py --print-tables  print every results table as Markdown, from outputs/
    python src/check_readme.py --update        rewrite the results block of README.md from outputs/

The results block of the README sits between two marker lines (RESULTS_BEGIN and RESULTS_END below). It is
written by --update and never typed by hand. The check works on whole table rows: every row printed by
--print-tables, and every generated sentence, must appear in the README exactly as printed, so a number in the
wrong row, a changed digit or a missing row is caught. The check also reads each saved verdict against its own
range, so a results file that disagrees with itself is caught.

Learning project. No claim about trading use, returns or profit.

Files that are missing are skipped with a note, because outputs/ is not committed (only final_test_k7.json is,
so the one-time test stays on record). --update refuses to run unless every file is there, so it can never
wipe a table. The check itself is tested on made-up results (tests/test_check_readme.py).

Exit code: 0 all numbers found, 1 a number is missing or different, 2 nothing to check against (or --update
could not run).
"""

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
K = 7  # the decision time used for the results (days after opening); the files are named ..._k7.json

RESULTS_BEGIN = "<!-- results:begin (written by src/check_readme.py --update; do not edit by hand) -->"
RESULTS_END = "<!-- results:end -->"

FILES = {
    "split": "split_report_k{k}.json",
    "baselines": "baselines_k{k}.json",
    "models": "models_k{k}.json",
    "tuning": "tuning_k{k}.json",
    "ablation": "ablation_k{k}.json",
    "final": "final_test_k{k}.json",
}

NAMES = {
    "raw_price": "Raw market price",
    "price_only_correction": "Price-only correction",
    "lr_price_only": "Price-only correction",
    "base_rate": "Base rate (train Yes rate)",
    "lr_all_features": "Logistic regression, all inputs",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "frozen_model": "Frozen model (logistic regression, all inputs, C=1.0)",
    "price_path_only": "Price path only",
    "price_path_plus_activity": "Price path + activity",
    "price_path_plus_category": "Price path + category",
    "price_path_plus_sibling_count": "Price path + sibling count",
    "all_minus_sibling_count": "All except sibling count",
    "all_columns": "All columns (the frozen model)",
}
SLICES = {"all": "All markets", "contested": "Contested"}
NO_DIFFERENCE = "cannot be told apart"


# ---------- number formats (one place, so the README and the check always agree) ----------

def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def s4(x):
    """Signed, 4 decimals. A value that rounds to zero is written 0.0000 with no sign."""
    r = round(x, 4)
    return "0.0000" if r == 0 else f"{r:+.4f}"


def pct1(x):
    return f"{x * 100:.1f}%"


def name_of(key):
    return NAMES.get(key, key)


def rng(d):
    return f"{f4(d['low'])} to {f4(d['high'])}"


def diff_cell(d):
    """A difference as 'point (low to high)', all signed."""
    return f"{s4(d['point'])} ({s4(d['low'])} to {s4(d['high'])})"


def rule_verdict(d):
    """The reading of a range, the same rule as verdict() in src/baselines.py.

    Every difference in the results is A minus B in a score where lower is better, so a range wholly
    above 0 means A is worse, wholly below 0 means A is better, and a range that includes 0 means
    the two cannot be told apart."""
    if d["low"] > 0:
        return "worse"
    if d["high"] < 0:
        return "better"
    return NO_DIFFERENCE


def table_md(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def dash(x, f=str):
    return "-" if x is None else f(x)


def params_text(params):
    return ", ".join(f"{k}={v}" for k, v in params.items()) if params else "none (nothing to tune)"


# ---------- reading the files ----------

def read_results(outputs: Path, k=K):
    """{'split': dict or None, ...}, one entry per file in FILES."""
    out = {}
    for key, pattern in FILES.items():
        path = outputs / pattern.format(k=k)
        out[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out


def iter_differences(node, path=""):
    """Every dict with low, high, point and verdict, with a readable path to it."""
    if isinstance(node, dict):
        if {"low", "high", "point", "verdict"} <= set(node):
            yield path, node
        for key, value in node.items():
            yield from iter_differences(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from iter_differences(value, f"{path}[{i}]")


def verdict_problems(results):
    """Saved verdicts that disagree with their own range. Returns [(file, path, saved, from_range)]."""
    problems = []
    for file_key, data in results.items():
        if data is None:
            continue
        for path, d in iter_differences(data):
            if d["verdict"] != rule_verdict(d):
                problems.append((file_key, path, d["verdict"], rule_verdict(d)))
    return problems


# ---------- the tables ----------
# Each returns (title, header, [(label, cells)]). The label only names the row in an error message.

def split_table(split):
    f = split["final"]
    rows = [(f"split: {s}", [s.capitalize(), f"{f['sizes'][s]:,}", f"{f['events'][s]:,}"])
            for s in ("train", "validation", "test")]
    return "Markets and events per split", ["Split", "Markets", "Events"], rows


def window_table(final):
    rows = []
    for s in ("train", "validation", "test"):
        w = final["window_check"][s]
        rows.append((f"window: {s}", [
            s.capitalize(), f"{w['markets']:,}", f"{w['t_first'][:10]} to {w['t_last'][:10]}",
            f"{w['days_t_to_close']['p50']:.2f}", pct1(w["share_closing_within_1_day"]),
            pct1(w["yes_rate"]), pct1(w["share_contested"])]))
    header = ["Split", "Markets", "Decision dates (UTC)", "Median days from decision to close",
              "Closing within a day", "Resolved Yes", "Contested"]
    return "The three windows side by side", header, rows


def score_rows(prefix, slices, keys, difference_of):
    """One row per slice and forecast: log loss, its range, Brier, and the log-loss difference from the raw price."""
    rows = []
    for sl in ("all", "contested"):
        s = slices[sl]
        for key in keys:
            diff = difference_of(s, key)
            rows.append((f"{prefix}: {sl} {key}", [
                SLICES[sl], name_of(key), f4(s["scores"][key]["log_loss"]), rng(s["wobble"][key]["log_loss"]),
                f4(s["scores"][key]["brier"]),
                "-" if diff is None else diff_cell(diff["log_loss"]),
                "-" if diff is None else diff["log_loss"]["verdict"]]))
    return rows


SCORE_HEADER = ["Slice", "Forecast", "Log loss", "Range (90%)", "Brier", "Log loss minus raw price", "Reading"]


def baselines_table(b):
    keys = ("raw_price", "price_only_correction", "base_rate")
    rows = score_rows("baselines", b["slices"], keys, lambda s, k: s["difference_vs_raw_price"].get(k))
    return "Baselines, validation", SCORE_HEADER, rows


def ladder_table(m):
    rows = []
    for key in m["order"]:
        va, vc = m["validation"]["all"], m["validation"]["contested"]
        da, dc = va["difference_vs_raw_price"].get(key), vc["difference_vs_raw_price"].get(key)
        rows.append((f"ladder: {key}", [
            name_of(key), f4(m["train"]["all"]["scores"][key]["log_loss"]), f4(va["scores"][key]["log_loss"]),
            dash(da, lambda d: diff_cell(d["log_loss"])), dash(da, lambda d: d["log_loss"]["verdict"]),
            f4(vc["scores"][key]["log_loss"]),
            dash(dc, lambda d: diff_cell(d["log_loss"])), dash(dc, lambda d: d["log_loss"]["verdict"])]))
    header = ["Model", "Train, all", "Validation, all", "Validation minus raw price, all", "Reading, all",
              "Validation, contested", "Validation minus raw price, contested", "Reading, contested"]
    return "Untuned model ladder (log loss; every model fitted on train)", header, rows


def tuning_table(t):
    rows = []
    for key in t["order"]:
        if key not in t["models"]:
            continue
        info = t["models"][key]
        best = info["settings"][info["best_index"]]
        rows.append((f"tuning: {key}", [
            name_of(key), str(len(info["settings"])), params_text(info["best_params"]),
            f4(best["validation"]["all"]), f4(info["library_default_validation_log_loss"]["all"]),
            f4(best["validation"]["contested"])]))
    header = ["Model", "Settings tried", "Chosen setting", "Validation, all", "Library default, all", "Validation, contested"]
    return "Light tuning (validation log loss of each model's best setting)", header, rows


def choice_table(t):
    c = t["choice"]
    rows = [
        ("choice: winner", ["Chosen on validation", f"{name_of(c['winner'])}, {params_text(c['params'])}"]),
        ("choice: runner-up", ["Runner-up", name_of(c["runner_up"])]),
        ("choice: validation all", ["Winner, validation log loss, all markets", f4(c["validation"]["all"])]),
        ("choice: margin", ["Runner-up minus winner, all markets",
                            f"{diff_cell(c['margin_runner_up_minus_winner']['all'])}, {c['margin_runner_up_minus_winner']['all']['verdict']}"]),
        ("choice: vs price all", ["Winner minus raw price, all markets",
                                  f"{diff_cell(c['winner_minus_raw_price']['all'])}, {c['winner_minus_raw_price']['all']['verdict']}"]),
        ("choice: vs price contested", ["Winner minus raw price, contested",
                                        f"{diff_cell(c['winner_minus_raw_price']['contested'])}, {c['winner_minus_raw_price']['contested']['verdict']}"]),
    ]
    return "The choice made on validation", ["Measure", "Value"], rows


def ablation_table(a):
    rows = []
    for key in a["order"]:
        va, vc = a["validation"]["all"], a["validation"]["contested"]
        da, dc = va["difference_vs_raw_price"].get(key), vc["difference_vs_raw_price"].get(key)
        n_inputs = str(len(a["rows"][key])) if key in a["rows"] else "-"
        rows.append((f"ablation: {key}", [
            name_of(key), n_inputs, f4(va["scores"][key]["log_loss"]),
            dash(da, lambda d: diff_cell(d["log_loss"])), dash(da, lambda d: d["log_loss"]["verdict"]),
            f4(vc["scores"][key]["log_loss"]),
            dash(dc, lambda d: diff_cell(d["log_loss"])), dash(dc, lambda d: d["log_loss"]["verdict"])]))
    header = ["Columns used", "Inputs", "Validation, all", "Minus raw price, all", "Reading, all",
              "Validation, contested", "Minus raw price, contested", "Reading, contested"]
    return "Column groups left out of the frozen model (validation log loss)", header, rows


def pair_counts(a):
    pairs = [(sl, p) for sl in ("all", "contested") for p in a["pairs"][sl]]
    flagged = [(sl, p) for sl, p in pairs if p["verdict"] != NO_DIFFERENCE]
    return pairs, flagged


def pairs_sentence(a):
    pairs, flagged = pair_counts(a)
    better = sum(1 for _sl, p in flagged if p["verdict"] == "better")
    worse = sum(1 for _sl, p in flagged if p["verdict"] == "worse")
    return (f"Pairwise comparisons in the ablation: {len(pairs)} (both slices); flagged as different: {len(flagged)} "
            f"(A better: {better}, A worse: {worse}).")


def flagged_pairs_table(a):
    _pairs, flagged = pair_counts(a)
    rows = [(f"flagged pair: {sl} {p['a']} vs {p['b']}", [
        SLICES[sl], name_of(p["a"]), name_of(p["b"]), diff_cell(p), p["verdict"]]) for sl, p in flagged]
    header = ["Slice", "A", "B", "A minus B, log loss", "Reading"]
    return "Ablation pairs flagged as different (A minus B; negative means A has the lower log loss)", header, rows


def final_scores_table(f):
    keys = ("raw_price", "price_only_correction", "base_rate", "frozen_model")
    rows = score_rows("test", f["test"]["slices"], keys, lambda s, k: s["difference_vs_raw_price"].get(k))
    return "One-time test: all four forecasts", SCORE_HEADER, rows


def headline_table(f):
    """Frozen model minus the raw price and minus the price-only correction, test next to validation."""
    def versus_price(window, sl):
        return f[window]["slices"][sl]["difference_vs_raw_price"]["frozen_model"]["log_loss"]

    def versus_correction(window, sl):
        return f[window]["slices"][sl]["frozen_minus_price_only_correction"]

    def cell(d):
        return f"{diff_cell(d)}, {d['verdict']}"

    rows = []
    for sl, label in (("contested", "Contested (the headline slice)"), ("all", "All markets")):
        for what, getter in (("raw market price", versus_price), ("price-only correction", versus_correction)):
            rows.append((f"headline: {sl} {what}", [
                f"{label}: frozen model minus {what}", cell(getter("test", sl)), cell(getter("validation", sl))]))
    header = ["Log loss difference (positive means the frozen model is worse)", "Test window", "Validation window"]
    return "Headline: frozen model against the market price", header, rows


def calibration_table(f):
    cal = f["test"]["slices"]["all"]["calibration"]["frozen_model"]
    rows = [(f"calibration: {c['band']}", [
        c["band"], f"{c['markets']:,}", dash(c["mean_forecast"], f3), dash(c["yes_rate"], f3)]) for c in cal]
    header = ["Forecast band", "Markets", "Mean forecast", "Share that resolved Yes"]
    return "Calibration of the frozen model on the test split, all markets", header, rows


def reading_sentences(f):
    """Generated sentences. Each must appear in the README as written."""
    h = f["headline"]
    return [
        ("reading", f"Reading, fixed before the run: **{h['reading']}** Validation and test {h['validation_and_test']}."),
        ("opened", "The test split was opened once, at "
                   f"{f['opened_at'][:16].replace('T', ' ')} UTC, on {f['test_markets']:,} markets in {f['test_events']:,} events."),
    ]


def build(results):
    """Return ([('table', title, header, rows)...], [(label, sentence)...], notes), in README order."""
    items, notes = [], []

    def add(key, make, note):
        if results[key] is None:
            notes.append(note)
        else:
            items.append(("table",) + make(results[key]))

    sentences = []
    add("split", split_table, "split_report: split sizes not checked")
    add("final", window_table, "final_test: window comparison not checked")
    add("baselines", baselines_table, "baselines: validation baselines not checked")
    add("models", ladder_table, "models: untuned ladder not checked")
    add("tuning", tuning_table, "tuning: tuning table not checked")
    if results["tuning"] is not None:
        items.append(("table",) + choice_table(results["tuning"]))
    add("ablation", ablation_table, "ablation: column-group table not checked")
    if results["ablation"] is not None:
        sentences.append(("ablation pairs", pairs_sentence(results["ablation"])))
        items.append(("table",) + flagged_pairs_table(results["ablation"]))
    if results["final"] is not None:
        sentences += reading_sentences(results["final"])
        items.append(("table",) + headline_table(results["final"]))
        items.append(("table",) + final_scores_table(results["final"]))
        items.append(("table",) + calibration_table(results["final"]))
    else:
        notes.append("final_test: the one-time test numbers are not checked")
    return items, sentences, notes


def results_markdown(results):
    """The whole results block, as it must stand between the markers."""
    items, sentences, _ = build(results)
    parts = []
    by_label = dict(sentences)
    for _kind, title, header, rows in items:
        parts.append(f"**{title}**\n\n" + table_md(header, [cells for _l, cells in rows]))
    # Sentences go where they read best: the pair count before the flagged pairs, the two test sentences before the headline table.
    ordered = []
    for part in parts:
        if part.startswith("**Ablation pairs flagged") and "ablation pairs" in by_label:
            ordered.append(by_label["ablation pairs"])
        if part.startswith("**Headline: frozen model"):
            ordered += [by_label["opened"], by_label["reading"]]
        ordered.append(part)
    return "\n\n".join(ordered)


def expected_strings(results):
    """Return ([(label, text that must appear in the README)], [notes about what was skipped])."""
    items, sentences, notes = build(results)
    wanted = []
    for _kind, _title, _header, rows in items:
        for label, cells in rows:
            wanted.append((label, "| " + " | ".join(str(c) for c in cells) + " |"))
    wanted += sentences
    return wanted, notes


def sections_of(readme_text):
    """{table title: the text under that title, up to the next bold title}. A title is a line of the form **Title**."""
    sections, title, lines = {}, None, []
    for line in readme_text.split("\n"):
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            if title is not None:
                sections[title] = "\n".join(lines)
            title, lines = line[2:-2], []
        elif title is not None:
            lines.append(line)
    if title is not None:
        sections[title] = "\n".join(lines)
    return sections


def find_problems(readme_text, results):
    """Every table must have its own header (and the line under it) and every row, each as a whole line under its own
    title, and every generated sentence must be in the README. The header matters as much as the rows: it says which
    window, slice or model each number belongs to."""
    items, sentences, _notes = build(results)
    sections = sections_of(readme_text)
    problems = []
    for _kind, title, header, rows in items:
        if title not in sections:
            problems.append(f"table '{title}' is not in README.md")
            continue
        lines = sections[title].split("\n")
        head_line, rule_line = table_md(header, []).split("\n")
        if head_line not in lines:
            problems.append(f"header of '{title}': expected {head_line!r}")
        elif lines.index(head_line) + 1 >= len(lines) or lines[lines.index(head_line) + 1] != rule_line:
            problems.append(f"header of '{title}': the line under the header is not {rule_line!r}")
        for label, cells in rows:
            text = "| " + " | ".join(str(c) for c in cells) + " |"
            if text not in lines:
                problems.append(f"{label}: expected {text!r} under '{title}'")
    for label, text in sentences:
        if text not in readme_text:
            problems.append(f"{label}: expected {text!r}")
    return problems


def block_of(readme_text):
    """The text between the markers, or None if a marker is missing."""
    if RESULTS_BEGIN not in readme_text or RESULTS_END not in readme_text:
        return None
    start = readme_text.index(RESULTS_BEGIN) + len(RESULTS_BEGIN)
    end = readme_text.index(RESULTS_END)
    return readme_text[start:end] if start <= end else None


def update_readme_text(readme_text, results):
    """The README with the results block rewritten. Raises ValueError if the markers are not both there."""
    if block_of(readme_text) is None:
        raise ValueError("the results markers are not in README.md")
    start = readme_text.index(RESULTS_BEGIN) + len(RESULTS_BEGIN)
    end = readme_text.index(RESULTS_END)
    return readme_text[:start] + "\n\n" + results_markdown(results) + "\n\n" + readme_text[end:]


def main(argv=None, project=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    project = Path(project) if project else PROJECT
    results = read_results(project / "outputs")
    present = [k for k, v in results.items() if v is not None]
    if not present:
        print("No results to check against: outputs/ has none of the saved result files.")
        return 2
    readme_path = project / "README.md"

    if "--print-tables" in argv:
        print(results_markdown(results))
        return 0

    if "--update" in argv:
        absent = [FILES[k].format(k=K) for k, v in results.items() if v is None]
        if absent:
            print("Not updating: these result files are missing, and a partial update could wipe a table:", ", ".join(absent))
            return 2
        problems = verdict_problems(results)
        if problems:
            print("Not updating: a saved verdict disagrees with its own range:", problems[0])
            return 1
        # newline="" keeps the README's own line endings (bytes in, bytes out)
        text = readme_path.read_bytes().decode("utf-8")
        eol = "\r\n" if "\r\n" in text else "\n"
        new = update_readme_text(text.replace("\r\n", "\n"), results).replace("\n", eol)
        readme_path.write_bytes(new.encode("utf-8"))
        print("README results block rewritten from outputs/.")
        return 0

    if not readme_path.exists():
        print("README.md not found.")
        return 2
    readme = readme_path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    wanted, notes = expected_strings(results)
    for note in notes:
        print("note:", note)
    failures = []
    for file_key, path, saved, from_range in verdict_problems(results):
        failures.append(f"{file_key}{path}: saved verdict {saved!r} but its range reads {from_range!r}")
    failures += find_problems(readme, results)
    if block_of(readme) is None:
        failures.append("the results markers are not in README.md")
    if failures:
        print(f"{len(failures)} problem(s):")
        for line in failures:
            print("  " + line)
        return 1
    print(f"README numbers match the saved results ({len(wanted)} rows and sentences checked, {len(present)} of {len(FILES)} files present).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
