"""Stage 6, part 2: open the TEST pile exactly once and grade the frozen choices.

Nothing is chosen here. The model, its settings and the cut-offs all come from frozen_choice.yaml
(written by select_final.py from train and validation only). The chosen model is refitted on TRAIN only,
exactly as validated, and graded on the test pile.

Safety:
  - without --confirm-open-test nothing is opened (a dry run that only prints the frozen choice);
  - a lock file is written the moment the test pile is opened; a second run is refused (see holdout.py);
  - everything that can fail without the test pile (checks, fitting) is done BEFORE it is opened.

Limits, stated up front: the wobble only measures sampling noise inside the test pile, not how the market
changes from period to period. The crash share differs between piles (validation 71%, test 59%), so the lazy
guess changes too and every score is shown next to its own pile's lazy guess. This is a risk flag, never a
trade signal; the costs are invented.

Run from the project folder:  python src/final_test.py --confirm-open-test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

import decision
import holdout
import metrics
import models
import piles
import select_final
import settings
import uncertainty


def load_frozen(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing. Run select_final.py first and commit that file.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_frozen(frozen: dict, config: dict, tuned_cfg: dict, train_columns) -> None:
    """Stop if the frozen file no longer matches the code, config, tuned settings or columns."""
    problems = []
    model = frozen["model"]
    if model not in config["final"]["candidates"]:
        problems.append(f"model {model!r} is not a candidate in config.yaml")
    elif frozen["model_params"] != tuned_cfg["models"][model]:
        problems.append("model settings differ from tuned_params.yaml (tuned again after freezing?)")
    if frozen["feature_groups"] != config["features"]["groups"]:
        problems.append("feature groups differ from config.yaml")
    if frozen["seed"] != config["seed"]:
        problems.append("seed differs from config.yaml")
    if frozen["columns_sha256"] != select_final.columns_hash(train_columns):
        problems.append("the columns the model would see are not the columns that were frozen")
    scenarios = config["final"]["scenarios"]
    if set(frozen["scenarios"]) != set(scenarios) or any(frozen["scenarios"][n]["costs"] != scenarios[n] for n in scenarios):
        problems.append("the cost scenarios differ from config.yaml")
    if problems:
        raise ValueError("the frozen choice does not match: " + "; ".join(problems) + ". Nothing was opened.")


def fit_scores(config: dict, tuned_cfg: dict, train_pile, cols, names) -> dict:
    """Fit each named model on TRAIN only and return the fitted models."""
    fitted = {}
    for name in names:
        model = models.build_model(name, tuned_cfg["models"][name], config["seed"])
        model.fit(train_pile[cols], train_pile["crash"])
        fitted[name] = model
    return fitted


def bootstrap_saving(y_true, scores, cutoff: float, scenario: dict, n_draws: int, seed: int) -> dict:
    """How much cheaper (per coin) the frozen cut-off is than the cheaper lazy option, on re-drawn exams."""
    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    savings = []
    for _ in range(n_draws):
        idx = rng.integers(0, len(y), len(y))
        counts = decision.counts_at(y[idx], s[idx], cutoff)
        lazy = min(decision.cost(decision.flag_nothing(y[idx]), scenario),
                   decision.cost(decision.flag_everything(y[idx]), scenario))
        savings.append((lazy - decision.cost(counts, scenario)) / len(y))
    savings = np.array(savings)
    summary = uncertainty._summary(savings)
    summary["share_of_draws_cheaper_than_lazy"] = round(float((savings > 0).mean()), 3)
    return summary


def evaluate(config: dict, frozen: dict, y_test, scores_by_model: dict, reference: str) -> dict:
    """All test numbers. The chosen model and every cut-off are read from the frozen file, never chosen here."""
    model = frozen["model"]
    fraction = config["evaluation"]["top_fraction"]
    draws = config["evaluation"]["bootstrap_draws"]
    chosen = np.asarray(scores_by_model[model], dtype=float)
    result = {
        "model": model,
        "reference": reference,
        "n_test": int(len(y_test)),
        "test_crash_share": round(float(np.mean(y_test)), 4),
        "ranking": {name: metrics.side_metrics(y_test, s, fraction) for name, s in scores_by_model.items()},
        "validation_for_comparison": frozen["validation"]["side_metrics"],
        "wobble": uncertainty.bootstrap(y_test, scores_by_model, draws, config["seed"], reference),
        "flag_nothing": decision.flag_nothing(y_test),
        "flag_everything": decision.flag_everything(y_test),
        "scenarios": {},
    }
    for name, frozen_sc in frozen["scenarios"].items():
        scenario, cutoff = frozen_sc["costs"], frozen_sc["cutoff"]
        counts = decision.counts_at(y_test, chosen, cutoff)
        result["scenarios"][name] = {
            "costs": scenario,
            "cutoff": cutoff,
            "counts": counts,
            "rates": decision.rates(counts),
            "cost_per_coin": round(decision.cost_per_coin(counts, scenario), 4),
            "cost_per_coin_flag_nothing": round(decision.cost_per_coin(result["flag_nothing"], scenario), 4),
            "cost_per_coin_flag_everything": round(decision.cost_per_coin(result["flag_everything"], scenario), 4),
            "saving_per_coin_vs_cheaper_lazy_option": bootstrap_saving(y_test, chosen, cutoff, scenario, draws, config["seed"]),
            "validation_cost_per_coin": frozen_sc["validation"]["cost_per_coin"],
        }
    return result


def format_result(result: dict) -> str:
    m, ref = result["model"], result["reference"]
    rk, val = result["ranking"], result["validation_for_comparison"]
    base = {side: rk[m][side]["baseline"] for side in ("crash", "survivor")}
    lines = [
        f"TEST PILE, opened once. Model {m} (frozen, fitted on train only). Risk flag, not a trade signal.",
        f"{result['n_test']:,} test coins, {100 * result['test_crash_share']:.1f}% crash. "
        f"Lazy guess on test: crash {base['crash']:.3f}, survivor {base['survivor']:.3f} "
        f"(validation was {val['crash']['baseline']:.3f} / {val['survivor']['baseline']:.3f}).",
        "\nranking quality, PR-AUC (times the lazy guess of the SAME pile in brackets):",
        f"{'':40} {'crash side':>20} {'survivor side':>20}",
    ]

    def cell(x):
        return f"{x['pr_auc']:.3f} ({x['pr_auc_times_baseline']:.2f}x)"

    lines.append(f"{m + ' on validation':40} {cell(val['crash']):>20} {cell(val['survivor']):>20}")
    for name in rk:
        label = name + (" on TEST" if name == m else " on TEST (reference)")
        lines.append(f"{label:40} {cell(rk[name]['crash']):>20} {cell(rk[name]['survivor']):>20}")
    w = result["wobble"]
    lines.append(f"\nwobble on the test pile ({w['n_draws']} re-drawn exams, middle 90%; sampling noise only):")
    for side in ("crash", "survivor"):
        c = w[f"pr_auc_{side}"][m]
        lines.append(f"  {side} side PR-AUC of {m}: {c['median']:.3f} ({c['low']:.3f} to {c['high']:.3f})")
        if m != ref:
            d = w[f"difference_{side}"][m]
            lines.append(f"    vs {ref}: {d['median']:+.3f} ({d['low']:+.3f} to {d['high']:+.3f})  {uncertainty.verdict(d)}")
    lines.append("\nthe frozen cut-offs on the test pile (P = precision, R = recall, in %):")
    lines.append("  " + decision.describe("flag nothing", np.inf, result["flag_nothing"]))
    lines.append("  " + decision.describe("flag everything", 0.0, result["flag_everything"]))
    for name, sc in result["scenarios"].items():
        c = sc["costs"]
        s = sc["saving_per_coin_vs_cheaper_lazy_option"]
        lines.append(f"\n  scenario {name} (INVENTED costs: holding a crasher = {c['holding_crasher']}, "
                     f"skipping a good coin = {c['skipped_good']})")
        lines.append("    " + decision.describe("frozen choice", sc["cutoff"], sc["counts"]))
        lines.append(f"    cost per coin: frozen cut-off {sc['cost_per_coin']:.4f}   flag nothing "
                     f"{sc['cost_per_coin_flag_nothing']:.4f}   flag everything {sc['cost_per_coin_flag_everything']:.4f}   "
                     f"(validation, frozen cut-off: {sc['validation_cost_per_coin']:.4f})")
        lines.append(f"    saving per coin vs the cheaper lazy option: {s['median']:+.4f} ({s['low']:+.4f} to {s['high']:+.4f}); "
                     f"cheaper in {100 * s['share_of_draws_cheaper_than_lazy']:.0f}% of re-drawn exams")
    lines.append("\nreading limits: the wobble covers sampling noise inside the test pile only, not period-to-period "
                 "change. The cost scenarios are assumptions. Nothing here says anything about trading results.")
    return "\n".join(lines)


def main(argv=None, config=None, data_dir=None, out_dir=None, params_path=None, frozen_path=None) -> int:
    args = argparse.ArgumentParser(description="Open the test pile once and grade the frozen choices.")
    args.add_argument("--confirm-open-test", action="store_true",
                      help="really open the test pile (it can be opened only once)")
    args = args.parse_args(argv)

    config = config or settings.load_config()
    data_dir = Path(data_dir) if data_dir else settings.DATA_DIR
    out_dir = Path(out_dir) if out_dir else settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    params_path = Path(params_path) if params_path else settings.PROJECT_DIR / config["tuning"]["tuned_params_file"]
    frozen_path = Path(frozen_path) if frozen_path else settings.PROJECT_DIR / config["final"]["frozen_file"]

    if holdout.is_locked(data_dir, config):
        print("REFUSED: the test pile was already opened. It is opened once. Nothing was done.")
        return 1
    try:
        frozen = load_frozen(frozen_path)
    except FileNotFoundError as error:
        print(error)
        return 1
    if not args.confirm_open_test:
        print(f"DRY RUN: the test pile was NOT opened. Frozen choice: model {frozen['model']}, "
              f"cut-offs " + ", ".join(f"{n} {s['cutoff']:.4f}" for n, s in frozen["scenarios"].items()) + ".")
        print("To open the test pile for real (once only), run:  python src/final_test.py --confirm-open-test")
        return 2

    try:
        pile_set = piles.load_piles(data_dir, config, parts=("train",))
    except FileNotFoundError as error:
        print(error)
        return 1
    train_pile = pile_set["train"]
    cols = piles.select_features(train_pile.columns, config)
    tuned_cfg = select_final.tuned_config(config, params_path)
    try:
        check_frozen(frozen, config, tuned_cfg, cols)
    except ValueError as error:
        print(f"REFUSED: {error}")
        return 1

    reference = config["evaluation"]["reference_model"]
    names = list(dict.fromkeys([frozen["model"], reference]))
    fitted = fit_scores(config, tuned_cfg, train_pile, cols, names)      # all fitting happens BEFORE the test pile is opened

    try:
        test = holdout.open_test_pile(data_dir, config, frozen_path, confirm=args.confirm_open_test)
    except (FileExistsError, FileNotFoundError, PermissionError) as error:
        print(f"REFUSED: {error}")
        return 1
    missing = [c for c in cols if c not in test.columns]
    if missing:
        print(f"the test pile lacks columns the model needs: {missing}")
        return 1

    y_test = test["crash"].to_numpy()
    scores = {name: model.predict_proba(test[cols])[:, 1] for name, model in fitted.items()}
    result = evaluate(config, frozen, y_test, scores, reference)
    print(format_result(result))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved final_test.json in {out_dir}. The test pile is now locked; it cannot be opened again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
