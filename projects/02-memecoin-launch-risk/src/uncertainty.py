"""How much would the scores wobble on a slightly different group of coins?

Method (bootstrap): write the validation coins on cards, draw as many cards as there are coins
(putting each card back after drawing), score the models on that shuffled exam, and repeat
many times. The spread of the scores shows the wobble. Every model gets the SAME shuffled exam,
so differences between models are compared fairly.

Limit: this measures wobble inside the validation pile only. It does not measure how results
change from period to period (the crash share is 71% in validation but 59% in test).

Run from the projects folder:  python 02-memecoin-launch-risk/src/uncertainty.py
"""

import json
import sys
from pathlib import Path

import numpy as np

import metrics
import piles
import settings
import train


def _summary(values) -> dict:
    """Middle 90% of the values: 5th percentile, median, 95th percentile."""
    low, median, high = np.percentile(values, [5, 50, 95])
    return {"low": round(float(low), 4), "median": round(float(median), 4), "high": round(float(high), 4)}


def bootstrap(y_true, scores_by_model: dict, n_draws: int, seed: int, reference: str | None = None) -> dict:
    """Score every model on n_draws re-drawn exams, on both sides, and summarise the spread."""
    y = np.asarray(y_true)
    n = len(y)
    names = list(scores_by_model)
    reference = reference or names[0]
    rng = np.random.default_rng(seed)

    crash = {m: [] for m in names}
    survivor = {m: [] for m in names}
    for _ in range(n_draws):
        idx = rng.integers(0, n, n)
        y_draw = y[idx]
        if y_draw.min() == y_draw.max():      # only one side present: PR-AUC is undefined, skip this draw
            continue
        for m in names:
            s_draw = np.asarray(scores_by_model[m])[idx]
            crash[m].append(metrics.pr_auc(y_draw, s_draw))
            survivor[m].append(metrics.pr_auc(1 - y_draw, 1 - s_draw))

    def differences(table):
        return {m: _summary(np.array(table[m]) - np.array(table[reference])) for m in names if m != reference}

    return {
        "n_draws": len(crash[reference]),
        "reference": reference,
        "pr_auc_crash": {m: _summary(crash[m]) for m in names},
        "pr_auc_survivor": {m: _summary(survivor[m]) for m in names},
        "difference_crash": differences(crash),
        "difference_survivor": differences(survivor),
    }


def verdict(difference: dict) -> str:
    """Plain-words reading of a difference range."""
    if difference["low"] > 0:
        return "better in nearly every re-drawn exam"
    if difference["high"] < 0:
        return "worse in nearly every re-drawn exam"
    return "cannot be told apart (range includes 0)"


def format_result(result: dict, n_coins: int) -> str:
    lines = [f"{result['n_draws']} re-drawn exams of {n_coins:,} validation coins. Ranges are the middle 90%.\n",
             f"{'model':20} {'crash PR-AUC':>26} {'survivor PR-AUC':>26}"]
    for name in result["pr_auc_crash"]:
        c, s = result["pr_auc_crash"][name], result["pr_auc_survivor"][name]
        lines.append(f"{name:20} {c['median']:.3f} ({c['low']:.3f} to {c['high']:.3f})"
                     f"   {s['median']:.3f} ({s['low']:.3f} to {s['high']:.3f})")
    for side in ("crash", "survivor"):
        lines.append(f"\n{side} side: PR-AUC difference compared with {result['reference']}:")
        for name, d in result[f"difference_{side}"].items():
            lines.append(f"{name:20} {d['median']:+.3f} ({d['low']:+.3f} to {d['high']:+.3f})  {verdict(d)}")
    return "\n".join(lines)


def main(argv=None, config=None, data_dir=None, out_dir=None) -> int:
    config = config or settings.load_config()
    data_dir = Path(data_dir) if data_dir else settings.DATA_DIR
    out_dir = Path(out_dir) if out_dir else settings.PROJECT_DIR / config["evaluation"]["outputs_folder"]
    try:
        pile_set = piles.load_piles(data_dir, config)
    except FileNotFoundError as error:
        print(error)
        return 1
    fitted = train.fit_models(config, pile_set)
    valid_scores = {name: s["validation"] for name, s in fitted["scores"].items()}
    result = bootstrap(fitted["y_validation"], valid_scores, config["evaluation"]["bootstrap_draws"],
                       config["seed"], config["evaluation"]["reference_model"])
    print(format_result(result, len(fitted["y_validation"])))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wobble.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved {out_dir / 'wobble.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
