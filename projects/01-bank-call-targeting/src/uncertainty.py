"""How much would the scores wobble on a slightly different group of clients?

Method (bootstrap): write the validation clients on cards, draw as many cards as
there are clients (putting each card back after drawing), score the models on that
shuffled exam, and repeat many times. The spread of the scores shows the wobble.

The same shuffled exam is given to every model, so differences between models are
compared fairly.

Limit: this measures wobble inside the validation month only. It does not measure
how results change from month to month.
"""

import json

import numpy as np

import evaluate
import split
import train
from settings import load_config


def _summary(values) -> dict:
    """Middle 90% of the values: 5th percentile, median, 95th percentile."""
    low, median, high = np.percentile(values, [5, 50, 95])
    return {"low": round(float(low), 4), "median": round(float(median), 4), "high": round(float(high), 4)}


def bootstrap(y_true, scores_by_model: dict, k: int, n_draws: int, seed: int, reference: str | None = None) -> dict:
    """Score every model on n_draws re-drawn exams and summarise the spread."""
    y = np.asarray(y_true)
    n = len(y)
    names = list(scores_by_model)
    reference = reference or names[0]
    rng = np.random.default_rng(seed)

    pr = {m: [] for m in names}
    lift = {m: [] for m in names}
    for _ in range(n_draws):
        idx = rng.integers(0, n, n)
        y_draw = y[idx]
        if y_draw.sum() == 0:      # no yes at all: PR-AUC is undefined, skip this draw
            continue
        for m in names:
            s_draw = np.asarray(scores_by_model[m])[idx]
            pr[m].append(evaluate.pr_auc(y_draw, s_draw))
            lift[m].append(evaluate.lift_at_k(y_draw, s_draw, k))

    return {
        "n_draws": len(pr[reference]),
        "reference": reference,
        "pr_auc": {m: _summary(pr[m]) for m in names},
        "lift_at_k": {m: _summary(lift[m]) for m in names},
        "pr_auc_minus_reference": {
            m: _summary(np.array(pr[m]) - np.array(pr[reference])) for m in names if m != reference
        },
    }


def verdict(difference: dict) -> str:
    """Plain-words reading of a difference range."""
    if difference["low"] > 0:
        return "better in nearly every re-drawn exam"
    if difference["high"] < 0:
        return "worse in nearly every re-drawn exam"
    return "cannot be told apart (range includes 0)"


def main() -> None:
    config = load_config()
    sets = split.prepare(config)
    scored = train.score_models(config, sets)
    y_valid = scored["y_validation"]
    k = config["evaluation"]["top_k"]
    draws = config["evaluation"]["bootstrap_draws"]

    valid_scores = {name: s["validation"] for name, s in scored["scores"].items()}
    result = bootstrap(y_valid, valid_scores, k, draws, config["seed"])

    print(f"{result['n_draws']} re-drawn exams of {len(y_valid):,} clients. Ranges are the middle 90%.\n")
    print(f"{'model':20} {'PR-AUC':>26} {'lift@' + str(k):>24}")
    for name in valid_scores:
        p, lf = result["pr_auc"][name], result["lift_at_k"][name]
        print(f"{name:20} {p['median']:.3f} ({p['low']:.3f} to {p['high']:.3f})"
              f" {lf['median']:10.2f} ({lf['low']:.2f} to {lf['high']:.2f})")

    print(f"\nPR-AUC difference compared with {result['reference']}:")
    for name, d in result["pr_auc_minus_reference"].items():
        print(f"{name:20} {d['median']:+.3f} ({d['low']:+.3f} to {d['high']:+.3f})  {verdict(d)}")

    train.OUTPUT_DIR.mkdir(exist_ok=True)
    (train.OUTPUT_DIR / "wobble.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
