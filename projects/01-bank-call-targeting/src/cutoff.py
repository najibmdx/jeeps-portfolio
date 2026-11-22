"""How many clients should the call team phone? One answer per cost scenario.

Money is measured in units: one subscription is worth 1 unit, and a phone call costs
cost_per_call units (an ASSUMPTION set in config.yaml, not the bank's real cost).

For a share of clients to phone (the top 1%, 2%, ... 100% by score):
    net value = (yes clients reached) - cost_per_call x (calls made)
The cut-off is the share with the highest net value, found on the VALIDATION set only.
The same share is then used on test, in the final check at the end.

Why a share and not a score threshold: yes-rates drift between periods (validation 9.0%,
test 44.5%), so a score threshold picked on validation would behave very differently on
test. A share of clients stays meaningful, and it matches how a call team is limited.

Limit: these are made-up costs on one validation month. The result shows HOW the advice
changes when calls get cheaper or dearer. It is not a profit claim.
"""

import json

import numpy as np

import evaluate
import split
import train
from settings import load_config


def best_share(y_true, scores, cost_per_call: float, step: float) -> dict:
    """Pick the share of clients to phone that gives the highest net value.

    Only the arrays passed in are used. Share 0 (phone nobody) is always allowed,
    so the answer is never worse than doing nothing.
    """
    y = np.asarray(y_true)
    n = len(y)
    order = np.argsort(-np.asarray(scores), kind="stable")
    yes_in_top = np.concatenate([[0], np.cumsum(y[order])])      # yes_in_top[k] = yes among the k best

    shares = np.linspace(0, 1, int(round(1 / step)) + 1)
    ks = np.array([0 if s == 0 else evaluate.top_k_from_fraction(n, s) for s in shares])
    net = yes_in_top[ks] - cost_per_call * ks

    i = int(np.argmax(net))                                      # first best = fewest calls on a tie
    k, found, total_yes = int(ks[i]), int(yes_in_top[ks[i]]), int(y.sum())
    return {
        "share": round(float(shares[i]), 4),
        "calls": k,
        "precision": round(found / k, 4) if k else None,
        "recall": round(found / total_yes, 4) if total_yes else None,
        "net_per_100_clients": round(100 * float(net[i]) / n, 2),
        "call_everyone_net_per_100_clients": round(100 * (total_yes - cost_per_call * n) / n, 2),
    }


def run(config: dict, sets: dict) -> dict:
    """Choose the cut-off for every scenario and model. Only train and validation are read."""
    scored = train.score_models(config, sets)
    y_valid = scored["y_validation"]
    step = config["evaluation"]["cutoff_share_step"]

    out = {}
    for scenario, settings in config["scenarios"].items():
        cost = settings["cost_per_call"]
        out[scenario] = {
            "cost_per_call": cost,
            "models": {
                model: best_share(y_valid, s["validation"], cost, step)
                for model, s in scored["scores"].items()
            },
        }
    return out


def main() -> None:
    config = load_config()
    result = run(config, split.prepare(config))

    print("ASSUMED costs (one subscription = 1 unit). Cut-off chosen on the validation set only.")
    print("net = units per 100 clients; phoning nobody = 0.\n")
    for scenario, info in result.items():
        print(f"{scenario}: one call costs {info['cost_per_call']:.2f} units")
        print(f"  {'model':20} {'share':>6} {'calls':>6} {'precision':>10} {'recall':>7} {'net':>7} {'call everyone':>14}")
        for model, m in info["models"].items():
            precision = f"{m['precision']:.3f}" if m["precision"] is not None else "-"
            recall = f"{m['recall']:.3f}" if m["recall"] is not None else "-"
            print(f"  {model:20} {m['share']:6.0%} {m['calls']:6,} {precision:>10} {recall:>7} "
                  f"{m['net_per_100_clients']:7.2f} {m['call_everyone_net_per_100_clients']:14.2f}")
        print()

    train.OUTPUT_DIR.mkdir(exist_ok=True)
    path = train.OUTPUT_DIR / "scenarios.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
