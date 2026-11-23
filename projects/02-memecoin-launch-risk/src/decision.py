"""Turn scores into a flag and count the two mistakes. Plain counting, no model in here.

A flag means "likely to crash": a coin is flagged when its score is AT OR ABOVE the cut-off.
It is a risk flag, never a trade signal.

The two mistakes, from a memecoin trader's point of view:
  skipped_good     flagged a coin that would have survived  (a good coin skipped)
  held_crasher     did not flag a coin that crashes          (a crashing coin held)

Both sides get precision and recall:
  crash side     of the flagged coins, the share that crash (precision); of all crashers, the share flagged (recall)
  survivor side  of the NOT flagged coins, the share that survive (precision); of all survivors, the share not flagged (recall)

Costs are invented and unit-free (config final.scenarios). They only show how the best cut-off moves.
Two lazy options are always shown next to any cut-off: flag nothing, flag everything.
"""

import numpy as np

COST_KEYS = ("holding_crasher", "skipped_good")


def counts_at(y_true, scores, cutoff: float) -> dict:
    """Count both mistakes (and both kinds of correct call) when coins at or above `cutoff` are flagged."""
    y = np.asarray(y_true)
    flagged = np.asarray(scores, dtype=float) >= cutoff
    crash = y == 1
    return {
        "n": int(len(y)),
        "crashers": int(crash.sum()),
        "survivors": int((~crash).sum()),
        "flagged": int(flagged.sum()),
        "flagged_crash": int((flagged & crash).sum()),        # flagged and it did crash: a correct flag
        "skipped_good": int((flagged & ~crash).sum()),        # flagged, but it survived: mistake 1
        "held_crasher": int((~flagged & crash).sum()),        # not flagged, but it crashed: mistake 2
        "kept_survivor": int((~flagged & ~crash).sum()),      # not flagged and it survived: a correct pass
    }


def _share(top: int, bottom: int):
    return None if bottom == 0 else round(top / bottom, 4)


def rates(counts: dict) -> dict:
    """Precision and recall on both sides. None where the denominator is 0 (nothing flagged, or nothing kept)."""
    unflagged = counts["n"] - counts["flagged"]
    return {
        "flag_share": round(counts["flagged"] / counts["n"], 4),
        "crash_precision": _share(counts["flagged_crash"], counts["flagged"]),
        "crash_recall": _share(counts["flagged_crash"], counts["crashers"]),
        "survivor_precision": _share(counts["kept_survivor"], unflagged),
        "survivor_recall": _share(counts["kept_survivor"], counts["survivors"]),
    }


def cost(counts: dict, scenario: dict) -> float:
    """Total invented cost: each mistake times what the scenario says it costs."""
    if set(scenario) != set(COST_KEYS):
        raise ValueError(f"a scenario needs exactly {COST_KEYS}")
    return scenario["holding_crasher"] * counts["held_crasher"] + scenario["skipped_good"] * counts["skipped_good"]


def cost_per_coin(counts: dict, scenario: dict) -> float:
    return cost(counts, scenario) / counts["n"]


def flag_nothing(y_true) -> dict:
    """Lazy option 1: never flag. Every crasher is held."""
    return counts_at(y_true, np.zeros(len(y_true)), np.inf)


def flag_everything(y_true) -> dict:
    """Lazy option 2: flag every coin. Every survivor is skipped."""
    return counts_at(y_true, np.zeros(len(y_true)), 0.0)


def best_lazy_cost(y_true, scenario: dict) -> float:
    """The cheaper of the two lazy options, as total cost."""
    return min(cost(flag_nothing(y_true), scenario), cost(flag_everything(y_true), scenario))


def curve(y_true, scores) -> dict:
    """Counts at every distinct cut-off, from 'flag nothing' (cut-off = inf) down to the lowest score.

    Row i flags the i-th group of coins from the highest score down; ties are never split.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0 or y.min() == y.max():
        raise ValueError("both crashes and survivors are needed to choose a cut-off")
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    last = np.append(np.where(np.diff(s_sorted) != 0)[0], len(s) - 1)     # last row of each distinct score
    cutoffs = np.concatenate([[np.inf], s_sorted[last]])
    flagged = np.concatenate([[0], last + 1])
    flagged_crash = np.concatenate([[0], np.cumsum(y_sorted)[last]])
    crashers = int(y.sum())
    return {
        "cutoff": cutoffs,
        "flagged": flagged,
        "skipped_good": flagged - flagged_crash,
        "held_crasher": crashers - flagged_crash,
        "n": len(y),
    }


def best_cutoff(y_true, scores, scenario: dict) -> dict:
    """The cut-off with the lowest total cost. A tie goes to the HIGHER cut-off (fewer coins flagged)."""
    c = curve(y_true, scores)
    total = scenario["holding_crasher"] * c["held_crasher"] + scenario["skipped_good"] * c["skipped_good"]
    i = int(np.argmin(total))                       # argmin takes the first minimum = the highest cut-off
    cutoff = float(c["cutoff"][i])
    return {"cutoff": cutoff, "counts": counts_at(y_true, scores, cutoff)}


def cutoff_at_flag_share(scores, share: float) -> float:
    """The score of the k-th highest coin, k = share of the pile. Flags about that share (ties flag a few more)."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    k = min(len(s), max(1, int(round(share * len(s)))))
    return float(s[k - 1])


def describe(label: str, cutoff, counts: dict) -> str:
    """One readable line: how many coins flagged, both mistakes, precision and recall on both sides."""
    r = rates(counts)

    def pct(x):
        return "  n/a" if x is None else f"{100 * x:5.1f}"

    cut = "  (none)" if cutoff == np.inf else f"{cutoff:8.4f}"
    return (f"{label:22} cut-off {cut}  flagged {100 * r['flag_share']:5.1f}%  "
            f"skipped good {counts['skipped_good']:5d}  held crasher {counts['held_crasher']:5d}  "
            f"crash side P/R {pct(r['crash_precision'])}/{pct(r['crash_recall'])}  "
            f"survivor side P/R {pct(r['survivor_precision'])}/{pct(r['survivor_recall'])}")
