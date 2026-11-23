"""Run the whole project in order, from raw data to results.

    python src/pipeline.py        (or:  bash run.sh)

Steps: get the data if it is missing, then train, wobble, leakage audit, ablations,
cost-scenario cut-offs and tuning. Every step uses the TRAINING and VALIDATION sets only.

The one-time test run (final.py) is NOT in this list, on purpose. It may be run once,
by hand, after every choice is made, and it refuses a second run. A test checks that
this file never imports it.
"""

import time

import ablation
import cutoff
import get_data
import leakage_audit
import train
import tune
import uncertainty

# (name, module). Each module has a main() that reads config.yaml and writes to outputs/.
STEPS = [
    ("get_data", get_data),
    ("train", train),
    ("uncertainty", uncertainty),
    ("leakage_audit", leakage_audit),
    ("ablation", ablation),
    ("cutoff", cutoff),
    ("tune", tune),
]


def run_steps(steps=None) -> list[tuple[str, float]]:
    """Run each step in order. Stop at the first failure, saying which step failed.

    Returns (name, seconds) for every step that finished.
    """
    steps = STEPS if steps is None else steps
    done = []
    for number, (name, module) in enumerate(steps, start=1):
        print(f"\n===== step {number} of {len(steps)}: {name} =====", flush=True)
        started = time.perf_counter()
        try:
            module.main()
        except Exception:
            print(f"\nFAILED in step {number} ({name}). Later steps were not run.", flush=True)
            raise
        done.append((name, round(time.perf_counter() - started, 1)))
    return done


def main() -> None:
    done = run_steps()
    print("\n===== done =====")
    for name, seconds in done:
        print(f"{name:15} {seconds:8.1f} s")
    print("The test set was not used. final.py was not run.")


if __name__ == "__main__":
    main()
