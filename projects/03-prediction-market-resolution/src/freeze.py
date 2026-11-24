"""Stage 7: write frozen_choice.yaml, the model chosen on validation, from the report of tune.py.

    python src/freeze.py [--data-dir DIR]          (needs outputs/tuning_k7.json from src/tune.py)

The file is committed BEFORE the test split is opened. It holds the model, its settings, the seed, the validation scores,
the margin over the runner-up and the date it was written. It holds no test number: none exists yet, and none was read.

This step refuses to run when
  - the main report is not there (run tune.py first), or it was made for a different k than the main run,
  - the test split has already been opened (the lock file exists): a choice made after that is not a choice,
  - frozen_choice.yaml already exists. A frozen choice is replaced only by deleting the file on purpose, by hand.
Stage 9 reads the file back with read_frozen(), which checks it against config.yaml.
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import yaml

import settings
import tune

KEYS = ("frozen_on", "k", "model", "params", "fixed_settings", "fitted_on", "train_markets", "validation_markets", "inputs",
        "validation", "runner_up", "margin_runner_up_minus_winner_all", "winner_minus_raw_price_all", "tie_at_4_decimals",
        "rule", "tuning_report", "test_split")
FITTED_ON = "train only (no refit on train + validation, so the test measures exactly this model)"
TEST_STATE = "not opened"
HEADER = """# The model chosen on validation (Stage 7). Written by src/freeze.py from {report}.
# Committed BEFORE the test split is opened. It holds no test number: the test split has not been read.
# The validation score of the winner is slightly flattering: it was picked on those markets. The one-time test is the honest number.
"""


def fixed_settings(name, config) -> dict:
    """The settings that were not tuned, as set in config.yaml."""
    m = config["models"]
    return {"lr_price_only": {}, "lr_all_features": {"seed": config["seed"], "logistic_max_iter": m["logistic_max_iter"]},
            "random_forest": {"seed": config["seed"], "forest_trees": m["forest_trees"]},
            "gradient_boosting": {"seed": config["seed"]}}[name]


def make_frozen(report, config, k, today) -> dict:
    """The contents of frozen_choice.yaml. Refuses a report that does not fit config.yaml (a stale or edited report)."""
    c = report["choice"]
    ladder = list(config["models"]["ladder"])
    if c["winner"] not in ladder:
        raise ValueError(f"the report's winner {c['winner']} is not on the ladder of config.yaml: {ladder}")
    if c["params"] not in tune.grid_settings(config, c["winner"]):
        raise ValueError("the report's winning settings are not in the grid of config.yaml; run src/tune.py again")
    if report["seed"] != config["seed"]:
        raise ValueError("the report was made with a different seed from config.yaml; run src/tune.py again")
    return {
        "frozen_on": today, "k": k, "model": c["winner"], "params": dict(c["params"]), "fixed_settings": fixed_settings(c["winner"], config),
        "fitted_on": FITTED_ON, "train_markets": report["train_markets"], "validation_markets": report["validation_markets"], "inputs": list(report["inputs"]),
        "validation": {"log_loss_all": c["validation"]["all"], "log_loss_contested": c["validation"]["contested"]},
        "runner_up": {"model": c["runner_up"], "params": dict(c["runner_up_params"]),
                      "log_loss_all": c["runner_up_validation"]["all"], "log_loss_contested": c["runner_up_validation"]["contested"]},
        "margin_runner_up_minus_winner_all": {k_: c["margin_runner_up_minus_winner"]["all"][k_] for k_ in ("point", "low", "high", "verdict")},
        "winner_minus_raw_price_all": {k_: c["winner_minus_raw_price"]["all"][k_] for k_ in ("point", "low", "high", "verdict")},
        "tie_at_4_decimals": c["tie_at_4_decimals"],
        "rule": report["rule"]["decides"] + "; the range is shown, never used to decide",
        "tuning_report": f"outputs/tuning_k{k}.json", "test_split": TEST_STATE,
    }


def frozen_text(frozen) -> str:
    return HEADER.format(report=frozen["tuning_report"]) + yaml.safe_dump(frozen, sort_keys=False, default_flow_style=False, allow_unicode=True)


def is_test_key(key) -> bool:
    """A key that names the test split, such as test_log_loss or log_loss_test. "contested" is not one."""
    return re.search(r"(^|_)test(_|$)", str(key).lower()) is not None


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _keys(value)


def read_frozen(path, config) -> dict:
    """Read frozen_choice.yaml and check it against config.yaml. Stage 9 uses this and nothing else to learn the chosen model."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path.name} is missing: the choice must be frozen (and committed) before the test split is opened")
    frozen = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(frozen, dict) or set(frozen) != set(KEYS):
        raise ValueError(f"{path.name} must hold exactly these entries: {list(KEYS)}")
    stray = [key for key in _keys(frozen) if is_test_key(key) and key != "test_split"]
    if stray or frozen["test_split"] != TEST_STATE:
        raise ValueError(f"{path.name} must hold no test numbers, and test_split must say {TEST_STATE!r}")
    name = frozen["model"]
    if name not in config["models"]["ladder"]:
        raise ValueError(f"{path.name}: {name} is not on the ladder of config.yaml (only the ladder models can be chosen)")
    if frozen["params"] not in tune.grid_settings(config, name):
        raise ValueError(f"{path.name}: the settings {frozen['params']} are not in the grid of config.yaml")
    if frozen["fixed_settings"] != fixed_settings(name, config):
        raise ValueError(f"{path.name}: the fixed settings differ from config.yaml, so this is not the model that was chosen")
    if not isinstance(frozen["frozen_on"], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", frozen["frozen_on"]):
        raise ValueError(f"{path.name}: frozen_on must be a date written as YYYY-MM-DD")
    return frozen


def main(argv=None, today=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the model chosen on validation")
    parser.add_argument("--data-dir", default=None, help="where the test lock file would be (default: ./data)")
    args = parser.parse_args(argv)
    config = settings.load_config()
    k = config["decision_time"]["days_after_open"]
    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_DIR
    report_path = settings.PROJECT_DIR / config["evaluation"]["outputs_folder"] / f"tuning_k{k}.json"
    frozen_path = settings.PROJECT_DIR / config["models"]["frozen_file"]
    if not report_path.exists():
        print(f"No tuning report at {report_path}. Run src/tune.py first (the main run, k={k}).")
        return 2
    if (data_dir / config["models"]["test_lock_file"]).exists():
        print("The test split has already been opened. A choice made now would not be a choice. Nothing was written.")
        return 1
    if frozen_path.exists():
        print(f"{frozen_path.name} already exists. A frozen choice is replaced only by deleting the file by hand, on purpose. Nothing was written.")
        return 1
    try:
        frozen = make_frozen(json.loads(report_path.read_text(encoding="utf-8")), config, k, today or datetime.date.today().isoformat())
    except (ValueError, KeyError) as e:
        print(f"PROBLEM {e}")
        return 1
    frozen_path.write_text(frozen_text(frozen), encoding="utf-8")
    try:
        read_frozen(frozen_path, config)  # what was written must read back and pass its own checks
    except ValueError as e:
        frozen_path.unlink()
        print(f"PROBLEM the file that was about to be written does not pass its own checks, so it was removed: {e}")
        return 1
    print(frozen_text(frozen))
    print(f"Written to {frozen_path.name}. Read it, then commit it BEFORE the test split is opened.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
