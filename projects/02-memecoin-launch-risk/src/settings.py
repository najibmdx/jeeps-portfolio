"""Read config.yaml and check it before anything uses it."""

import re
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
DATA_DIR = PROJECT_DIR / "data"

OUTCOME_COLUMNS = ("min_ratio", "return_ratio", "manipulated", "label")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_config(config: dict) -> dict:
    """Stop early, in plain words, if a setting is impossible. Returns the config."""
    label = config["label"]
    if not 0 < label["crash_line"] < 1:
        raise ValueError("label.crash_line must be between 0 and 1")

    split = config["split"]
    shares = [split["train"], split["validation"], split["test"]]
    if any(s <= 0 for s in shares):
        raise ValueError("split: every share must be above 0")
    if abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError(f"split: train + validation + test must add up to 1, got {sum(shares)}")

    scope = config["scope"]
    if scope["min_holders"] < 0 or scope["min_valid_swap_seconds"] < 0:
        raise ValueError("scope: limits cannot be negative")

    missing = [c for c in OUTCOME_COLUMNS if c not in config["columns"]["banned_as_features"]]
    if missing:
        raise ValueError(f"columns.banned_as_features must include {missing}: they are measured after migration")

    for name in ("feature", "label"):
        entry = config["data"][name]
        if not _SHA256.match(entry["sha256"]):
            raise ValueError(f"data.{name}.sha256 must be 64 lowercase hex characters")
        if entry["bytes"] <= 0:
            raise ValueError(f"data.{name}.bytes must be above 0")

    if not isinstance(config["seed"], int):
        raise ValueError("seed must be a whole number")
    return config


def load_config(path=None) -> dict:
    """Load and check the settings file."""
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return validate_config(yaml.safe_load(f))
