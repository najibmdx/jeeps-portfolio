"""Read config.yaml and check it before anything uses it."""

import re
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
DATA_DIR = PROJECT_DIR / "data"

TABLES = ("events", "markets", "predictions", "ohlcv_1d")
# Fields that describe the future of a market. They must stay on the excluded list.
REQUIRED_EXCLUDED = ("close_time", "events.volume", "events.open_interest", "events.n_questions")
KNOWN_BASELINES = ("raw_price", "price_only_correction", "base_rate")
KNOWN_MODELS = ("lr_price_only", "lr_all_features", "random_forest", "gradient_boosting")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_config(config: dict) -> dict:
    """Stop early, in plain words, if a setting is impossible. Returns the config."""
    q = config["question"]
    if sorted(q["outcomes"]) != ["No", "Yes"] or q["n_outcomes"] != 2:
        raise ValueError("question: only Yes/No markets with exactly 2 outcomes are built")

    t = config["decision_time"]
    for key in ("days_after_open", "robustness_days_after_open"):
        if not isinstance(t[key], int) or isinstance(t[key], bool) or t[key] < 1:
            raise ValueError(f"decision_time.{key} must be a whole number of days, at least 1")
    if t["days_after_open"] == t["robustness_days_after_open"]:
        raise ValueError("decision_time: the robustness run must use a different number of days from the main run")
    if t["bars_strictly_before_t"] is not True:
        raise ValueError("decision_time.bars_strictly_before_t must stay true: bars on or after t leak the outcome")

    u = config["universe"]
    if u["min_trades_before_t"] < 1:
        raise ValueError("universe.min_trades_before_t must be at least 1")
    if not 0 < u["contested_low"] < u["contested_high"] < 1:
        raise ValueError("universe: need 0 < contested_low < contested_high < 1")

    split = config["split"]
    shares = [split["train"], split["validation"], split["test"]]
    if any(s <= 0 for s in shares):
        raise ValueError("split: every share must be above 0")
    if abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError(f"split: train + validation + test must add up to 1, got {sum(shares)}")
    if split["order_by"] != "t":
        raise ValueError("split.order_by must be t (the decision time)")

    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be a whole number")

    files = config["data"]["files"]
    if tuple(files) != TABLES:
        raise ValueError(f"data.files must list exactly these tables, in this order: {list(TABLES)}")
    for name, entry in files.items():
        if entry["bytes"] <= 0:
            raise ValueError(f"data.files.{name}.bytes must be above 0")
        if entry["sha256"] is not None and not _SHA256.match(str(entry["sha256"])):
            raise ValueError(f"data.files.{name}.sha256 must be null or 64 lowercase hex characters")
        if not entry["url"].startswith("https://"):
            raise ValueError(f"data.files.{name}.url must start with https://")

    f = config["features"]
    used = list(f["price_path"]) + list(f["activity"]) + list(f["context"])
    if len(used) != len(set(used)):
        raise ValueError("features: a feature is listed twice")
    missing = [c for c in REQUIRED_EXCLUDED if c not in f["excluded"]]
    if missing:
        raise ValueError(f"features.excluded must include {missing}: they describe the future of the market")
    if set(used) & set(f["excluded"]):
        raise ValueError("features: an excluded field is also listed as a feature")
    if set(used) & set(f["held_back"]) or set(f["held_back"]) & set(f["excluded"]):
        raise ValueError("features: a held-back field cannot also be a feature or excluded")

    if list(config["baselines"]) != list(KNOWN_BASELINES):
        raise ValueError(f"baselines must be exactly {list(KNOWN_BASELINES)}")

    m = config["metrics"]
    if m["decides"] != "log_loss":
        raise ValueError("metrics.decides must be log_loss (the only rule that is built)")
    if list(m["slices"]) != ["all", "contested"]:
        raise ValueError("metrics.slices must be [all, contested]")

    models = config["models"]
    if not models["ladder"] or not set(models["ladder"]) <= set(KNOWN_MODELS):
        raise ValueError(f"models.ladder must be a non-empty list drawn from {list(KNOWN_MODELS)}")
    if models["selection"] != "lowest_validation_log_loss":
        raise ValueError("models.selection must be lowest_validation_log_loss (the only rule that is built)")
    return config


def load_config(path=None) -> dict:
    """Load and check the settings file."""
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return validate_config(yaml.safe_load(f))
