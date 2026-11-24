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
KNOWN_AT_T = ("yes", "assumed", "no", "unverified")
# Stage 7: what can be tuned, the library default of each setting (the grid must hold it), how the setting is typed,
# and the direction a grid is listed in so that "the first of equals wins a tie" means "the simpler setting wins".
TUNABLE = {"lr_all_features": ("C",), "random_forest": ("min_samples_leaf",), "gradient_boosting": ("learning_rate", "min_samples_leaf")}
LIBRARY_DEFAULTS = {"lr_all_features": {"C": 1.0}, "random_forest": {"min_samples_leaf": 1}, "gradient_boosting": {"learning_rate": 0.1, "min_samples_leaf": 20}}
PARAM_TYPES = {"C": float, "learning_rate": float, "min_samples_leaf": int}
GRID_DIRECTION = {"C": "ascending", "learning_rate": "ascending", "min_samples_leaf": "descending"}
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
    if split["purge_unknown_labels"] is not True:
        raise ValueError("split.purge_unknown_labels must stay true: a market whose outcome is unknown when the next period starts leaks it")
    if split["events_whole"] is not True:
        raise ValueError("split.events_whole must stay true: markets of one event must never sit in two splits")

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

    d = config["dataset"]
    if "{k}" not in d["file"]:
        raise ValueError("dataset.file must contain {k} so the k=7 and k=3 tables never overwrite each other")
    if "{k}" not in d["split_file"] or "{part}" not in d["split_file"]:
        raise ValueError("dataset.split_file must contain {part} and {k} so no split file overwrites another")
    if set(d["meta"]) & set(used):
        raise ValueError("dataset.meta: a column cannot be both a model input and a meta column")
    for needed in ("market_id", "event_key", "t", "close_time", "label"):
        if needed not in d["meta"]:
            raise ValueError(f"dataset.meta must include {needed}")

    if list(config["baselines"]) != list(KNOWN_BASELINES):
        raise ValueError(f"baselines must be exactly {list(KNOWN_BASELINES)}")

    m = config["metrics"]
    if m["decides"] != "log_loss":
        raise ValueError("metrics.decides must be log_loss (the only rule that is built)")
    if list(m["slices"]) != ["all", "contested"]:
        raise ValueError("metrics.slices must be [all, contested]")

    if not isinstance(m["price_clip"], (int, float)) or isinstance(m["price_clip"], bool) or not 0 < m["price_clip"] < 0.5:
        raise ValueError("metrics.price_clip must be above 0 and below 0.5")
    bands = m["calibration_bands"]
    if (not isinstance(bands, list) or len(bands) < 3 or bands[0] != 0 or bands[-1] != 1
            or any(not isinstance(b, (int, float)) or isinstance(b, bool) for b in bands) or any(b >= c for b, c in zip(bands, bands[1:]))):
        raise ValueError("metrics.calibration_bands must be a strictly increasing list of at least 3 numbers, from 0 to 1")
    ev = config["evaluation"]
    if not isinstance(ev["bootstrap_draws"], int) or isinstance(ev["bootstrap_draws"], bool) or ev["bootstrap_draws"] < 1:
        raise ValueError("evaluation.bootstrap_draws must be a whole number, at least 1")
    if not isinstance(ev["interval"], (int, float)) or isinstance(ev["interval"], bool) or not 0 < ev["interval"] < 1:
        raise ValueError("evaluation.interval must be above 0 and below 1")

    models = config["models"]
    if not models["ladder"] or not set(models["ladder"]) <= set(KNOWN_MODELS):
        raise ValueError(f"models.ladder must be a non-empty list drawn from {list(KNOWN_MODELS)}")
    if models["selection"] != "lowest_validation_log_loss":
        raise ValueError("models.selection must be lowest_validation_log_loss (the only rule that is built)")
    for key in ("forest_trees", "logistic_max_iter"):
        if not isinstance(models[key], int) or isinstance(models[key], bool) or models[key] < 1:
            raise ValueError(f"models.{key} must be a whole number, at least 1")
    dropped = models["dropped_features"]
    if not isinstance(dropped, dict) or any(not str(why).strip() for why in dropped.values()):
        raise ValueError("models.dropped_features must map each dropped feature to a reason")
    unknown = [c for c in dropped if c not in used]
    if unknown:
        raise ValueError(f"models.dropped_features names columns that are not features: {unknown}")
    if len(dropped) >= len(used):
        raise ValueError("models.dropped_features leaves no model input")
    odds = models["log_odds_columns"]
    if not isinstance(odds, list) or any(c not in used for c in odds) or len(odds) != len(set(odds)):
        raise ValueError("models.log_odds_columns must be a list of features, each once")
    _validate_tuning(config)
    _validate_leakage(config)
    return config


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_tuning(config: dict) -> None:
    """Stage 7 grids: one per tunable model of the ladder, the same number of settings for each, each holding the library default."""
    models = config["models"]
    tuning = models["tuning"]
    wanted = [m for m in models["ladder"] if m in TUNABLE]
    if not isinstance(tuning, dict) or set(tuning) != set(wanted):
        raise ValueError(f"models.tuning must have a grid for exactly these models of the ladder: {wanted}")
    sizes = {}
    for name in wanted:
        grid = tuning[name]
        if not isinstance(grid, dict) or set(grid) != set(TUNABLE[name]):
            raise ValueError(f"models.tuning.{name} must list exactly these settings: {list(TUNABLE[name])}")
        size = 1
        for param in TUNABLE[name]:
            values = grid[param]
            where = f"models.tuning.{name}.{param}"
            if not isinstance(values, list) or not values:
                raise ValueError(f"{where} must be a non-empty list")
            if any(not _is_number(v) for v in values):
                raise ValueError(f"{where} must hold numbers only")
            if param == "min_samples_leaf" and any(not isinstance(v, int) or v < 1 for v in values):
                raise ValueError(f"{where} must hold whole numbers, at least 1")
            if param == "C" and any(v <= 0 for v in values):
                raise ValueError(f"{where} must hold numbers above 0")
            if param == "learning_rate" and any(not 0 < v <= 1 for v in values):
                raise ValueError(f"{where} must hold numbers above 0 and at most 1")
            ordered = sorted(values, reverse=GRID_DIRECTION[param] == "descending")
            if values != ordered or len(set(values)) != len(values):
                raise ValueError(f"{where} must be listed {GRID_DIRECTION[param]}, each value once: the first of equals wins a tie, so the simplest setting comes first")
            if LIBRARY_DEFAULTS[name][param] not in values:
                raise ValueError(f"{where} must include the library default {LIBRARY_DEFAULTS[name][param]}, so tuning cannot come out worse than untuned")
            size *= len(values)
        sizes[name] = size
    if len(set(sizes.values())) > 1:
        raise ValueError(f"models.tuning: every model gets the same number of settings, got {sizes}")
    if sizes and next(iter(sizes.values())) < 2:
        raise ValueError("models.tuning: each model needs at least 2 settings, or there is nothing to tune")


def _validate_leakage(config: dict) -> None:
    """The leakage settings and the table that says, for every column, whether it is known at t."""
    lk = config["leakage"]
    for key in ("min_markets_in_segment", "min_events_in_segment"):
        if not isinstance(lk[key], int) or isinstance(lk[key], bool) or lk[key] < 1:
            raise ValueError(f"leakage.{key} must be a whole number, at least 1")
    if not isinstance(lk["clock_like_correlation"], (int, float)) or not 0 < lk["clock_like_correlation"] <= 1:
        raise ValueError("leakage.clock_like_correlation must be above 0 and at most 1")
    if not isinstance(lk["shift_number_difference"], (int, float)) or isinstance(lk["shift_number_difference"], bool) or lk["shift_number_difference"] <= 0:
        raise ValueError("leakage.shift_number_difference must be a number above 0")
    if not isinstance(lk["shift_share_change"], (int, float)) or isinstance(lk["shift_share_change"], bool) or not 0 < lk["shift_share_change"] <= 1:
        raise ValueError("leakage.shift_share_change must be above 0 and at most 1")
    edges = lk["sibling_bucket_edges"]
    if not edges or any(not isinstance(e, int) or isinstance(e, bool) or e < 1 for e in edges) or edges != sorted(set(edges)):
        raise ValueError("leakage.sibling_bucket_edges must be whole numbers from 1, strictly increasing")

    f = config["features"]
    features = list(f["price_path"]) + list(f["activity"]) + list(f["context"])
    role_of = {name: "meta" for name in config["dataset"]["meta"]}
    role_of.update({name: "excluded" for name in f["excluded"]})
    role_of.update({name: "held_back" for name in f["held_back"]})
    role_of.update({name: "feature" for name in features})

    table = config["leakage_table"]
    missing = [n for n in role_of if n not in table]
    if missing:
        raise ValueError(f"leakage_table has no row for: {missing}")
    extra = [n for n in table if n not in role_of]
    if extra:
        raise ValueError(f"leakage_table has rows for names nothing uses: {extra}")
    for name, row in table.items():
        if set(row) != {"role", "source", "known_at_t", "reason"}:
            raise ValueError(f"leakage_table.{name} must have exactly role, source, known_at_t and reason")
        if row["role"] != role_of[name]:
            raise ValueError(f"leakage_table.{name}: role is {row['role']}, the other settings make it {role_of[name]}")
        if row["known_at_t"] not in KNOWN_AT_T:
            raise ValueError(f"leakage_table.{name}.known_at_t must be one of {list(KNOWN_AT_T)} (as text), got {row['known_at_t']!r}")
        if not str(row["source"]).strip() or not str(row["reason"]).strip():
            raise ValueError(f"leakage_table.{name} needs a source and a reason")
        allowed = {"feature": ("yes", "assumed"), "held_back": ("unverified",), "excluded": ("no",)}.get(row["role"])
        if allowed and row["known_at_t"] not in allowed:
            raise ValueError(f"leakage_table.{name}: a {row['role']} column must be {' or '.join(allowed)} at t, not {row['known_at_t']}")


def load_config(path=None) -> dict:
    """Load and check the settings file."""
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return validate_config(yaml.safe_load(f))
