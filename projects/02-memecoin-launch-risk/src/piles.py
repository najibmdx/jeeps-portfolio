"""Load the saved piles for modelling, and choose which columns a scorer may see.

Modelling stages open ONLY the train and validation piles. The test pile is refused here
on purpose: it is opened once, at the very end, by its own step.
"""

from pathlib import Path

import pandas as pd

import columns
import prepare

MODELLING_PARTS = ("train", "validation")


def load_piles(data_dir, config: dict, parts=MODELLING_PARTS) -> dict:
    """Read the saved feature files of the chosen piles. Refuses the test pile."""
    if "test" in parts:
        raise ValueError("the test pile is not opened by modelling steps; it is opened once, at the very end")
    folder = Path(data_dir) / config["data"]["prepared_folder"]
    piles = {}
    for name in parts:
        path = folder / f"{name}_features.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run prepare.py first.")
        pile = prepare.load_features(path)
        if "crash" not in pile or not set(pile["crash"].unique()) <= {0, 1}:
            raise ValueError(f"{name}: the crash column must exist and hold only 0 and 1")
        piles[name] = pile
    return piles


def select_features(all_columns, config: dict) -> list:
    """The columns the scorer may see: allowed feature columns that belong to the chosen groups."""
    groups = config["features"]["groups"]
    allowed = columns.feature_columns(all_columns, config)
    chosen = [c for c in allowed if c.split("_")[0] in groups]
    for group in groups:
        if not any(c.split("_")[0] == group for c in chosen):
            raise ValueError(f"no columns found for group {group}")
    columns.check_no_banned(chosen, config)
    return chosen
