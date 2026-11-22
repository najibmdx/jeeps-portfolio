"""Read the settings file, so that experiments change config.yaml and not the code."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Return the settings as a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
