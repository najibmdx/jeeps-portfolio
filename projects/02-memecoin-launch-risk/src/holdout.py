"""The ONLY place the test pile can be opened. It is kept apart from piles.load_piles on purpose.

Rules, all enforced here:
  1. Only with the explicit confirm flag.
  2. Only after the choices were frozen in a file (select_final.py writes it, and it should be committed first).
  3. Only once. The moment the pile is opened a lock file is written (created in 'exclusive' mode, so two runs
     cannot both get through). Any later attempt is refused.

If a run crashed because of a bug AFTER the lock was written, the lock stays. Deleting it by hand is possible
but is a decision you must disclose in the write-up: the test pile was then opened more than once.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import prepare


def lock_path(data_dir, config: dict) -> Path:
    return Path(data_dir) / config["final"]["test_lock_file"]


def is_locked(data_dir, config: dict) -> bool:
    """True once the test pile has been opened."""
    return lock_path(data_dir, config).exists()


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def open_test_pile(data_dir, config: dict, frozen_path, confirm: bool = False):
    """Write the lock, then read the test features. Refuses without confirm, without a frozen file, or a second time."""
    if not confirm:
        raise PermissionError("the test pile is opened only with the explicit confirm flag (--confirm-open-test)")
    frozen_path = Path(frozen_path)
    if not frozen_path.exists():
        raise FileNotFoundError(f"{frozen_path.name} is missing. Run select_final.py and commit that file first.")
    lock = lock_path(data_dir, config)
    if lock.exists():
        raise FileExistsError(
            f"the test pile was already opened ({lock.name} exists). It is opened once. "
            "If an earlier run died from a bug, deleting the lock by hand is possible but must be disclosed.")
    path = Path(data_dir) / config["data"]["prepared_folder"] / "test_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run prepare.py first.")

    record = {"opened_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
              "frozen_file": frozen_path.name, "frozen_sha256": file_sha256(frozen_path)}
    with open(lock, "x", encoding="utf-8") as f:          # "x" = fail if it already exists
        json.dump(record, f, indent=2)

    pile = prepare.load_features(path)
    if "crash" not in pile or not set(pile["crash"].unique()) <= {0, 1}:
        raise ValueError("test: the crash column must exist and hold only 0 and 1")
    return pile
