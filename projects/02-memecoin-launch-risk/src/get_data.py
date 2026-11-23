"""Get the two data files and check them. Never overwrites a file that is already there.

label.csv     : downloaded from the MELT repository, then checked against its hash.
feature.pkl   : NOT downloaded by this script (the authors' Google Drive shows a
                virus-scan page). The script only checks that the file you saved
                is the exact file we inspected.

Run from the projects folder:  python 02-memecoin-launch-risk/src/get_data.py (on Windows cmd use backslashes)
Add --check-only to skip the download and only check.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import settings


def sha256_of(path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in pieces so a big file does not fill memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def check_file(path, expected_bytes: int, expected_sha256: str) -> str:
    """Return 'missing', 'wrong_size', 'wrong_hash' or 'ok'."""
    path = Path(path)
    if not path.exists():
        return "missing"
    if path.stat().st_size != expected_bytes:
        return "wrong_size"
    if sha256_of(path) != expected_sha256:
        return "wrong_hash"
    return "ok"


def _download(url: str, dest) -> None:
    """Save a web address to a file."""
    with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)


def fetch_label(config: dict, data_dir, download=_download):
    """Make sure label.csv is in data_dir. Returns (ok, message). Never overwrites."""
    entry = config["data"]["label"]
    data_dir = Path(data_dir)
    final = data_dir / entry["file"]

    status = check_file(final, entry["bytes"], entry["sha256"])
    if status == "ok":
        return True, f"{entry['file']}: already here and the hash matches. Not touched."
    if status != "missing":
        return False, (
            f"{entry['file']}: a file is already there but it does not match "
            f"({status}). I will not overwrite it. Rename or move it, then run again."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    part = data_dir / (entry["file"] + ".part")
    try:
        download(entry["url"], part)
    except OSError as error:
        part.unlink(missing_ok=True)
        return False, f"{entry['file']}: download failed ({error}). Nothing was saved."

    status = check_file(part, entry["bytes"], entry["sha256"])
    if status != "ok":
        part.unlink(missing_ok=True)
        return False, (
            f"{entry['file']}: the download does not match the file we checked "
            f"({status}). It was thrown away. Nothing was saved."
        )
    if final.exists():  # someone saved one while we were downloading
        part.unlink(missing_ok=True)
        return False, f"{entry['file']}: a file appeared meanwhile. I will not overwrite it."
    os.replace(part, final)
    return True, f"{entry['file']}: downloaded and the hash matches."


def check_feature(config: dict, data_dir):
    """Check feature.pkl. Returns (ok, message). Never downloads, never overwrites."""
    entry = config["data"]["feature"]
    path = Path(data_dir) / entry["file"]
    status = check_file(path, entry["bytes"], entry["sha256"])
    if status == "ok":
        return True, f"{entry['file']}: found and the hash matches."
    if status == "missing":
        return False, (
            f"{entry['file']}: not found at {path}. Download it from the {entry['source']}, "
            f"save it as that exact path, then run this again."
        )
    return False, (
        f"{entry['file']}: found at {path} but it does not match the file we checked "
        f"({status}). Do NOT load it. Download it again from the official link."
    )


def main(argv=None, config=None, data_dir=None, download=_download) -> int:
    """Return 0 if both files are in place and correct, otherwise 1."""
    argv = sys.argv[1:] if argv is None else argv
    config = config or settings.load_config()
    data_dir = Path(data_dir) if data_dir else settings.DATA_DIR

    print(f"data folder: {data_dir}")
    print("checking hashes (the big file takes a few seconds)...")
    if "--check-only" in argv:
        entry = config["data"]["label"]
        status = check_file(data_dir / entry["file"], entry["bytes"], entry["sha256"])
        label_ok, label_msg = status == "ok", f"{entry['file']}: {status}"
    else:
        label_ok, label_msg = fetch_label(config, data_dir, download)
    feature_ok, feature_msg = check_feature(config, data_dir)

    print(label_msg)
    print(feature_msg)
    return 0 if (label_ok and feature_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
