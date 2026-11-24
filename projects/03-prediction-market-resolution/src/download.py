"""Download the four data files from Hugging Face and check each one.

Every file is checked against the size and sha256 in config.yaml. A file that is already
in data/ and passes the check is left alone. Run:  python src/download.py [--data-dir DIR]
"""

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

import settings

_CHUNK = 1024 * 1024


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def file_problem(path, expected_bytes, expected_sha256):
    """None if the file is exactly as expected, otherwise a plain sentence."""
    path = Path(path)
    if not path.exists():
        return "not there"
    size = path.stat().st_size
    if size != expected_bytes:
        return f"is {size} bytes, expected {expected_bytes}"
    if expected_sha256 is not None and sha256_of(path) != expected_sha256:
        return "has the right size but a different sha256"
    return None


def download_file(url, dest, expected_bytes, expected_sha256, say=print) -> None:
    """Fetch url into dest through a .part file. Raises ValueError if the result is not as expected."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "jeeps-portfolio-project-03"})
    done, next_report, start = 0, 0.2, time.time()
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        while block := response.read(_CHUNK):
            out.write(block)
            done += len(block)
            if expected_bytes and done / expected_bytes >= next_report:
                say(f"  {int(next_report * 100)}% ({done / 1e6:.0f} MB, {time.time() - start:.0f}s)")
                next_report += 0.2
    problem = file_problem(part, expected_bytes, expected_sha256)
    if problem:
        part.unlink()
        raise ValueError(f"download {problem}")
    part.replace(dest)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download and check the four data files")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    config = settings.load_config()
    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_DIR
    failed = False
    for table, entry in config["data"]["files"].items():
        path = data_dir / entry["file"]
        if file_problem(path, entry["bytes"], entry["sha256"]) is None:
            print(f"{table}: already there and checked (size and sha256).")
            continue
        print(f"{table}: downloading {entry['bytes'] / 1e6:.0f} MB ...")
        try:
            download_file(entry["url"], path, entry["bytes"], entry["sha256"])
        except Exception as error:
            failed = True
            print(f"PROBLEM {table}: {error}")
            continue
        print(f"{table}: downloaded and checked (size and sha256).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
