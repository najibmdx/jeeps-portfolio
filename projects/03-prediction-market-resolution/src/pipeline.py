"""Entry point: check the settings and the data layout, then build the dataset. Nothing is modelled yet.

    python src/pipeline.py --sample          # made-up data, no download
    python src/pipeline.py --data-dir DIR    # real files in DIR (default: ./data)
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

import build
import schema
import settings
from sample import write_sample


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Project C pipeline")
    parser.add_argument("--sample", action="store_true", help="use made-up data, no download")
    parser.add_argument("--data-dir", default=None, help="folder holding the four parquet files")
    args = parser.parse_args(argv)

    config = settings.load_config()
    print("Settings: config.yaml read and checked.")

    with tempfile.TemporaryDirectory() as tmp:
        if args.sample:
            data_dir = write_sample(Path(tmp) / "sample", seed=config["seed"])
            check_sizes = False
            print("SAMPLE MODE: made-up data, written to a temporary folder outside the project.")
            print("The numbers mean nothing.")
        else:
            data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_DIR
            check_sizes = True

        missing = [t for t in settings.TABLES if not (data_dir / f"{t}.parquet").exists()]
        if missing:
            print(f"Missing data files in {data_dir}: {', '.join(m + '.parquet' for m in missing)}")
            return 2

        failed = False
        for table in settings.TABLES:
            path = data_dir / f"{table}.parquet"
            problems = schema.check_file(table, path)
            if check_sizes:
                expected = config["data"]["files"][table]["bytes"]
                actual = path.stat().st_size
                if actual != expected:
                    problems.append(f"file is {actual} bytes, expected {expected}")
            if problems:
                failed = True
                for p in problems:
                    print(f"PROBLEM {table}: {p}")
                continue
            rows = pq.ParquetFile(path).metadata.num_rows
            print(f"{table}: layout OK, {rows:,} rows")
        if failed:
            return 1

        k = config["decision_time"]["days_after_open"]
        dataset, funnel = build.build_dataset(data_dir, config, k)
        path = build.write_dataset(dataset, funnel, data_dir, config, k)
        print(f"Built the dataset for k={k}: " + ", ".join(f"{name} {n:,}" for name, n in funnel.items()))
        print(f"Dataset: {len(dataset):,} rows, written to {path.name} (not committed).")

    if any(config["data"]["files"][t]["sha256"] is None for t in settings.TABLES):
        print("Note: file fingerprints (sha256) are not recorded yet. Stage 1 records them.")
    print("Stage 2: the dataset is built. The split and model steps do not exist yet.")
    print("The test split was not opened by this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
