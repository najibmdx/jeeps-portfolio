"""Entry point: check the settings and the data layout, build the dataset, split it, audit for leakage and train-versus-validation shift, score the baselines and the untuned model ladder. Nothing is modelled yet.

    python src/pipeline.py --sample          # made-up data, no download
    python src/pipeline.py --data-dir DIR    # real files in DIR (default: ./data)
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

import baselines
import build
import leakage
import models
import schema
import settings
import split
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

        table, report = split.assign(dataset, config)
        split.write_splits(table, data_dir, config, k)
        print("\n".join(split.summary_lines(report)))

        # The leakage audit sees the rows that were written to the train and validation files. Test is not passed in.
        kept = lambda part: table[(table["split"] == part) & ~table["purged"]].drop(columns=split.HELPERS)  # noqa: E731
        audit_report = leakage.audit(kept("train"), kept("validation"), config)
        print("\n".join(leakage.summary_lines(audit_report, config)))
        if audit_report["problems"]:
            return 1

        # The baselines are fitted on the train rows and scored on the validation rows. Test is not passed in.
        try:
            baseline_report = baselines.run(kept("train"), kept("validation"), config)
        except ValueError as e:
            print(f"PROBLEM baselines: {e}")
            return 1
        print("\n".join(baselines.summary_lines(baseline_report)))

        # The untuned models are fitted on the train rows and scored on train and validation. Test is not passed in.
        try:
            model_report = models.run(kept("train"), kept("validation"), config)
        except ValueError as e:
            print(f"PROBLEM models: {e}")
            return 1
        print("\n".join(models.summary_lines(model_report)))

    if any(config["data"]["files"][t]["sha256"] is None for t in settings.TABLES):
        print("Note: file fingerprints (sha256) are not recorded yet. Stage 1 records them.")
    print("Stage 6: the dataset is built, split, checked for leakage and shift, and the baselines and the untuned model ladder are scored. Nothing is tuned or chosen yet.")
    print("The test split was not opened by this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
