"""A small MADE-UP sample with the same layout as the real files.

Why: the real feature.pkl is about 500 MB and cannot be downloaded in CI, so tests
and CI use this instead. Every value here is random. Nothing in it is real MELT data
and nothing learned from it means anything.

Layout copied from the authors' code (src/feat_gen.py and src/dataset.py):
  feature file = a LIST of dicts, one per coin. Each dict holds the tabular columns,
  plus mint_ts, mint_address, ts (a 360 x 5 float32 array) and ts_len.
  label file   = a CSV with mint_address, min_ratio, manipulated, return_ratio, label.

Only a FEW tabular columns are included (the three the scope filter and the split
need, plus two demo columns). The real file has 122. Add real names later, only
after they have been read from the real file's inspection report.
"""

import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np

import settings

START_TS = 1733702400  # 9 Dec 2024 00:00 UTC, as a round starting point
LABEL_COLUMNS = ["mint_address", "min_ratio", "manipulated", "return_ratio", "label"]


def make_sample(n_coins: int = 80, n_unlabelled: int = 3, seed: int = 42):
    """Return (feature_list, label_rows). Same seed gives the same sample."""
    rng = np.random.default_rng(seed)
    features, labels = [], []

    for i in range(n_coins):
        address = f"SAMPLE{i:04d}"
        # holders and valid-swap seconds: some coins fall below the scope filter
        holders = int(rng.integers(20, 400))
        valid_seconds = float(rng.integers(5, 600))
        time_span = valid_seconds + float(rng.integers(0, 300))
        ts_len = int(rng.integers(1, 361))
        bars = np.zeros((360, 5), dtype=np.float32)
        bars[:ts_len] = rng.random((ts_len, 5), dtype=np.float32)

        coin = OrderedDict()
        coin["group3_holder_num"] = holders
        coin["group3_time_span"] = time_span
        coin["group3_time_span_valid"] = valid_seconds
        coin["group2_demo_a"] = float(rng.random())
        coin["group2_demo_b"] = float(rng.random())
        coin["mint_ts"] = START_TS + i * 3600 + int(rng.integers(0, 3000))
        coin["mint_address"] = address
        coin["ts"] = bars
        coin["ts_len"] = ts_len
        features.append(coin)

        if i < n_coins - n_unlabelled:  # the last few coins get no label, like the real 233
            min_ratio = float(rng.random())
            labels.append(
                {
                    "mint_address": address,
                    "min_ratio": round(min_ratio, 6),
                    "manipulated": str(rng.choice(["yes", "no", "unknown"])),
                    "return_ratio": round(float(rng.random() * 3), 6),
                    "label": str(rng.choice(["high", "medium", "low"])),
                }
            )
    return features, labels


def write_sample(folder, **kwargs):
    """Write sample_feature.pkl and sample_label.csv into folder. Returns both paths."""
    import pickle

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    features, labels = make_sample(**kwargs)

    feature_path = folder / "sample_feature.pkl"
    with open(feature_path, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    label_path = folder / "sample_label.csv"
    with open(label_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(labels)
    return feature_path, label_path


if __name__ == "__main__":
    paths = write_sample(settings.DATA_DIR / "sample")
    print("wrote made-up sample files:", *paths, sep="\n  ")
