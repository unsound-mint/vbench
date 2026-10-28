"""Decode a dataset's clips into uint8 memmap caches (see vbench.data).

Usage: build_cache.py <dataset> [workers]

Writes the fit+validation cache and, when the manifest has test clips, a
separate test cache, plus a local row index for each. Caches are derived data
under data/ and are never published.
"""

import csv
import sys

from vbench.data import build_cache, read_manifest
from vbench.run import DATASETS


def build(rows: list[dict], videos, cache, index, workers: int) -> None:
    for i, r in enumerate(rows):
        r["cache_idx"] = i
    build_cache(rows, videos, cache, workers)
    with index.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("cached", len(rows), "clips ->", cache, f"{cache.stat().st_size / 2**30:.2f} GiB", flush=True)


def main() -> None:
    ds = DATASETS[sys.argv[1]]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rows = read_manifest(ds["manifest"])
    train = [r for r in rows if r["split"] in ("fit", "val")]
    test = [r for r in rows if r["split"] == "test"]
    build(train, ds["videos"], ds["cache"], ds["index"], workers)
    if test and "test_cache" in ds:
        build(test, ds["videos"], ds["test_cache"], ds["test_index"], workers)


if __name__ == "__main__":
    main()
