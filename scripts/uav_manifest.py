"""Build the UAV-Human (RGB, Cross-Subject-v1) split manifest.

Probes every extracted training clip and assigns a stratified fit/validation
split per action class (seed 17). The validation fraction matches Iteration 1's
fit/validation sizes (12,773 / 3,512); its exact assignment is unknown, so the
split here is our own and clips of one subject can fall on both sides.

Writes manifests/uavhuman_csv1_split.csv (relative paths only).
"""

import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from vbench.data import probe_video

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/uavhuman/train")
OUT = Path("manifests/uavhuman_csv1_split.csv")
SEED = 17
VAL_FRACTION = 3512 / 16285


def main() -> None:
    rows, bad = [], []
    for p in sorted(ROOT.rglob("*.avi")):
        try:
            n, w, h = probe_video(p)
        except Exception as e:
            bad.append((p, repr(e)))
            continue
        label = re.search(r"(A\d{3})R", p.name).group(1)
        rows.append(dict(path=str(p.relative_to(ROOT)), label=label, official="train",
                         frames=n, width=w, height=h))
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r)
    rng = random.Random(SEED)
    for label in sorted(by_class):
        items = sorted(by_class[label], key=lambda r: r["path"])
        rng.shuffle(items)
        n_val = round(len(items) * VAL_FRACTION)
        for i, r in enumerate(items):
            r["split"] = "val" if i < n_val else "fit"
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "official", "split", "frames", "width", "height"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["path"]))
    counts = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    print("classes:", len(by_class), "counts:", dict(counts), "unreadable:", len(bad))
    for p, e in bad:
        print("  BAD", p, e)
    fr = sorted(r["frames"] for r in rows)
    print("frames min/median/max:", fr[0], fr[len(fr) // 2], fr[-1])
    print("sizes:", sorted({(r["width"], r["height"]) for r in rows})[:6])


if __name__ == "__main__":
    main()
