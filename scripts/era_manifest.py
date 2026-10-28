"""Build the ERA split manifest: probe every video, assign a stratified 80/20
fit/validation split of the official training set (seed 17).

Writes manifests/era_split.csv (relative paths only, no video content).
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from vbench.data import probe_video as probe

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/era/Videos")
OUT = Path("manifests/era_split.csv")
SEED = 17
VAL_FRACTION = 0.2
OFFICIAL = {"Tra": "train", "Test": "test"}  # the archive's training folder is named "Tra"


def main() -> None:
    rows, bad = [], []
    for folder, official in OFFICIAL.items():
        for p in sorted((ROOT / folder).rglob("*.mp4")):
            try:
                n, w, h = probe(p)
            except Exception as e:
                bad.append((p, repr(e)))
                continue
            rows.append(dict(path=str(p.relative_to(ROOT)), label=p.parent.name, official=official,
                             frames=n, width=w, height=h))
    by_class = defaultdict(list)
    for r in rows:
        if r["official"] == "train":
            by_class[r["label"]].append(r)
    rng = random.Random(SEED)
    for label in sorted(by_class):
        items = sorted(by_class[label], key=lambda r: r["path"])
        rng.shuffle(items)
        n_val = round(len(items) * VAL_FRACTION)
        for i, r in enumerate(items):
            r["split"] = "val" if i < n_val else "fit"
    for r in rows:
        r.setdefault("split", "test")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "official", "split", "frames", "width", "height"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["path"]))
    counts = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    print("classes:", len({r["label"] for r in rows}), "counts:", dict(counts), "unreadable:", len(bad))
    for p, e in bad:
        print("  BAD", p, e)
    fr = sorted(r["frames"] for r in rows)
    print("frames min/median/max:", fr[0], fr[len(fr) // 2], fr[-1])
    print("sizes:", sorted({(r["width"], r["height"]) for r in rows})[:6])


if __name__ == "__main__":
    main()
