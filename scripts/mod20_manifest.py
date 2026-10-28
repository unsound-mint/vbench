"""Build the MOD20 split manifest from the authors' official split files.

Usage: mod20_manifest.py [split 1|2|3] [videos dir]

The official train/test index files number the clips globally: classes in
alphabetical order, clips sorted by file name inside a class. The test set of
split k is the first/middle/last 30 % of each class, which keeps clips of one
source video out of both train and test.

Validation is carved from the official training part the same way: the last
20 % of each class's training clips, in file order, so neighbouring clips of
one source video stay mostly on one side (a random split would leak them).

Needs scipy (`uv run --group data`). Writes manifests/mod20_split.csv.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import scipy.io

from vbench.data import probe_video

SPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ROOT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/mod20/videos")
OUT = Path("manifests/mod20_split.csv")
VAL_FRACTION = 0.2


def main() -> None:
    clips = [(p.parent.name, p) for c in sorted(d.name for d in (ROOT / "dataset").iterdir())
             for p in sorted((ROOT / "dataset" / c).glob("*.mp4"))]
    ids = {k: set(scipy.io.loadmat(ROOT / "split_sets" / f"{k}_idx_split{SPLIT}.mat")[f"{k}_idx"].ravel().tolist())
           for k in ("train", "test")}
    assert len(clips) == len(ids["train"]) + len(ids["test"]) and not ids["train"] & ids["test"], \
        "split files do not cover the clips"

    rows, by_class = [], {}
    for i, (label, p) in enumerate(clips, start=1):  # MATLAB indices are 1-based
        n, w, h = probe_video(p)
        official = "test" if i in ids["test"] else "train"
        r = dict(path=str(p.relative_to(ROOT)), label=label, official=official,
                 split="test" if official == "test" else "fit", frames=n, width=w, height=h)
        rows.append(r)
        if official == "train":
            by_class.setdefault(label, []).append(r)
    for items in by_class.values():
        for r in items[len(items) - round(len(items) * VAL_FRACTION):]:
            r["split"] = "val"
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "official", "split", "frames", "width", "height"])
        w.writeheader()
        w.writerows(rows)
    print("split", SPLIT, "classes:", len(by_class), dict(Counter(r["split"] for r in rows)))
    fr = sorted(r["frames"] for r in rows)
    print("frames min/median/max:", fr[0], fr[len(fr) // 2], fr[-1], "sizes:", sorted({(r["width"], r["height"]) for r in rows}))
    # sanity: test must be a per-class prefix/middle/suffix block, never scattered
    for label in sorted({r["label"] for r in rows})[:3]:
        flags = "".join("T" if r["official"] == "test" else "." for r in rows if r["label"] == label)
        print(f"  {label[:12]:12} {flags}")


if __name__ == "__main__":
    main()
