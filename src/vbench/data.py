"""ERA clip cache and GPU-side batching.

Clips are decoded once into a uint8 memmap (16 uniformly spaced frames,
resized to 224x224). Training then needs no CPU decoding workers, which keeps
host RAM use small and removes decode time from the measured training run.
The cache is derived from a dataset with redistribution limits: it stays under
`data/` (gitignored) and must never be published.
"""

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

FRAMES, SIZE = 16, 224


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def probe_video(path: str | Path) -> tuple[int, int, int]:
    """(frame count, width, height) of a video file."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        n = s.frames or sum(1 for _ in c.decode(video=0))
        return n, s.codec_context.width, s.codec_context.height


def class_names(rows: list[dict]) -> list[str]:
    return sorted({r["label"] for r in rows})


def _decode(args: tuple[str, int]) -> np.ndarray:
    path, n = args
    want = set(np.linspace(0, n - 1, FRAMES).round().astype(int).tolist())
    frames = []
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                frames.append(fr.to_ndarray(width=SIZE, height=SIZE, format="rgb24"))
    while len(frames) < FRAMES:  # clip shorter than its header claims: repeat last frame
        frames.append(frames[-1])
    return np.stack(frames[:FRAMES])


def build_cache(rows: list[dict], video_root: Path, out: Path, workers: int = 6) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8,
                                   shape=(len(rows), FRAMES, SIZE, SIZE, 3))
    jobs = [(str(video_root / r["path"]), int(r["frames"])) for r in rows]
    with ProcessPoolExecutor(workers) as ex:
        for i, arr in enumerate(ex.map(_decode, jobs, chunksize=8)):
            mm[i] = arr
    mm.flush()


class ClipCache:
    """Memmapped clips + integer labels for a subset of manifest rows."""

    def __init__(self, cache_path: Path, rows: list[dict], classes: list[str]):
        self.clips = np.load(cache_path, mmap_mode="r")
        self.rows = rows
        index = {c: i for i, c in enumerate(classes)}
        self.labels = torch.tensor([index[r["label"]] for r in rows])
        self.cache_idx = np.array([int(r["cache_idx"]) for r in rows])

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, idx: np.ndarray, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        order = np.argsort(self.cache_idx[idx])  # sorted reads are friendlier to the memmap
        x = np.empty((len(idx), FRAMES, SIZE, SIZE, 3), np.uint8)
        for j in order:
            x[j] = self.clips[self.cache_idx[idx[j]]]
        x = torch.from_numpy(x).to(device, non_blocking=True)
        return x, self.labels[idx].to(device)


def preprocess(x: torch.Tensor, mean: tuple, std: tuple, train: bool,
               scale: tuple[float, float] = (0.7, 1.0), generator=None) -> torch.Tensor:
    """uint8 (B,T,H,W,C) on GPU -> float (B,T,C,H,W), normalized.

    Train: one random square crop of area scale*224^2 per clip, resized back to
    224, plus horizontal flip with p=0.5 (same crop/flip for all frames of a clip).
    """
    x = x.permute(0, 1, 4, 2, 3).float().div_(255)
    b, t, c, h, w = x.shape
    if train:
        out = torch.empty_like(x)
        for i in range(b):
            s = float(torch.empty(1).uniform_(*scale, generator=generator))
            side = max(8, int(round(h * s**0.5)))
            top = int(torch.randint(0, h - side + 1, (1,), generator=generator))
            left = int(torch.randint(0, w - side + 1, (1,), generator=generator))
            crop = x[i, :, :, top:top + side, left:left + side]
            crop = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)
            if torch.rand(1, generator=generator) < 0.5:
                crop = crop.flip(-1)
            out[i] = crop
        x = out
    m = torch.tensor(mean, device=x.device).view(1, 1, 3, 1, 1)
    s_ = torch.tensor(std, device=x.device).view(1, 1, 3, 1, 1)
    return (x - m) / s_
