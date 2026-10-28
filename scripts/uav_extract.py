"""Extract the UAV-Human RGB clips of the Cross-Subject-v1 training subjects.

The archive is a raw split zip (all_rgb.zip.001 ...), read through one
seekable view over the parts so no concatenated copy is written. Only the
training subjects listed in manifests/uavhuman_csv1_train_subjects.txt are
extracted; the test subjects are not used for selection.
"""

import io
import os
import re
import sys
import zipfile
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/uavhuman")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "data/uavhuman/train")
SUBJECTS = Path("manifests/uavhuman_csv1_train_subjects.txt")


class SplitFile(io.RawIOBase):
    """Read-only seekable concatenation of raw file parts."""

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.sizes = [os.path.getsize(p) for p in paths]
        self.total = sum(self.sizes)
        self.pos = 0
        self._fh, self._idx = None, -1

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = (0, self.pos, self.total)[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, b) -> int:
        n, out = len(b), 0
        while out < n and self.pos < self.total:
            start = 0
            for i, size in enumerate(self.sizes):
                if self.pos < start + size:
                    break
                start += size
            if i != self._idx:
                if self._fh:
                    self._fh.close()
                self._fh, self._idx = open(self.paths[i], "rb"), i
            self._fh.seek(self.pos - start)
            chunk = self._fh.read(min(n - out, start + size - self.pos))
            b[out:out + len(chunk)] = chunk
            out += len(chunk)
            self.pos += len(chunk)
        return out


def main() -> None:
    train = {int(s) for s in SUBJECTS.read_text().split()}
    parts = sorted(SRC.glob("all_rgb.zip.0*"))
    z = zipfile.ZipFile(io.BufferedReader(SplitFile(parts), 1 << 22))
    members = [m for m in z.infolist()
               if m.filename.endswith(".avi") and int(re.search(r"/P(\d+)S", m.filename).group(1)) in train]
    print("training-subject clips:", len(members), flush=True)
    DST.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(members):
        out = DST / m.filename
        if out.exists() and out.stat().st_size == m.file_size:
            continue
        z.extract(m, DST)
        if i % 1000 == 0:
            print(i, flush=True)
    print("done", len(members))


if __name__ == "__main__":
    main()
