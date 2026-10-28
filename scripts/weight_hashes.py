"""Print the sha256 of every pretrained checkpoint the models load, for the methodology doc."""

import hashlib

from vbench.models import MODELS


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


for name, spec in MODELS.items():
    for path in spec.weight_files():
        print(f"{name}\t{path.rsplit('/', 1)[-1]}\t{sha256(path)}")
