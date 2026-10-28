"""Training-feasibility sweep: which model/method/precision/checkpointing combos fit."""

import itertools
import json
import sys
from pathlib import Path

from vbench.hardware import snapshot
from vbench.models import MODELS
from vbench.training import probe_step

OUT = Path("runs/training_probe.json")


def main() -> None:
    rows = []
    grid = itertools.product(MODELS, ("full", "lora", "bitfit"), ("fp32", "bf16"), (False, True))
    for model, method, precision, ckpt in grid:
        r = probe_step(model, method, precision, ckpt)
        rows.append(r)
        print(
            f"{model:14s} {method:7s} {precision:5s} ckpt={int(ckpt)}  "
            f"{r['status']:5s} {r.get('peak_mib', '')!s:>8} MiB  {r.get('trainable_m', '')!s:>8} M  "
            f"{r.get('error', '')}",
            flush=True,
        )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"hardware": snapshot(), "rows": rows}, indent=1))
    print("wrote", OUT, file=sys.stderr)


if __name__ == "__main__":
    main()
