"""Run planned experiments, one subprocess per run (clean GPU state), resumable.

Plans:
  smoke     every supported combination on a few clips: support + timing check
  all       per model: staged search (method at BF16, then precision for the winner)
            plus the Full FT / LoRA x FP32 / BF16-AMP ablation
  uav       reduced UAV-Human plan: full search + ablation for VideoMAE-B, then
            LoRA BF16 and Full FT BF16 for the other models
  confirm   repeat each model's best configuration, best PEFT configuration and
            Full FT BF16 over 3 seeds with test evaluation and checkpoints
            (configurations are picked from the dataset's existing 3-epoch runs)

Usage: run_experiments.py <plan> [epochs] [dataset]
"""

import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

from vbench.models import MODELS
from vbench.run import RunConfig, check_supported

EPOCHS = 3
DATASET = "era"  # set from argv


def run_id(cfg: RunConfig) -> str:
    return cfg.run_id


def execute(cfg: RunConfig) -> dict:
    out = cfg.out_dir / f"{cfg.run_id}.json"
    if not out.exists():
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "vbench.run", "--model", cfg.model, "--method", cfg.method,
               "--precision", cfg.precision, "--dataset", cfg.dataset, "--epochs", str(cfg.epochs),
               "--limit-clips", str(cfg.limit_clips), "--seed", str(cfg.seed)]
        cmd += ["--test"] * cfg.test + ["--save-checkpoint"] * cfg.save_checkpoint
        t0 = time.time()
        with open(cfg.out_dir / f"{cfg.run_id}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"{cfg.run_id}: {time.time() - t0:.0f}s", flush=True)
    r = json.loads(out.read_text()) if out.exists() else {"status": "crashed"}
    print(f"  {cfg.run_id} -> {r['status']}"
          + (f" top1={r['validation']['top1']:.1f} NS#={r['netscore']['NS_hash']:.2f}" if r["status"] == "ok" else
             f" {r.get('error', '')[:100]}"), flush=True)
    return r


def valid_combos():
    for method, precision in itertools.product(("full", "lora", "qlora", "adalora", "bitfit"),
                                               ("fp32", "fp16", "bf16", "int8")):
        cfg = RunConfig("videomae-b", method, precision)
        try:
            check_supported(cfg)
        except NotImplementedError:
            continue
        yield method, precision


def smoke(limit: int = 32) -> None:
    for model in MODELS:
        for method, precision in valid_combos():
            execute(RunConfig(model, method, precision, epochs=1, limit_clips=limit, dataset=DATASET))


MODEL_ORDER = ("videomae-b", "hiera-b", "videoswin-b", "timesformer-b")  # cheapest first
PEFT = ("lora", "qlora", "adalora", "bitfit")


def supported(model: str, method: str, precision: str, epochs: int) -> RunConfig | None:
    cfg = RunConfig(model, method, precision, epochs, dataset=DATASET)
    try:
        check_supported(cfg)
    except NotImplementedError:
        return None
    return cfg


def search(model: str, epochs: int = EPOCHS) -> None:
    """Stage 1: PEFT method at BF16. Stage 2: precision for the best method (by NS#)."""
    stage1 = {}
    for method in PEFT:
        precision = "bf16"
        r = execute(supported(model, method, precision, epochs))
        if r["status"] == "ok":
            stage1[method] = r["netscore"]["NS_hash"]
    if not stage1:
        return
    winner = max(stage1, key=stage1.get)
    print(f"{model}: stage-1 winner {winner} {stage1}", flush=True)
    for precision in ("fp32", "fp16", "int8"):
        cfg = supported(model, winner, precision, epochs)
        if cfg:
            execute(cfg)


def ablation(model: str, epochs: int = EPOCHS) -> None:
    """Full FT / LoRA x FP32 / BF16 AMP (shared runs are reused)."""
    for method, precision in (("full", "fp32"), ("lora", "fp32"), ("full", "bf16"), ("lora", "bf16")):
        execute(supported(model, method, precision, epochs))


def everything(epochs: int = EPOCHS) -> None:
    for model in MODEL_ORDER:
        search(model, epochs)
        ablation(model, epochs)


SEEDS = (17, 18, 19)


def confirm(epochs: int, search_epochs: int = EPOCHS) -> None:
    """Seeds + test evaluation for the configurations the search singled out."""
    out_dir = RunConfig("videomae-b", "lora", "bf16", dataset=DATASET).out_dir
    for model in MODEL_ORDER:
        found = []
        for p in out_dir.glob(f"{model}__*__e{search_epochs}.json"):  # excludes lr/seed/test/smoke variants
            r = json.loads(p.read_text())
            if r["status"] == "ok":
                c = r["config"]
                found.append((r["netscore"]["NS_hash"], c["method"], c["precision"]))
        if not found:
            continue
        picks = {max(found)[1:], max(f for f in found if f[1] != "full")[1:], ("full", "bf16")}
        for seed in SEEDS:
            for method, precision in sorted(picks):
                cfg = supported(model, method, precision, epochs)
                cfg.seed, cfg.test, cfg.save_checkpoint = seed, True, seed == SEEDS[0]
                execute(cfg)


def reduced(epochs: int = EPOCHS) -> None:
    """Cheaper plan for datasets where a full grid is too slow."""
    search("videomae-b", epochs)
    ablation("videomae-b", epochs)
    for model in MODEL_ORDER[1:]:
        for method in ("lora", "full"):
            execute(supported(model, method, "bf16", epochs))


if __name__ == "__main__":
    plan = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS
    DATASET = sys.argv[3] if len(sys.argv) > 3 else "era"
    {"smoke": lambda: smoke(), "all": lambda: everything(epochs), "uav": lambda: reduced(epochs), "confirm": lambda: confirm(epochs)}[plan]()
