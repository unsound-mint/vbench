"""One fine-tuning run on ERA with full resource measurement.

Phases are timed separately so energy can be attributed: training (all
optimizer steps), validation passes, and prepared model-only inference.
Training memory is the peak allocated bytes during training steps only.
"""

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .adapt import apply_method, trainable_params_m
from .data import ClipCache, class_names, preprocess, read_manifest
from .hardware import snapshot
from .inference import measure_inference
from .models import MODELS
from .netscore import ns_combined, ns_e, ns_m
from .environment import disturbance, gpu_state, idle_baseline
from .power import PowerMeter
from .quant import quantize_linears

# Per dataset: frame cache, cache row index, and where run JSONs land.
# test_* is the reserved test split, cached separately and used only by --test runs.
DATASETS = {
    "era": dict(manifest=Path("manifests/era_split.csv"), videos=Path("data/era/Videos"),
                cache=Path("data/era/cache_trainval.npy"), index=Path("data/era/cache_index.csv"),
                test_cache=Path("data/era/cache_test.npy"), test_index=Path("data/era/cache_test_index.csv"),
                out_dir=Path("runs/it2")),
    "mod20": dict(manifest=Path("manifests/mod20_split.csv"), videos=Path("data/mod20/videos"),
                  cache=Path("data/mod20/cache_train.npy"), index=Path("data/mod20/cache_index.csv"),
                  test_cache=Path("data/mod20/cache_test.npy"), test_index=Path("data/mod20/cache_test_index.csv"),
                  out_dir=Path("runs/it2_mod20")),
    # consistency check only: the test subjects are not extracted, so no test split
    "uavhuman": dict(manifest=Path("manifests/uavhuman_csv1_split.csv"), videos=Path("data/uavhuman/train"),
                     cache=Path("data/uavhuman/cache_train.npy"), index=Path("data/uavhuman/cache_index.csv"),
                     out_dir=Path("runs/it2_uavhuman")),
}
CHECKPOINT_DIR = Path("checkpoints")
SEED = 17
EFFECTIVE_BATCH = 64
METHODS = ("full", "lora", "qlora", "adalora", "bitfit")
PRECISIONS = ("fp32", "fp16", "bf16", "int8")


@dataclass
class RunConfig:
    model: str
    method: str
    precision: str
    epochs: int = 3
    limit_clips: int = 0  # smoke tests only: use the first N fit clips
    seed: int = SEED
    dataset: str = "era"
    lr_override: float = 0.0  # learning-rate sensitivity runs only; 0 keeps the protocol default
    test: bool = False  # also evaluate the final model once on the reserved test split
    save_checkpoint: bool = False  # keep the trained (trainable) weights under checkpoints/

    @property
    def out_dir(self) -> Path:
        return DATASETS[self.dataset]["out_dir"]

    @property
    def run_id(self) -> str:
        tag = f"{self.model}__{self.method}__{self.precision}__e{self.epochs}"
        tag += f"__lr{self.lr_override:g}" if self.lr_override else ""
        tag += f"__s{self.seed}" if self.seed != SEED else ""
        tag += "__test" if self.test else ""  # confirmation runs never collide with search runs
        return tag + (f"__n{self.limit_clips}" if self.limit_clips else "")


def check_supported(cfg: RunConfig) -> None:
    if cfg.method not in METHODS or cfg.precision not in PRECISIONS:
        raise ValueError("unknown method or precision")
    if cfg.method == "qlora" and cfg.precision not in ("bf16", "fp16"):
        raise NotImplementedError("qlora needs a bf16/fp16 compute dtype")
    if cfg.precision == "int8" and cfg.method in ("full", "qlora"):
        raise NotImplementedError("int8 base weights are frozen; needs an adapter method")


def hyperparams(method: str) -> tuple[float, float, float]:
    """(learning rate, weight decay, warmup fraction), after the Iteration 1 protocol."""
    return (2.5e-4, 0.05, 0.05) if method == "full" else (1e-3, 0.0, 0.10)


def _autocast(precision: str):
    if precision == "fp32":
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def _energy_between(meter: PowerMeter, t0: float, t1: float) -> float:
    # The sampler thread appends time then power, so a read can catch one more time than power.
    n = min(len(meter.times_s), len(meter.power_w))
    t, p = np.array(meter.times_s[:n]), np.array(meter.power_w[:n])
    grid = np.concatenate([[t0], t[(t > t0) & (t < t1)], [t1]])
    return float(np.trapezoid(np.interp(grid, t, p), grid))


def _metrics(pred: np.ndarray, true: np.ndarray, n_classes: int) -> dict:
    recalls, f1s = [], []
    for c in range(n_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        if (tp + fn) == 0:
            continue
        recalls.append(tp / (tp + fn))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"top1": 100 * float((pred == true).mean()),
            "balanced_acc": 100 * float(np.mean(recalls)), "macro_f1": 100 * float(np.mean(f1s))}


@torch.no_grad()
def evaluate(model, cache: ClipCache, spec, precision: str, n_classes: int, bs: int = 8) -> dict:
    model.eval()
    preds = []
    for i in range(0, len(cache), bs):
        x, _ = cache.batch(np.arange(i, min(i + bs, len(cache))))
        x = preprocess(x, spec.mean, spec.std, train=False)
        with _autocast(precision):
            preds.append(model(x).argmax(-1).cpu())
    torch.cuda.synchronize()
    return _metrics(torch.cat(preds).numpy(), cache.labels.numpy(), n_classes)


def prepare_for_deployment(model: nn.Module, cfg: RunConfig) -> tuple[nn.Module, torch.dtype]:
    """Merge adapters where the base is full precision, and cast to the run's dtype."""
    quantized = cfg.method == "qlora" or cfg.precision == "int8"
    if cfg.method in ("lora", "adalora") and not quantized:
        model = model.merge_and_unload()
    dtype = {"fp32": torch.float32, "fp16": torch.float16}.get(cfg.precision, torch.bfloat16)
    return model.to(dtype), dtype


def run(cfg: RunConfig) -> dict:
    check_supported(cfg)
    env_start = idle_baseline()  # before this process touches the GPU
    spec = MODELS[cfg.model]
    ds = DATASETS[cfg.dataset]
    rows = read_manifest(ds["index"])
    classes = class_names(rows)
    fit_rows = [r for r in rows if r["split"] == "fit"]
    if cfg.limit_clips:
        fit_rows = fit_rows[:: max(1, len(fit_rows) // cfg.limit_clips)][: cfg.limit_clips]
    val_rows = [r for r in rows if r["split"] == "val"]
    fit, val = ClipCache(ds["cache"], fit_rows, classes), ClipCache(ds["cache"], val_rows, classes)

    steps_per_epoch = math.ceil(len(fit) / EFFECTIVE_BATCH)
    total_steps = steps_per_epoch * cfg.epochs
    lr, wd, warmup = hyperparams(cfg.method)
    lr = cfg.lr_override or lr

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = spec.build(True, len(classes))
    compute_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16
    if cfg.method == "qlora":
        quantize_linears(model, spec, "nf4", compute_dtype)
    elif cfg.precision == "int8":
        quantize_linears(model, spec, "int8")
    model = apply_method(model, spec, cfg.method, total_steps).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:  # bitsandbytes casts biases of quantized layers to half; train in fp32
        if p.dtype != torch.float32:
            p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    warm = max(1, int(warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    scaler = torch.amp.GradScaler(enabled=cfg.precision == "fp16")

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {cfg.run_id} {msg}", flush=True)

    train_s = train_j = val_j = 0.0
    train_peak = 0
    epochs_log, final_val_j, final_val_s = [], 0.0, 0.0
    wall0 = time.perf_counter()
    torch.cuda.synchronize()
    with PowerMeter() as meter:
        step = 0
        for epoch in range(cfg.epochs):
            model.train()
            perm = np.random.RandomState(cfg.seed + epoch).permutation(len(fit))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            loss_sum = 0.0
            for s in range(steps_per_epoch):
                group = perm[s * EFFECTIVE_BATCH:(s + 1) * EFFECTIVE_BATCH]
                for j in group:  # micro-batch 1, gradients accumulated to the effective batch
                    x, y = fit.batch(np.array([j]))
                    x = preprocess(x, spec.mean, spec.std, train=True, generator=gen)
                    with _autocast(cfg.precision):
                        loss = nn.functional.cross_entropy(model(x).float(), y) / len(group)
                    scaler.scale(loss).backward()
                    loss_sum += loss.item() * len(group)
                for p in params:  # bitsandbytes can return half-precision grads for fp32 params
                    if p.grad is not None and p.grad.dtype != p.dtype:
                        p.grad = p.grad.to(p.dtype)
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if cfg.method == "adalora":
                    model.base_model.update_and_allocate(step)
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            train_peak = max(train_peak, torch.cuda.max_memory_allocated())
            train_s += t1 - t0
            train_j += _energy_between(meter, t0, t1)

            v0 = time.perf_counter()
            m = evaluate(model, val, spec, cfg.precision, len(classes))
            v1 = time.perf_counter()
            final_val_s, final_val_j = v1 - v0, _energy_between(meter, v0, v1)
            val_j += final_val_j
            epochs_log.append({"epoch": epoch + 1, "train_loss": loss_sum / len(fit),
                               "train_s": round(t1 - t0, 1), **{k: round(v, 2) for k, v in m.items()}})
            log(f"epoch {epoch + 1}/{cfg.epochs} loss {loss_sum / len(fit):.3f} "
                f"val top1 {m['top1']:.2f} train {t1 - t0:.0f}s")
    total_s = time.perf_counter() - wall0

    trainable = trainable_params_m(model)
    test_result = {}
    if cfg.test:  # once, after the last epoch: no selection ever looks at test data
        test_rows = read_manifest(ds["test_index"])
        test = ClipCache(ds["test_cache"], test_rows, classes)
        test_result = {"test": evaluate(model, test, spec, cfg.precision, len(classes)),
                       "n_test": len(test)}
    if cfg.save_checkpoint:
        CHECKPOINT_DIR.joinpath(cfg.dataset).mkdir(parents=True, exist_ok=True)
        keep = {n for n, p in model.named_parameters() if p.requires_grad}
        torch.save({n: t.cpu() for n, t in model.state_dict().items() if n in keep},
                   CHECKPOINT_DIR / cfg.dataset / f"{cfg.run_id}.pt")
    model, dtype = prepare_for_deployment(model, cfg)
    inf = measure_inference(model, dtype=dtype)

    env_end = gpu_state()
    acc = epochs_log[-1]["top1"]
    peak_mib = train_peak / 2**20
    return {
        "config": asdict(cfg) | {"run_id": cfg.run_id, "micro_batch": 1,
                                 "effective_batch": EFFECTIVE_BATCH, "lr": lr, "weight_decay": wd,
                                 "n_fit": len(fit), "n_val": len(val)},
        "status": "ok",
        "validation": epochs_log[-1] | {"top1": acc},
        **{k: ({m: round(v, 2) for m, v in x.items()} if isinstance(x, dict) else x)
           for k, x in test_result.items()},
        "trainable_m": round(trainable, 3),
        "adaptation": {
            "train_time_min": round(train_s / 60, 2),
            "run_time_min": round(total_s / 60, 2),
            "peak_train_mem_mib": round(peak_mib, 1),
            "train_energy_kwh": round(train_j / 3.6e6, 5),
            "val_energy_total_kwh": round(val_j / 3.6e6, 5),
        },
        "deployment": {
            "latency_ms": round(inf.latency_ms, 3),
            "energy_j_per_clip": round(inf.energy_j_per_clip, 4),
            "mean_power_w": round(inf.mean_power_w, 1),
            "val_pass_energy_j": round(final_val_j, 1),
            "val_pass_s": round(final_val_s, 2),
        },
        "netscore": {
            "NS_E": round(ns_e(acc, final_val_j), 3),
            "NS_M": round(ns_m(acc, peak_mib), 3),
            "NS_hash": round(ns_combined(acc, final_val_j, peak_mib), 3),
        },
        "environment": {
            "idle_power_w": env_start["idle_power_w"],
            "temp_start_c": env_start["temp_c"], "temp_end_c": env_end["temp_c"],
            "other_procs_start": env_start["other_procs"], "other_procs_end": env_end["other_procs"],
            "disturbed": disturbance(env_start, env_end),
        },
        "epochs": epochs_log,
        "hardware": snapshot(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--precision", required=True, choices=PRECISIONS)
    ap.add_argument("--dataset", choices=list(DATASETS), default="era")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--save-checkpoint", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = RunConfig(a.model, a.method, a.precision, a.epochs, a.limit_clips, seed=a.seed, dataset=a.dataset,
                    lr_override=a.lr, test=a.test, save_checkpoint=a.save_checkpoint)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir / f"{cfg.run_id}.json"
    if out.exists() and not a.force:
        print("exists, skipping", out)
        return
    try:
        result = run(cfg)
    except NotImplementedError as e:
        result = {"config": asdict(cfg) | {"run_id": cfg.run_id}, "status": "unsupported", "error": str(e)}
    except torch.OutOfMemoryError:
        result = {"config": asdict(cfg) | {"run_id": cfg.run_id}, "status": "oom"}
    except Exception as e:
        traceback.print_exc()
        result = {"config": asdict(cfg) | {"run_id": cfg.run_id}, "status": "error",
                  "error": f"{type(e).__name__}: {str(e)[:300]}"}
    out.write_text(json.dumps(result, indent=1))
    print("wrote", out, result["status"])
    sys.exit(0 if result["status"] in ("ok", "unsupported") else 1)


if __name__ == "__main__":
    main()
