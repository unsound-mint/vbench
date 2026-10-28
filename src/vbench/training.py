"""Single-step training feasibility probe: peak allocated memory or OOM."""

import gc
import time
from contextlib import nullcontext

import torch
from torch import nn

from .adapt import apply_method, enable_checkpointing, trainable_params_m
from .models import MODELS, NUM_CLASSES

PRECISIONS = ("fp32", "bf16", "fp16")  # bf16/fp16 are autocast over fp32 master weights


def _autocast(precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def assert_all_trainable_have_grad(model: nn.Module) -> None:
    """A trainable parameter without a gradient means the method is not wired
    into the forward pass (a silent no-op), which invalidates its measurements."""
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        raise RuntimeError(f"{len(missing)} trainable params got no grad, e.g. {missing[0]}")


def probe_step(
    model_name: str, method: str, precision: str, checkpointing: bool,
    batch: int = 1, frames: int = 16,
) -> dict:
    spec = MODELS[model_name]
    result = {
        "model": model_name, "method": method, "precision": precision,
        "checkpointing": checkpointing, "batch": batch, "frames": frames,
    }
    model = opt = x = y = None
    try:
        model = spec.build(False)
        if checkpointing:
            enable_checkpointing(model, spec.block_class)
        model = apply_method(model, spec, method).cuda().train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4)
        scaler = torch.amp.GradScaler(enabled=precision == "fp16")
        x = torch.randn(batch, frames, 3, 224, 224, device="cuda")
        y = torch.randint(0, NUM_CLASSES, (batch,), device="cuda")

        def step():
            opt.zero_grad(set_to_none=True)
            with _autocast(precision):
                loss = nn.functional.cross_entropy(model(x).float(), y)
            scaler.scale(loss).backward()
            assert_all_trainable_have_grad(model)
            scaler.step(opt)
            scaler.update()

        step()  # first step allocates optimizer state; measure the steady state
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        result.update(
            status="ok",
            peak_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
            step_s=round(time.perf_counter() - t0, 3),
            trainable_m=round(trainable_params_m(model), 3),
        )
    except torch.OutOfMemoryError:
        result.update(status="oom")
    except Exception as e:  # unsupported combination: record why, keep sweeping
        result.update(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
    finally:
        del model, opt, x, y
        gc.collect()
        torch.cuda.empty_cache()
    return result
