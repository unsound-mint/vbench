"""Prepared model-only inference measurement.

Protocol: a zero-valued tensor already resident on the GPU, 5 warmup passes,
latency as the mean of 30 synchronized passes, and energy per clip as the mean
board power over a sustained loop multiplied by the mean latency.
"""

import time
from dataclasses import asdict, dataclass

import torch

from .power import PowerMeter

WARMUP_PASSES = 5
TIMED_PASSES = 30
SUSTAINED_S = 12.0


@dataclass
class InferenceResult:
    latency_ms: float
    energy_j_per_clip: float
    mean_power_w: float
    peak_memory_mib: float

    def to_dict(self) -> dict:
        return asdict(self)


@torch.inference_mode()
def measure_inference(
    model: torch.nn.Module,
    input_shape: tuple[int, ...] = (1, 16, 3, 224, 224),
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    sustained_s: float = SUSTAINED_S,
) -> InferenceResult:
    model.eval()
    x = torch.zeros(input_shape, dtype=dtype, device=device)
    for _ in range(WARMUP_PASSES):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(TIMED_PASSES):
        model(x)
    torch.cuda.synchronize()
    latency_s = (time.perf_counter() - start) / TIMED_PASSES

    with PowerMeter() as meter:
        end = time.perf_counter() + sustained_s
        while time.perf_counter() < end:
            model(x)
        torch.cuda.synchronize()

    return InferenceResult(
        latency_ms=latency_s * 1000,
        energy_j_per_clip=meter.mean_power_w * latency_s,
        mean_power_w=meter.mean_power_w,
        peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
    )
