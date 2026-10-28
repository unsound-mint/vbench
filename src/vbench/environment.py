"""Idle GPU baseline and other GPU users, to flag runs whose measurements may be disturbed."""

import os

import pynvml

from .power import PowerMeter

IDLE_SECONDS = 4.0  # about 8 independent readings on GPUs whose counter refreshes every 0.5 s
MIN_OTHER_MIB = 64  # the desktop compositor holds a few MiB; anything larger is real GPU work
MAX_IDLE_W = 25.0  # the RTX 4070 Laptop idles near 10 W


def _other_processes(handle) -> list[dict]:
    seen = {}
    for query in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
        for p in query(handle):
            if p.pid == os.getpid():
                continue
            name = pynvml.nvmlSystemGetProcessName(p.pid)
            seen[p.pid] = {"name": os.path.basename(name if isinstance(name, str) else name.decode()),
                           "mem_mib": round((p.usedGpuMemory or 0) / 2**20)}
    return list(seen.values())


def gpu_state(device_index: int = 0) -> dict:
    """Temperature and every other process currently using the GPU."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return {"temp_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                "other_procs": _other_processes(handle)}
    finally:
        pynvml.nvmlShutdown()


def idle_baseline(seconds: float = IDLE_SECONDS) -> dict:
    """Mean board power with this process not yet using the GPU, plus the GPU state."""
    with PowerMeter() as meter:
        meter._stop.wait(seconds)
    return {"idle_power_w": round(meter.mean_power_w, 1), **gpu_state()}


def disturbance(start: dict, end: dict) -> list[str]:
    """Reasons the run's energy and timing may not be comparable; empty when clean."""
    reasons = []
    if start["idle_power_w"] > MAX_IDLE_W:
        reasons.append(f"idle power {start['idle_power_w']} W before the run")
    for when, state in (("start", start), ("end", end)):
        for p in state["other_procs"]:
            if p["mem_mib"] >= MIN_OTHER_MIB:
                reasons.append(f"{p['name']} using {p['mem_mib']} MiB at {when}")
    return reasons
