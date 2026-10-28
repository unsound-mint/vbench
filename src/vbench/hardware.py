"""Hardware/software snapshot for result files.

Result files are published, so this records only what is needed to compare
runs and deliberately omits GPU UUID/serial, hostname and username.
"""

import platform

import pynvml
import torch


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def snapshot(device_index: int = 0) -> dict:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(handle)
        vram_mib = pynvml.nvmlDeviceGetMemoryInfo(handle).total // 2**20
        driver = pynvml.nvmlSystemGetDriverVersion()
    finally:
        pynvml.nvmlShutdown()
    return {
        "gpu": name,
        "gpu_vram_mib": vram_mib,
        "driver": driver,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "cpu": _cpu_model(),
        "os": platform.system() + " " + platform.release(),
        "python": platform.python_version(),
    }
