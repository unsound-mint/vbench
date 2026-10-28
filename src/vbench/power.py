"""GPU board-power sampling and energy integration via NVML."""

import threading
import time
from contextlib import contextmanager

import pynvml

# The methodology samples at 0.02 s, but the NVML counter itself refreshes far
# more slowly on some GPUs (about 0.5 s on the RTX 4070 Laptop), so the extra
# samples repeat the same reading. Energy over long windows is unaffected;
# windows shorter than a few seconds are quantized to the counter's refresh.
SAMPLE_INTERVAL_S = 0.02


def trapezoid_energy_j(times_s: list[float], power_w: list[float]) -> float:
    """Energy in joules from consecutive (time, power) samples."""
    return sum(
        (times_s[i] - times_s[i - 1]) * (power_w[i] + power_w[i - 1]) / 2
        for i in range(1, len(times_s))
    )


class PowerMeter:
    """Samples board power in a background thread while the context is open."""

    def __init__(self, device_index: int = 0, interval_s: float = SAMPLE_INTERVAL_S):
        self.device_index = device_index
        self.interval_s = interval_s
        self.times_s: list[float] = []
        self.power_w: list[float] = []
        self._handle = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self, handle) -> None:
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            self.times_s.append(time.perf_counter())
            self.power_w.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)
            next_tick += self.interval_s
            self._stop.wait(max(0.0, next_tick - time.perf_counter()))

    def __enter__(self) -> "PowerMeter":
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._thread = threading.Thread(target=self._run, args=(self._handle,), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        # Closing sample, so the window ends at the context exit rather than up
        # to one interval earlier.
        self.times_s.append(time.perf_counter())
        self.power_w.append(pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000)
        pynvml.nvmlShutdown()

    @property
    def energy_j(self) -> float:
        return trapezoid_energy_j(self.times_s, self.power_w)

    @property
    def mean_power_w(self) -> float:
        return sum(self.power_w) / len(self.power_w)

    @property
    def duration_s(self) -> float:
        return self.times_s[-1] - self.times_s[0]


@contextmanager
def measure_energy(device_index: int = 0):
    with PowerMeter(device_index) as meter:
        yield meter
