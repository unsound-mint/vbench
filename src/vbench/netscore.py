"""NetScore variants (Wong, 2019, arXiv:1806.05512) with 1/8-weight efficiency terms.

Higher is better. `accuracy` is top-1 in percentage points, `energy_j` is the
energy of one complete validation pass in joules, `memory_mib` is peak
allocated training memory in MiB.
"""

import math

_EXPONENT = 1 / 8


def ns_e(accuracy: float, energy_j: float) -> float:
    return 20 * math.log10(accuracy**2 / energy_j**_EXPONENT)


def ns_m(accuracy: float, memory_mib: float) -> float:
    return 20 * math.log10(accuracy**2 / memory_mib**_EXPONENT)


def ns_combined(accuracy: float, energy_j: float, memory_mib: float) -> float:
    return 20 * math.log10(accuracy**2 / (energy_j * memory_mib) ** _EXPONENT)
