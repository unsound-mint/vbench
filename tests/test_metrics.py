import math

import pytest

from vbench.netscore import ns_combined, ns_e, ns_m
from vbench.power import PowerMeter, trapezoid_energy_j
from vbench.run import _energy_between


def test_trapezoid_constant_power():
    assert trapezoid_energy_j([0.0, 1.0, 2.0], [10.0, 10.0, 10.0]) == pytest.approx(20.0)


def test_trapezoid_linear_ramp():
    # 0 W -> 10 W over 2 s is 10 J
    assert trapezoid_energy_j([0.0, 2.0], [0.0, 10.0]) == pytest.approx(10.0)


def test_netscore_matches_published_row():
    # VideoMAE-B Full FT on UAV-Human (63.04 %, 1829.9 MiB) reports NS_M = 63.83
    assert ns_m(63.04, 1829.9) == pytest.approx(63.83, abs=0.01)


def test_combined_identity():
    # NS# = NS_E + NS_M - 20*log10(a^2): the accuracy term is counted once
    a, e, m = 70.0, 3.0, 2000.0
    assert ns_combined(a, e, m) == pytest.approx(
        ns_e(a, e) + ns_m(a, m) - 20 * math.log10(a**2)
    )


def test_energy_between_tolerates_sampler_mid_append():
    # The sampler thread appends the timestamp before the power reading, so a
    # reader can see one more time than power.
    meter = PowerMeter()
    meter.times_s = [0.0, 1.0, 2.0, 3.0]
    meter.power_w = [10.0, 10.0, 10.0]
    assert _energy_between(meter, 0.0, 2.0) == pytest.approx(20.0)


def test_run_ids_do_not_collide():
    # search runs, seed repeats, test confirmations and lr probes must write separate files
    from vbench.run import RunConfig

    base = RunConfig("videomae-b", "lora", "bf16", epochs=3)
    variants = [base, RunConfig(**{**base.__dict__, "seed": 18}),
                RunConfig(**{**base.__dict__, "test": True}),
                RunConfig(**{**base.__dict__, "lr_override": 1e-4}),
                RunConfig(**{**base.__dict__, "limit_clips": 32})]
    assert len({v.run_id for v in variants}) == len(variants)
    assert base.run_id == "videomae-b__lora__bf16__e3"  # stable: existing results are keyed by it


def test_disturbance_flags_real_gpu_users_only():
    from vbench.environment import disturbance

    quiet = {"idle_power_w": 10.0, "temp_c": 45, "other_procs": [{"name": "gnome-shell", "mem_mib": 3}]}
    assert disturbance(quiet, quiet) == []  # the desktop compositor is always there
    busy = {**quiet, "other_procs": [{"name": "firefox", "mem_mib": 900}]}
    assert any("firefox" in r for r in disturbance(quiet, busy))
    assert any("idle power" in r for r in disturbance({**quiet, "idle_power_w": 60.0}, quiet))
