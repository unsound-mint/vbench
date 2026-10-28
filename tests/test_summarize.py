import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from summarize import confirm_table  # noqa: E402


def _row(seed, test_top1):
    return {"model": "m", "method": "lora", "precision": "bf16", "epochs": 10, "seed": seed,
            "val_top1": 70.0, "test_top1": test_top1, "test_macro_f1": 60.0, "NS_hash": 55.0,
            "train_min": 5.0, "peak_mem_mib": 1000.0, "train_kwh": 0.01}


def test_confirm_table_reports_mean_and_sample_std():
    table = confirm_table([_row(17, 70.0), _row(18, 72.0), _row(19, 74.0)])
    assert "72.00 ± 2.00" in table  # sample std (n-1) of 70, 72, 74
    assert "| 3 |" in table


def test_confirm_table_single_seed_has_no_spread():
    assert "±" not in confirm_table([_row(17, 70.0)])
