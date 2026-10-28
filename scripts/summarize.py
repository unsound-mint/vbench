"""Aggregate a dataset's run JSONs into results/<prefix>_summary.md and <prefix>_runs.csv.

Usage: summarize.py [epochs] [dataset]

Confirmation runs (several seeds, test split) are aggregated as mean +- std in
their own table and never compete with the protocol runs.

The aggregated tables contain only measurements and the sanitized hardware
record, so they are safe to publish.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from vbench.run import DATASETS

OUT = Path("results")
PREFIX = {"era": "it2", "mod20": "mod20", "uavhuman": "uavhuman"}
COLS = ["lr", "model", "method", "precision", "epochs", "top1", "macro_f1", "trainable_m", "train_min",
        "peak_mem_mib", "train_kwh", "latency_ms", "infer_j_per_clip", "val_j", "NS_E", "NS_M", "NS_hash", "idle_w", "disturbed"]


def load(runs: Path, epochs: int | None) -> tuple[list[dict], list[dict], dict]:
    rows, confirm, hardware = [], [], {}
    for p in sorted(runs.glob("*.json")):
        r = json.loads(p.read_text())
        c = r["config"]
        if r["status"] != "ok" or c.get("limit_clips"):
            continue
        hardware = r["hardware"]
        env = r.get("environment", {})  # absent in runs made before idle logging
        if c.get("test"):
            confirm.append({
                "model": c["model"], "method": c["method"], "precision": c["precision"], "epochs": c["epochs"],
                "seed": c["seed"], "val_top1": r["validation"]["top1"], "test_top1": r["test"]["top1"],
                "test_macro_f1": r["test"]["macro_f1"], "NS_hash": r["netscore"]["NS_hash"],
                "train_min": r["adaptation"]["train_time_min"],
                "peak_mem_mib": r["adaptation"]["peak_train_mem_mib"],
                "train_kwh": r["adaptation"]["train_energy_kwh"],
                "run_id": c["run_id"], "why": "; ".join(env.get("disturbed", [])),
            })
            continue
        if epochs and c["epochs"] != epochs:
            continue
        rows.append({
            "lr": c.get("lr_override") or "default",
            "model": c["model"], "method": c["method"], "precision": c["precision"], "epochs": c["epochs"],
            "top1": r["validation"]["top1"], "macro_f1": r["validation"]["macro_f1"],
            "trainable_m": r["trainable_m"], "train_min": r["adaptation"]["train_time_min"],
            "peak_mem_mib": r["adaptation"]["peak_train_mem_mib"], "train_kwh": r["adaptation"]["train_energy_kwh"],
            "latency_ms": r["deployment"]["latency_ms"], "infer_j_per_clip": r["deployment"]["energy_j_per_clip"],
            "val_j": r["deployment"]["val_pass_energy_j"], "NS_E": r["netscore"]["NS_E"],
            "NS_M": r["netscore"]["NS_M"], "NS_hash": r["netscore"]["NS_hash"],
            "idle_w": env.get("idle_power_w", "n/a"),
            "disturbed": "n/a" if not env else ("yes" if env["disturbed"] else "no"),
            "run_id": c["run_id"], "why": "; ".join(env.get("disturbed", [])),
        })
    return rows, confirm, hardware


def table(rows: list[dict]) -> str:
    head = "| " + " | ".join(COLS) + " |\n|" + "---|" * len(COLS) + "\n"
    return head + "\n".join("| " + " | ".join(str(r[c]) for c in COLS) + " |" for r in rows)


CONFIRM_METRICS = ["val_top1", "test_top1", "test_macro_f1", "NS_hash", "train_min", "peak_mem_mib", "train_kwh"]


def confirm_table(confirm: list[dict]) -> str:
    groups = defaultdict(list)
    for r in confirm:
        groups[(r["model"], r["method"], r["precision"], r["epochs"])].append(r)
    cols = ["model", "method", "precision", "epochs", "n_seeds"] + CONFIRM_METRICS
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for key in sorted(groups):
        rs = groups[key]
        cells = [f"{statistics.mean(r[m] for r in rs):.2f} ± {statistics.stdev(r[m] for r in rs):.2f}"
                 if len(rs) > 1 else f"{rs[0][m]:.2f}" for m in CONFIRM_METRICS]
        lines.append("| " + " | ".join([*map(str, key), str(len(rs)), *cells]) + " |")
    return "\n".join(lines)


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    dataset = sys.argv[2] if len(sys.argv) > 2 else "era"
    rows, confirm, hw = load(DATASETS[dataset]["out_dir"], epochs)
    prefix = PREFIX[dataset]
    OUT.mkdir(exist_ok=True)
    with (OUT / f"{prefix}_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = [f"# {dataset} fine-tuning measurements", "", "Hardware: " + json.dumps(hw), ""]
    # Learning-rate sensitivity runs are reported but never compete for "best configuration".
    sens = [r for r in rows if r["lr"] != "default"]
    rows = [r for r in rows if r["lr"] == "default"]
    md += ["## All runs", "", table(sorted(rows, key=lambda r: (r["model"], r["method"], r["precision"]))), ""]
    if sens:
        md += ["## Learning-rate sensitivity", "", table(sorted(sens, key=lambda r: (r["model"], r["method"], r["precision"], r["lr"]))), ""]
    disturbed = [(r["run_id"], r["why"]) for r in (*rows, *confirm) if r.get("why")]
    if disturbed:
        md += ["## Runs flagged as possibly disturbed", "",
               "Another process was using the GPU or its idle power was high; "
               "energy and timing of these runs may not be comparable.", "",
               *[f"- `{run_id}`: {why}" for run_id, why in disturbed], ""]
    if confirm:
        md += ["## Confirmation runs (mean ± std over seeds, test split)", "", confirm_table(confirm), ""]
    for name, key in (("NS#", "NS_hash"), ("NS_M", "NS_M"), ("NS_E", "NS_E")):
        md += [f"## Best configuration per model by {name}", ""]
        best = {}
        for r in rows:
            if r["model"] not in best or r[key] > best[r["model"]][key]:
                best[r["model"]] = r
        md += [table(list(best.values())), ""]
    (OUT / f"{prefix}_summary.md").write_text("\n".join(md))
    print(f"{len(rows)} runs -> {OUT}/{prefix}_summary.md")


if __name__ == "__main__":
    main()
