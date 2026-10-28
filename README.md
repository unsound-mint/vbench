# vbench

Energy, latency and memory measurement for fine-tuning and inference of video
classification models on consumer GPUs.

Parameter-efficient fine-tuning (PEFT) cuts trainable parameters, but not
necessarily training memory, time or energy. This repo measures those directly
and combines them with accuracy into NetScore-style composite scores.

## Setup

```sh
uv sync
uv run pytest
```

Requires an NVIDIA GPU for the measurement scripts (tests run on CPU).

## What is measured

- **Power and energy:** NVML board power, sampled every 0.02 s, trapezoidal
  integration (`vbench.power`). Some GPUs refresh the NVML counter far more
  slowly (about 0.5 s on an RTX 4070 Laptop), so energy over windows shorter
  than a few seconds is quantized.
- **Prepared inference:** zero tensor `1x16x3x224x224` already on the GPU,
  5 warmup passes, latency as the mean of 30 synchronized passes, energy per
  clip as mean power over a 12 s sustained loop times mean latency
  (`vbench.inference`).
- **Training:** wall time and GPU energy of the optimizer-step phases, and peak
  allocated memory during them (`vbench.run`); validation is metered
  separately.
- **Composite scores:** `NS_E`, `NS_M`, `NS#` (`vbench.netscore`), the 1/8-weight
  variants of NetScore (Wong, 2019).
- **Hardware record:** GPU model, VRAM, driver, CUDA, torch, CPU, OS. No
  serials, UUIDs or hostnames (`vbench.hardware`).

## Models and methods

Open-weight video Transformers behind one `(B, T, C, H, W)` interface
(`vbench.models`): VideoMAE-B, TimeSformer-B, Video Swin-B (torchvision),
Hiera-B. Methods: Full FT, LoRA, QLoRA, AdaLoRA, BitFit; precisions FP32,
FP16, BF16 (autocast) and INT8 (frozen 8-bit base) (`vbench.adapt`,
`vbench.quant`).

Notes that affect results:

- LoRA on a fused `qkv` projection adapts q, k and v together; VideoMAE uses
  separate `query`/`value` targets.
- torchvision's Swin3D calls `F.linear` on `qkv.weight` directly, which bypasses
  peft wrappers. `vbench.models` patches it, and
  `tests/test_adapt.py` checks every trainable parameter receives a gradient.
- TimeSformer is natively 8 frames and is fed 16 here (time embeddings
  interpolate).

## Running the experiments

Get the datasets yourself (they are not redistributed). The commands below use
ERA; `mod20` and `uavhuman` work the same way (`scripts/mod20_manifest.py`,
`scripts/uav_extract.py` and `scripts/uav_manifest.py` build their manifests).

```sh
uv run python scripts/run_experiments.py all 3 mod20      # search on the primary dataset
uv run python scripts/run_experiments.py confirm 10 mod20 # 10 epochs, seeds 17-19, test split
uv run python scripts/summarize.py 3 mod20
```

ERA, step by step:

```sh
uv run python scripts/era_manifest.py      # manifests/era_split.csv
uv run python scripts/build_cache.py era   # decode once to uint8 frame caches
uv run python scripts/run_experiments.py smoke
uv run python scripts/run_experiments.py all 3   # staged search + ablation, 3 epochs
uv run python scripts/summarize.py 3       # results/it2_summary.md, results/it2_runs.csv
```

One run is one process: `python -m vbench.run --model hiera-b --method lora
--precision bf16`. Per-run JSON (config, validation, optional test metrics, adaptation cost,
deployment cost, NetScore variants, hardware, idle-power and GPU-sharing record)
lands in `runs/it2/` (`runs/it2_mod20/`, `runs/it2_uavhuman/`).
`docs/methodology.md` defines the protocol, how it differs from the
reference study, and the energy-measurement caveats; read it before comparing
numbers.

## Results (ERA, RTX 4070 Laptop 8 GB, 3 epochs, one seed)

Full tables: [`results/it2_summary.md`](results/it2_summary.md),
[`results/it2_runs.csv`](results/it2_runs.csv). Best configuration per model by
NS# (protocol learning rates):

| Model | Best by NS# | Top-1 % | Peak train mem MiB | NS# |
|---|---|---|---|---|
| VideoMAE-B | Full FT, BF16 | 72.95 | 2126 | 59.60 |
| Video Swin-B | Full FT, BF16 | 79.79 | 4306 | 59.09 |
| TimeSformer-B | Full FT, BF16 | 80.14 | 4721 | 58.82 |
| Hiera-B | BitFit, BF16 | 62.33 | 1560 | 56.15 |

Absolute accuracies are low because of the 3-epoch budget; read these as
rankings under a fixed small budget. See `docs/methodology.md` for caveats
(single seed, Hiera Full FT learning-rate sensitivity, energy-counter
resolution).

### UAV-Human (RGB, Cross-Subject-v1 training subjects, same setup)

Reduced plan (full search for VideoMAE-B; LoRA and Full FT in BF16 for the
others), so "best" is within the runs made. Tables:
[`results/uavhuman_summary.md`](results/uavhuman_summary.md).

| Model | Best of runs made, by NS# | Top-1 % | Peak train mem MiB | NS# |
|---|---|---|---|---|
| Hiera-B | LoRA, BF16 | 51.04 | 1597 | 49.98 |
| VideoMAE-B | QLoRA, FP16 | 44.81 | 619 | 49.46 |
| Video Swin-B | Full FT, BF16 | 46.43 | 4308 | 46.91 |
| TimeSformer-B | Full FT, BF16 | 32.64 | 4723 | 40.56 |

Hiera-B Full FT at the shared learning rate diverged (0.7 % top-1); at 1e-4 it
reaches 47.82 % (NS# 48.33, reported separately). TimeSformer-B is still
converging after 3 epochs (loss 3.1 at epoch 2), so its numbers mostly reflect
the short budget. The validation split is not subject-disjoint.

## Feasibility probe

```sh
uv run python scripts/probe_training.py   # writes runs/training_probe.json
```

One forward+backward+optimizer step per model, method, precision and
checkpointing setting; reports peak memory, or OOM.

Datasets and model weights are not included and are not redistributed; follow
each dataset's own license and access process.

## License

MIT
