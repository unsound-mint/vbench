# Measurement methodology

How `vbench` fine-tunes and measures, and where it deliberately differs from
the protocol it follows (a published-style efficiency study that ran on an
RTX 5090 with full-length schedules). Read this before comparing numbers from
this repo with numbers from anywhere else.

## Task and data

Three aerial-video datasets are cached. MOD20 is the primary one (search,
confirmation, test evaluation); ERA is the second dataset for the confirmation;
UAV-Human is a consistency check on validation accuracy only. None of them is
redistributed.

- Dataset: ERA (aerial event recognition), 25 classes. Official training
  folder (1,473 clips) is split 80/20 per class with seed 17 into 1,181 fit and
  292 validation clips (`manifests/era_split.csv`). The official test split is
  not used for selection.
- Each clip is decoded once to 16 uniformly spaced frames, resized
  (not cropped) from 640x640 to 224x224, and cached as uint8
  (`scripts/build_cache.py`). The cache is derived data under `data/`, which is
  git-ignored, and must not be published.
- Training augmentation (GPU): one random square crop with area in [0.7, 1.0]
  of the frame, resized back to 224, plus horizontal flip p=0.5, shared by all
  frames of a clip. Validation: no augmentation.

## MOD20

20 action classes, 2,324 clips of 720x720 at 29.97 fps (academic use; no licence
is stated, so the clips and caches are not redistributed). The official split 1
(`split_sets/*.mat`) is used: the test set is the first 30 % of each class,
giving 701 test clips. The remaining 1,623 training clips are split into 1,297
fit and 326 validation by taking the last 20 % of each class's training clips in
file order, which keeps clips from the same source video together as far as the
file order allows (`scripts/mod20_manifest.py`). Frames are cached as for ERA;
the test clips go to a separate cache that selection never reads.

## UAV-Human (consistency check)

RGB clips of the Cross-Subject-v1 protocol: the 89 official training subjects
(`manifests/uavhuman_csv1_train_subjects.txt`) give 16,285 clips in 155 action
classes; the test subjects are not used. Clips are 1920x1080 and are resized
(not cropped, so the aspect ratio is squashed) to 224x224 when cached.
`scripts/uav_extract.py`, `uav_manifest.py` and `build_cache.py` build the cache.
The fit/validation split is stratified per class with seed 17 and a 21.57 %
validation share, giving 12,768 / 3,517 clips (Iteration 1: 12,773 / 3,512; its
exact assignment is not known). It is not subject-disjoint, so validation
accuracy is optimistic relative to a subject-held-out split. Because a full
grid would take days, only a reduced plan is run: the full staged search and
ablation for VideoMAE-B, then LoRA BF16 and Full FT BF16 for the other models.
There is no test evaluation, because the test subjects are not extracted.

## Training protocol

- AdamW, cosine decay, warmup 5% (Full FT) or 10% (PEFT), gradient clipping 1.0,
  seed 17.
- Effective batch 64 by accumulating micro-batches of 1 (identical memory
  accounting for every method). 3 epochs (57 optimizer steps), not the 10 of
  the reference protocol, to fit an 8 GB laptop GPU. Absolute accuracies are
  therefore lower than a full schedule would give; rankings under a fixed
  small budget are what these runs compare.
- Seeds: the search uses seed 17. The confirmation (below) repeats the selected
  configurations with seeds 17, 18, 19 and reports mean and sample standard
  deviation. The seed changes the head initialisation, adapter initialisation,
  data order and augmentation; the train/validation split is fixed.
- Learning rate / weight decay: Full FT 2.5e-4 / 0.05; every PEFT method
  1e-3 / 0. Hyperparameters were not tuned per model; they come from the
  reference protocol's VideoMAE/Video Swin settings and are reused for
  TimeSformer and Hiera, which the reference does not cover.

## Methods

| Method | What trains | Base weights |
|---|---|---|
| Full FT | everything | full precision |
| LoRA | rank-8 adapters (alpha 8) + head | full precision |
| QLoRA | LoRA as above | 4-bit NF4 (bitsandbytes), bf16/fp16 compute |
| AdaLoRA | rank 12 pruned to 8 + head | full precision |
| BitFit | bias terms + head | full precision |

- LoRA targets `query`/`value` where they are separate (VideoMAE) and the fused
  `qkv` otherwise, which adapts q, k and v together.
- AdaLoRA is run without its orthogonality regularizer (peft applies it only
  for models that return a `.loss`).
- Quantized base layers exclude the classifier. Video Swin's attention
  `qkv`/`proj` stay unquantized because torchvision reads their weights
  directly.

## Precisions

- **FP32:** no autocast, TF32 off (PyTorch default), so this is true FP32.
- **FP16 / BF16:** autocast over FP32 master weights ("AMP"); FP16 uses a
  gradient scaler.
- **INT8:** frozen base layers quantized to 8-bit (LLM.int8 via bitsandbytes)
  with adapter training under BF16 autocast. This is our reading of "INT8" for a
  training study; it is not post-training quantization of a finished model.

## What is measured

- **Training time** and **training energy**: summed over the optimizer-step
  phases only; validation passes are excluded and reported separately.
- **Peak training memory**: `torch.cuda.max_memory_allocated` during training
  steps (weights + optimizer state + activations; not CUDA context).
- **Validation pass energy `E_v`**: GPU energy of the final complete validation
  pass over cached clips (includes host-to-device copy and preprocessing, not
  video decoding, which the cache removes).
- **Prepared inference**: after merging adapters (LoRA/AdaLoRA on full-precision
  bases; kept unmerged on quantized bases) and casting to the run's compute
  dtype, a zero tensor `1x16x3x224x224`, 5 warmup + 30 timed synchronized
  passes; energy per clip is mean power over a 12 s loop times mean latency.
- **Accuracy**: top-1 on the validation split after the last epoch (no
  checkpoint selection); macro-F1 and balanced accuracy are also stored.
- **Test accuracy** (confirmation runs only, `--test`): one pass over the test
  cache after the last epoch, outside the energy meter. Test data are never used
  for selection.
- **Checkpoints** (`--save-checkpoint`, seed 17 of each confirmed configuration):
  the trainable parameters only, saved under `checkpoints/` (git-ignored).
- **Environment record**: before the process touches the GPU, a 4 s idle power
  baseline, the GPU temperature, and every other process using the GPU, again
  at the end of the run. A run is flagged `disturbed` when the idle power is
  above 25 W or another process holds at least 64 MiB at the start or end.
  Flagged runs are listed in the summary and should be repeated. Runs recorded
  before this was added show `n/a`.
- **NS_E, NS_M, NS#**: 1/8-weight NetScore variants (`vbench.netscore`) from
  top-1 (%), `E_v` (J) and peak training memory (MiB).

## Energy caveats

- Energy is GPU board power only (NVML). CPU, DRAM and display power are not
  included.
- The NVML counter of the laptop GPU used here refreshes about every 0.5 s, so
  the 0.02 s sampling oversamples it. Phases shorter than a few seconds are
  quantized; the 12 s inference loop holds about 24 independent readings.
- Laptop GPUs throttle with temperature and power state. Runs are sequential
  with a fresh process each. The search is single-seed; spread comes only from
  the confirmation seeds. Keep other GPU work off the machine during runs: light
  use shows up in the environment record, heavy use changes time and energy.

## Selection ("best supported configuration")

Staged search per model, on the primary dataset, 3 epochs:

1. Method search at BF16 AMP: LoRA, QLoRA, AdaLoRA, BitFit.
2. Precision search (FP32, FP16, INT8; BF16 already run) for the method with
   the highest NS#.
3. Best configuration per model by NS# (also reported by NS_M and NS_E).

This is a coordinate search, not the full 4x4 grid, to keep total GPU time
tractable. A combination that cannot run is recorded as `unsupported` or
`error` and skipped.

Confirmation (`scripts/run_experiments.py confirm 10 <dataset>`): per model, the
best configuration by NS#, the best non-full-fine-tuning configuration by NS#
and Full FT BF16, chosen from the 3-epoch search, are rerun for 10 epochs with
seeds 17, 18, 19 and evaluated on the test split.

## Ablation

Per model: Full FT FP32 (baseline), LoRA FP32, Full FT BF16 AMP, LoRA BF16 AMP.

## Learning-rate sensitivity

Hiera-B Full FT at the shared 2.5e-4 was unstable (FP32 50.0 %, BF16 60.3 %
top-1, against about 62 % for its PEFT runs). Both were rerun at 1e-4
(FP32 62.0 %, BF16 64.7 %; `lr_override` in the run JSON). These are reported
in a separate table and never enter the "best configuration" selection, which
uses the protocol learning rates only. The other models were not re-tuned, so
their Full FT numbers may also be sub-optimal.

On UAV-Human, Hiera-B Full FT BF16 diverged at 2.5e-4 (0.7 % top-1) and reached
47.8 % at 1e-4, the same pattern as on ERA.

## Pretrained checkpoints

`scripts/weight_hashes.py` prints these; all are Kinetics-400 fine-tuned.

| Model | File | sha256 |
|---|---|---|
| VideoMAE-B | `MCG-NJU/videomae-base-finetuned-kinetics` `model.safetensors` | `065cfde56f97671da4d196e1a3a6d3cf9304dcad92eb671e3632a6c6cdd04f73` |
| TimeSformer-B | `facebook/timesformer-base-finetuned-k400` `pytorch_model.bin` | `bdb264469ffc423fb86c78932268a290bfbd69bd768b7ce8595c047a2dc54089` |
| Video Swin-B | torchvision `swin3d_b_1k-24f7c7c6.pth` | `24f7c7c6fd97dd52439d72c2208e0785081811389185a4781405b482404e4b9b` |
| Hiera-B | `hiera_base_16x224.pth` (`mae_k400_ft_k400`) | `50ac81f8048019905341c176054c86c85b60ba7e9076e048c1dbe2039e56103d` |

## Known limitations

- Single device. The search is single-seed with a 3-epoch budget; only the confirmation has seeds and a 10-epoch budget.
- Hiera's normalization constants (0.45/0.225) are assumed, not documented by
  its package.
- TimeSformer is natively 8 frames and is fed 16 (time embeddings interpolate).
- MViTv2-B was a listed candidate but has no maintained open checkpoint in the
  libraries used here, so it is not covered.
