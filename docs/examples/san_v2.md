# san-v2 model

[**san-v2**](https://github.com/dkagramanyan/san-v2) is a fork of Sony's
StyleSAN-XL (Slicing Adversarial Network, StyleGAN3 + Projected GAN) used to
generate WC-Co microstructure SEM images. Its training evaluation is wired into
combra: on every **snapshot** tick it scores generated samples with combra's
sharded split-API metrics (numerically equivalent to
{py:func}`combra.metrics.compute_all_metrics`) and logs the results to TensorBoard.

The combra integration is **optional** and controlled by its own flag,
`--combra-metrics` (default `true`) — **independent of `--metrics`**, so you can run
combra with `--metrics none` (skip FID) or vice-versa. san-v2 does not depend on
combra: the import is guarded, so training runs unchanged when combra is not
installed — but if `--combra-metrics=true` and the package is missing, training emits
a **warning** at startup (and skips the metrics) so it is never silently ignored. To
enable the metrics, install combra alongside san-v2:

```bash
pip install combra            # or: pip install 'san-v2[combra]'
```

The full **install → test → train → generate** guide lives in the san-v2
[README](https://github.com/dkagramanyan/san-v2#readme); the essentials are below.

## Installation

Create a `python=3.12` conda env, install the latest PyTorch (CUDA 13.2 wheels), the
CUDA compiler (`nvcc`) and ninja **from conda** (both needed to build the custom CUDA
ops — a pip ninja conflicts with conda's, and the torch wheel ships no `nvcc`), then the
remaining deps. torch and ninja are intentionally kept out of `requirements.txt` so the
last step won't disturb them:

```bash
conda create -n san python=3.12 -y && conda activate san
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
conda install -c nvidia cuda-nvcc -y     # match torch's CUDA major (13.x)
conda install anaconda::ninja -y
pip install -r requirements.txt          # from the san-v2 repo root
```

The `sbatch/` scripts load no system CUDA module — they set `CUDA_HOME=$CONDA_PREFIX`
so the ops compile against this conda toolkit.

## Test the build

The custom CUDA ops are JIT-compiled on first use. Compile and check them (including
the H200 `sm_90` path) before training, and run the CPU SAN-layer unit tests:

```bash
python tests/test_cuda_ops.py    # CUDA-op compile + correctness check (needs a GPU)
python -m pytest tests/ -v       # SAN-layer unit tests (CI; GPU tests auto-skip on CPU)
```

## Training

Models are trained **progressively** (low → high resolution), each stage resuming
from the previous stage's `best_model.pkl`. The 16² stem trains from scratch; every
higher resolution is a super-resolution stage (`--superres --up_factor 2`):

```bash
# Stage 0 — 16x16 stem
python train.py --outdir=./runs/wc-cv_h200 --cfg=stylegan3-r --cond True \
        --data=./datasets/imagenet_9to4_1024x1024_16x16.zip \
        --gpus=2 --mirror=0 --snap 500 --batch-gpu 320 --kimg 20000 --syn_layers 6

# Stage N — superres, resuming from the previous stage
python train.py --outdir=./runs/wc-cv_h200 --cfg=stylegan3-r --cond True \
        --data=./datasets/imagenet_9to4_1024x1024_32x32.zip \
        --gpus=2 --mirror=0 --snap 100 --batch-gpu 96 --kimg 20000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 7 \
        --path_stem ./runs/wc-cv_h200/00000-.../best_model.pkl
```

On the cluster the per-stage scripts in `sbatch/` are submitted from the sbatch folder
onto the `rocky` partition (2× H200 each):

```bash
cd sbatch
sbatch train_16x16.sbatch        # … through train_1024x1024.sbatch
```

Training can also be launched via Hydra (`train_hydra.py`), which shares the same
`build_config()` logic as the click CLI. Because the click CLI is the single source of
truth for defaults, any flag below works as a Hydra override too (e.g.
`save_inference_only=true`).

The production `sbatch/train_*.sbatch` scripts pass `--save-inference-only 0`, so a prod
run keeps only the single full checkpoint `network-snapshot-latest.pt` — one big file
**overwritten in place** every snapshot tick (it never accumulates), holding `G`/`D`/`G_ema`
plus resume `progress` for `--resume` — together with the best-FID `best_model.pkl`. It
does **not** accumulate per-tick artifacts. Pass `--save-inference-only 1` to also write a
small `network-snapshot-<kimg>-inference.pkl` each tick holding **only `G_ema`** (no
discriminator, no resume state) — the smallest artifact, exactly what `gen_images.py` and
the combra evaluation consume — or `--save-weights-only 1`
for per-tick `G`/`D`/`G_ema` when the discriminator is needed.

```{warning}
`--save-inference-only` means the **opposite** here than in the other three
models: in san-v2 the rolling full `network-snapshot-latest.pt` is written
regardless and `1` *adds* the per-tick inference `.pkl`s, whereas in
{doc}`DiffiT-v2 <diffit>`, {doc}`EDM2-v2 <edm2>` and {doc}`StyleSwin-v2 <styleswin>`
inference snapshots are always written and the flag *skips* the rolling full
checkpoint. See the {doc}`API scheme <models_api>` before copying the flag
between repos.
```

```{note}
Checkpoints that embed the projected discriminator's `timm` feature networks
(`best_model.pkl` stems, `network-snapshot-latest.pt`) are `timm`-version-sensitive.
`san-v2` now uses **timm 1.x**; stems saved under the old `timm==0.4.12` pin will not
unpickle and must be regenerated. The inference-only `G_ema` snapshots carry no `timm`
modules and are unaffected.
```

During training, the combra metrics use the **whole training set** as the fixed
reference, scored against a fixed **`COMBRA_NUM_GEN = 10 000`** images generated
from `G_ema` (their labels sampled from the training-set class distribution, seeded
identically on every rank). combra's image-feature metrics estimate per-side
statistics, so the two counts need not match. The reference set is fixed across
training, so its features (pooled angles, FID/DINOv2 `(mu, sigma)`, CLIP embedding)
are extracted **once before the training loop** — sharded across ranks (each rank
processes its deterministic slice of the reals) and gathered to rank 0 — then
cached; only the generated side is recomputed each evaluation tick.

This mirrors the standard StyleGAN `fid50k_full` (all reals vs a fixed sample of
fakes) but with a 10 000-image generated side, and it adds the angle-distribution
and CMMD/FD-DINOv2 metrics on top of FID.

`G_ema` emits float images in `[-1, 1]`, while the reals are `uint8` `[0, 255]`;
the loop denormalizes the fakes to `uint8` first so both sides are scored
explicitly on the same scale. (combra also rescales float inputs internally — both
the image-feature and the angle-density metrics map `[-1, 1]`/`[0, 1]` to `uint8`
the same way — but denormalizing in the loop keeps the comparison unambiguous.)
Each tick computes **both** the angle-density metrics (Wasserstein `w1`, `w2`,
`circular_w1`, `circular_w2` and the bimodal-Gaussian `mu/sigma/amp` errors) **and**
the image-feature metrics `fid`, `cmmd`, `fd_dinov2`. All are logged to TensorBoard
under `Metrics/combra_*` and printed to the run log; the three image-feature metrics
carry their 10k sample size in the key (`combra_fid10k`, `combra_cmmd10k`,
`combra_fd_dinov2_10k`).

On a **multi-GPU** run **all** the per-image extraction is **sharded across ranks** —
both the image-feature metrics and the angle-density / Gaussian-fit metrics. Each
rank generates its own shard of the fakes and, from that shard, extracts the CLIP /
DINOv2 / InceptionV3 features and pools the vertex angles; the feature rows and the
pooled-angle arrays are gathered to rank 0, where the Fréchet / MMD distances and
the angle Wasserstein / Gaussian metrics are taken against the precomputed
reference. This uses combra's split APIs — the feature halves
({py:func}`combra.metrics.fid_features` + {py:func}`combra.metrics.fid_from_features`,
and the `cmmd_*` / `fd_dinov2_*` analogues) and the angle halves
({py:func}`combra.metrics.images_to_pooled_angles` +
`angle_density_metrics_from_pooled`) — and is numerically identical to the
single-GPU `compute_all_metrics(image_metrics=True)` path. Sharding the angle
extraction also fixes a multi-GPU hang: when it ran rank-0-only over the full
gathered batch, at 512/1024 px it could exceed NCCL's collective-timeout watchdog
while the other ranks idled, aborting the job. On a single GPU the three image
metrics instead run concurrently in a thread pool. Any whose optional backend is
unavailable (e.g. no network to fetch the DINOv2/CLIP weights on an offline compute
node) is recorded as `nan` rather than aborting — the angle metrics always come back. To
avoid those `nan`s, pre-download the weights once on a login node with
`bash download_models.sh` (or `python tests/test_san_modules.py`); the weights cache
under `$HOME`, shared with the compute nodes.

When `--combra-metrics` is enabled (the default), `fid50k_full` is dropped from
`--metrics` so the expensive 50k-image FID pass is not computed twice, and
`combra_fid10k` becomes the FID that drives best-model checkpoint selection
(`best_model.pkl`).

```{note}
Because the reference is the entire dataset and an equal number of fakes is
generated each evaluation tick, both the reference and the generated extraction
(angle pooling + the InceptionV3 / CLIP / DINOv2 backbones) run on every rank over
its own shard, so that cost is shared across GPUs; only the small gathered feature
rows and pooled-angle arrays land on rank 0 for the final distances. Each snapshot
still costs more than a full FID pass. For very large or high-resolution datasets this is
memory- and compute-intensive; pass `--combra-metrics False` on runs where you
don't want it.
```

## Evaluation

Standalone metrics for a trained checkpoint use the StyleGAN-XL runners:

```bash
python calc_metrics.py --metrics=fid50k_full --network=<path_to_checkpoint>
python calc_metrics.py --metrics=is50k       --network=<path_to_checkpoint>
```

To score arbitrary real/generated image batches with the combra metrics directly
(the same ones logged during training), call
{py:func}`combra.metrics.compute_all_metrics`:

```python
>>> from combra.metrics import compute_all_metrics
>>> results = compute_all_metrics(reference_images, generated_images, image_metrics=True)
>>> results
{'w1': ..., 'w2': ..., 'circular_w1': ..., 'circular_w2': ...,
 'mu1': ..., 'mu2': ..., 'sigma1': ..., 'sigma2': ..., 'amp1': ..., 'amp2': ...,
 'fid': ..., 'cmmd': ..., 'fd_dinov2': ...}
```

`image_metrics=True` is what adds the `fid`/`cmmd`/`fd_dinov2` keys (and is what the
training loop passes); without it, only the angle-Wasserstein and bimodal-Gaussian
metrics come back. Batches may be numpy arrays or torch tensors in `NCHW`/`NHWC`,
with pixel values in `uint8`, `[0, 1]`, or `[-1, 1]`; combra rescales each side to
`uint8` by inferring the range (any negative ⇒ `[-1, 1]`, else max ≤ 1 ⇒ `[0, 1]`).
Because
that inference is per-batch, prefer passing `uint8` (or an explicit, consistent
scale) when you can — as the training loop does — so an unusual batch (e.g. a
generated batch that happens to be all-positive) cannot be misread.

## Inference

Generate samples from a trained network with `gen_images.py`. Images are written
per class into `class_<id>/<class>_<index>.png`; pass `--gpus` (or launch with
`torchrun`) to distribute generation:

```bash
python gen_images.py \
  --outdir=./generated/ \
  --trunc=0.7 \
  --samples-per-class 1000 \
  --classes 0,1,2 \
  --gpus 2 \
  --batch-gpu 60 \
  --network=<path_to_checkpoint>
```

### Class index → grain class

The `--classes` integer selects a grain morphology. For the SAN model trained on the
WC-Co dataset the indices map as:

| index | grain class | morphology |
|---|---|---|
| `0` | `Ultra_Co25` | medium grain (средние зёрна) |
| `1` | `Ultra_Co11` | small grain (мелкие зёрна) |
| `2` | `Ultra_Co6_2` | large grain (крупные зёрна) |

```{warning}
**The other generators most likely share this order, not the alphabetical
one.** The other repos' dataset *tools* nominally derive labels
alphabetically (`0 → Ultra_Co11`, `1 → Ultra_Co25` — indices 0 and 1 swapped
relative to SAN; `2 → Ultra_Co6_2` matches), but the on-disk
`imagenet_9to4_*` archives their real runs consumed carry the **same swapped
order as the SAN zips**, and all four repos take zip labels verbatim at
train time (see {doc}`diffit`, {doc}`edm2`, {doc}`styleswin`). Classify
every checkpoint by the dataset path in its `training_options.json` before
comparing across models; remap with combra's `CLASS_MAP` (consumed by
{py:func}`combra.angles.resolve_overlay_rows` and
{py:func}`combra.angles.build_overlay_grid` as `gen_name_for_mode`) **only**
where the audit shows the conventions actually differ.
```

**Why the order differs.** The dataset stores only an integer label per image — never
the `Ultra_Co*` name — and the two pipelines derive that integer by *different rules*:

- **SAN** (`dataset_tool.py`) copies the label **verbatim from the preprocessed
  source's `dataset.json`**, which was written `Ultra_Co25 → 0`, `Ultra_Co11 → 1`,
  `Ultra_Co6_2 → 2` — an order that is **not** alphabetical.
- **DiffiT** (`dataset_tool_for_imagenet.py`) ignores that file and re-derives labels
  from the **alphabetical order of the class-folder names** (`sorted` then
  `enumerate`): `Ultra_Co11 → 0`, `Ultra_Co25 → 1`, `Ultra_Co6_2 → 2`.

Because the source `dataset.json` lists `Co25` before `Co11` while the folder sort puts
`Co11` first, the two conventions disagree on labels 0 and 1 — the `Co11`↔`Co25` swap.
`Ultra_Co6_2` is last under both rules, so it stays `2`. **Which rule a checkpoint
follows depends on which zip it trained on** — the shipped `imagenet_9to4_*` archives
carry the SAN-order labels, whichever tool nominally built them. Since neither pipeline
records the grain name downstream (the zips hold neither `class_names` nor original
filenames, and generated h5s carry only `class_0/1/2`), the correspondence has to be
recovered per run from `training_options.json` and pinned in combra's `CLASS_MAP`.
