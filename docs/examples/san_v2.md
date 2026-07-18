# san-v2 model

[**san-v2**](https://github.com/dkagramanyan/san-v2) is a fork of Sony's
StyleSAN-XL (Slicing Adversarial Network, StyleGAN3 + Projected GAN) used to
generate WC-Co microstructure SEM images. As of **v0.2.0** it implements the v2
model API convention specified in {doc}`models_api_proposal` (§12); this page
documents that API. On every **snapshot** tick it scores generated samples with
combra's sharded split-API metrics (numerically equivalent to
{py:func}`combra.metrics.compute_all_metrics`) and logs the results to both
TensorBoard and `stats.jsonl`.

The combra integration is **optional** and controlled by `--combra-metrics`
(default `True`). san-v2 does not depend on combra: the import is guarded, so
training runs unchanged when combra is not installed — but if
`--combra-metrics=True` and the package is missing, training emits a **warning**
at startup (and skips the metrics) so it is never silently ignored. To enable the
metrics, install the optional extra (pulled from the private repo over
`git+https`):

```bash
pip install 'san-v2[combra]'
```

The full **install → test → train → generate** guide lives in the san-v2
[README](https://github.com/dkagramanyan/san-v2#readme); the essentials are below.

## Installation

`pyproject.toml` is the only dependency declaration (there is no
`requirements.txt`) and `pip install -e .` is the one install path. Create a conda
env, install the latest PyTorch (CUDA wheels), the CUDA compiler (`nvcc`) and
ninja **from conda** (both needed to build the custom CUDA ops — a pip ninja
conflicts with conda's, and the torch wheel ships no `nvcc`), then the package.
torch and ninja are intentionally kept out of `pyproject.toml` so the last step
won't disturb them:

```bash
conda create -n san-v2 python=3.12 -y && conda activate san-v2
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
conda install -c nvidia cuda-nvcc -y     # match torch's CUDA major (13.x)
conda install anaconda::ninja -y
pip install -e ".[combra]"               # from the san-v2 repo root; drop [combra] to skip metrics
```

Installing the package exposes four console scripts: **`san-train`**,
**`san-gen-images`**, **`san-eval`**, **`san-prepare-data`**. The `sh/` launch
scripts set `CUDA_HOME=$CONDA_PREFIX` so the ops compile against this conda
toolkit, and set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` for offline compute
nodes (prefetch backbones once on a login node with `bash download_models.sh`).

## Test the build

The custom CUDA ops are JIT-compiled on first use. Compile and check them (including
the H200 `sm_90` path) before training, and run the CPU tests (CLI-contract smoke
tests + SAN-layer unit tests):

```bash
python tests/test_cuda_ops.py    # CUDA-op compile + correctness check (needs a GPU)
python -m pytest tests/ -v       # CPU smoke + SAN-layer unit tests (GPU tests auto-skip)
```

## Dataset preparation

`san-prepare-data` is a click group; its `convert` subcommand builds a
StyleGAN-style zip from a folder of class subfolders. Labels are derived from the
**alphabetical** order of the class-folder names, an index-aligned `class_names`
list is written into `dataset.json`, and grayscale sources are converted to RGB at
build time. A single unlabeled image in an otherwise-labeled tree is a hard error
(never a silent drop of the whole label set):

```bash
san-prepare-data convert --source ./raw/wc_co --dest ./datasets/wc_co_256.zip \
    --transform center-crop --resolution 256x256
```

## Training

Models are trained **progressively** (low → high resolution). The 16² stem trains
from scratch; every higher resolution is a super-resolution stage that
**weights-only warm-starts** from the previous stage's inference snapshot via
`--path-stem` (there is no resume — runs go start-to-finish):

```bash
# Stage 0 — 16x16 stem
san-train --outdir ./runs --cfg stylegan3-r --cond True \
    --data ./datasets/wc_co_16x16.zip \
    --gpus 2 --mirror False --snap 500 --batch-gpu 320 --kimg 20000 --syn-layers 6

# Stage N — superres, warm-starting from the previous stage's snapshot
san-train --outdir ./runs --cfg stylegan3-r --cond True \
    --data ./datasets/wc_co_32x32.zip \
    --gpus 2 --mirror False --snap 100 --batch-gpu 96 --kimg 20000 --syn-layers 6 \
    --superres --up-factor 2 --head-layers 7 \
    --path-stem ./runs/00000-stylegan3-r-gpus2-batch640/san-snapshot-020000-inference.pt
```

Ready-made per-resolution launch scripts live in `sh/` (`train_256.sh`,
`generate_512.sh`, …); they carry only the compute-node environment plus one
console-command call, with SLURM specifics supplied at submission time:

```bash
DATA=./datasets/wc_co_256.zip GPUS=2 bash sh/train_256.sh
sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_256.sh   # same script on a cluster
```

### Checkpoint scheme

There is exactly one artifact kind: an **EMA-only inference snapshot**
`san-snapshot-<kimg:06d>-inference.pt`, a `.pt` **state dict** (no pickled
modules — loading never depends on the `timm` version that trained the
discriminator). It is written **atomically** every snapshot tick **and always at
the last tick**, so the newest snapshot *is* the final model, and the history is
pruned to `--snapshot-keep-last` (default 3, `0` = keep all). Each snapshot carries
self-describing metadata `{n_classes, resolution, class_names, cur_nimg}`.

There is **no resume, no `best_model`, no rolling `latest` checkpoint** — pick the
best snapshot post-hoc from `stats.jsonl` (the combra metrics are mirrored there,
not TensorBoard-only). Size `--kimg` (or split stages) so a run fits its job's time
limit; an interrupted run cannot be continued.

```{warning}
This is a **breaking change** from pre-0.2.0 san-v2: `.pkl` artifacts,
`--resume`, `best_model.pkl`, `--save-inference-only`/`--save-weights-only`,
`--fp32`/`--nobench`, the `--metrics` registry and the Hydra entry point are all
gone. Precision is now `--precision {fp32,fp16,bf16}` with `--tf32`/`--bench`;
`--mirror` is a stochastic loader-level flip (no longer dataset x-flip doubling).
```

### combra metrics during training

Each snapshot tick scores `--num-fid-samples` fakes (default 10 000, `0` disables
eval) generated from `G_ema` against the training set as the fixed reference
(cap it to a seeded random subset with `--combra-ref-count`). The reference
features (pooled angles, FID/DINOv2 `(mu, sigma)`, CLIP embedding) are extracted
**once before the loop**, sharded across ranks and cached; only the generated side
recomputes each tick. Both the image-feature metrics (`fid`, `cmmd`, `fd_dinov2`)
and the angle-density metrics (Wasserstein `w1`/`w2`/`circular_*` + bimodal-Gaussian
fit errors) are logged under `Metrics/combra_*` in TensorBoard **and** `stats.jsonl`;
the three image metrics carry their 10k sample size literally in the key
(`combra_fid10k`, `combra_cmmd10k`, `combra_fd_dinov2_10k`).

`G_ema` emits float images in `[-1, 1]`; the loop denormalizes fakes to `uint8`
with the single normalize/denormalize pair (asserted to round-trip) so reals and
fakes are scored on the identical `uint8` scale.

On a **multi-GPU** run **all** per-image extraction is **sharded across ranks** —
each rank generates its shard of the fakes, extracts the CLIP / DINOv2 / InceptionV3
features and pools the vertex angles; the feature rows and pooled-angle arrays are
gathered to rank 0, which takes the Fréchet / MMD distances and the angle metrics
against the cached reference. This uses combra's split APIs — the feature halves
({py:func}`combra.metrics.fid_features` + {py:func}`combra.metrics.frechet_from_features`,
and the `cmmd_*` / `fd_dinov2_*` analogues) and the pooled-angle extractor
({py:func}`combra.metrics.images_to_pooled_angles`) — and is numerically identical to the
single-GPU `compute_all_metrics(image_metrics=True)` path. Any metric whose optional
backend is unavailable (e.g. no network to fetch DINOv2/CLIP weights) is recorded as
`nan` rather than aborting. Pre-download the weights once on a login node with
`bash download_models.sh`.

## Evaluation

Standalone metrics for a trained snapshot use `san-eval` (the native metric
registry is legacy; `--data` is required since snapshots carry no dataset kwargs):

```bash
san-eval --metrics fid50k_full --data ./datasets/wc_co_256.zip \
    --network ./runs/00001-stylegan3-r-gpus2-batch168/san-snapshot-020000-inference.pt
```

To score arbitrary real/generated image batches with the combra metrics directly,
call {py:func}`combra.metrics.compute_all_metrics`:

```python
>>> from combra.metrics import compute_all_metrics
>>> results = compute_all_metrics(reference_images, generated_images, image_metrics=True)
>>> results
{'w1': ..., 'w2': ..., 'circular_w1': ..., 'circular_w2': ...,
 'mu1': ..., 'mu2': ..., 'sigma1': ..., 'sigma2': ..., 'amp1': ..., 'amp2': ...,
 'fid': ..., 'cmmd': ..., 'fd_dinov2': ...}
```

Prefer passing `uint8` batches (as the training loop does) so an unusual batch
(e.g. an all-positive `[-1, 1]` batch) cannot be misread by combra's range
inference.

## Inference

Generate samples from a trained snapshot with `san-gen-images`. `--gpus N`
self-spawns one worker per GPU (the same launch model as training; no `torchrun`).
`--classes` accepts indices, ranges, **or grain-class names**, validated against
the checkpoint's `class_names` metadata:

```bash
san-gen-images \
    --network ./runs/00001-stylegan3-r-gpus2-batch168/san-snapshot-020000-inference.pt \
    --outdir ./generated \
    --classes Ultra_Co11,Ultra_Co6_2 \
    --samples-per-class 1000 \
    --gpus 2 --batch-gpu 60 --trunc 0.7 --save-mode hdf5
```

With `--save-mode hdf5` (default) each rank writes a shard in the `RankH5Writer`
layout (`class_<c>/images|seeds`, uint8 NHWC), merged by rank 0 into
`<outdir>/<desc>.h5` — exactly what the wc_cv angle pipeline consumes. Every shard
and the merged file carry `format="generated_images_shard"`, `schema_version=1` and
the `class_names`, and the merge **hard-fails** if any sample slot is missing (a
crashed run never feeds zero-filled black images downstream). `--save-mode dir`
writes `class_<c>/idx_<i>_seed_<s>.png` plus a `classes.json` manifest. Generation
is deterministic: `seed = base + class·samples_per_class + idx`.

### Class index → grain class

New zips built by `san-prepare-data` derive labels **alphabetically** and stamp
`class_names` into the artifact, so downstream code matches by name and the mapping
is self-describing:

| index | grain class | morphology |
|---|---|---|
| `0` | `Ultra_Co11` | small grain (мелкие зёрна) |
| `1` | `Ultra_Co25` | medium grain (средние зёрна) |
| `2` | `Ultra_Co6_2` | large grain (крупные зёрна) |

```{warning}
**Legacy artifacts (pre-0.2.0) use a different, swapped order.** The on-disk
`imagenet_9to4_*` archives that the existing checkpoints trained on carry the
non-alphabetical order `Ultra_Co25 → 0`, `Ultra_Co11 → 1`, `Ultra_Co6_2 → 2`
(the `Co11`↔`Co25` swap), and they record no `class_names`. Those checkpoints
and everything generated from them stay under combra's legacy `CLASS_MAP` until
retrained on rebuilt zips — classify each run by the dataset path in its
`training_options.json` before remapping. Once san-v2 is retrained on
`san-prepare-data`-built zips, all artifacts are self-describing by name and the
class-map warning becomes a historical note. See the {doc}`label contract
<models_api_proposal>` (§5) and the current-state {doc}`API scheme <models_api>`.
```
