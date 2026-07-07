# StyleSwin-v2 model

[**StyleSwin-v2**](https://github.com/dkagramanyan/StyleSwin-v2) is a fork of Microsoft's
[StyleSwin](https://github.com/microsoft/StyleSwin) (a Swin-transformer StyleGAN) specialised
for generating WC-Co microstructure SEM images. Upstream StyleSwin is **unconditional**; this
fork adds **class-conditional** generation over the three grain classes and wires the training
evaluation into combra: on every **snapshot** tick it scores generated samples with
{py:func}`combra.metrics.compute_all_metrics` and logs the results to TensorBoard — the same
integration as {doc}`san_v2` and {doc}`diffit`.

Conditioning reuses the san-v2 techniques: the generator embeds the one-hot label into the
mapping network (2nd-moment normalised alongside `z`), and the discriminator adds a Miyato &
Koyama projection term to StyleSwin's unchanged logistic + R1 loss. Enable it with
`--cond True`; `n_classes` is read from the dataset's `dataset.json` (`n_classes = 0` keeps the
unconditional path). The generator/discriminator update math is unchanged from upstream — the
conditioning and the san-v2-style tooling wrap around it.

The combra integration is **optional** and controlled by `--combra-metrics` (default `true`).
The import is guarded, so training runs unchanged when combra is not installed — but if
`--combra-metrics=true` and the package is missing, training emits a **warning** at startup (and
skips the metrics) so it is never silently ignored:

```bash
pip install -e '.[combra]'          # from the StyleSwin repo root; see Installation below
```

## Installation

Create a `python=3.12` conda env named `styleswin`, install the latest PyTorch (CUDA 13.x
wheels), the CUDA compiler (`nvcc`) and ninja **from conda** (both needed to build StyleSwin's
custom CUDA ops — `fused_act` / `upfirdn2d`), then the remaining deps:

```bash
conda create -n styleswin python=3.12 -y && conda activate styleswin
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
conda install -c nvidia cuda-nvcc -y     # match torch's CUDA major (13.x)
conda install anaconda::ninja -y
pip install -e .                         # from the StyleSwin repo root (reads pyproject.toml)
pip install -e '.[combra]'               # optional: adds the combra metrics package
```

combra lives in a **private** repo, so the `.[combra]` extra clones it over `git+https` and only
succeeds when you are authenticated to GitHub — sign in once with the GitHub CLI
(`gh auth login` → github.com → HTTPS) and `pip` inherits its credential helper. On hosts with
the nexus PyPI proxy, a plain `pip install combra` resolves it without GitHub.

The `sbatch/` scripts load no system CUDA module — they set `CUDA_HOME=$CONDA_PREFIX` and
`TORCH_CUDA_ARCH_LIST=9.0` so the ops compile against this conda toolkit for the H200 (`sm_90`).

## Dataset

Training consumes an **ImageNet-style zip** of `(uint8 image, one-hot label)` — the same format
as san-v2. The zip holds class subfolders of PNGs plus a root `dataset.json`
(`{"labels": [[path, class_idx], ...]}`); labels are stored as integer indices and expanded to
one-hot at read time, and `n_classes` is derived as `max(label) + 1`. The WC-Co archives are
3-class (2880 images each):

```text
datasets/imagenet_9to4_1024x1024_256x256.zip     # 256px
datasets/imagenet_9to4_1024x1024_512x512.zip     # 512px
datasets/imagenet_9to4_1024x1024_1024x1024.zip   # 1024px
```

Place the zips directly in the repo's `datasets/` folder (the `sbatch/` scripts reference
`./datasets/…`). Build new ones from a raw ImageNet-style tree with
`dataset_tool_for_imagenet.py` (labels derived from the **alphabetical** class-folder order), or
from a generic labelled folder with `dataset_tool.py convert-dataset`.

## Training

`train.py` is the primary entry point — a san-v2-style `click` CLI (`--outdir/--data/--gpus/
--cfg/--cond/--kimg/--snap/--combra-metrics/--save-inference-only/--resume` plus StyleSwin's own
model flags). Checkpoints and logs are written under `--outdir`; the WC-Co runs use
`./runs/wc-cv_h200`.

StyleSwin builds all layers at the target resolution at once, so unlike san-v2's progressive
super-resolution stages **each resolution is trained independently** (no stage-to-stage resume
chain). Pick the resolution with a `--cfg` preset:

```bash
python train.py --outdir=./runs/wc-cv_h200 \
        --cfg styleswin-256 \
        --data=./datasets/imagenet_9to4_1024x1024_256x256.zip \
        --gpus=2 --cond True --combra-metrics True --save-inference-only True \
        --kimg 25000 --snap 50
```

### Resolution presets (`--cfg`)

The generator and discriminator are **resolution-parametric** — built from the dataset
resolution — so there is no separate per-resolution model file. The per-resolution knobs instead
live in a single `RESOLUTION_CONFIGS` dict in `train.py`, selected by `--cfg`:

| preset | resolution | batch/GPU | total batch (2 GPUs) |
|---|---|---|---|
| `styleswin-256`  | 256²  | 16 | 32 |
| `styleswin-512`  | 512²  | 8  | 16 |
| `styleswin-1024` | 1024² | 4  | 8  |

Today the presets differ only in the memory-bound batch size (the other model/optimizer knobs
are bundled in so each resolution has one place to tune). Any explicit CLI flag overrides the
preset — e.g. `--cfg styleswin-256 --batch-gpu 8` keeps the preset's other values but forces a
batch of 8. `--cfg` also cross-checks its resolution against the dataset and refuses a mismatched
`--data` zip. Omit `--cfg` and pass `--batch-gpu` directly to train at an arbitrary resolution.

On the cluster the per-resolution scripts in `sbatch/` are submitted from the sbatch folder onto
the `rocky` partition (2× H200 each):

```bash
cd sbatch
sbatch train_256x256.sbatch        # … train_512x512.sbatch, train_1024x1024.sbatch
```

The scripts pass `--save-inference-only True`, so every snapshot tick also writes a small
`network-snapshot-<kimg>-inference.pt` holding **only `G_ema`** — the smallest artifact and
exactly what `gen_images.py` and the combra evaluation consume. This is written *in addition to*
the full `network-snapshot-<kimg>.pt` (`G` + `D` + `G_ema` + both optimizers), which is kept so a
run can be resumed with `--resume`; `best_model.pt` (selected by `combra_fid10k`) is a full
checkpoint too. All three carry `g_ema`/`n_classes`/`size`, so any of them can drive
`gen_images.py`.

## Metrics

Training runs a **kimg/tick** loop (matching san-v2): a per-run `log.txt`, `stats.jsonl`,
TensorBoard events and the `tick … kimg … sec/tick …` status line. Reported **every tick** and
written to all three sinks: the generator/discriminator losses and R1 (`Loss/*`), timing
(`Timing/*`), CPU/GPU memory (`Resources/*`) and the effective learning rates
(`LearningRate/G`, `LearningRate/D`). The **combra metrics** are computed **every snapshot tick**
(and step-held on TensorBoard between snapshots); the running best FID is logged as
`Metrics/combra_fid10k_best`. Each snapshot also logs the `G_ema` sample grid to TensorBoard
(tag `Fakes`) alongside the on-disk `fakes<kimg>.png`.

combra uses the **whole training set** as the fixed reference, scored against a fixed
`COMBRA_NUM_GEN = 10 000` images generated from `G_ema` (labels sampled from the training-set
class distribution, seeded identically on every rank). Each snapshot computes **both** the
angle-density metrics (Wasserstein `w1`, `w2`, `circular_w1`, `circular_w2` and the
bimodal-Gaussian `mu/sigma/amp` errors) **and** the image-feature metrics `fid`, `cmmd`,
`fd_dinov2`. All are logged to TensorBoard under `Metrics/combra_*` and printed to the run log;
the three image-feature metrics carry their 10k sample size in the key (`combra_fid10k`,
`combra_cmmd10k`, `combra_fd_dinov2_10k`). `combra_fid10k` drives best-model checkpoint selection
(`best_model.pt`, with the winning image count in `best_nimg.txt`).

On a **multi-GPU** run all per-image extraction is **sharded across ranks** — each rank generates
its own shard of the fakes and extracts the CLIP / DINOv2 / InceptionV3 features and pools the
vertex angles from it; the feature rows and pooled-angle arrays are gathered to rank 0 for the
final Fréchet / MMD / Wasserstein distances against the precomputed reference. This is
numerically identical to the single-GPU `compute_all_metrics(image_metrics=True)` path. Any
image metric whose optional backend is unavailable (e.g. no network to fetch the DINOv2/CLIP
weights on an offline compute node) is recorded as `nan` rather than aborting; the angle metrics
always come back. Each snapshot is compute-intensive at high resolution — pass
`--combra-metrics False` on runs where you don't want it.

## Inference

Generate samples from a trained checkpoint with `gen_images.py`. Images are written per class
into `class_<id>/class_<id>_<index>.png`; pass `--gpus` to distribute generation:

```bash
python gen_images.py \
  --network=./runs/wc-cv_h200/00000-styleswin-…-cond/best_model.pt \
  --outdir=./generated/256x256 \
  --trunc=0.7 \
  --classes 0,1,2 \
  --samples-per-class 1000 \
  --gpus 2 \
  --batch-gpu 32
```

The `sbatch/generate_256x256.sbatch` (… `512`, `1024`) scripts wrap this per resolution.

### Class index → grain class

The `--classes` integer selects a grain morphology. StyleSwin consumes the same
`imagenet_9to4_*` archives as DiffiT (built by `dataset_tool_for_imagenet.py`, which derives
labels from the **alphabetical** class-folder order), so it uses the **DiffiT** index convention:

| index | grain class | morphology |
|---|---|---|
| `0` | `Ultra_Co11` | small grain (мелкие зёрна) |
| `1` | `Ultra_Co25` | medium grain (средние зёрна) |
| `2` | `Ultra_Co6_2` | large grain (крупные зёрна) |

```{warning}
**The index order differs from the SAN generator.** san-v2 numbers the same grains as
`0 → Ultra_Co25`, `1 → Ultra_Co11` (indices 0 and 1 swapped relative to StyleSwin/DiffiT;
`2 → Ultra_Co6_2` matches — see {doc}`san_v2`). So the same index does **not** generate the same
morphology across StyleSwin and SAN. When comparing a generator against the real classes — e.g.
in the `co_angles` notebooks — remap per model with combra's `CLASS_MAP`, which
{py:func}`combra.angles.resolve_overlay_rows` and {py:func}`combra.angles.build_overlay_grid`
consume as `gen_name_for_mode`.
```
