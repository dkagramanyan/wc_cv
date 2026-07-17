# StyleSwin-v2 model

[**StyleSwin-v2**](https://github.com/dkagramanyan/StyleSwin-v2) is a fork of Microsoft's
[StyleSwin](https://github.com/microsoft/StyleSwin) (a Swin-transformer StyleGAN) specialised
for generating WC-Co microstructure SEM images. Upstream StyleSwin is **unconditional**; this
fork adds **class-conditional** generation over the three grain classes and **implements the
shared generative-model API convention** ({doc}`models_api_proposal`): the console commands,
training flags, checkpoint format and generated-artifact layout match the sibling repos, and its
output feeds the wc_cv angle pipeline with zero conversion. On every **snapshot** tick it scores
generated samples with combra's sharded split-API metrics (numerically equivalent to
{py:func}`combra.metrics.compute_all_metrics`) and logs the results to **both** TensorBoard and
`stats.jsonl` — the same integration as {doc}`san_v2` and {doc}`diffit`.

Conditioning reuses the san-v2 techniques: the generator embeds the one-hot label into the
mapping network (2nd-moment normalised alongside `z`), and the discriminator adds a Miyato &
Koyama projection term to StyleSwin's unchanged logistic + R1 loss. Enable it with
`--cond True`; `n_classes` and `class_names` are read from the dataset's `dataset.json`
(`n_classes = 0` keeps the unconditional path). The generator/discriminator update math is
unchanged from upstream — the conditioning and the shared-convention tooling wrap around it.

The combra integration is **optional** and controlled by `--combra-metrics` (default `true`).
The import is guarded, so training runs unchanged when combra is not installed — but if
`--combra-metrics=true` and the package is missing, training emits a **warning** at startup (and
skips the metrics) so it is never silently ignored. Install it via the `[combra]` extra — see
[Installation](#installation).

## Installation

StyleSwin's custom CUDA ops (`fused_act` / `upfirdn2d`) are JIT-compiled by torch on first
import, so the env needs `nvcc` and `ninja`. `nvcc` comes from the system CUDA module
(`module load CUDA/13.1`, loaded by the `sh/` scripts); `ninja` — torch's build backend — is
installed from conda (a pip `ninja` conflicts with conda's). torch comes from the CUDA wheel index:

```bash
conda create -n styleswin python=3.12 -y && conda activate styleswin
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
conda install anaconda::ninja -y         # torch's JIT build backend
pip install -e .                         # base deps + console scripts, from the repo root
pip install -e '.[combra]'               # optional: combra in-training metrics
```

`pip install -e .` exposes the shared console-script family: `styleswin-train`,
`styleswin-gen-images`, `styleswin-eval`, `styleswin-prepare-data` and
`styleswin-download-models` (which prefetches the InceptionV3 / CLIP / DINOv2 combra backbones for
offline / cluster nodes).

combra lives in a **private** repo, so the `.[combra]` extra clones it over `git+https` and only
succeeds when you are authenticated to GitHub — sign in once with the GitHub CLI
(`gh auth login` → github.com → HTTPS) and `pip` inherits its credential helper.

The `sh/` scripts `module load CUDA/13.1`, derive `CUDA_HOME` from the loaded `nvcc`, set
`TORCH_CUDA_ARCH_LIST=9.0` (H200 / `sm_90`) and force `HF_HUB_OFFLINE=1` so the combra CLIP load
reads the prefetched cache on offline nodes.

## Dataset

Training consumes an **ImageNet-style zip** of `(uint8 RGB image, one-hot label)` — the same
format as san-v2. The zip holds class subfolders of PNGs plus a root `dataset.json`
(`{"labels": [[path, class_idx], ...], "class_names": [...]}`); integer labels are expanded to
one-hot at read time. Following the label contract, the integer label is the class folder's index
in **alphabetical** order, and the class **names travel with the artifact** (`class_names` in
`dataset.json`, copied into every checkpoint and every generated h5). The dataset class also
**asserts 3-channel RGB** — grayscale is converted once at build time, never silently at runtime.
The WC-Co archives are 3-class (2880 images each):

```text
datasets/imagenet_9to4_1024x1024_256x256.zip     # 256px
datasets/imagenet_9to4_1024x1024_512x512.zip     # 512px
datasets/imagenet_9to4_1024x1024_1024x1024.zip   # 1024px
```

Build one from a labelled folder (class = top-level subfolder) with the click group:

```bash
styleswin-prepare-data convert --source /path/to/wc_co_source \
    --dest ./datasets/imagenet_9to4_256x256.zip --transform center-crop --resolution 256x256
```

## Training

`styleswin-train` is the primary entry point — the shared-convention `click` CLI
(`--outdir/--data/--gpus/--batch-gpu/--cfg/--cond/--kimg/--tick/--snap`, the
`--precision/--tf32/--bench` scheme, the single `--mirror` loader-level flip, `--grad-accum`,
`--combra-metrics/--num-fid-samples/--combra-ref-count/--snapshot-keep-last`) plus StyleSwin's own
model flags. StyleSwin builds all layers at the target resolution at once, so **each resolution is
trained independently** (no stage-to-stage resume chain). Pick the resolution with a `--cfg`
preset:

```bash
styleswin-train --outdir=./runs/wc-cv \
        --cfg styleswin-256 \
        --data=./datasets/imagenet_9to4_1024x1024_256x256.zip \
        --gpus=2 --cond True --combra-metrics True --snapshot-keep-last 3 \
        --kimg 25000 --snap 50
```

### Resolution presets (`--cfg`)

The generator and discriminator are **resolution-parametric**, so the per-resolution knobs live
in a single `RESOLUTION_CONFIGS` dict in `train.py`, selected by `--cfg`:

| preset | resolution | batch/GPU | total batch (2 GPUs) |
|---|---|---|---|
| `styleswin-256`  | 256²  | 64 | 128 |
| `styleswin-512`  | 512²  | 32 | 64  |
| `styleswin-1024` | 1024² | 4  | 8   |

Today the presets differ only in the memory-bound batch size. Any explicit CLI flag overrides the
preset, and `--cfg` cross-checks its resolution against the `--data` zip. The total batch is
`batch_gpu × gpus × grad_accum`, and the run directory is named
`<id:05d>-<cfg>-gpus<G>-batch<B>[-desc]` (no dataset name spliced in). On the cluster:

```bash
sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_256.sh   # or 512 / 1024
bash sh/train_256.sh                                                 # same script on a workstation
```

### Checkpoints

The run follows the **checkpoint contract**: exactly one artifact kind,
`network-snapshot-<kimg:06d>-inference.pt`, holding **only `G_ema`** plus self-describing metadata
(`n_classes`, `resolution`, `class_names`, `cur_nimg`, and the `arch` hyperparameters needed to
rebuild the generator). It is written **atomically** (temp file + `os.replace`) every snapshot
tick **and always at the last tick**, so the newest snapshot *is* the final model; history is
pruned to the most recent `--snapshot-keep-last` (default `3`; `0` = keep all). There is **no
resume, no rolling `latest`, and no `best_model.pt`** — pick the best snapshot post-hoc from
`stats.jsonl` (`Metrics/combra_fid10k`). Because runs are unrecoverable by design, size `--kimg`
(or split stages) to fit the job's time limit.

## Metrics

Training runs a **kimg/tick** loop: a per-run `<runname>.log` (rank-0 only), `stats.jsonl`,
TensorBoard events and the `tick … kimg … sec/tick …` status line. Reported **every tick** and
written to `stats.jsonl` + TensorBoard: the generator/discriminator losses and R1 (`Loss/*`),
timing (`Timing/*`), CPU/GPU memory (`Resources/*`) and the effective learning rates
(`LearningRate/G`, `LearningRate/D`). The **combra metrics** are computed **every snapshot tick**
(step-held between); the global step is `cur_nimg`. Each snapshot also logs the class-sorted
`G_ema` sample grid to TensorBoard (tag `Fakes`) alongside the on-disk `fakes<kimg>.png`, and the
run writes `reals.png` + `fakes_init.png` once at startup.

combra uses the **whole training set** as the reference (raw, unflipped uint8 pixels; a
`--combra-ref-count` cap takes a **seeded random** subset, never the first N), scored against
`--num-fid-samples` (default 10 000) images generated from `G_ema` (labels sampled from the
training-set class distribution, latents seeded from `--seed` alone so the metric set is identical
at any `--gpus`). Each snapshot computes **both** the angle-density metrics (Wasserstein `w1`,
`w2`, `circular_w1`, `circular_w2` and the bimodal-Gaussian `mu/sigma/amp` errors) **and** the
image-feature metrics `fid`, `cmmd`, `fd_dinov2` (keys `combra_fid10k`, `combra_cmmd10k`,
`combra_fd_dinov2_10k`; the running best is `combra_fid10k_best`). All are **mirrored into
`stats.jsonl`** (`Metrics/combra_*`) as well as TensorBoard, so post-hoc best-snapshot selection
survives the loss of the tfevents file; a **failed** eval tick clears the row instead of re-logging
the previous tick's values. `styleswin-eval` scores a checkpoint standalone.

On a **multi-GPU** run all per-image extraction is **sharded across ranks** — each rank generates
its own shard of the fakes, extracts the CLIP / DINOv2 / InceptionV3 features and pools the vertex
angles; the feature rows and pooled-angle arrays are gathered to rank 0 for the final distances
against the precomputed reference. This is numerically identical to the single-GPU
`compute_all_metrics(image_metrics=True)` path. Any image metric whose backend is unavailable is
recorded as `nan`; the angle metrics always come back. Pass `--combra-metrics False` (or
`--num-fid-samples 0`) to skip it on runs where you don't want it.

## Inference

Generate samples from a trained checkpoint with `styleswin-gen-images`. `--save-mode hdf5`
(default) writes per-rank shards merged into `<desc>.h5` in the RankH5Writer layout the angle
pipeline consumes (per-class `images` as uint8 NHWC + `seeds`, root `format`/`schema_version`/
`class_names` attributes, a per-sample `written` mask, and a `missing_count` on which the merge
**hard-fails**); `--save-mode dir` writes `class_<c>/idx_<i:06d>_seed_<s>.png` plus a
`classes.json` manifest. `--classes` accepts names or indices, validated against the checkpoint:

```bash
styleswin-gen-images \
  --network=./runs/wc-cv/00000-styleswin-256-gpus2-batch128-cond/network-snapshot-025000-inference.pt \
  --outdir=./generated/256x256 \
  --save-mode hdf5 \
  --classes Ultra_Co11,Ultra_Co25,Ultra_Co6_2 \
  --samples-per-class 1000 \
  --trunc=0.7 \
  --gpus 2 \
  --batch-gpu 32
```

Per-image seeds are deterministic (`seed = base + class·samples_per_class + idx`), so any subset
reproduces in isolation. The `sh/generate_256.sh` (… `512`, `1024`) scripts wrap this per
resolution.

### Class index → grain class

The `--classes` integer selects a grain morphology. StyleSwin consumes the same
`imagenet_9to4_*` archives as DiffiT and takes their labels **verbatim**. Under the
nominal **alphabetical** convention the indices map as:

| index | grain class | morphology |
|---|---|---|
| `0` | `Ultra_Co11` | small grain (мелкие зёрна) |
| `1` | `Ultra_Co25` | medium grain (средние зёрна) |
| `2` | `Ultra_Co6_2` | large grain (крупные зёрна) |

```{warning}
**The trained checkpoints most likely do NOT follow this table.** The on-disk
`imagenet_9to4_*` archives carry labels in **SAN's swapped order**
(`0 → Ultra_Co25`, `1 → Ultra_Co11`, `2 → Ultra_Co6_2` — see {doc}`san_v2`), not the
alphabetical order the build tool nominally produces — and StyleSwin trains on the zip
labels verbatim. Classify each checkpoint by the dataset path in its
`training_options.json` before assuming either convention, and do **not** apply a
`CLASS_MAP` remap (via {py:func}`combra.angles.resolve_overlay_rows` /
{py:func}`combra.angles.build_overlay_grid` `gen_name_for_mode`) to a checkpoint
trained on those archives — it would introduce the very swap it is meant to fix.
Newly built zips (via `styleswin-prepare-data`) record `class_names`, which travel into
every checkpoint and generated h5, so new artifacts are self-describing and this ambiguity
does not recur.
```
