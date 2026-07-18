# edm2 model

[**EDM2-v2**](https://github.com/dkagramanyan/edm2-v2) is a refresh of NVlabs'
EDM2 ("Analyzing and Improving the Training Dynamics of Diffusion Models") used to
generate WC-Co microstructure SEM images. Like {doc}`DiffiT-v2 <diffit>`, its
training evaluation is wired into combra: every snapshot tick scores generated
samples with combra's sharded split-API metrics, computed across all GPU ranks
(numerically equivalent to {py:func}`combra.metrics.compute_all_metrics`).

```{note}
**EDM2-v2 implements the v2 model-API convention** specified in
{doc}`models_api_proposal` (unified CLI, EMA-only `.pt` inference snapshots, HDF5
class-batch generation, raw-pixel combra reference). The pages below describe the
**current** EDM2-v2 CLI; the other three repos have not yet adopted the convention.
```

The combra integration is **optional** — EDM2 does not depend on combra. The
import is guarded, so training runs unchanged when combra is not installed. To
enable the metrics, install EDM2 with the `[combra]` extra — see
[Installation](#from-scratch) below.

## Training

`edm2-train` (equivalently `python train_edm2.py`) trains the model; it spawns
its own per-GPU worker processes via `--gpus`, so **no `torchrun` is needed for
training**. EDM2 operates in the Stable-Diffusion VAE latent space, so the same
pipeline serves 256 / 512 / 1024 resolutions (32² / 64² / 128² latents).

```{warning}
**Runs are not resumable by design.** A crash or SLURM walltime kill cannot be
continued, and every launch allocates a **fresh** run id. Size `--kimg` (or split
resolution stages) so a run fits its job's time limit. Two guarantees make this
safe: snapshots are written **atomically** (a snapshot present under its final name
is always complete), and the **last tick always snapshots**, so a finished run
always ends in a usable model.
```

### From scratch

1. **Install EDM2-v2.** combra is **optional** — EDM2's import of it is guarded,
   so training runs unchanged without it. Enable the metrics with the `[combra]`
   extra, which pulls combra from its **private** GitHub repo over `git+https`
   (log in once with `gh auth login` → github.com → HTTPS and `pip` inherits the
   credential helper):

   ```bash
   pip install -e '.[combra]'       # run in the edm2-v2 checkout; omit [combra] to skip metrics
   ```

   This puts the `edm2-train`, `edm2-prepare-data`, `edm2-download-models`,
   `edm2-sample`, `edm2-gen-images`, `edm2-eval` and `edm2-compare-samplers`
   commands on your `PATH`. `pyproject.toml` is the only dependency declaration —
   there is no `requirements.txt` and no Hydra entry point.

2. **(Optional) prefetch model weights** — handy for offline / cluster nodes.
   Training needs the Stable-Diffusion VAE (plus InceptionV3 / CLIP / DINOv2 for the
   combra metrics); fetch everything up front with `edm2-download-models`.

3. **Prepare the dataset.** `edm2-prepare-data convert` center-crops a folder of
   images into a `.zip` the trainer reads, writing index-aligned `class_names`
   (alphabetical folder order) into `dataset.json`:

   ```bash
   edm2-prepare-data convert --source ./raw_wc_co_images \
       --dest ./datasets/wc_co_256x256.zip \
       --resolution 256x256 --transform center-crop-dhariwal
   ```

   Class labels come from the top-level directory names (alphabetical → integer);
   grayscale SEM images are converted to RGB at build time.

4. **Launch training.** The required flags are `--outdir` and `--data`; select the
   architecture with `--cfg` (`edm2-img256-s`, `edm2-img512-s`, `edm2-img1024-s`).
   The total batch is `--batch-gpu × --gpus × --grad-accum`; progress is counted in
   **kimg and ticks** (`--kimg`, `--tick`, `--snap`):

   ```bash
   # 256² from scratch, 2 GPUs
   edm2-train \
       --outdir=./runs \
       --cfg=edm2-img256-s \
       --data=./datasets/wc_co_256x256.zip \
       --gpus 2 --batch-gpu 64 \
       --tick 128 --snap 64
   ```

   Each run gets a **fresh** directory under `--outdir`, named
   `<id:05d>-<cfg>-gpus<N>-batch<B>[-desc]` (e.g.
   `runs/00000-edm2-img256-s-gpus2-batch128/`).

   Each snapshot tick (and the last tick) writes EMA-only **`.pt` state-dict**
   inference snapshots `edm2-snapshot-<kimg:06d>-<ema_std>-inference.pt` (one per EMA
   std, e.g. `edm2-snapshot-000200-0.100-inference.pt`), pruned to the newest
   `--snapshot-keep-last` (default 3). Every snapshot carries self-describing
   `{n_classes, resolution, class_names, cur_nimg}` metadata, so loading rebuilds the
   model from current code. There is no resume checkpoint, no `best_model.pt` — the
   newest snapshot *is* the final model.

### Quick example (single-GPU smoke test)

A short run to confirm the pipeline works end-to-end. It uses one GPU, a tiny
`--kimg`, and `--num-fid-samples 0` to skip evaluation entirely (so it needs neither
combra nor the metric backbones):

```bash
python train_edm2.py --outdir ./runs --cfg edm2-img256-s \
    --data ./datasets/wc_co_256x256.zip \
    --batch-gpu 8 --kimg 200 --tick 20 --snap 1 --num-fid-samples 0
```

Watch it in TensorBoard with `tensorboard --logdir ./runs`. Add `-n` /
`--dry-run` to print the resolved options and exit without training. Drop
`--num-fid-samples 0` to re-enable the combra eval (DPM-Solver++(2M) at 25 steps by
default).

### Precision, TF32 and mirror

`--precision {fp32,fp16,bf16}` selects the training precision (default `fp16`);
`--tf32 True/False` (default `True`) controls the cuDNN / matmul TF32 paths.
`--mirror True/False` (default `False`) is a stochastic per-item horizontal flip in
the **training** loader only — the eval and combra-reference loaders never flip.

### Logging

Every run directory contains exactly these artifacts:

| file | contents |
| --- | --- |
| `training_options.json` | the resolved launch config |
| `<id>-<cfg>-gpus<N>-batch<B>[-desc].log` | rank-0 console transcript, one `[YYYY-MM-DD HH:MM:SS]` timestamp per line |
| `stats.jsonl` | machine-readable scalars + combra metrics — **scalar rows only** |
| `events.out.tfevents.*` | TensorBoard scalars, image grids and text (rank 0 only; run name as `filename_suffix`) |
| `reals.png`, `fakes_init.png`, `fakes<kimg>.png` | class-sorted sample grids (reals are raw dataset pixels) |

There is no `progress.csv` / `progress.json` and no per-rank log file: the console
transcript is rank-0-only. Watch a run with `tensorboard --logdir ./runs`.

### Progressive training

Each higher-resolution stage is trained independently with the matching preset
(`edm2-img512-s`, `edm2-img1024-s`) and dataset zip. Because runs are not resumable,
there is no in-place stage continuation; retrain per resolution.

During training, at each snapshot tick the loop generates a batch of images from
the EMA model by running the configured reverse-diffusion sampler (DPM-Solver++(2M)
at 25 steps by default — see the Samplers section below) in VAE latent space,
decodes them to pixels, and scores them with the combra suite (when combra is
installed) — the sharded equivalent of `compute_all_metrics(reals, fakes)`.
All returned metrics — angle-Wasserstein `w1`, `w2`, `circular_w1`, `circular_w2`,
the bimodal-Gaussian relative errors `mu1`/`mu2`/`sigma1`/`sigma2`/`amp1`/`amp2`,
and the image-feature metrics `fid`, `cmmd`, `fd_dinov2` — are logged to
TensorBoard under `Metrics/combra_*`, written to `stats.jsonl`, and printed to the
run log. Metrics whose optional backends are unavailable (e.g. no network to fetch
DINOv2 weights) are recorded as `nan`; the angle metrics always come back.

On a multi-GPU run **all** the per-image extraction work is spread across every
rank. Each rank generates its own shard of the fakes and, from that shard, extracts
the CLIP / DINOv2 / InceptionV3 features and pools the vertex angles; the feature
rows and the pooled-angle arrays are gathered to rank 0, where the Fréchet / MMD
distances and the angle metrics are computed once against the reference. The
reference side is the **raw dataset pixels** (never VAE round-tripped), extracted
**once before training** (each rank processes its deterministic slice of the reals),
then cached on rank 0. This uses combra's split APIs
({py:func}`combra.metrics.fid_features` + {py:func}`combra.metrics.frechet_from_features`
and the `cmmd_*` / `fd_dinov2_*` analogues, plus
{py:func}`combra.metrics.images_to_pooled_angles`) — numerically identical to the single-GPU
`compute_all_metrics` path.

**Sample count.** Each combra eval scores `--num-fid-samples` generated images
(default 10000) against the training set as the real reference (`--combra-ref-count`
caps the reference to a **seeded random** subset, or `--num-fid-samples 0` disables
eval entirely).

### Samplers

The reverse-diffusion sampler used for evaluation **and** inference is selectable,
all in the native EDM σ-space on the `net(x, σ, labels)` denoiser:

- **`dpm++`** *(default, 25 steps)* — DPM-Solver++(2M) in log-σ space. 2nd-order
  accurate at **one** denoiser evaluation per step, the cheapest route to
  near-converged quality.
- **`edm`** — EDM 2nd-order Heun sampler; the only one supporting stochasticity
  (via `S_churn`). Costs 2 denoiser evaluations per step.
- **`ddim`** — deterministic DDIM (η=0), the first-order EDM step (≡ `euler`).
- **`euler`** — 1st-order deterministic Euler.

Pass `--eval-sampler` / `--eval-sampling-steps` during training; `edm2-gen-images`
takes the same choice via `--sampler` / `--steps`. Use `edm2-compare-samplers` to
confirm the step count on your own data ({doc}`sampler_comparison`).

## Generation

`edm2-gen-images` generates images **per class** into the HDF5 layout the wc_cv
angle pipeline (`co_angles/generate_class_samples.py`,
{py:meth}`combra.data.PobeditDataset.generate_angles`) consumes directly. `--gpus`
self-spawns per-GPU workers (no torchrun); `--classes` accepts indices, ranges, or
class names:

```bash
edm2-gen-images \
    --network=./runs/00000-.../edm2-snapshot-000200-0.100-inference.pt \
    --outdir=./generated/256 \
    --classes=0,1,2 --samples-per-class=1000 \
    --gpus=2 --batch-gpu=32 \
    --save-mode=hdf5 --sampler=dpm++ --steps=25
```

Each rank writes a shard `shards/rank_NNN.h5` in the `RankH5Writer` layout
(`class_<c>/images|seeds`, images stored as **uint8 NHWC**), merged by rank 0 into
`<desc>.h5` carrying `format="generated_images_shard"`, `schema_version=1` and
`class_names`. The merge **hard-fails** if any shard is incomplete, so a crashed
generation run never feeds zero-filled (black) slots downstream. The per-image seed
is `base + class·samples_per_class + idx`, so any subset of the output is
reproducible in isolation. `--save-mode=dir` writes
`class_<c>/idx_<i:06d>_seed_<s>.png` + a `classes.json` manifest instead.

### Conditioning

EDM2 is class-conditional; conditioning is expressed with a few composable methods:

- **One-hot class labels** — the model is built with `label_dim = <num classes>`
  (from the dataset) and conditioned via `net(x, σ, class_labels)`. Train
  conditional with `--cond=True` (default); an unlabeled dataset + `--cond=False`
  gives an unconditional model.
- **Specific classes** — `edm2-gen-images --classes <spec>` selects which grain
  classes to generate.
- **Classifier-free / auto-guidance** — steer generation with a second *guiding*
  network via `--gnet` and `--guidance` (strength `> 1`). `--guidance 1` (default)
  disables it. Honored by every sampler and by the training-time combra eval.

### Class index → grain class

`edm2-prepare-data convert` assigns each class an integer from the **alphabetical
order of the class-folder names** and records the names as `class_names` in
`dataset.json`; those names travel into every checkpoint and every generated h5, so
combra matches generated images to grain classes by **name**:

| index | grain class | morphology |
|---|---|---|
| `0` | `Ultra_Co11` | small grain (мелкие зёрна) |
| `1` | `Ultra_Co25` | medium grain (средние зёрна) |
| `2` | `Ultra_Co6_2` | large grain (крупные зёрна) |

```{warning}
**The table holds only for zips EDM2's own tool built from class folders.**
Training takes zip labels **verbatim**, and the shared `imagenet_9to4_*`
archives (consumed by the real DiffiT and StyleSwin runs) carry labels in
**SAN's swapped order** (`0 → Ultra_Co25`, `1 → Ultra_Co11`) — a zip's
provenance, not the repo, decides the convention. Classify each checkpoint
by the dataset path in its `training_options.json` before comparing across
models or remapping with combra's `CLASS_MAP`. New zips built with the current
`edm2-prepare-data` carry `class_names`, so this ambiguity does not arise for them.
```
