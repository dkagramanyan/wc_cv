# diffit model

[**DiffiT-v2**](https://github.com/dkagramanyan/DiffiT-v2) is a performance
refresh of NVlabs' DiffiT (Diffusion Vision Transformers) used to generate WC-Co
microstructure SEM images. Its training evaluation is wired into combra: every
evaluation tick scores generated samples with combra's sharded split-API
metrics — numerically equivalent to
{py:func}`combra.metrics.compute_all_metrics`.

The combra integration is **optional** — DiffiT does not depend on combra. The
import is guarded, so training runs unchanged when combra is not installed. To
enable the metrics, install combra alongside DiffiT:

```bash
pip install combra            # or: pip install 'diffit[combra]'
```

## Training

`diffit-train` trains the model; **"from scratch" simply means launching without
`--resume`** (random initialisation). DiffiT-v2 can also be trained progressively
(low → high resolution) — RoPE-2D lets a 256² checkpoint be re-used at higher
resolutions.

### From scratch

1. **Install combra and DiffiT-v2** into the *same* environment. Both are
   source-only packages (neither is on PyPI), and combra is **optional** — DiffiT's
   import of it is guarded, so training runs unchanged without it. To enable the
   combra metrics, install combra first (so DiffiT can find it), then DiffiT-v2:

   ```bash
   pip install -e /path/to/combra   # the combra checkout (or wc_cv/combra submodule)
   pip install -e .                 # run in the DiffiT-v2 checkout
   ```

   Without combra, just run the second line. This puts the `diffit-train`,
   `diffit-prepare-data`, `diffit-download-models`, `diffit-sample`,
   `diffit-gen-images` and `diffit-eval` commands on your `PATH`.

2. **(Optional) prefetch model weights** — handy for offline / cluster nodes.
   Training needs the Stable-Diffusion VAE (plus InceptionV3 for FID/IS); the VAE
   auto-downloads on first run, or fetch everything up front:

   ```bash
   diffit-download-models
   ```

3. **Prepare the dataset.** `diffit-prepare-data` center-crops and resizes a folder
   of images into a `.zip` the trainer reads:

   ```bash
   diffit-prepare-data --source ./raw_wc_co_images \
       --dest ./datasets/wc_co_256x256.zip \
       --resolution 256x256 --transform center-crop
   ```

   Class labels come from a `dataset.json` inside the `.zip`
   (`{"labels": [["img.png", classID], ...]}`); if you instead point `--data` at a
   plain directory, the class is the filename prefix before `_` (e.g. `wc_001.png` →
   class `wc`). Grayscale SEM images are supported, and single-class data is fine
   (all files share one prefix / label).

4. **Launch training.** The required flags are `--outdir`, `--cfg`, `--data`,
   `--gpus`, `--batch-gpu`; the presets are `diffit-256`, `diffit-512`,
   `diffit-1024`:

   ```bash
   # 256² from scratch (no --resume → random init)
   diffit-train --outdir=./training-runs \
       --cfg=diffit-256 \
       --data=./datasets/wc_co_256x256.zip \
       --gpus 2 \
       --batch-gpu 96
   ```

### Quick example (single-GPU smoke test)

A short run to confirm the whole pipeline works end-to-end before committing to a
long job. It uses one GPU, a tiny `--kimg` budget, and `--num-fid-samples 0` to skip
evaluation entirely (so it needs neither combra nor InceptionV3):

```bash
diffit-train --outdir ./training-runs --cfg diffit-256 \
    --data ./datasets/wc_co_256x256.zip \
    --gpus 1 --batch-gpu 8 \
    --kimg 200 --tick 5 --snap 5 --num-fid-samples 0
```

Watch it in TensorBoard with `tensorboard --logdir ./training-runs`. Add `-n` /
`--dry-run` to print the resolved options and exit without training (see the Dry run
section below). Drop `--num-fid-samples 0` to re-enable the combra/FID eval (DDIM by
default).

### Progressive finetuning

Each higher-resolution stage resumes from the previous stage's `network-final.pt`:

```bash
# 256² → 512² finetune
diffit-train --outdir=./training-runs \
    --cfg=diffit-512 \
    --data=./datasets/wc_co_512x512.zip \
    --gpus 2 --batch-gpu 64 \
    --resume ./training-runs/00000-diffit-256-*/network-final.pt \
    --lr 5e-5 --lr-warmup 500 --kimg 100000
```

### Checkpoints

At every snapshot tick (every `--snap` ticks) the run directory accumulates a
small inference snapshot for history and keeps single, non-growing full
checkpoints for resume:

| File | Contents | Cadence |
| --- | --- | --- |
| `network-snapshot-<kimg>-inference.pt` | G_ema weights only (smallest artifact for `diffit-sample` / `diffit-gen-images`) | every tick; only the newest `--snapshot-keep-last` (default 3) are kept |
| `network-snapshot-latest.pt` | full `{model, ema, opt, scaler?, cur_nimg}` for `--resume` | overwritten in place each tick (skipped under `--save-inference-only`) |
| `best_model.pt` | full checkpoint of the lowest-FID tick (`combra_fid10k`, else `FID`) | rewritten only when FID improves; never pruned |
| `network-final.pt` / `network-final-inference.pt` | full / G_ema-only final model | once at end |

This keeps disk bounded — the full optimizer state lives in exactly one rolling
file rather than one per tick. Resume from whichever full checkpoint you want:
`network-snapshot-latest.pt` for the most recent step, or `best_model.pt` for the
best-scoring one. `--save-inference-only` skips the rolling `network-snapshot-latest.pt`
(so a run writes no repeated optimizer state), but `best_model.pt` and the final
`network-final.pt` are still full checkpoints, so resume — and progressive
finetuning from a previous stage — always work.

During training, at each evaluation tick (every `--snap` ticks) the loop
generates a batch of images from the EMA model by running the configured
reverse-diffusion sampler (DDIM by default — see the Samplers section below) in
VAE latent space, decodes them to pixels, and scores them with the combra suite
(when combra is installed) — the sharded equivalent of
`compute_all_metrics(reals, fakes, image_metrics=True)` (see below). All returned metrics — angle-Wasserstein `w1`, `w2`, `circular_w1`,
`circular_w2`, the bimodal-Gaussian relative errors
`mu1`/`mu2`/`sigma1`/`sigma2`/`amp1`/`amp2`, and the image-feature metrics `fid`,
`cmmd`, `fd_dinov2` — are logged to TensorBoard under `Metrics/combra_*`, written
to `stats.jsonl`, and printed to the run log. Enabling the combra metrics (the
default `--combra-metrics=true`) **replaces** DiffiT's own Inception suite: when
they are on, IS / FID / sFID / Precision / Recall are **not** computed (so FID is
not measured twice); pass `--combra-metrics=false` to compute that Inception
suite instead. Metrics whose optional backends are unavailable (e.g. no network
to fetch DINOv2 weights) are recorded as `nan`; the angle metrics always come
back. The reference batch is scored once and cached, so only the generated-side
work is repeated each tick.

On a multi-GPU run **all** the per-image extraction work is spread across every
rank — both the image-feature metrics (`fid`, `cmmd`, `fd_dinov2`) and the
angle-density / Gaussian-fit metrics. Each rank generates its own shard of the
fakes and, from that shard, extracts the CLIP / DINOv2 / InceptionV3 features and
pools the vertex angles; the feature rows and the pooled-angle arrays are gathered
to rank 0, where the Fréchet / MMD distances and the angle Wasserstein / Gaussian
metrics are computed once against the reference. The reference side is sharded the
same way and extracted **once before training** (each rank processes its
deterministic slice of the reals), then cached on rank 0 — so no reference work or
collective recurs per tick. This uses combra's split APIs — the feature halves
({py:func}`combra.metrics.fid_features` + {py:func}`combra.metrics.fid_from_features`
and the `cmmd_*` / `fd_dinov2_*` analogues) and the angle halves
({py:func}`combra.metrics.images_to_pooled_angles` +
`angle_density_metrics_from_pooled`) — and is numerically identical to the
single-GPU `compute_all_metrics` path. Sharding the angle extraction this way also
fixes a multi-GPU hang: when it ran rank-0-only over the full gathered batch, at
512²/1024² it could take longer than NCCL's watchdog while the other ranks idled,
aborting the job.

**Sample count.** When combra is **not** installed, the eval generates
`--num-fid-samples` images (default 10000), unchanged. When combra **is**
installed, each evaluation tick instead scores the **whole training dataset**:
the reference is every real image used once, and DiffiT generates a matching
number of images. This makes the combra metrics a full-dataset measurement rather
than a 10k subsample. (Set `--num-fid-samples 0` to disable eval entirely.)

To keep the larger per-tick generation affordable, the EMA model's sampling
forward is `torch.compile`d, which speeds up the reverse-diffusion latent
generation that dominates evaluation time.

### Samplers

The reverse-diffusion sampler used for evaluation **and** inference is selectable.
All three reuse the same trained model — they only differ in how the reverse
process is integrated:

- **`ddim`** *(default)* — deterministic DDIM. Reproducible (low-variance) metric
  curves and good quality at moderate step counts; the recommended default for the
  training-time eval signal and for final inference.
- **`dpm++`** — DPM-Solver++(2M). Fastest (near-converged quality in ~25 steps);
  use it for the cheapest possible training-time eval.
- **`ddpm`** — stochastic ancestral sampling. Most faithful / most diverse at high
  step counts (~250), but the slowest.

During training, pick the eval sampler with `--eval-sampler` and its step count
with `--eval-sampling-steps` (per-sampler defaults: `dpm++`=25, `ddim`=100,
`ddpm`=250):

```bash
diffit-train --outdir=./training-runs --cfg=diffit-256 \
    --data=./datasets/wc_co_256.zip --gpus 2 --batch-gpu 96 \
    --eval-sampler ddim --eval-sampling-steps 100
```

The standalone `scripts.sample` and `diffit-gen-images` take the same choice via
`--sampler` (which replaces the former `--use-ddim` flag).

### Dry run

To validate the config and data wiring without starting a real run, pass `-n` /
`--dry-run` — it prints the resolved training options (resolution, batch, sample
count, etc.) and exits before training:

```bash
diffit-train --outdir=./training-runs \
    --cfg=diffit-256 \
    --data=./datasets/wc_co_256x256.zip \
    --gpus 1 --batch-gpu 4 \
    -n
```

## Evaluation

A full FID-50K evaluation of a trained checkpoint uses the standalone evaluator
on bulk-sampled `.npz` batches:

```bash
torchrun --nproc_per_node=4 -m scripts.sample \
    --model-path ./training-runs/00000-diffit-256-*/network-final.pt \
    --outdir ./samples/256 --image-size 256 \
    --cfg-scale 4.4 --num-samples 50000 \
    --sampler ddim --num-sampling-steps 250 --cfg-cond

# reference: a folder of real WC-Co images (--ref-path) or a prebuilt .npz batch (--ref-batch)
diffit-eval \
    --ref-path ./raw_wc_co_images \
    --sample-batch ./samples/256/samples_50000x256x256x3.npz
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

`image_metrics=True` is what adds the `fid`/`cmmd`/`fd_dinov2` keys; without it,
only the angle-Wasserstein and bimodal-Gaussian metrics come back. Batches may be
numpy arrays or torch tensors in `NCHW`/`NHWC`, with pixel values in `uint8`,
`[0, 1]`, or `[-1, 1]`.

## Inference

Generate individual images from a trained checkpoint with `diffit-gen-images`.
Images are written per class into `class_<id>/...`; pass `--gpus` (or launch with
`torchrun`) to distribute generation:

```bash
diffit-gen-images \
    --model-path ./training-runs/00000-diffit-256-*/network-final.pt \
    --outdir ./generated/ \
    --image-size 256 \
    --samples-per-class 1000 \
    --classes 0,1,2 \
    --cfg-scale 4.4 \
    --sampler ddim --num-sampling-steps 250 \
    --gpus 0,1,2,3 \
    --batch-size 32
```

### Class index → grain class

The `--classes` / `--class-idx` integer selects a grain morphology. Under the
**alphabetical convention of `dataset_tool_for_imagenet.py`** the indices map
as:

| index | grain class | morphology |
|---|---|---|
| `0` | `Ultra_Co11` | small grain (мелкие зёрна) |
| `1` | `Ultra_Co25` | medium grain (средние зёрна) |
| `2` | `Ultra_Co6_2` | large grain (крупные зёрна) |

```{warning}
**The trained checkpoints most likely do NOT follow this table.** The table
is the dataset *tool's* convention, but training takes zip labels
**verbatim** (the startup class probe remaps sorted ints to contiguous ints —
an identity map for `0,1,2`), and the on-disk `imagenet_9to4_*` archives the
real runs consumed (e.g. run 00017, per its `training_options.json`) carry
labels in **SAN's swapped order** (`0 → Ultra_Co25`, `1 → Ultra_Co11`,
`2 → Ultra_Co6_2` — see {doc}`san_v2`). Classify each checkpoint by the
dataset path in its `training_options.json` before assuming either
convention, and do **not** apply a `CLASS_MAP` remap
(via {py:func}`combra.angles.resolve_overlay_rows` /
{py:func}`combra.angles.build_overlay_grid` `gen_name_for_mode`) to a
checkpoint trained on those archives — it would introduce the very swap it
is meant to fix.
```

**Why the order differs.** The dataset stores only an integer label per image — never
the `Ultra_Co*` name — and the two pipelines derive that integer by *different rules*:

- **DiffiT** (`dataset_tool_for_imagenet.py`) assigns each class an integer from the
  **alphabetical order of the class-folder names** (`sorted(train_dirs)` then
  `enumerate`): `Ultra_Co11 → 0`, `Ultra_Co25 → 1`, `Ultra_Co6_2 → 2`.
- **SAN** (`dataset_tool.py`) copies the label **verbatim from the preprocessed
  source's `dataset.json`**, which was written `Ultra_Co25 → 0`, `Ultra_Co11 → 1`,
  `Ultra_Co6_2 → 2` — an order that is **not** alphabetical.

Because the source `dataset.json` lists `Co25` before `Co11` while the folder sort puts
`Co11` first, the two conventions disagree on labels 0 and 1 — the `Co11`↔`Co25` swap.
`Ultra_Co6_2` is last under both rules, so it stays `2`. **Which rule a checkpoint
follows depends on which zip it trained on** — the shipped `imagenet_9to4_*` archives
carry the SAN-order labels, whichever tool nominally built them. Since neither pipeline
records the grain name downstream (the zips hold neither `class_names` nor original
filenames, and generated h5s carry only `class_0/1/2`), the correspondence has to be
recovered per run from `training_options.json` and pinned in combra's `CLASS_MAP`.
