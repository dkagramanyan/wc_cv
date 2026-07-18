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
`--init-weights`** (random initialisation). DiffiT-v2 can also be trained
progressively (low → high resolution) — RoPE-2D lets a 256² checkpoint's EMA
weights warm-start a higher-resolution stage via `--init-weights` (weights only,
fresh optimizer). There is no `--resume`: runs go start-to-finish (see
Checkpoints).

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

   `diffit-prepare-data` is a click group; the `convert` subcommand builds the zip:

   ```bash
   diffit-prepare-data convert --source ./raw_wc_co_images \
       --dest ./datasets/wc_co_256x256.zip \
       --resolution 256x256 --transform center-crop
   ```

   The transform set is `center-crop` / `center-crop-wide` / `center-crop-dhariwal`.
   The `.zip`'s `dataset.json` carries both the integer `labels` and an
   index-aligned `class_names` list (§5 label contract), where the integer label
   is the class-folder index in **alphabetical** order. `class_names` then travels
   into every checkpoint and every generated `.h5`, so grain-class identity is
   recoverable without a `CLASS_MAP`. Grayscale SEM images are converted to RGB
   **at build time** (the loader asserts 3 channels rather than converting
   silently). If you point `--data` at a plain directory instead, the class is the
   immediate parent folder name (alphabetical). A dataset with any unlabeled image
   is rejected rather than silently demoted to unconditional.

4. **Launch training.** The required flags are `--outdir`, `--cfg`, `--data`,
   `--gpus`, `--batch-gpu`; the presets are `diffit-256`, `diffit-512`,
   `diffit-1024`:

   ```bash
   # 256² from scratch (no --init-weights → random init)
   diffit-train --outdir=./training-runs \
       --cfg=diffit-256 \
       --data=./datasets/wc_co_256x256.zip \
       --gpus 2 \
       --batch-gpu 96
   ```

   The total batch is `batch-gpu × gpus × grad-accum`. Precision is chosen with
   `--precision {fp32,fp16,bf16}` (default `bf16`); boolean flags take an explicit
   value (`--tf32 True`, `--bench True`, `--mirror False`, `--cache-in-ram True`).

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

Each higher-resolution stage warm-starts from the previous stage's EMA snapshot
via `--init-weights` (weights only, fresh optimizer — not a resume):

```bash
# 256² → 512² finetune
diffit-train --outdir=./training-runs \
    --cfg=diffit-512 \
    --data=./datasets/wc_co_512x512.zip \
    --gpus 2 --batch-gpu 64 \
    --init-weights ./training-runs/00000-diffit-256-*/diffit-snapshot-000400-inference.pt \
    --lr 5e-5 --lr-warmup 500 --kimg 100000
```

### Checkpoints

There is exactly one checkpoint kind — no resume, no best-model tracking, no
separate final checkpoint:

| File | Contents | Cadence |
| --- | --- | --- |
| `diffit-snapshot-<kimg>-inference.pt` | EMA weights only + self-describing metadata (`n_classes`, `resolution`, `class_names`, `cur_nimg`) | every `--snap` ticks **and always at the last tick**; only the newest `--snapshot-keep-last` (default 3, `0` = keep all) are kept |

Only EMA weights ever touch disk — raw model weights and optimizer state are
never saved. Every snapshot is written **atomically** (temp file + `os.replace`),
so a snapshot present under its final name is always complete, and the last tick
always snapshots, so the newest snapshot *is* the final model. Runs are
unrecoverable by design (a crash or SLURM walltime kill cannot be resumed) — size
`--kimg` (or split into progressive stages via `--init-weights`) so a run fits its
job's time limit. Pick the best checkpoint post-hoc from `stats.jsonl` against the
snapshot history (set `--snapshot-keep-last 0` to keep them all).

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
({py:func}`combra.metrics.fid_features` + {py:func}`combra.metrics.frechet_from_features`
and the `cmmd_*` / `fd_dinov2_*` analogues) and the pooled-angle extractor
({py:func}`combra.metrics.images_to_pooled_angles`) — and is numerically identical to the
single-GPU `compute_all_metrics` path. Sharding the angle extraction this way also
fixes a multi-GPU hang: when it ran rank-0-only over the full gathered batch, at
512²/1024² it could take longer than NCCL's watchdog while the other ranks idled,
aborting the job.

**Sample count.** `--num-fid-samples` (default 10000) governs the number of fakes
generated each tick for both the combra and the Inception path. The combra
reference is the **whole training set** by default; `--combra-ref-count N` caps it
to a **seeded random subset** of `N` reals (never the first N — dataset zips are
class-sorted, so a first-N slice would be class-biased). The `10k` suffix in the
metric keys (`Metrics/combra_fid10k`, …) is literal and does not change with
`--num-fid-samples`. Set `--num-fid-samples 0` to disable eval entirely.

All per-tick scalars are written to `stats.jsonl` (one JSON line per tick,
scalar rows only) and mirrored to TensorBoard under the `Loss/*`,
`LearningRate/*`, `Timing/*`, `Resources/*`, `Metrics/*` and `Fakes` namespaces,
with the global step counted in images (`cur_nimg`).

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

For WC-Co work, the in-training combra metrics and `diffit-gen-images` → the
angle pipeline are the evaluation path. A legacy FID-50K evaluation of a trained
checkpoint on bulk-sampled `.npz` batches is still available via the standalone
evaluator, but `scripts.sample` is a **legacy** bulk-`.npz` sampler outside the v2
generation contract (it carries no contract guarantees):

```bash
torchrun --nproc_per_node=4 -m scripts.sample \
    --model-path ./training-runs/00000-diffit-256-*/diffit-snapshot-000400-inference.pt \
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

Generate images from a trained checkpoint with `diffit-gen-images`. In the
default `hdf5` save mode each GPU worker writes a per-rank shard and rank 0 merges
them into `<desc>.h5` (the RankH5Writer layout the angle pipeline consumes, with
`format="generated_images_shard"` / `schema_version=1` and `class_names` stamped
in); `--save-mode dir` writes per-class PNGs plus a `classes.json`. `--gpus N`
self-spawns one worker per GPU (no `torchrun`). Every image has its own seed
(`base-seed + class·samples_per_class + idx`), so any subset is reproducible in
isolation, and the merge hard-fails if any shard is incomplete:

```bash
diffit-gen-images \
    --network ./training-runs/00000-diffit-256-*/diffit-snapshot-000400-inference.pt \
    --outdir ./generated/ \
    --image-size 256 \
    --samples-per-class 1000 \
    --classes Ultra_Co11,Ultra_Co25,Ultra_Co6_2 \
    --cfg-scale 4.4 \
    --sampler ddim --steps 250 \
    --gpus 4 \
    --batch-gpu 32 \
    --desc wc_co_256
```

`--network` (alias `--model-path`) takes the checkpoint; `--steps` (alias
`--num-sampling-steps`) sets the sampler steps; `--batch-gpu` is the per-GPU batch;
`--classes` accepts class names or indices, validated against the checkpoint's
`class_names` / `n_classes` metadata.

### Class index → grain class

The `--classes` selector accepts class **names** directly (validated against the
checkpoint), so the index↔grain mapping no longer has to be memorised. Under the
alphabetical convention now written into every new dataset's `class_names` the
indices map as:

| index | grain class | morphology |
|---|---|---|
| `0` | `Ultra_Co11` | small grain (мелкие зёрна) |
| `1` | `Ultra_Co25` | medium grain (средние зёрна) |
| `2` | `Ultra_Co6_2` | large grain (крупные зёрна) |

```{note}
**New (post-contract) runs are self-describing.** `diffit-prepare-data` writes an
index-aligned `class_names` into `dataset.json`; it flows into every checkpoint
and every generated `.h5`, so combra matches by name and no `CLASS_MAP` is
needed.
```

```{warning}
**Legacy checkpoints do NOT carry names and most likely do NOT follow this
table.** Runs trained before the label contract took zip labels **verbatim**, and
the on-disk `imagenet_9to4_*` archives the real runs consumed (e.g. run 00017, per
its `training_options.json`) carry labels in **SAN's swapped order**
(`0 → Ultra_Co25`, `1 → Ultra_Co11`, `2 → Ultra_Co6_2` — see {doc}`san_v2`).
Classify each legacy checkpoint by the dataset path in its `training_options.json`
before assuming either convention, and do **not** apply a `CLASS_MAP` remap
(via {py:func}`combra.angles.resolve_overlay_rows` /
{py:func}`combra.angles.build_overlay_grid` `gen_name_for_mode`) to a checkpoint
trained on those archives — it would introduce the very swap it is meant to fix.
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
`Ultra_Co6_2` is last under both rules, so it stays `2`. **Which rule a legacy
checkpoint follows depends on which zip it trained on** — the shipped
`imagenet_9to4_*` archives carry the SAN-order labels, whichever tool nominally built
them. For those legacy artifacts, which recorded neither `class_names` nor original
filenames, the correspondence has to be recovered per run from
`training_options.json` and pinned in combra's `CLASS_MAP`. New DiffiT-v2 runs avoid
the problem entirely: `class_names` is written into the zip, the checkpoint and every
generated `.h5`, so identity travels with the artifact.
