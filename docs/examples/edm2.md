# edm2 model

[**EDM2-v2**](https://github.com/dkagramanyan/edm2-v2) is a refresh of NVlabs'
EDM2 ("Analyzing and Improving the Training Dynamics of Diffusion Models") used to
generate WC-Co microstructure SEM images. Like {doc}`DiffiT-v2 <diffit>`, its
training evaluation is wired into combra: every snapshot tick scores generated
samples with {py:func}`combra.metrics.compute_all_metrics` (computed across all
GPU ranks).

The combra integration is **optional** — EDM2 does not depend on combra. The
import is guarded, so training runs unchanged when combra is not installed. To
enable the metrics, install combra alongside EDM2:

```bash
pip install combra            # or: pip install 'edm2[combra]'
```

## Training

`edm2-train` (equivalently `torchrun … train_edm2.py`) trains the model.
**"From scratch" simply means launching against an empty `--outdir`** (random
initialisation); re-running the same command resumes automatically. EDM2 operates
in the Stable-Diffusion VAE latent space, so the same pipeline serves 256 / 512 /
1024 resolutions (32² / 64² / 128² latents).

### From scratch

1. **Install combra and EDM2-v2** into the *same* environment. Both are
   source-only packages (neither is on PyPI), and combra is **optional** — EDM2's
   import of it is guarded, so training runs unchanged without it. To enable the
   combra metrics, install combra first (so EDM2 can find it), then EDM2-v2:

   ```bash
   pip install -e /path/to/combra   # the combra checkout (or wc_cv/combra submodule)
   pip install -e .                 # run in the edm2-v2 checkout
   ```

   Without combra, just run the second line. This puts the `edm2-train`,
   `edm2-prepare-data`, `edm2-download-models`, `edm2-sample`, `edm2-gen-images`,
   `edm2-eval` and `edm2-compare-samplers` commands on your `PATH`.

2. **(Optional) prefetch model weights** — handy for offline / cluster nodes.
   Training needs the Stable-Diffusion VAE (plus InceptionV3 / CLIP / DINOv2 for the
   combra metrics); the VAE auto-downloads on first run, or fetch everything up
   front:

   ```bash
   edm2-download-models
   ```

3. **Prepare the dataset.** `edm2-prepare-data` center-crops and VAE-encodes a
   folder of images into a `.zip` the trainer reads:

   ```bash
   edm2-prepare-data --source ./raw_wc_co_images \
       --dest ./datasets/wc_co_256x256.zip \
       --resolution 256x256 --transform center-crop-dhariwal
   ```

   Class labels come from a `dataset.json` inside the `.zip`
   (`{"labels": [["img.png", classID], ...]}`); if none is present, the class is
   taken from the top-level directory name (alphabetical order → integer). Grayscale
   SEM images and single-class data are both supported.

4. **Launch training.** The required flags are `--outdir` and `--data`; select the
   architecture with `--preset` (`edm2-img256-s`, `edm2-img512-s`,
   `edm2-img1024-s`), and cap per-GPU memory with `--batch-gpu` (the preset's global
   batch is reached via gradient accumulation automatically):

   ```bash
   # 256² from scratch, 2 GPUs
   torchrun --standalone --nproc_per_node=2 train_edm2.py \
       --outdir=./training-runs \
       --preset=edm2-img256-s \
       --data=./datasets/wc_co_256x256.zip \
       --batch-gpu 96
   ```

   To **resume**, run the exact same command again — training picks up the latest
   `training-state-*.pt` in `--outdir`.

### Quick example (single-GPU smoke test)

A short run to confirm the whole pipeline works end-to-end before committing to a
long job. It uses one GPU, a tiny `--duration`, and `--num-fid-samples 0` to skip
evaluation entirely (so it needs neither combra nor the metric backbones):

```bash
python train_edm2.py --outdir ./training-runs --preset edm2-img256-s \
    --data ./datasets/wc_co_256x256.zip \
    --batch-gpu 8 \
    --duration 200Ki --status 20Ki --snapshot 20Ki --num-fid-samples 0
```

Watch it in TensorBoard with `tensorboard --logdir ./training-runs`. Add `-n` /
`--dry-run` to print the resolved options and exit without training (see the Dry
run section below). Drop `--num-fid-samples 0` to re-enable the combra eval (EDM
Heun sampler by default).

### Progressive training

Each higher-resolution stage starts from the previous stage's snapshot by pointing
`--outdir` at a fresh directory seeded with that checkpoint (EDM2 auto-resumes from
`training-state-*.pt`), or simply retrain per resolution with the matching preset
(`edm2-img512-s`, `edm2-img1024-s`) and dataset zip.

During training, at each snapshot tick the loop generates a batch of images from
the EMA model by running the configured reverse-diffusion sampler (EDM Heun by
default — see the Samplers section below) in VAE latent space, decodes them to
pixels, and runs `compute_all_metrics(reals, fakes)` (when combra is installed).
All returned metrics — angle-Wasserstein `w1`, `w2`, `circular_w1`, `circular_w2`,
the bimodal-Gaussian relative errors `mu1`/`mu2`/`sigma1`/`sigma2`/`amp1`/`amp2`,
and the image-feature metrics `fid`, `cmmd`, `fd_dinov2` — are logged to
TensorBoard under `Metrics/combra_*`, written to `stats.jsonl`, and printed to the
run log. Metrics whose optional backends are unavailable (e.g. no network to fetch
DINOv2 weights) are recorded as `nan`; the angle metrics always come back. The
reference batch is scored once and cached, so only the generated-side work is
repeated each tick.

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
single-GPU `compute_all_metrics` path.

**Sample count.** Each combra eval scores `--num-fid-samples` generated images
(default 10000) against the **whole training set** as the real reference (set
`--combra-ref-count` to cap the reference, or `--num-fid-samples 0` to disable eval
entirely).

### Samplers

The reverse-diffusion sampler used for evaluation **and** inference is selectable.
All reuse the same trained model — they differ only in how the reverse-time ODE is
integrated, all in the native EDM σ-space on the `net(x, σ, labels)` denoiser:

- **`edm`** *(default)* — EDM 2nd-order Heun sampler (supports stochasticity via
  `S_churn`). The reference EDM2 sampler; strong quality at moderate step counts.
- **`dpm++`** — DPM-Solver++(2M) in log-σ space. Fastest (near-converged quality in
  ~25 steps); use it for the cheapest possible training-time eval.
- **`ddim`** — deterministic DDIM (η=0), which for the EDM probability-flow ODE is
  the first-order deterministic step (≡ `euler`).
- **`euler`** — 1st-order deterministic Euler.

During training, pick the eval sampler with `--sampler` and its step count with
`--sampling-steps`:

```bash
torchrun --standalone --nproc_per_node=2 train_edm2.py \
    --outdir=./training-runs --preset=edm2-img256-s \
    --data=./datasets/wc_co_256x256.zip --batch-gpu 96 \
    --sampler dpm++ --sampling-steps 25
```

`edm2-gen-images` and `sample_images.py` take the same choice via `--sampler` /
`--steps`.

### Dry run

To validate the config and data wiring without starting a real run, pass `-n` /
`--dry-run` — it prints the resolved training options (preset, batch, sample count,
etc.) and exits before training:

```bash
python train_edm2.py --outdir=./training-runs \
    --preset=edm2-img256-s \
    --data=./datasets/wc_co_256x256.zip \
    --batch-gpu 4 \
    -n
```

## Evaluation

A full FID-style evaluation of a trained checkpoint bulk-samples a `.npz` batch and
scores it with the offline evaluator (`edm2-eval` = `calculate_metrics.py`):

```bash
torchrun --standalone --nproc_per_node=4 sample_images.py \
    --net ./training-runs/00000-*/network-snapshot-final.pkl \
    --outdir ./samples/256 --num-samples 50000 \
    --sampler dpm++ --steps 25

edm2-eval calc --images ./samples/256 --ref <dataset-ref>.pkl
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

### Optimal number of sampling steps

`edm2-compare-samplers` answers *how many reverse-diffusion steps each sampler
needs*: it sweeps samplers × step counts, scores every batch against a fixed real
reference with {py:func}`combra.metrics.compare_samplers`, and writes a table + plot
via {py:func}`combra.metrics.plot_sampler_comparison`. The metric-vs-steps curve
plateaus at the optimal step count per sampler (see {doc}`sampler_comparison`):

```bash
python compare_samplers.py \
    --net ./training-runs/00000-*/network-snapshot-final.pkl \
    --data ./datasets/wc_co_256x256.zip \
    --samplers edm,euler,ddim,dpm++ --k-values 5,10,20,50,100,250 \
    --num-samples 512 --outdir ./sampler-comparison/256
# -> sampler_comparison.parquet + sampler_comparison.png
```

## Inference

Generate individual images from a trained checkpoint with `edm2-gen-images`. Images
are written as `<seed>.png`; launch with `torchrun` to distribute generation across
GPUs:

```bash
edm2-gen-images \
    --net ./training-runs/00000-*/network-snapshot-final.pkl \
    --outdir ./generated/256 \
    --seeds 0-999 \
    --class 0 \
    --sampler dpm++ --steps 25
```

### Conditioning

EDM2 is class-conditional; conditioning is expressed with a few composable methods:

- **One-hot class labels** — the model is built with `label_dim = <num classes>`
  (from the dataset) and conditioned via `net(x, σ, class_labels)`. Train
  conditional with `--cond=True` (default); an unlabeled dataset + `--cond=False`
  gives an unconditional model.
- **Specific class** — `edm2-gen-images --class <idx>` (and `sample_images.py
  --class <idx>`) fixes the label; omit it to sample a random class per image.
- **Null label** — `class_labels=None` evaluates the model unconditionally; this is
  what the guiding network uses.
- **Classifier-free / auto-guidance** — steer generation with a second *guiding*
  network via `--gnet` and `--guidance` (strength `> 1`); the denoiser output is
  extrapolated away from the guiding network's, `D = lerp(D_guide, D_main,
  guidance)`. `--guidance 1` (default) disables it. Honored by every sampler and by
  the training-time combra eval.

### Class index → grain class

The `--class` integer selects a grain morphology. EDM2's `dataset_tool.py` assigns
each class an integer from the **alphabetical order of the class-folder names**
(`sorted(...)` then `enumerate`) — the same rule as DiffiT — so the index order
matches DiffiT's:

| index | grain class | morphology |
|---|---|---|
| `0` | `Ultra_Co11` | small grain (мелкие зёрна) |
| `1` | `Ultra_Co25` | medium grain (средние зёрна) |
| `2` | `Ultra_Co6_2` | large grain (крупные зёрна) |

```{warning}
**The index order differs from SAN.** SAN numbers the same grains as
`0 → Ultra_Co25`, `1 → Ultra_Co11` (indices 0 and 1 swapped; `2 → Ultra_Co6_2`
matches — see {doc}`san_v2`), because SAN copies the label verbatim from a
`dataset.json` written in non-alphabetical order. EDM2 and DiffiT both derive the
label from the alphabetical folder sort, so they agree with each other but not with
SAN. When comparing a generator against the real classes, remap per model with
combra's `CLASS_MAP`.
```
