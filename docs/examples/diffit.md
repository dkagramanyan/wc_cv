# diffit model

[**DiffiT-v2**](https://github.com/dkagramanyan/DiffiT-v2) is a performance
refresh of NVlabs' DiffiT (Diffusion Vision Transformers) used to generate WC-Co
microstructure SEM images. Its training evaluation is wired into combra: every
evaluation tick scores generated samples with
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

During training, at each evaluation tick (every `--snap` ticks) the loop
generates a batch of images from the EMA model by running the configured
reverse-diffusion sampler (DDIM by default — see the Samplers section below) in
VAE latent space, decodes them to pixels, and runs
`compute_all_metrics(reals, fakes)` (when combra is installed). All returned
metrics — angle-Wasserstein `w1`, `w2`, `circular_w1`, `circular_w2`, the
bimodal-Gaussian relative errors `mu1`/`mu2`/`sigma1`/`sigma2`/`amp1`/`amp2`, and
the image-feature metrics `fid`, `cmmd`, `fd_dinov2` — are logged to TensorBoard
under `Metrics/combra_*`, written to `stats.jsonl`, and printed to the run log,
alongside DiffiT's own IS / FID / sFID / Precision / Recall. Metrics whose
optional backends are unavailable (e.g. no network to fetch DINOv2 weights) are
recorded as `nan`; the angle metrics always come back. The reference batch is
scored once and cached, so only the generated-side work is repeated each tick.

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
    --data=./datasets/imagenet_256x256.zip \
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

diffit-eval \
    --ref-batch ./VIRTUAL_imagenet256_labeled.npz \
    --sample-batch ./samples/256/samples_50000x256x256x3.npz
```

To score arbitrary real/generated image batches with the combra metrics directly
(the same ones logged during training), call
{py:func}`combra.metrics.compute_all_metrics`:

```python
>>> from combra.metrics import compute_all_metrics
>>> results = compute_all_metrics(reference_images, generated_images)
>>> results
{'w1': ..., 'w2': ..., 'circular_w1': ..., 'circular_w2': ...,
 'fid': ..., 'cmmd': ..., 'fd_dinov2': ...}
```

Batches may be numpy arrays or torch tensors in `NCHW`/`NHWC`, with pixel values
in `uint8`, `[0, 1]`, or `[-1, 1]`.

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
