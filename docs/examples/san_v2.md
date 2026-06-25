# san-v2 model

[**san-v2**](https://github.com/dkagramanyan/san-v2) is a fork of Sony's
StyleSAN-XL (Slicing Adversarial Network, StyleGAN3 + Projected GAN) used to
generate WC-Co microstructure SEM images. Its training evaluation is wired into
combra: every evaluation tick scores generated samples with
{py:func}`combra.metrics.compute_all_metrics`.

The combra integration is **optional** — san-v2 does not depend on combra. The
import is guarded, so training runs unchanged when combra is not installed. To
enable the metrics, install combra alongside san-v2:

```bash
pip install combra            # or: pip install 'san-v2[combra]'
```

## Training

Models are trained progressively (low → high resolution), each stage resuming
from the previous stage's `best_model.pkl`:

```bash
python train.py --outdir=./training-runs/wc --cfg=stylegan3-r --data=./data/wc64.zip \
        --gpus=8 --mirror=1 --snap 10 --batch-gpu 8 --syn_layers 6
```

Training can also be launched via Hydra (`train_hydra.py`), which shares the same
`build_config()` logic as the click CLI.

During training, the combra metrics use the **whole training set** as the fixed
reference, scored against a fixed **`COMBRA_NUM_GEN = 10 000`** images generated
from `G_ema` (their labels sampled from the training-set class distribution, seeded
identically on every rank). combra's image-feature metrics estimate per-side
statistics, so the two counts need not match. The reference set is fixed across
training, so its features (angle density, FID/DINOv2 `(mu, sigma)`, CLIP embedding)
are computed once and reused via combra's `reference_cache`; only the generated
side is recomputed each evaluation tick.

This mirrors the standard StyleGAN `fid50k_full` (all reals vs a fixed sample of
fakes) but with a 10 000-image generated side, and it adds the angle-distribution
and CMMD/FD-DINOv2 metrics on top of FID.

`G_ema` emits float images in `[-1, 1]`, while the reals are `uint8` `[0, 255]`;
the loop denormalizes the fakes to `uint8` first so both sides are scored
explicitly on the same scale. (combra also rescales float inputs internally — both
the image-feature and the angle-density metrics map `[-1, 1]`/`[0, 1]` to `uint8`
the same way — but denormalizing in the loop keeps the comparison unambiguous.)
All returned metrics — angle-Wasserstein `w1`, `w2`, `circular_w1`, `circular_w2`,
the bimodal-Gaussian relative errors `mu1`/`mu2`/`sigma1`/`sigma2`/`amp1`/`amp2`,
and the image-feature metrics `fid`, `cmmd`, `fd_dinov2` — are logged to
TensorBoard under `Metrics/combra_*` and printed to the run log; the three
image-feature metrics carry their 10k sample size in the TensorBoard key
(`combra_fid10k`, `combra_cmmd10k`, `combra_fd_dinov2_10k`). Metrics whose
optional backends are unavailable (e.g. no network to fetch DINOv2 weights) are
recorded as `nan`; the angle metrics always come back.

When `--combra-metrics` is enabled (the default), `fid50k_full` is dropped from
`--metrics` so the expensive 50k-image FID pass is not computed twice, and
`combra_fid10k` becomes the FID that drives best-model checkpoint selection
(`best_model.pkl`).

```{note}
Because the reference is the entire dataset and 10 000 fakes are generated each
evaluation tick, both the real images and the generated batch are held in memory
on rank 0. For very large or high-resolution datasets this is
memory- and compute-intensive (comparable to a full FID pass every snapshot).
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
  --gpus 4 \
  --batch-gpu 60 \
  --network=<path_to_checkpoint>
```
