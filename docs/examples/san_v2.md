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
        --gpus=8 --batch=256 --mirror=1 --snap 10 --batch-gpu 8 --syn_layers 6
```

Training can also be launched via Hydra (`train_hydra.py`), which shares the same
`build_config()` logic as the click CLI.

During training, at each evaluation tick the loop generates a batch from `G_ema`
and runs `compute_all_metrics(reals, fakes)` (when combra is installed). `G_ema`
emits float images in `[-1, 1]`, while the reals are `uint8` `[0, 255]`; the loop
denormalizes the fakes to `uint8` first so both sides are scored explicitly on the
same scale. (combra also rescales float inputs internally — both the image-feature
and the angle-density metrics now map `[-1, 1]`/`[0, 1]` to `uint8` the same way —
but denormalizing in the loop keeps the comparison unambiguous.) All returned
metrics — angle-Wasserstein `w1`, `w2`,
`circular_w1`, `circular_w2` and the image-feature metrics `fid`, `cmmd`,
`fd_dinov2` — are logged to TensorBoard under `Metrics/combra_*` and printed to
the run log. Metrics whose optional backends are unavailable (e.g. no network to
fetch DINOv2 weights) are recorded as `nan`; the angle metrics always come back.

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
>>> results = compute_all_metrics(reference_images, generated_images)
>>> results
{'w1': ..., 'w2': ..., 'circular_w1': ..., 'circular_w2': ...,
 'fid': ..., 'cmmd': ..., 'fd_dinov2': ...}
```

Batches may be numpy arrays or torch tensors in `NCHW`/`NHWC`, with pixel values
in `uint8`, `[0, 1]`, or `[-1, 1]`; combra rescales each side to `uint8` by
inferring the range (any negative ⇒ `[-1, 1]`, else max ≤ 1 ⇒ `[0, 1]`). Because
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
