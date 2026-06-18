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

DiffiT-v2 is trained progressively (low → high resolution) — RoPE-2D lets a 256²
checkpoint be re-used at higher resolutions, so each stage resumes from the
previous stage's `network-final.pt`:

```bash
# 256² from scratch
diffit-train --outdir=./training-runs \
    --cfg=diffit-256 \
    --data=./datasets/imagenet_256x256.zip \
    --gpus 2 \
    --batch-gpu 96

# 256² → 512² finetune
diffit-train --outdir=./training-runs \
    --cfg=diffit-512 \
    --data=./datasets/imagenet_512x512.zip \
    --gpus 2 --batch-gpu 64 \
    --resume ./training-runs/00000-diffit-256-*/network-final.pt \
    --lr 5e-5 --lr-warmup 500 --kimg 100000
```

During training, at each evaluation tick (every `--snap` ticks) the loop
generates a batch from the EMA model and runs `compute_all_metrics(reals, fakes)`
(when combra is installed). All returned metrics — angle-Wasserstein `w1`, `w2`,
`circular_w1`, `circular_w2` and the image-feature metrics `fid`, `cmmd`,
`fd_dinov2` — are logged to TensorBoard under `Metrics/combra_*`, written to
`stats.jsonl`, and printed to the run log, alongside DiffiT's own IS / FID / sFID
/ Precision / Recall. Metrics whose optional backends are unavailable (e.g. no
network to fetch DINOv2 weights) are recorded as `nan`; the angle metrics always
come back. The reference batch is scored once and cached, so only the
generated-side work is repeated each tick.

## Evaluation

A full FID-50K evaluation of a trained checkpoint uses the standalone evaluator
on bulk-sampled `.npz` batches:

```bash
torchrun --nproc_per_node=4 -m scripts.sample \
    --model-path ./training-runs/00000-diffit-256-*/network-final.pt \
    --outdir ./samples/256 --image-size 256 \
    --cfg-scale 4.4 --num-samples 50000 --num-sampling-steps 250 --cfg-cond

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
    --num-sampling-steps 250 \
    --gpus 0,1,2,3 \
    --batch-size 32
```
