# StyleSwin-v2 model

[**StyleSwin-v2**](https://github.com/dkagramanyan/StyleSwin-v2) is a fork of Microsoft's
[StyleSwin](https://github.com/microsoft/StyleSwin) (a transformer/Swin-based, StyleGAN-style
GAN) specialised to generate WC-Co microstructure SEM images. It is brought to parity with
{doc}`san_v2` so the two generators can be compared under matching tooling: its training
evaluation is wired into combra, and on every **snapshot** tick it scores generated samples
with {py:func}`combra.metrics.compute_all_metrics` and logs the results to TensorBoard.

Like san-v2, the combra integration is controlled by `--combra-metrics` (default `true`) and
is optional: the import is guarded, so training runs unchanged when combra is not installed —
but if `--combra-metrics=true` and the package is missing, training emits a **warning** at
startup (and skips the metrics) so it is never silently ignored. To enable the metrics, install
combra alongside StyleSwin-v2:

```bash
pip install -e ../wc_cv/combra
```

## Class conditioning

Upstream StyleSwin is **unconditional**. This fork makes it **class-conditional** on the three
WC-Co grain classes, so it is comparable to the (already conditional) {doc}`san_v2` generator.
Two standard techniques are used, enabled with `--cond True` (`n_classes` is read from the
dataset's `dataset.json`; `n_classes = 0` keeps the original unconditional path):

- **Generator — san-v2 mapping conditioning.** The one-hot class label is embedded by a linear
  layer to `style_dim`; both the latent `z` and the label embedding are 2nd-moment-normalised
  (StyleSwin's `PixelNorm`, identical to san-v2's `normalize_2nd_moment`), concatenated, and fed
  to the mapping MLP, so the style vector `w` becomes class-dependent. Only the mapping-network
  **input** changes — the Swin transformer blocks, ToRGB and attention are untouched.
- **Discriminator — projection (Miyato & Koyama, 2018).** The label embedding is projected onto
  the discriminator's pre-logit feature `h` and `(embed(y)·h)/√dim` is added to the scalar logit.
  This rides inside StyleSwin's existing **logistic + R1** objective (no SAN discriminator, so
  the update/loss logic is unchanged).

Fake labels are sampled from the **empirical** class distribution by default
(`--fake-label-sampling empirical`) so imbalanced classes are not over-represented in training
or in the combra generated set. The generator/discriminator update math is otherwise identical
to upstream StyleSwin.

The full **install → data → train → generate** guide lives in the StyleSwin-v2
[README](https://github.com/dkagramanyan/StyleSwin-v2#readme); the essentials are below.

## Installation

Create a `python=3.12` conda env, install the latest PyTorch (CUDA 13.2 wheels), the CUDA
compiler (`nvcc`) and ninja **from conda** (both needed to build the custom CUDA ops in `op/` —
a pip ninja conflicts with conda's, and the torch wheel ships no `nvcc`), then the remaining
deps:

```bash
conda create -n styleswin python=3.12 -y && conda activate styleswin
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
conda install -c nvidia cuda-nvcc -y     # match torch's CUDA major (13.x)
conda install anaconda::ninja -y
pip install -r requirements.txt          # from the StyleSwin-v2 repo root
```

The `sbatch/` scripts load no system CUDA module — they set `CUDA_HOME=$CONDA_PREFIX` so the
ops compile against this conda toolkit.

## Training

Build one ImageNet-style zip per resolution with `dataset_tool.py` (same format and label
convention as san-v2), then train:

```bash
python train.py --outdir=./runs/wc-cv_h200 \
        --data=./datasets/imagenet_9to4_256x256.zip \
        --gpus=2 --batch-gpu 16 --cond True \
        --combra-metrics True --save-inference-only True \
        --kimg 25000 --snap 50
```

On the cluster the per-resolution scripts in `sbatch/` are submitted from the sbatch folder onto
the `rocky` partition (2× H200 each):

```bash
cd sbatch
sbatch train_256x256.sbatch        # … train_512x512.sbatch, train_1024x1024.sbatch
```

Each run writes to `runs/.../NNNNN-<desc>/` with `log.txt` (stdout capture), `stats.jsonl`
(per-tick stats), TensorBoard events, and `network-snapshot-<kimg>.pt` checkpoints. The status
line printed each tick matches san-v2 (`tick … kimg … time … sec/tick … gpumem …`).
`--save-inference-only True` additionally writes a small `network-snapshot-<kimg>-inference.pt`
holding **only `G_ema`** — the smallest artifact, and exactly what `gen_images.py` and
{py:func}`combra.metrics.compute_all_metrics` evaluation consume.

During training, the combra metrics use the **whole training set** as the fixed reference,
scored against a fixed **10 000** images generated from `G_ema` (labels sampled from the
training-set class distribution, seeded identically on every rank). The reference set is fixed
across training, so its features (pooled angles, FID/DINOv2 `(mu, sigma)`, CLIP embedding) are
extracted **once before the training loop** — sharded across ranks and gathered to rank 0 — then
cached; only the generated side is recomputed each evaluation tick.

`G_ema` emits images in StyleSwin's ImageNet-normalised space; the loop reverses that
normalization to `uint8` `[0, 255]` before scoring, so both the reals (raw `uint8` pixels) and
the fakes are compared on the same scale. Each tick computes **both** the angle-density metrics
(Wasserstein `w1`, `w2`, `circular_w1`, `circular_w2` and the bimodal-Gaussian `mu/sigma/amp`
errors) **and** the image-feature metrics `fid`, `cmmd`, `fd_dinov2`. All are logged to
TensorBoard under `Metrics/combra_*` and printed to the run log; the three image-feature metrics
carry their 10k sample size in the key (`combra_fid10k`, `combra_cmmd10k`, `combra_fd_dinov2_10k`),
and `combra_fid10k` drives best-model checkpoint selection (`best_model.pt`).

On a **multi-GPU** run **all** the per-image extraction is **sharded across ranks** — both the
image-feature metrics and the angle-density / Gaussian-fit metrics. Each rank generates its own
shard of the fakes and, from that shard, extracts the CLIP / DINOv2 / InceptionV3 features and
pools the vertex angles; the feature rows and the pooled-angle arrays are gathered to rank 0,
where the Fréchet / MMD distances and the angle Wasserstein / Gaussian metrics are taken against
the precomputed reference. This uses combra's split APIs — the feature halves
({py:func}`combra.metrics.fid_features` + {py:func}`combra.metrics.fid_from_features`, and the
`cmmd_*` / `fd_dinov2_*` analogues) and the angle halves
({py:func}`combra.metrics.images_to_pooled_angles` + `angle_density_metrics_from_pooled`) — and
is numerically identical to the single-GPU `compute_all_metrics(image_metrics=True)` path. The
implementation is ported directly from san-v2, so the two models compute the metrics the same
way.

```{note}
Because the reference is the entire dataset and an equal number of fakes is generated each
evaluation tick, both the reference and the generated extraction run on every rank over its own
shard, so that cost is shared across GPUs. For very large or high-resolution datasets this is
memory- and compute-intensive; pass `--combra-metrics False` on runs where you don't want it.
```

## Inference

Generate samples from a trained network with `gen_images.py`. For a conditional model images are
written per class into `class_<id>/class_<id>_<index>.png`; pass `--gpus` to shard generation:

```bash
python gen_images.py \
  --network=./runs/wc-cv_h200/00000-.../best_model.pt \
  --outdir=./generated/ \
  --trunc=0.7 \
  --classes 0,1,2 \
  --samples-per-class 1000 \
  --gpus 2 \
  --batch-gpu 32
```

The `sbatch/generate_{256x256,512x512,1024x1024}.sbatch` scripts wrap this for the cluster.

To score arbitrary real/generated image batches with the combra metrics directly (the same ones
logged during training), call {py:func}`combra.metrics.compute_all_metrics`:

```python
>>> from combra.metrics import compute_all_metrics
>>> results = compute_all_metrics(reference_images, generated_images, image_metrics=True)
```

`image_metrics=True` adds the `fid`/`cmmd`/`fd_dinov2` keys (what the training loop passes);
without it, only the angle-Wasserstein and bimodal-Gaussian metrics come back. Batches may be
numpy arrays or torch tensors in `NCHW`/`NHWC`, with pixel values in `uint8`, `[0, 1]`, or
`[-1, 1]`.
