# combra.metrics

The `combra.metrics` module bundles three families of metrics:

- **Training-loop / batch metrics** — score generated images on the fly (no parquet round-trip): classic InceptionV3 FID (`compute_fid`), CLIP-MMD (`compute_cmmd`), Fréchet distance on DINOv2 features (`compute_fd_dinov2`), the four angle-Wasserstein distances between two samples' angle densities (`compute_wasserstein_metrics`), and the bimodal-Gaussian fit relative errors (`compute_gauss_metrics`). Every metric takes **in-memory images** (a numpy array or torch tensor), not a folder of files. Each side may be a **single image or a batch** (Wasserstein, Gaussian and CMMD are valid on one image; the Fréchet-distance metrics FID/FD-DINOv2 need ≥ 2). `compute_all_metrics` runs the whole set — FID included — in one call. Every two-input metric takes the *reference* first, then the *generated*.
- **Distribution comparison helpers** — for comparing per-class angle distributions stored as parquet files.
- **Convergence analysis** — N-sweep aggregation, Kendall trend tests, plateau fits, and the convergence-grid / gain-distribution plots used by `3_metrics_convergence.ipynb`.

```python
from combra import metrics
```

## Training-loop metrics

These score **in-memory images** (a numpy array or torch tensor) and return a
scalar, so generated samples can be evaluated on the fly during training without
writing them to disk. They live in `combra.metrics.training`.

**Batches vs single data points.** Every metric accepts both a **single image**
(`(H, W)`) and a **batch** (`(N, H, W)`, `(N, C, H, W)`, or `(N, H, W, C)`), but
they differ in what a single image *means*:

- `compute_wasserstein_metrics`, `compute_gauss_metrics` and `compute_cmmd` are
  valid on a **single image** — the angle metrics reduce each side to an angle
  density (then, for the Gaussian metrics, fit it), and CMMD uses kernel means,
  none of which needs a sample of images.
- The **Fréchet-distance** metrics — `compute_fd_dinov2` and the InceptionV3 FID
  (`compute_fid`, also computed inside `compute_all_metrics`) — estimate a
  per-side **covariance**, which is undefined for one image, so they require
  **≥ 2 images per side** and raise `ValueError` otherwise (inside
  `compute_all_metrics` this surfaces as a `nan` for that metric).
- `compute_wasserstein_metrics` and `compute_gauss_metrics` additionally accept a
  precomputed `(x, y)` angle density in place of images on either side.

The backbone models (CLIP / DINOv2 / InceptionV3) are loaded once per process and
memoised with `functools.cache`. Each two-input metric also takes an optional
`reference_cache` dict: pass the **same** dict across calls that share one fixed
reference batch and the reference-side work (CLIP embedding, DINOv2 / Inception
`(mu, sigma)`, angle density) runs only once. The cache is keyed by metric and
parameters, **not** by the reference content — use one cache per reference batch.

### Image-feature metrics

These compare the deep-feature distributions of a *reference* and a *generated*
image set: classic InceptionV3 **FID** (`compute_fid`), CLIP-MMD (`compute_cmmd`),
and the Fréchet distance on DINOv2 features (`compute_fd_dinov2`). FID is just one
of these feature metrics — `compute_fid` scores two in-memory image batches, and
`compute_all_metrics` computes the same FID alongside the others.
`compute_cmmd` and `compute_fd_dinov2` need the `gen-metrics` extra
(`pip install ".[gen-metrics]"`); the DINOv2 backbone is fetched from `torch.hub`
on first use.

````{py:function} combra.metrics.compute_fid(reference_images, generated_images, device=None, batch_size=50, dims=2048, reference_cache=None) -> float

Classic InceptionV3 FID between two **in-memory image batches** (a numpy array or torch tensor), computed with [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (a core dependency, so no `gen-metrics` extra needed). Runs every image through InceptionV3 and returns the Fréchet distance between the two activation distributions. Like every Fréchet-distance metric it estimates a per-side covariance, so it needs **≥ 2 images per side** and raises `ValueError` otherwise. The weights are downloaded and cached by pytorch-fid on first use. `compute_all_metrics` computes this same metric under its `fid` key. Runs on PyTorch, selecting CUDA automatically when available.

:param reference_images: Batch of reference (real) images.
:type reference_images: ndarray or torch.Tensor
:param generated_images: Batch of generated images.
:type generated_images: ndarray or torch.Tensor
:param device: Torch device to run on (e.g. `'cuda'` or `'cpu'`). When `None`, picks `'cuda'` if available, otherwise `'cpu'`. Default: `None`.
:type device: str or torch.device or None, optional
:param batch_size: Forward-pass batch size. Default: `50`.
:type batch_size: int, optional
:param dims: Dimensionality of the InceptionV3 feature layer (`64`, `192`, `768`, or `2048`). Default: `2048`.
:type dims: int, optional
:param reference_cache: Opt-in dict memoising the reference `(mu, sigma)` across calls. Default: `None`.
:type reference_cache: dict or None, optional
:returns: **fid** (*float*) – The Fréchet (FID) distance. `0.0` when both batches are identical.
:rtype: float

**Example**

```python
>>> from combra.metrics import compute_fid
>>> fid = compute_fid(real_batch, generated_batch)  # CUDA when available, else CPU
>>> print(f'FID = {fid:.4f}')
```

A full multi-resolution loop is shown in the {doc}`FID example </examples/fid>`. FID is always computed from images; combra does not expose the raw mean/covariance (`mu`/`sigma`) form of the Fréchet distance.
````

````{py:function} combra.metrics.compute_cmmd(reference_images, generated_images, model_name='ViT-L-14-336', pretrained='openai', device=None, batch_size=64, sigma=10.0, scale=1000.0, reference_cache=None) -> float

CLIP-MMD (CMMD) between a reference and a generated image set. Features come from a ready CLIP model loaded with [open_clip](https://github.com/mlfoundations/open_clip); the distance is the Gaussian-RBF Maximum Mean Discrepancy of [Google's CMMD](https://github.com/google-research/google-research/tree/master/cmmd) (`sigma=10`, `scale=1000`). CMMD uses kernel means (no covariance), so each side may be a **single image or a batch**.

:param reference_images: Batch of reference (real) images.
:type reference_images: ndarray or torch.Tensor
:param generated_images: Batch of generated images.
:type generated_images: ndarray or torch.Tensor
:param model_name: open_clip architecture name. Default: `'ViT-L-14-336'`.
:type model_name: str, optional
:param pretrained: open_clip pretrained-weights tag. Default: `'openai'`.
:type pretrained: str, optional
:param device: Torch device. When `None`, picks `'cuda'` if available, else `'cpu'`. Default: `None`.
:type device: str or torch.device or None, optional
:param batch_size: Forward-pass batch size for the CLIP encoder. Default: `64`.
:type batch_size: int, optional
:param sigma: Gaussian-RBF kernel bandwidth. Default: `10.0`.
:type sigma: float, optional
:param scale: Multiplier applied to the MMD². Default: `1000.0`.
:type scale: float, optional
:param reference_cache: Opt-in dict memoising the reference CLIP embedding across calls. Default: `None`.
:type reference_cache: dict or None, optional
:returns: **cmmd** (*float*) – The scaled CLIP-MMD distance.
:rtype: float

**Example**

```python
>>> from combra.metrics import compute_cmmd
>>> cmmd = compute_cmmd(real_batch, generated_batch)   # CUDA when available
>>> print(f'CMMD = {cmmd:.4f}')
```
````

````{py:function} combra.metrics.compute_fd_dinov2(reference_images, generated_images, model_name='dinov2_vitb14', device=None, batch_size=64, image_size=224, reference_cache=None) -> float

Fréchet distance between the [DINOv2](https://github.com/facebookresearch/dinov2) features of two image sets. The backbone is loaded from `torch.hub` (`facebookresearch/dinov2`); the Fréchet distance itself is computed with pytorch-fid's `calculate_frechet_distance`. Like FID, it estimates a per-side covariance, so it needs **≥ 2 images per side** and raises `ValueError` on a single image.

:param reference_images: Batch of reference (real) images.
:type reference_images: ndarray or torch.Tensor
:param generated_images: Batch of generated images.
:type generated_images: ndarray or torch.Tensor
:param model_name: DINOv2 `torch.hub` entrypoint. Default: `'dinov2_vitb14'`.
:type model_name: str, optional
:param device: Torch device. When `None`, picks `'cuda'` if available, else `'cpu'`. Default: `None`.
:type device: str or torch.device or None, optional
:param batch_size: Forward-pass batch size. Default: `64`.
:type batch_size: int, optional
:param image_size: Square size images are resized to before the backbone. Default: `224`.
:type image_size: int, optional
:param reference_cache: Opt-in dict memoising the reference `(mu, sigma)` across calls. Default: `None`.
:type reference_cache: dict or None, optional
:returns: **fd** (*float*) – Fréchet distance between the two DINOv2 feature sets.
:rtype: float
````

### Angle-Wasserstein metrics

A single entry point, `compute_wasserstein_metrics`, reduces *both* the reference
and the generated sample to their angle densities (the same image → angles →
histogram pipeline used to build the parquet datasets) and returns all four
Wasserstein distances at once — the linear `w1`/`w2` and the circular
`circular_w1`/`circular_w2` (which treat 359° and 1° as neighbours). Each side
may be a single image, a batch, or a precomputed `(x, y)` angle density.

````{py:function} combra.metrics.compute_wasserstein_metrics(reference, generated, step=None, reference_cache=None, **angle_kw) -> dict

All four angle-Wasserstein distances in a single pass, returned as a dict `{'w1', 'w2', 'circular_w1', 'circular_w2'}` (all in degrees). Each side is reduced to its angle density once, so the four distances share that work rather than recomputing it per metric.

`reference` and `generated` may each be **a single `(H, W)` image**, **a batch of images**, or **a precomputed `(x, y)` angle density** (a length-2 pair of 1-D arrays) — image input is reduced to a density, a density is used as-is.

:param reference: Reference sample — image, image batch, or `(x, y)` density.
:type reference: ndarray or torch.Tensor or tuple
:param generated: Generated sample to score — image, image batch, or `(x, y)` density.
:type generated: ndarray or torch.Tensor or tuple
:param step: Histogram bin width in degrees (image input only). Defaults to `5.0`. Default: `None`.
:type step: float or None, optional
:param reference_cache: Opt-in dict memoising the reference angle density across calls. Default: `None`.
:type reference_cache: dict or None, optional
:param angle_kw: Extra keyword args forwarded to `images_to_angle_density` (`border_eps`, `tol`, `min_segment_len`).
:returns: **metrics** (*dict*) – Keys `w1`, `w2`, `circular_w1`, `circular_w2`, all in degrees.
:rtype: dict

**Example**

```python
>>> from combra.metrics import compute_wasserstein_metrics
>>> dists = compute_wasserstein_metrics(real_batch, generated_batch)   # batches
>>> one   = compute_wasserstein_metrics(real_image, generated_image)   # single images
>>> dists['w1'], dists['circular_w2']
```

The density-level core (used by the parquet comparison path) lives at `combra.metrics.wasserstein.wasserstein_density_metrics` if you need to call it on `(x, y)` densities directly.
````

````{py:function} combra.metrics.images_to_angle_density(images, step=None, border_eps=5, tol=3, min_segment_len=10.0) -> tuple[ndarray, ndarray]

Reduce a batch of images to a single angle density `(x, y)`. Runs `_preprocess_image → get_angles` on each image, pools all vertex angles, then histograms them with {py:func}`combra.stats.stats_preprocess` at `step` degrees.

:param images: Image batch.
:type images: ndarray or torch.Tensor
:param step: Histogram bin width in degrees. Defaults to `5.0`. Default: `None`.
:type step: float or None, optional
:param border_eps: Border margin passed to {py:func}`combra.angles.get_angles`. Default: `5`.
:type border_eps: int, optional
:param tol: Polygon-approximation tolerance passed to `get_angles`. Default: `3`.
:type tol: int, optional
:param min_segment_len: Minimum segment length passed to `get_angles`. Default: `10.0`.
:type min_segment_len: float, optional
:returns: **density** (*tuple[ndarray, ndarray]*) – `(x, y)` angle-bin centres and densities.
:rtype: tuple(ndarray, ndarray)
````

### Gaussian-fit metrics

`compute_gauss_metrics` mirrors `compute_wasserstein_metrics`, but instead of a
transport distance it fits a **bimodal Gaussian** to each side's angle density
({py:func}`combra.approx.bimodal_gauss_approx` — two modes, each with a mean `mu`,
width `sigma` and amplitude `amp`) and reports the **per-mode relative error**
`(generated − reference) / reference` of every fitted parameter. These are the
same six numbers (`mu1`, `mu2`, `sigma1`, `sigma2`, `amp1`, `amp2`) the parquet
comparison path computes from its stored fits, made available directly from
in-memory images.

````{py:function} combra.metrics.compute_gauss_metrics(reference, generated, step=None, reference_cache=None, **angle_kw) -> dict

Bimodal-Gaussian relative-error metrics between a reference and a generated sample, returned as a flat dict `{'mu1', 'mu2', 'sigma1', 'sigma2', 'amp1', 'amp2'}` — mode 1 then mode 2 for each fitted parameter. Each side is reduced to its angle density once and fitted with {py:func}`combra.approx.bimodal_gauss_approx`; each value is `(generated − reference) / reference` for that parameter.

`reference` and `generated` may each be **a single `(H, W)` image**, **a batch of images**, or **a precomputed `(x, y)` angle density** (a length-2 pair of 1-D arrays) — image input is reduced to a density, a density is used as-is. The `reference_cache` is keyed the same way as `compute_wasserstein_metrics`, so a cache shared between the two reuses the one reference angle density.

:param reference: Reference sample — image, image batch, or `(x, y)` density.
:type reference: ndarray or torch.Tensor or tuple
:param generated: Generated sample to score — image, image batch, or `(x, y)` density.
:type generated: ndarray or torch.Tensor or tuple
:param step: Histogram bin width in degrees (image input only). Defaults to `5.0`. Default: `None`.
:type step: float or None, optional
:param reference_cache: Opt-in dict memoising the reference angle density across calls. Default: `None`.
:type reference_cache: dict or None, optional
:param angle_kw: Extra keyword args forwarded to `images_to_angle_density` (`border_eps`, `tol`, `min_segment_len`).
:returns: **metrics** (*dict*) – Keys `mu1`, `mu2`, `sigma1`, `sigma2`, `amp1`, `amp2` — the per-mode relative errors of the fitted means, widths and amplitudes.
:rtype: dict

**Example**

```python
>>> from combra.metrics import compute_gauss_metrics
>>> errs = compute_gauss_metrics(real_batch, generated_batch)   # batches
>>> one  = compute_gauss_metrics(real_image, generated_image)   # single images
>>> errs['mu1'], errs['sigma2']
```

The density-level core (used by the parquet comparison path) lives at `combra.metrics.gauss.gauss_density_metrics`; the raw per-mode error math, shared with `compute_metrics`, is `combra.metrics.gauss.gauss_relative_errors`.
````

### Unified entry point

````{py:function} combra.metrics.compute_all_metrics(reference_images, generated_images, *, step=None, device=None, angle_kw=None, reference_cache=None) -> dict

Run every batch metric for a (reference, generated) pair in parallel and return one flat dict. The angle-density metrics — the Wasserstein distances (`w1`, `w2`, `circular_w1`, `circular_w2`) and the bimodal-Gaussian relative errors (`mu1`, `mu2`, `sigma1`, `sigma2`, `amp1`, `amp2`) — compare the two samples' angle densities; `fid` (classic InceptionV3 FID on the in-memory images), `cmmd`, and `fd_dinov2` compare their deep features. An image-feature metric that cannot run (missing optional dependency, no network, **or fewer than 2 images per side** for the Fréchet-distance `fid`/`fd_dinov2`) is recorded as `nan` and logged, so the angle metrics still come back. A single image therefore yields real `w*`/`mu*`/`sigma*`/`amp*`/`cmmd` values and `nan` for `fid`/`fd_dinov2`.

:param reference_images: Batch of reference (real) images.
:type reference_images: ndarray or torch.Tensor
:param generated_images: Batch of generated images.
:type generated_images: ndarray or torch.Tensor
:param step: Histogram bin width in degrees. Defaults to `5.0`. Default: `None`.
:type step: float or None, optional
:param device: Torch device for the image-feature metrics. Default: `None`.
:type device: str or torch.device or None, optional
:param angle_kw: Extra keyword args forwarded to `images_to_angle_density`. Default: `None`.
:type angle_kw: dict or None, optional
:param reference_cache: Opt-in dict reused across calls to compute the reference-side features once. Default: `None`.
:type reference_cache: dict or None, optional
:returns: **results** (*dict*) – Flat `{metric_name: value}` dict with keys `w1`, `w2`, `circular_w1`, `circular_w2`, `mu1`, `mu2`, `sigma1`, `sigma2`, `amp1`, `amp2`, `fid`, `cmmd`, `fd_dinov2`.
:rtype: dict

**Example**

```python
>>> from combra.metrics import compute_all_metrics
>>> scores = compute_all_metrics(real_batch, generated_batch)
>>> scores['cmmd'], scores['w1'], scores['mu1']
```
````

## Distribution comparison

These helpers compare angle-distribution parquet files (as produced by {py:meth}`combra.data.PobeditDataset.generate_angles`) against an "ethalon" reference. The lower-level plumbing they build on — `index_by_name_step`, `find_ethalon`, `parquet_has_step`, and the row-pair `compute_metrics` — is not part of the public surface; import it from `combra.metrics.compare` if you need it directly.

````{py:function} combra.metrics.load_rows(parquet_path) -> list[dict]

Read a parquet file and return a flat list of `{'meta': {..., 'step'}, 'prep': {...}}` rows. Both the legacy schema (one row per `(class, step)` with `meta.step`) and the newer per-class `prep_per_step` schema are flattened to the same uniform shape, so callers don't need to branch.

:param parquet_path: Path to an angles parquet.
:type parquet_path: str or Path
:returns: **rows** (*list[dict]*) – Each entry has `meta` (with synthesised `step`) and `prep` (single step's fit + density curves).
:rtype: list[dict]

**Example**

```python
>>> from combra.metrics import load_rows
>>> rows = load_rows('./data/angles/o_bc_left_4x_1536_1024x1024_256x256_rgb_N360_msl5/angles_n360.parquet')
>>> print(f'{len(rows)} rows;  first = name={rows[0]["meta"]["name"]} step={rows[0]["meta"]["step"]}')
```
````

````{py:function} combra.metrics.compare_folders(folder_paths, ethalon_path, class_map=None, steps=None, coef=1000, verbose=True, fid_by_kimg=None) -> list[dict]

Walk every parquet under each folder in `folder_paths`, look each row up in `ethalon_path`, print/collect the metrics.

:param folder_paths: Folders, each holding one generator's parquets. The folder name suffix after `_kimg_` becomes the `kimg` tag in records (full name used if absent). May also pass `.parquet` paths directly to process exactly one file.
:type folder_paths: Iterable[str]
:param ethalon_path: Parquet file with reference distributions.
:type ethalon_path: str
:param class_map: `{fake_class: real_class}` mapping when names don't match. Default: `None`.
:type class_map: dict[str, str] or None, optional
:param steps: Subset of steps to keep — others are skipped. Default: `None`.
:type steps: Iterable[float] or None, optional
:param coef: Multiplier applied to `w1` in the printed table only (records keep raw values). Default: `1000`.
:type coef: float, optional
:param verbose: Print a per-row table. Default: `True`.
:type verbose: bool, optional
:param fid_by_kimg: Optional `{kimg_key: fid}` map — prints a `[kimg=… FID=…]` header above each checkpoint's rows. Default: `None`.
:type fid_by_kimg: dict[str, float] or None, optional
:returns: **records** (*list[dict]*) – `{'kimg', 'class', 'step', 'w1', 'w2', 'circular_w1', 'circular_w2', 'mus_m', 'sig_m', 'amp_m'}` per matched row (the `w*`/`circular_w*` keys are the angle-Wasserstein distances, as in {py:func}`combra.metrics.compute_wasserstein_metrics`).
:rtype: list[dict]

**Example**

Adapted from `co_angles/2_comparison.ipynb` — compare every diffit checkpoint against a shared real ethalon at one step:

```python
>>> from combra.metrics import compare_folders
>>> ethalon = './grid_results/o_bc_left_4x_1536_1024x1024_256x256_rgb_N360_msl5/angles_n360.parquet'
>>> folder_paths = [
...     './grid_results/00017-diffit-256-gpus2-batch192_kimg_004435_msl5/angles_n1000.parquet',
...     './grid_results/00017-diffit-256-gpus2-batch192_kimg_008064_msl5/angles_n1000.parquet',
...     './grid_results/00017-diffit-256-gpus2-batch192_kimg_016128_msl5/angles_n1000.parquet',
... ]
>>> class_map = {'class_0': 'class_Ultra_Co11',
...              'class_1': 'class_Ultra_Co25',
...              'class_2': 'class_Ultra_Co6_2'}
>>> recs = compare_folders(folder_paths, ethalon, class_map=class_map,
...                        steps=[2], coef=1)  # coef=1 prints raw degrees
```
````

````{py:function} combra.metrics.compare_pairs(pairs, step, coef=1000, verbose=True, label_header='label') -> list[dict]

Like `compare_folders` but accepts a list of explicit `(label, ethalon_pq, fake_pq, class_map)` tuples — one row per pair, exactly one step. Used by `4_grid_plot.ipynb` to print one row per resolution.

:param pairs: Each tuple is `(label, orig_pq, gen_pq, class_map)`. `class_map` is `{orig_class: gen_class}`.
:type pairs: list[tuple]
:param step: The single step to compare at.
:type step: float
:param coef: Multiplier applied to `w1` in the printed table only. Default: `1000`.
:type coef: float, optional
:param verbose: Print a per-row table. Default: `True`.
:type verbose: bool, optional
:param label_header: Column header used in the printed table for the first column. Default: `'label'`.
:type label_header: str, optional
:returns: **records** (*list[dict]*) – `{'label', 'class', 'step', 'w1', 'w2', 'circular_w1', 'circular_w2', 'mus_m', 'sig_m', 'amp_m'}` (the `w*`/`circular_w*` keys are the angle-Wasserstein distances, as in {py:func}`combra.metrics.compute_wasserstein_metrics`).
:rtype: list[dict]

**Example**

Adapted from `co_angles/4_grid_plot.ipynb` — one row per resolution, each row pairs its own orig + diffit:

```python
>>> from combra.metrics import compare_pairs
>>> class_map = {'class_Ultra_Co11': 'class_0',
...              'class_Ultra_Co25': 'class_1',
...              'class_Ultra_Co6_2': 'class_2'}
>>> pairs = [
...     ('256', './data/angles/orig_256/angles_n360.parquet',
...             './data/angles/diffit_256/angles_n10000.parquet', class_map),
...     ('512', './data/angles/orig_512/angles_n360.parquet',
...             './data/angles/diffit_512/angles_n10000.parquet', class_map),
... ]
>>> recs = compare_pairs(pairs, step=2, coef=1, label_header='res')
```
````

````{py:function} combra.metrics.metrics_vs_n(folder, ethalon_path, class_map=None, step=5.0, allowed_ns=None) -> list[dict]

Walk every parquet under `folder` (each assumed to be the same generator at a different sample size) and emit one record per `(parquet, class)` with every metric: the Wasserstein distances (`w1`, `w2`, `circular_w1`, `circular_w2`) plus the Gaussian-fit relative errors `(mu1, mu2, sigma1, sigma2, amp1, amp2)`. N is read from `meta.n_images`, so the filename convention does not matter.

:param folder: Sweep folder containing one parquet per N.
:type folder: str or Path
:param ethalon_path: Reference parquet (typically `find_ethalon(folder)` for the same generator, or a separate "real" parquet for cross-generator comparisons).
:type ethalon_path: str
:param class_map: `{fake_class: real_class}` mapping when names don't match between folders. Default: `None`.
:type class_map: dict[str, str] or None, optional
:param step: Histogram step to filter rows on. Default: `5.0`.
:type step: float, optional
:param allowed_ns: If given, restrict N values to this set — stray parquets at other sample sizes are skipped. Default: `None`.
:type allowed_ns: set[int] or None, optional
:returns: **records** (*list[dict]*) – Sorted by `(class, n_images)`. Each entry: `{'n_images', 'class', 'w1', 'w2', 'circular_w1', 'circular_w2', 'mu1', 'mu2', 'sigma1', 'sigma2', 'amp1', 'amp2'}`.
:rtype: list[dict]

**Example**

```python
>>> from combra.metrics import metrics_vs_n
>>> from combra.metrics.compare import find_ethalon
>>> ethalon = find_ethalon('./sweeps/real_msl5')
>>> recs = metrics_vs_n('./sweeps/diffit_msl5', str(ethalon),
...                     class_map={'class_0': 'class_Ultra_Co11'},
...                     step=2.0, allowed_ns={100, 250, 1000, 10000})
```
````

## Convergence analysis

Tools that aggregate `metrics_vs_n` records into per-curve statistics (Kendall trend test, endpoint relative error, plateau fit) and turn them into the tables and figures consumed by `3_metrics_convergence.ipynb`.

````{py:function} combra.metrics.convergence_stats(df_metrics, metrics, endpoints_by_kind, expected_points=None, pre_endpoints_by_kind=None) -> pandas.DataFrame

Per `(kind, resolution, class)` curve, compute trend significance, endpoint relative error, a power-law decay exponent, and a plateau fit $|m|(N) = a + b \cdot N^{-1/2}$.

:param df_metrics: Long-form table with columns `kind`, `resolution`, `class`, `n_images`, and one column per metric in `metrics`.
:type df_metrics: pd.DataFrame
:param metrics: Metric column names to analyse (e.g. `['w1', 'mu1', 'mu2', 'sigma1', 'sigma2', 'amp1', 'amp2']`).
:type metrics: list[str]
:param endpoints_by_kind: `{kind: (n_lo, n_hi)}` used for the `rel_err_abs_%` computation.
:type endpoints_by_kind: dict[str, tuple[int, int]]
:param expected_points: `{kind: expected_n_points_per_curve}`. When given, prints a warning if a curve has the wrong count. Default: `None`.
:type expected_points: dict[str, int] or None, optional
:param pre_endpoints_by_kind: Optional `{kind: (n_lo, n_hi)}` for an *earlier* N-range (e.g. `360 → 1000` for `san`/`diffit`), reported alongside the main range. The `pre_*` columns are `NaN` for any kind absent from this mapping. Default: `None`.
:type pre_endpoints_by_kind: dict[str, tuple[int, int]] or None, optional
:returns: **result** (*pd.DataFrame*) – One row per `(kind, resolution, class, metric)`. Columns: `kendall_p`, `rel_err_abs_%`, `pre_rel_err_abs_%`, `alpha`, `pre_alpha`, `m_at_nhi`, `a_hat`, `a_se`, `b_hat`, `fit_r2`, `gain_abs`, `gain_pct`, `vs_a_pct`.
:rtype: pandas.DataFrame

The returned columns mean:

- `kendall_p` — one-sided Kendall τ p-value for "|metric| decreases with N".
- `rel_err_abs_%` — `(|m|@N_lo − |m|@N_hi) / |m|@N_lo × 100` over the main endpoints. Positive = improvement between the two N endpoints. `pre_rel_err_abs_%` is the same over `pre_endpoints_by_kind` (`NaN` when no pre pair).
- `alpha` — power-law decay exponent in `|m| ~ N^(-alpha)`, estimated from the two endpoints as `log(|m|@N_lo / |m|@N_hi) / log(N_hi / N_lo)`. `0.5` is ideal Monte-Carlo decay, `0` means no improvement, `< 0` means |metric| grew with N. `pre_alpha` is the same over the pre endpoints.
- `m_at_nhi` — `|metric|` at `N_hi`.
- `a_hat`, `a_se`, `b_hat` — the plateau (bias floor `a`), its standard error, and the `N^{-1/2}` slope `b` from {py:func}`combra.approx.fit_plateau`.
- `fit_r2` — coefficient of determination of the plateau fit against the observed `|metric|` curve.
- `gain_pct` — sampling-only gain from `N_hi` to infinity, in percent of `|m|@N_hi`.
- `vs_a_pct` — excess of `|m|@N_lo` over the bias floor `a_hat`, in percent.

**Example**

From `co_angles/3_metrics_convergence.ipynb`:

```python
>>> from combra.metrics import convergence_stats
>>> METRICS  = ['w1', 'mu1', 'mu2', 'sigma1', 'sigma2', 'amp1', 'amp2']
>>> ENDPTS   = {'real': (100, 300), 'san': (360, 10000), 'diffit': (360, 10000)}
>>> PRE      = {'san': (360, 1000), 'diffit': (360, 1000)}
>>> result = convergence_stats(df_metrics, METRICS, ENDPTS,
...                            expected_points={'real': 5, 'san': 8, 'diffit': 8},
...                            pre_endpoints_by_kind=PRE)
>>> result.head(3)
```
````

````{py:function} combra.metrics.print_convergence_report(result, metrics, kinds, endpoints_by_kind, alpha=0.05, step=None, kind_display=None, pre_endpoints_by_kind=None) -> None

Print three human-readable tables from a `convergence_stats` result:

- **T1 (kendall_byclass)** — per-class Kendall p-values + `rel_err_abs_%`, with Fisher's combined $\chi^2 = -2 \cdot \sum \log p$ across the 3 classes.
- **T2 (kendall_fisher_byres)** — per-resolution Fisher across classes; one panel per resolution.
- **T3 (gains+alpha)** — finite-range gain and power-law-exponent columns side by side: `gain_%_a->b = (|m|@a − |m|@b)/|m|@a × 100` (positive = error shrank) and `alpha_a->b = log(|m|@a / |m|@b) / log(b/a)` (`0.5` = ideal Monte-Carlo, `0` = no gain). When `pre_endpoints_by_kind` is given, the pre-range columns are shown next to the main-range ones.

:param result: Output of `convergence_stats`.
:type result: pd.DataFrame
:param metrics: Metric order in the printed tables.
:type metrics: list[str]
:param kinds: Generators to include (e.g. `['san', 'diffit']`).
:type kinds: list[str]
:param endpoints_by_kind: Same mapping passed to `convergence_stats`; used to annotate each subsection with its `(N_lo, N_hi)`.
:type endpoints_by_kind: dict[str, tuple[int, int]]
:param alpha: Significance threshold used for the per-metric `rej/n` counts. Default: `0.05`.
:type alpha: float, optional
:param step: If provided, prefixed to the header line for traceability. Default: `None`.
:type step: float or None, optional
:param kind_display: Display names (e.g. `{'san': 'SAN', 'diffit': 'DiffiT'}`). Default: `None`.
:type kind_display: dict[str, str] or None, optional
:param pre_endpoints_by_kind: Optional `{kind: (n_lo, n_hi)}` earlier-N range; when given, T3 prints its `gain_%`/`alpha` columns alongside the main range. Default: `None`.
:type pre_endpoints_by_kind: dict[str, tuple[int, int]] or None, optional
:returns: Nothing. Output goes to stdout.
:rtype: None

**Example**

```python
>>> from combra.metrics import print_convergence_report
>>> print_convergence_report(
...     result, METRICS, kinds=['san', 'diffit'],
...     endpoints_by_kind={'san': (360, 10000), 'diffit': (360, 10000)},
...     pre_endpoints_by_kind={'san': (360, 1000), 'diffit': (360, 1000)},
...     step=2.0, kind_display={'san': 'SAN', 'diffit': 'DiffiT'},
... )
```
````

````{py:function} combra.metrics.summarize_metric_distribution(result, metric, kinds, resolutions) -> dict[str, str]

Per `(kind, resolution)`, summarise the distribution of one column of a `convergence_stats` result. Useful as a one-line companion to `plot_metric_distribution`.

:param result: Output of `convergence_stats`.
:type result: pd.DataFrame
:param metric: Column to summarise (typically `'gain_pct'`).
:type metric: str
:param kinds: Generators to include.
:type kinds: list[str]
:param resolutions: Resolutions to include.
:type resolutions: list[int]
:returns: **summary** (*dict[str, str]*) – `{f'{kind}_{res}': 'mean=... median=... std=... n=...'}`.
:rtype: dict[str, str]

**Example**

```python
>>> from combra.metrics import summarize_metric_distribution
>>> summary = summarize_metric_distribution(result, 'gain_pct',
...                                         kinds=['san', 'diffit'],
...                                         resolutions=[256, 512])
>>> for tag, line in summary.items():
...     print(f'  {tag}: {line}')
# san_256:  mean=11.42 median=10.30 std=4.21 n=7
# diffit_512: mean=15.87 median=14.10 std=5.30 n=7
```
````

````{py:function} combra.metrics.plot_wdist_convergence_grid(records_by_panel, classes, kind_labels=None, grain_labels=None, row_keys=None, col_label_fn=None, title_fn=None, save_path=None, png_meta=None, fonts=None, height_per_row=560, width_per_col=720, metric='w1', y_label='W-dist', zero_line=False, panel_annotations=None, annot_kind_labels=None) -> plotly.graph_objects.Figure

Plotly grid of metric-vs-N curves. Rows are resolutions (or arbitrary `row_keys`), columns are classes. Curves on the same axes share a kind color/legend group; the legend is shown only once. Despite the name it plots any per-record metric, not just W-dist (see `metric`).

:param records_by_panel: `{(row_key, kind): [records]}` where each record carries `n_images`, `class`, and the chosen `metric` (as emitted by `metrics_vs_n`).
:type records_by_panel: dict[tuple, list[dict]]
:param classes: Column ordering.
:type classes: list[str]
:param kind_labels: `{kind: legend_label}`. Default: `None`.
:type kind_labels: dict[str, str] or None, optional
:param grain_labels: `{class: column_label}`. Default: `None`.
:type grain_labels: dict[str, str] or None, optional
:param row_keys: Ordered row identifiers. Defaults to sorted unique keys from `records_by_panel`. Default: `None`.
:type row_keys: list or None, optional
:param col_label_fn: Override `class → column label`. Default: `None`.
:type col_label_fn: callable or None, optional
:param title_fn: Override `(row_key, class) → subplot title`. Default: `None`.
:type title_fn: callable or None, optional
:param save_path: PNG output path. `None` skips saving. Default: `None`.
:type save_path: str or None, optional
:param png_meta: `{key: value}` written as PNG tEXt chunks. Default: `None`.
:type png_meta: dict or None, optional
:param fonts: Override the `title/axis/tick/legend` font sizes. Default: `None`.
:type fonts: dict or None, optional
:param height_per_row: Per-cell height in pixels. Default: `560`.
:type height_per_row: int, optional
:param width_per_col: Per-cell width in pixels. Default: `720`.
:type width_per_col: int, optional
:param metric: Record key to plot on the y-axis. Default: `'w1'`.
:type metric: str, optional
:param y_label: Y-axis title. Default: `'W-dist'`.
:type y_label: str, optional
:param zero_line: Draw a horizontal reference line at `y = 0`. Default: `False`.
:type zero_line: bool, optional
:param panel_annotations: `{(row_key, kind): text}` annotations drawn inside the matching panel. Default: `None`.
:type panel_annotations: dict or None, optional
:param annot_kind_labels: Display labels used when rendering `panel_annotations`. Default: `None`.
:type annot_kind_labels: dict[str, str] or None, optional
:returns: **fig** (*plotly.graph_objects.Figure*) – The grid figure.
:rtype: plotly.graph_objects.Figure

**Example**

```python
>>> from combra.metrics import plot_wdist_convergence_grid
>>> CLASSES = ['class_Ultra_Co11', 'class_Ultra_Co25', 'class_Ultra_Co6_2']
>>> GRAIN   = {'class_Ultra_Co11': 'small', 'class_Ultra_Co25': 'medium',
...            'class_Ultra_Co6_2': 'large'}
>>> fig = plot_wdist_convergence_grid(
...     records_by_panel, classes=CLASSES,
...     kind_labels={'real': 'original', 'san': 'SAN', 'diffit': 'DiffiT'},
...     grain_labels=GRAIN, row_keys=[256, 512, 1024],
...     save_path='wdist_convergence.png',
... )
>>> fig.show()
```
````

````{py:function} combra.metrics.plot_metric_distribution(result, metric, kinds, resolutions, kind_display=None, res_style=None, bin_step=1.5, x_range=None, save_path=None, png_meta=None, fonts=None, height=900, width=900) -> plotly.graph_objects.Figure

Per-kind distribution of one `convergence_stats` column (one subplot per kind, colored by resolution). Companion plot to `summarize_metric_distribution`.

:param result: Output of `convergence_stats`.
:type result: pd.DataFrame
:param metric: Column to bin (e.g. `'gain_pct'`).
:type metric: str
:param kinds: Generators — one subplot each.
:type kinds: list[str]
:param resolutions: Resolutions to overlay.
:type resolutions: list[int]
:param kind_display: Subplot titles. Default: `None`.
:type kind_display: dict[str, str] or None, optional
:param res_style: `{res: {'color': str, 'label': str}}`. Defaults to an auto-palette. Default: `None`.
:type res_style: dict[int, dict] or None, optional
:param bin_step: Histogram bin width on the x-axis. Default: `1.5`.
:type bin_step: float, optional
:param x_range: Clamp the x-axis. `None` → data-driven. Default: `None`.
:type x_range: tuple[float, float] or None, optional
:param save_path: PNG output path. Default: `None`.
:type save_path: str or None, optional
:param png_meta: PNG tEXt metadata. Default: `None`.
:type png_meta: dict or None, optional
:param fonts: Override `title/axis/tick/legend` font sizes. Default: `None`.
:type fonts: dict or None, optional
:param height: Figure height in pixels. Default: `900`.
:type height: int, optional
:param width: Figure width in pixels. Default: `900`.
:type width: int, optional
:returns: **fig** (*plotly.graph_objects.Figure*) – The distribution figure.
:rtype: plotly.graph_objects.Figure

**Example**

End-to-end: drive `metrics_vs_n` over every sweep folder, run `convergence_stats`, then print + plot.

```python
>>> from combra.metrics import (
...     metrics_vs_n, convergence_stats,
...     print_convergence_report, summarize_metric_distribution,
...     plot_wdist_convergence_grid, plot_metric_distribution,
... )
>>> from combra.metrics.compare import find_ethalon
>>> # … walk SOURCES → records_by_panel and df_metrics …
>>> result = convergence_stats(df_metrics, METRICS, ENDPOINTS_BY_KIND,
...                            expected_points={'real': 5, 'san': 8, 'diffit': 8})
>>> print_convergence_report(result, METRICS, KINDS, ENDPOINTS_BY_KIND,
...                          step=STEP, kind_display=KIND_DISPLAY)
>>> fig = plot_wdist_convergence_grid(records_by_panel, classes=CLASSES,
...                                   kind_labels=LABELS, save_path='wdist.png')
>>> fig = plot_metric_distribution(result, metric='gain_pct',
...                                kinds=KINDS, resolutions=[256, 512],
...                                save_path='gain.png')
```

A full notebook walkthrough is in `co_angles/3_metrics_convergence.ipynb`.
````

## See also

- {py:meth}`combra.data.PobeditDataset.generate_angles` — produces the angles parquets these comparators consume.
- {py:func}`combra.angles.angles_plot_grid` — visualise the same comparisons as overlaid grids.
- {py:func}`combra.approx.fit_plateau` — the plateau fitter used inside `convergence_stats`.
- {doc}`combra.stats <stats>` — Kendall + Fisher primitives.
- {doc}`FID example </examples/fid>` — multi-resolution loop using `compute_fid`.
