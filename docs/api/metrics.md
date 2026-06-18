# combra.metrics

The `combra.metrics` module bundles four families of metrics:

- **FID (Fréchet Inception Distance)** — image-level distance between a real and a generated set. combra does not implement FID itself; `combra.metrics.fid` delegates to the reference library [pytorch-fid](https://github.com/mseitzer/pytorch-fid), which ships as a core dependency and downloads/caches its own InceptionV3 weights on first use, so no manual model setup is required.
- **Training-loop / batch metrics** — score an in-memory batch of generated images on the fly (no parquet round-trip): CLIP-MMD (`compute_cmmd`), Fréchet distance on DINOv2 features (`compute_fd_dinov2`), and angle-Wasserstein distances between two batches' angle densities (`compute_w1`/`compute_w2`/circular variants). Every two-input metric takes the *reference* batch first, then the *generated* batch.
- **Distribution comparison helpers** — for comparing per-class angle distributions stored as parquet files.
- **Convergence analysis** — N-sweep aggregation, Kendall trend tests, plateau fits, and the convergence-grid / gain-distribution plots used by `3_metrics_convergence.ipynb`.

```python
from combra import metrics
```

## FID

`combra.metrics.fid` is a thin wrapper over [pytorch-fid](https://github.com/mseitzer/pytorch-fid) — it exposes a single convenience entry point, `compute_fid`. Everything runs on PyTorch; `compute_fid` selects CUDA automatically when available and falls back to CPU.

````{py:function} combra.metrics.compute_fid(reference_folder, generated_folder, batch_size=50, dims=2048, device=None, num_workers=1) -> float

Classic folder-vs-folder FID, computed with [pytorch-fid](https://github.com/mseitzer/pytorch-fid). Walks both folders, runs every image through InceptionV3, and returns the Fréchet distance between the two activation distributions. The weights are downloaded and cached by pytorch-fid on first use.

:param reference_folder: Path to the reference (real) image folder.
:type reference_folder: str or Path
:param generated_folder: Path to the generated image folder.
:type generated_folder: str or Path
:param batch_size: Forward-pass batch size. Default: `50`.
:type batch_size: int, optional
:param dims: Dimensionality of the InceptionV3 feature layer (`64`, `192`, `768`, or `2048`). Default: `2048`.
:type dims: int, optional
:param device: Torch device to run on (e.g. `'cuda'` or `'cpu'`). When `None`, picks `'cuda'` if available, otherwise `'cpu'`. Default: `None`.
:type device: str or torch.device or None, optional
:param num_workers: Number of dataloader workers. Default: `1`.
:type num_workers: int, optional
:returns: **fid** (*float*) – The Fréchet (FID) distance. `0.0` when both folders are identical.
:rtype: float

**Example**

```python
>>> from combra.metrics import compute_fid
>>> fid = compute_fid('data/real', 'data/gen')  # CUDA when available, else CPU
>>> print(f'FID = {fid:.4f}')
```

A full multi-resolution loop is shown in the {doc}`FID example </examples/fid>`.
````

`compute_fid` is a thin convenience wrapper over [pytorch-fid](https://github.com/mseitzer/pytorch-fid)'s folder-level primitive, adding sensible defaults and automatic device selection. FID is always computed from images; combra does not expose the raw mean/covariance (`mu`/`sigma`) form of the Fréchet distance.

## Training-loop metrics

These score an **in-memory batch of images** (a numpy array or torch tensor) and
return a scalar, so generated samples can be evaluated on the fly during training
without writing them to disk. They live in `combra.metrics.training` and accept a
batch shaped `(H, W)`, `(N, H, W)`, `(N, C, H, W)`, or `(N, H, W, C)`.

```python
from combra import metrics
```

### Image-feature metrics

`compute_cmmd` and `compute_fd_dinov2` take a *reference* and a *generated* image
batch and compare deep-feature distributions. Both need the `gen-metrics` extra
(`pip install ".[gen-metrics]"`); the DINOv2 backbone is fetched from `torch.hub`
on first use.

````{py:function} combra.metrics.compute_cmmd(reference_images, generated_images, model_name='ViT-L-14-336', pretrained='openai', device=None, batch_size=64, sigma=10.0, scale=1000.0) -> float

CLIP-MMD (CMMD) between a reference and a generated image batch. Features come from a ready CLIP model loaded with [open_clip](https://github.com/mlfoundations/open_clip); the distance is the Gaussian-RBF Maximum Mean Discrepancy of [Google's CMMD](https://github.com/google-research/google-research/tree/master/cmmd) (`sigma=10`, `scale=1000`).

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
:returns: **cmmd** (*float*) – The scaled CLIP-MMD distance.
:rtype: float

**Example**

```python
>>> from combra.metrics import compute_cmmd
>>> cmmd = compute_cmmd(real_batch, generated_batch)   # CUDA when available
>>> print(f'CMMD = {cmmd:.4f}')
```
````

````{py:function} combra.metrics.compute_fd_dinov2(reference_images, generated_images, model_name='dinov2_vitb14', device=None, batch_size=64, image_size=224) -> float

Fréchet distance between the [DINOv2](https://github.com/facebookresearch/dinov2) features of two image batches. The backbone is loaded from `torch.hub` (`facebookresearch/dinov2`); the Fréchet distance itself is computed with pytorch-fid's `calculate_frechet_distance`.

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
:returns: **fd** (*float*) – Fréchet distance between the two DINOv2 feature sets.
:rtype: float
````

### Angle-Wasserstein metrics

These reduce *both* a reference and a generated batch to their angle densities
(the same image → angles → histogram pipeline used to build the parquet datasets)
and compare them.

````{py:function} combra.metrics.compute_w1(reference_images, generated_images, step=None, **angle_kw) -> float

1-Wasserstein distance (in degrees) between the reference and generated angle densities. `compute_w2`, `compute_circular_w1`, and `compute_circular_w2` share this exact signature (`reference_images, generated_images, step=None, **angle_kw`) and differ only in the distance returned.

:param reference_images: Reference image batch.
:type reference_images: ndarray or torch.Tensor
:param generated_images: Generated image batch to score.
:type generated_images: ndarray or torch.Tensor
:param step: Histogram bin width in degrees. Defaults to `config.STEP` (`5.0`). Default: `None`.
:type step: float or None, optional
:param angle_kw: Extra keyword args forwarded to `images_to_angle_density` (`border_eps`, `tol`, `min_segment_len`).
:returns: **dist** (*float*) – The Wasserstein distance in degrees.
:rtype: float

**Example**

```python
>>> from combra.metrics import compute_w1, compute_circular_w2
>>> w1  = compute_w1(real_batch, generated_batch)
>>> cw2 = compute_circular_w2(real_batch, generated_batch)
```
````

````{py:function} combra.metrics.compute_w2(reference_images, generated_images, step=None, **angle_kw) -> float

2-Wasserstein distance (in degrees) between the reference and generated angle densities. Same arguments as {py:func}`combra.metrics.compute_w1`.
````

````{py:function} combra.metrics.compute_circular_w1(reference_images, generated_images, step=None, **angle_kw) -> float

Circular 1-Wasserstein distance (angles wrap at 360°) between the reference and generated angle densities. Same arguments as {py:func}`combra.metrics.compute_w1`.
````

````{py:function} combra.metrics.compute_circular_w2(reference_images, generated_images, step=None, **angle_kw) -> float

Circular 2-Wasserstein distance (angles wrap at 360°) between the reference and generated angle densities. Same arguments as {py:func}`combra.metrics.compute_w1`.
````

````{py:function} combra.metrics.images_to_angle_density(images, step=None, border_eps=5, tol=3, min_segment_len=10.0) -> tuple[ndarray, ndarray]

Reduce a batch of images to a single angle density `(x, y)`. Runs `_preprocess_image → get_angles` on each image, pools all vertex angles, then histograms them with {py:func}`combra.stats.stats_preprocess` at `step` degrees.

:param images: Image batch.
:type images: ndarray or torch.Tensor
:param step: Histogram bin width in degrees. Defaults to `config.STEP` (`5.0`). Default: `None`.
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

````{py:function} combra.metrics.wasserstein_density_metrics(x_reference, y_reference, x_generated, y_generated) -> dict

Linear and circular Wasserstein distances between two angle densities (in degrees). Both densities are resampled onto a shared dense grid and normalised to unit mass; the four distances are delegated to [POT](https://pythonot.github.io/) (linear via `ot.wasserstein_1d`, circular via `ot.wasserstein_circle`), with `w_dist` kept as the historical scipy value for backwards compatibility.

:param x_reference: Reference angle-bin locations (degrees).
:type x_reference: ndarray
:param y_reference: Reference densities at `x_reference`.
:type y_reference: ndarray
:param x_generated: Generated angle-bin locations (degrees).
:type x_generated: ndarray
:param y_generated: Generated densities at `x_generated`.
:type y_generated: ndarray
:returns: **metrics** (*dict*) – Keys `w_dist`, `w1`, `w2`, `circular_w1`, `circular_w2`, all in degrees.
:rtype: dict
````

```{warning}
The reference distributions in `combra.metrics.config` are currently **random
placeholders**. Replace `REFERENCE_GAUSS_PARAMS` / `REFERENCE_ANGLE_DENSITY` with
the real per-resolution / per-grain-class values before relying on the
`reference_density` results.
```

````{py:function} combra.metrics.reference_density(resolution, grain_class, mode='gauss') -> tuple[ndarray, ndarray]

Return the fixed reference angle density `(x, y)` for a `(resolution, grain_class)`. `mode='gauss'` builds it from the bimodal-Gaussian parameters in `config.REFERENCE_GAUSS_PARAMS`; `mode='empirical'` returns the stored measured distribution. `y` is normalised to unit mass in both cases.

:param resolution: Reference key (e.g. `256`).
:type resolution: int
:param grain_class: Reference key (e.g. `'Ultra_small'`).
:type grain_class: str
:param mode: `'gauss'` or `'empirical'`. Default: `'gauss'`.
:type mode: str, optional
:returns: **density** (*tuple[ndarray, ndarray]*) – `(x, y)` reference angle density.
:rtype: tuple(ndarray, ndarray)
````

### Unified entry point

````{py:function} combra.metrics.compute_all_metrics(reference_images, generated_images, *, step=None, device=None, angle_kw=None) -> dict

Run every batch metric for a (reference, generated) batch pair in parallel and return one flat dict. The angle-Wasserstein metrics (`w1`, `w2`, `circular_w1`, `circular_w2`) compare the two batches' angle densities; `fid` (classic InceptionV3 FID on the in-memory batches), `cmmd`, and `fd_dinov2` compare their deep features. An image-feature metric that cannot run (missing optional dependency or no network) is recorded as `nan` and logged, so the angle metrics still come back.

:param reference_images: Batch of reference (real) images.
:type reference_images: ndarray or torch.Tensor
:param generated_images: Batch of generated images.
:type generated_images: ndarray or torch.Tensor
:param step: Histogram bin width in degrees. Defaults to `config.STEP`. Default: `None`.
:type step: float or None, optional
:param device: Torch device for the image-feature metrics. Default: `None`.
:type device: str or torch.device or None, optional
:param angle_kw: Extra keyword args forwarded to `images_to_angle_density`. Default: `None`.
:type angle_kw: dict or None, optional
:returns: **results** (*dict*) – Flat `{metric_name: value}` dict with keys `w1`, `w2`, `circular_w1`, `circular_w2`, `fid`, `cmmd`, `fd_dinov2`.
:rtype: dict

**Example**

```python
>>> from combra.metrics import compute_all_metrics
>>> scores = compute_all_metrics(real_batch, generated_batch)
>>> scores['cmmd'], scores['w1']
```
````

## Distribution comparison

These helpers compare angle-distribution parquet files (as produced by {py:meth}`combra.data.PobeditDataset.generate_angles`) against an "ethalon" reference.

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

````{py:function} combra.metrics.index_by_name_step(rows) -> dict

Build a `{(name, step): row}` lookup from rows returned by `load_rows`. Used internally to align generated rows with their ethalon counterparts.

:param rows: Output of `load_rows`.
:type rows: list[dict]
:returns: **index** (*dict*) – `{(name, step): row}`.
:rtype: dict

**Example**

```python
>>> from combra.metrics import load_rows, index_by_name_step
>>> rows = load_rows('ethalon.parquet')
>>> idx = index_by_name_step(rows)
>>> row = idx[('class_Ultra_Co11', 2.0)]   # exact (class, step) lookup
```
````

````{py:function} combra.metrics.find_ethalon(sweep_dir) -> Path

Locate the largest-N parquet inside a sweep folder. Used as the reference distribution for convergence comparisons (each `metrics_vs_n` curve compares smaller-N parquets against this file).

:param sweep_dir: Folder containing `angles_n*.parquet` files.
:type sweep_dir: str or Path
:returns: **parquet** (*Path or None*) – Path of the parquet with the largest `meta.n_images`, or `None` if the folder is empty.
:rtype: Path

**Example**

From `co_angles/3_metrics_convergence.ipynb`:

```python
>>> from combra.metrics import find_ethalon
>>> ethalon = find_ethalon('./data/angles/o_bc_left_4x_1536_1024x1024_256x256_rgb_N360_msl5')
>>> print(ethalon)   # …/angles_n360.parquet
```
````

````{py:function} combra.metrics.parquet_has_step(parquet_path, step) -> bool

True iff the parquet exists AND contains rows tagged with `step`. Used to decide whether a previously-generated sweep parquet is still usable when `STEP` changes (`generate_angles` overwrites, does not append).

:param parquet_path: Parquet to inspect.
:type parquet_path: str or Path
:param step: Step value to look for.
:type step: float
:returns: **has_step** (*bool*) – Whether the file contains a row at this step.
:rtype: bool

**Example**

```python
>>> from combra.metrics import parquet_has_step
>>> # Skip re-generation if the parquet already has rows at STEP=2.0.
>>> if not parquet_has_step('out/angles_n1000.parquet', step=2.0):
...     ...  # regenerate
```
````

````{py:function} combra.metrics.compute_metrics(real, fake) -> tuple[float, ndarray, ndarray, ndarray]

Given a matched `(real, fake)` row pair, compute the per-step Wasserstein distance and bimodal-Gauss parameter deltas.

:param real: Single row from `load_rows` for the reference distribution.
:type real: dict
:param fake: Single row from `load_rows`, expected to have matching `meta.step`.
:type fake: dict
:returns: **w_dist** (*float*) – Wasserstein distance between the angle density curves, in degrees; and **mus_m** (*ndarray[2]*) – Relative differences `(fake - real) / real` on the two Gaussian means; and **sig_m** (*ndarray[2]*) – Same for the sigmas; and **amp_m** (*ndarray[2]*) – Same for the amplitudes.
:rtype: tuple(float, ndarray, ndarray, ndarray)

**Example**

```python
>>> from combra.metrics import load_rows, index_by_name_step, compute_metrics
>>> real_idx = index_by_name_step(load_rows('ethalon.parquet'))
>>> fake_idx = index_by_name_step(load_rows('gen.parquet'))
>>> real, fake = real_idx[('class_Ultra_Co11', 2.0)], fake_idx[('class_Ultra_Co11', 2.0)]
>>> w_dist, mus_m, sig_m, amp_m = compute_metrics(real, fake)
>>> print(f'w_dist={w_dist:.4f}°   |mu1 rel|={abs(mus_m[0])*100:.2f}%')
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
:param coef: Multiplier applied to `w_dist` in the printed table only (records keep raw values). Default: `1000`.
:type coef: float, optional
:param verbose: Print a per-row table. Default: `True`.
:type verbose: bool, optional
:param fid_by_kimg: Optional `{kimg_key: fid}` map — prints a `[kimg=… FID=…]` header above each checkpoint's rows. Default: `None`.
:type fid_by_kimg: dict[str, float] or None, optional
:returns: **records** (*list[dict]*) – `{'kimg', 'class', 'step', 'w_dist', 'mus_m', 'sig_m', 'amp_m'}` per matched row.
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
:param coef: Multiplier applied to `w_dist` in the printed table only. Default: `1000`.
:type coef: float, optional
:param verbose: Print a per-row table. Default: `True`.
:type verbose: bool, optional
:param label_header: Column header used in the printed table for the first column. Default: `'label'`.
:type label_header: str, optional
:returns: **records** (*list[dict]*) – `{'label', 'class', 'step', 'w_dist', 'mus_m', 'sig_m', 'amp_m'}`.
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

Walk every parquet under `folder` (each assumed to be the same generator at a different sample size) and emit one record per `(parquet, class)` with all seven metrics: `w_dist` plus the Gaussian-fit relative errors `(mu1, mu2, sigma1, sigma2, amp1, amp2)`. N is read from `meta.n_images`, so the filename convention does not matter.

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
:returns: **records** (*list[dict]*) – Sorted by `(class, n_images)`. Each entry: `{'n_images', 'class', 'w_dist', 'mu1', 'mu2', 'sigma1', 'sigma2', 'amp1', 'amp2'}`.
:rtype: list[dict]

**Example**

```python
>>> from combra.metrics import metrics_vs_n, find_ethalon
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
:param metrics: Metric column names to analyse (e.g. `['w_dist', 'mu1', 'mu2', 'sigma1', 'sigma2', 'amp1', 'amp2']`).
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
>>> METRICS  = ['w_dist', 'mu1', 'mu2', 'sigma1', 'sigma2', 'amp1', 'amp2']
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

````{py:function} combra.metrics.plot_wdist_convergence_grid(records_by_panel, classes, kind_labels=None, grain_labels=None, row_keys=None, col_label_fn=None, title_fn=None, save_path=None, png_meta=None, fonts=None, height_per_row=560, width_per_col=720, metric='w_dist', y_label='W-dist', zero_line=False, panel_annotations=None, annot_kind_labels=None) -> plotly.graph_objects.Figure

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
:param metric: Record key to plot on the y-axis. Default: `'w_dist'`.
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
...     metrics_vs_n, find_ethalon, convergence_stats,
...     print_convergence_report, summarize_metric_distribution,
...     plot_wdist_convergence_grid, plot_metric_distribution,
... )
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
