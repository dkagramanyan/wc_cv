---
title: "Metrics"
weight: 11
---

The `combra.metrics` module bundles two families of metrics:

- **FID (Fréchet Inception Distance)** — image-level distance between a real and a generated set, on top of `pytorch_fid`.
- **Distribution comparison helpers** — for comparing per-class angle distributions stored as parquet files.

```python
from combra import metrics
```

## FID

### `combra.metrics.load_inception`

```python
load_inception(dims=2048, device=None)
```

Build an InceptionV3 feature extractor and move it to the chosen device.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `dims` | `int` | `2048` | Feature dimension. `2048` is the standard FID setting. |
| `device` | `str \| None` | `None` | Torch device string. When `None`, uses CUDA if available. |

**Returns** `(model, device)` — pass both to `compute_fid` to reuse the model across calls.

---

### `combra.metrics.collect_images`

```python
collect_images(folder)
```

Recursively walk `folder` and return a sorted list of image paths whose extensions are recognised by `pytorch_fid` (handles per-class subdirectories).

**Returns** `list[Path]`.

---

### `combra.metrics.compute_stats`

```python
compute_stats(folder, model, device, dims=2048, batch_size=50, num_workers=8)
```

Compute the activation statistics `(mu, sigma, n_files)` for every image collected from `folder`.

**Returns** `(mu, sigma, n_files)` — mean activation, covariance, and image count.

---

### `combra.metrics.compute_fid`

```python
compute_fid(real_folder, gen_folder, model=None, device=None,
            dims=2048, batch_size=50, num_workers=8)
```

End-to-end FID between two folders.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `real_folder`, `gen_folder` | `str` | — | Paths to the real and generated image folders. |
| `model`, `device` | from `load_inception` | `None` | Pre-built InceptionV3 / device pair. If omitted, a fresh one is created on each call — pass them in when looping over many folder pairs. |
| `dims` | `int` | `2048` | Feature dimension. |
| `batch_size`, `num_workers` | `int` | `50`, `8` | Forwarded to the `pytorch_fid` data loader. |

**Returns**

| name | type | description |
| --- | --- | --- |
| `fid` | `float` | Distance value. |
| `n_real` | `int` | Number of images contributing on the real side. |
| `n_gen` | `int` | Same on the generated side. |

**Example**

```python
from combra.metrics import load_inception, compute_fid

model, device = load_inception()
fid, n_real, n_gen = compute_fid('data/real', 'data/gen', model=model, device=device)
print(f'FID = {fid:.4f}  (real={n_real}, gen={n_gen})')
```

A full multi-resolution loop is shown in the [FID example]({{< relref "/examples/fid" >}}).

---

## Distribution comparison

These helpers compare angle-distribution parquet files (as produced by `PobeditDataset.generate_angles`) against an "ethalon" reference.

### `combra.metrics.load_rows`

```python
load_rows(parquet_path)
```

Read a parquet file and return a flat list of `{'meta': {..., 'step'}, 'prep': {...}}` rows. Both the legacy schema (one row per `(class, step)` with `meta.step`) and the newer per-class `prep_per_step` schema are flattened to the same uniform shape, so callers don't need to branch.

**Returns** `list[dict]` — each entry has `meta` (with synthesised `step`) and `prep` (single step's fit + density curves).

---

### `combra.metrics.index_by_name_step`

```python
index_by_name_step(rows)
```

Build a `{(name, step): row}` lookup from rows returned by `load_rows`. Used internally to align generated rows with their ethalon counterparts.

---

### `combra.metrics.compute_metrics`

```python
compute_metrics(real, fake)
```

Given a matched `(real, fake)` row pair, compute the per-step Wasserstein distance and bimodal-Gauss parameter deltas.

**Returns**

| name | type | description |
| --- | --- | --- |
| `w_dist` | `float` | Wasserstein distance between the angle density curves, in degrees. |
| `mus_m` | `ndarray[2]` | Relative differences `(fake - real) / real` on the two Gaussian means. |
| `sig_m` | `ndarray[2]` | Same for the sigmas. |
| `amp_m` | `ndarray[2]` | Same for the amplitudes. |

---

### `combra.metrics.compare_folders`

```python
compare_folders(folder_paths, ethalon_path, class_map=None, steps=None,
                coef=1000, verbose=True, fid_by_kimg=None)
```

Walk every parquet under each folder in `folder_paths`, look each row up in `ethalon_path`, print/collect the metrics.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `folder_paths` | `Iterable[str]` | — | Folders, each holding one generator's parquets. The folder name suffix after `_kimg_` becomes the `kimg` tag in records (full name used if absent). May also pass `.parquet` paths directly to process exactly one file. |
| `ethalon_path` | `str` | — | Parquet file with reference distributions. |
| `class_map` | `dict[str, str] \| None` | `None` | `{fake_class: real_class}` mapping when names don't match. |
| `steps` | `Iterable[float] \| None` | `None` | Subset of steps to keep — others are skipped. |
| `coef` | `float` | `1000` | Multiplier applied to `w_dist` in the printed table only (records keep raw values). |
| `verbose` | `bool` | `True` | Print a per-row table. |
| `fid_by_kimg` | `dict[str, float] \| None` | `None` | Optional `{kimg_key: fid}` map — prints a `[kimg=… FID=…]` header above each checkpoint's rows. |

**Returns** `list[dict]` — records `{'kimg', 'class', 'step', 'w_dist', 'mus_m', 'sig_m', 'amp_m'}`.

---

### `combra.metrics.compare_pairs`

```python
compare_pairs(pairs, step, coef=1000, verbose=True, label_header='label')
```

Like `compare_folders` but accepts a list of explicit `(label, ethalon_pq, fake_pq, class_map)` tuples — one row per pair, exactly one step. Used by `4_grid_plot.ipynb` to print one row per resolution.

---

### `combra.metrics.wdist_vs_n`

```python
wdist_vs_n(folder, ethalon_path, class_map=None, step=5.0)
```

Sweep every parquet under `folder` (each assumed to be the same generator at a different sample size) and return one record per `(parquet, class)` with `n_images` (read from `meta.n_images`) and the Wasserstein distance to the ethalon.

**Returns** `list[dict]` — sorted by `(class, n_images)`, each entry `{'n_images', 'class', 'w_dist'}`.

**Example**

```python
from combra.metrics import compare_folders, wdist_vs_n

records = compare_folders(
    ['runs/gen_kimg_1000', 'runs/gen_kimg_2000'],
    ethalon_path='ethalon.parquet',
    steps=[5.0],
)

curve = wdist_vs_n('runs/n_sweep', ethalon_path='ethalon.parquet', step=5.0)
```

---

## See also

- [`combra.data.PobeditDataset.generate_angles`]({{< relref "/docs/data#pobeditdatasetgenerate_angles" >}}) — produces the angles parquets these comparators consume.
- [`combra.angles.angles_plot_grid`]({{< relref "/docs/angles" >}}) — visualise the same comparisons as overlaid grids.
- [FID example]({{< relref "/examples/fid" >}}) — multi-resolution loop using `load_inception` + `compute_fid`.
