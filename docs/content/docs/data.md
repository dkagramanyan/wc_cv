---
title: "Data"
weight: 2
---

The `combra.data` module bundles example assets shipped with the package and two dataset classes — `PobeditDataset` and `PoliamidDataset` — that orchestrate preprocessing and per-class metric generation into parquet.

```python
from combra import data
```

## Bundled fetchers

Zero-argument helpers that load assets shipped under `combra/data/`. They're meant for smoke tests and minimum-reproducible examples — point them at your own data for real work.

### `combra.data.microstructure_images`

```python
combra.data.microstructure_images() → list[tuple[str, ndarray]]
```

Load the five SEM example images shipped with combra (`Ultra_Co{6_2, 8, 11, 15, 25}-001_part_1.jpeg`).

**Returns**

- **samples** (*list[tuple[str, ndarray]]*) – One `(filename, BGR uint8 image)` tuple per class.

**Return type**

*list[tuple(str, ndarray)]*

**Example**

```python
>>> from combra import data
>>> samples = data.microstructure_images()
>>> for name, img in samples:
...     print(name, img.shape, img.dtype)
```

---

### `combra.data.microstructure_class_path`

```python
combra.data.microstructure_class_path() → str
```

Return the path to the bundled 5-class WC/Co directory tree (one subfolder per class). Pass it to `PobeditDataset(path=...)` for end-to-end tests.

**Returns**

- **class_root** (*str*) – Absolute path to the class-folder root.

**Return type**

*str*

**Example**

```python
>>> from combra import data
>>> class_root = data.microstructure_class_path()
>>> ds = data.PobeditDataset(path=class_root, max_images_num_per_class=50)
```

---

### `combra.data.crack_images` / `combra.data.crack_labeled_json`

```python
combra.data.crack_images() → list[tuple[str, ndarray]]
combra.data.crack_labeled_json() → dict
```

Load the bundled crack image and matching label-studio annotation. `crack_images()` returns `[(filename, ndarray)]`; `crack_labeled_json()` returns the parsed JSON dict.

**Returns**

- **images** (*list[tuple[str, ndarray]]*) – From `crack_images`.
- **labels** (*dict*) – From `crack_labeled_json`.

**Return type**

*list[tuple(str, ndarray)]* / *dict*

**Example**

```python
>>> from combra import data
>>> (name, img), labels = data.crack_images()[0], data.crack_labeled_json()
>>> print(name, img.shape, list(labels.keys())[:3])
```

---

## `combra.data.PobeditDataset`

```python
CLASS combra.data.PobeditDataset(path=None, max_images_num_per_class=100,
                                 print_depth=3, compression='lzf')
```

Manages WC/Co microstructure images. The constructor opens an HDF5 file or converts a folder-of-classes to one; the first call to a `generate_*` method then builds a preprocessed-image cache (`.npy` memmap) that subsequent calls reuse.

**Layout convention** — when an h5 lives at `<data>/h5/<stem>.h5`, the cache is written to `<data>/cache/<stem>/<task>_n<N>.npy`. Legacy layout (cache next to the h5 with a `prep_cache_` filename prefix) is still supported for backwards compatibility.

**Parameters**

- **path** (*str or Path, optional*) – Either a folder containing `class_*/` subfolders of images, or an existing `*.h5` produced by combra. When a folder is passed, the constructor converts it to HDF5 next to the source on first run. Default: `None`.
- **max_images_num_per_class** (*int or None, optional*) – Cap per class. `None` uses every available image. Each class is then truncated to the smallest class count to keep the cache rectangular. Default: `100`.
- **print_depth** (*int, optional*) – Tree-print depth of the source structure at construction time. Default: `3`.
- **compression** (*str or None, optional*) – HDF5 compression used when converting a folder. Pass `None` for uncompressed (faster prep cache builds, larger file). Default: `'lzf'`.

**Example**

```python
>>> from combra import data
>>> ds = data.PobeditDataset(
...     path=data.microstructure_class_path(),
...     max_images_num_per_class=360,
... )
```

---

### `PobeditDataset.generate_angles`

```python
PobeditDataset.generate_angles(save_path, types_dict, step, workers=20,
                               angles_tol=3, min_segment_len=10.0,
                               keep_contours=False, chunksize=64,
                               run_meta=None) → Path
```

Compute angle distributions for every image and write them as parquet. One row per class; per-step Gaussian-fit results are stored as a list under `prep_per_step`.

The output parquet's `run_meta` struct column carries full provenance — model tag, source h5, code commit, generation timestamp, and the exact extraction params used.

**Parameters**

- **save_path** (*str or Path*) – Directory for the output. The filename is computed as `angles_n{N}.parquet`.
- **types_dict** (*dict[str, str]*) – Maps class names (e.g. `'Ultra_Co11'`) to display labels used in legends.
- **step** (*float or list[float]*) – Histogram-binning step(s) in degrees. Pass a list to fit multiple steps in one pass; all are stored under `prep_per_step`.
- **workers** (*int, optional*) – Multiprocessing pool size. Default: `20`.
- **angles_tol** (*float, optional*) – Douglas–Peucker simplification tolerance applied before angle extraction. Default: `3`.
- **min_segment_len** (*float, optional*) – Vertices producing shorter neighbouring segments are iteratively removed before angle calculation. Higher → smoother distributions, fewer angles. Recommended: 5–20 px. Default: `10.0`.
- **keep_contours** (*bool, optional*) – If `True`, the heavy `contours_angles` / `contours_angles_per_image` columns are populated. Off by default to keep parquets small. Default: `False`.
- **chunksize** (*int, optional*) – Worker chunk size for `pool.imap_unordered`. Default: `64`.
- **run_meta** (*dict or None, optional*) – Caller-supplied provenance. May set `family`, `resolution`, `tags`, `notes`. Everything else (`model_tag`, `kimg`, `source_h5`, `code_commit`, `generated_at`, `extraction_params`) is filled automatically. Default: `None`.

**Returns**

- **out_path** (*Path*) – The written parquet path.

**Return type**

*Path*

**Example**

```python
>>> from combra import data
>>> ds = data.PobeditDataset(
...     path='./data/h5/00017-diffit-256-gpus2-batch192_N10000.h5',
...     max_images_num_per_class=1000,
... )
>>> ds.generate_angles(
...     save_path='./data/angles/00017-diffit-256-gpus2-batch192_N10000_msl5',
...     types_dict={'Ultra_Co11': 'small grain',
...                 'Ultra_Co25': 'medium grain',
...                 'Ultra_Co6_2': 'large grain'},
...     step=[0.1, 0.5, 1, 2, 3, 4, 5],
...     workers=20, min_segment_len=5.0, keep_contours=False,
...     run_meta={'family': 'diffit', 'resolution': 256, 'tags': ['kimg-sweep'], 'notes': ''},
... )
```

Inspecting the output:

```python
>>> import pyarrow.parquet as pq
>>> t = pq.read_table('.../angles_n1000.parquet', columns=['run_meta'])
>>> print(t['run_meta'][0].as_py()['model_tag'])
>>> print(t['run_meta'][0].as_py()['code_commit'])
```

---

### `PobeditDataset.generate_beams`

```python
PobeditDataset.generate_beams(save_path, types_dict, step, pixel,
                              start=2, end=-3, workers=20, mvee_tol=0.2,
                              keep_contours=False, run_meta=None) → Path
```

Compute Minimum Volume Enclosing Ellipse (MVEE) beam parameters for every image and write them as parquet. One row per `(class, step)`; the per-step linear-fit results live alongside the raw beam/centroid arrays.

**Parameters**

- **save_path** (*str or Path*) – Output directory. Filename is `beams_n{N}.parquet`.
- **types_dict** (*dict[str, str]*) – Class-name to display-label map.
- **step** (*float or list[float]*) – Histogram-binning step(s) for the beam-length distribution.
- **pixel** (*float*) – Physical pixel size (`pixel2meter` in the output `meta`). Beam lengths are scaled by this.
- **start** (*int, optional*) – Lower slice bound applied to the binned log-density before the linear fit (trims poorly-sampled extremes). Default: `2`.
- **end** (*int, optional*) – Upper slice bound applied to the binned log-density before the linear fit (trims poorly-sampled extremes). Default: `-3`.
- **workers** (*int, optional*) – Multiprocessing pool size. Default: `20`.
- **mvee_tol** (*float, optional*) – Convergence tolerance for the MVEE solve. Lower → tighter ellipses, slower. Default: `0.2`.
- **keep_contours** (*bool, optional*) – If `True`, populate `contours_mvee`. Default: `False`.
- **run_meta** (*dict or None, optional*) – Caller-supplied provenance (`family`, `resolution`, `tags`, `notes`). The rest is filled automatically. The beams-flavour `extraction_params` records `pixel`, `start`, `end`, `mvee_tol`, `keep_contours`. Default: `None`.

**Returns**

- **out_path** (*Path*) – The written parquet path.

**Return type**

*Path*

**Example**

```python
>>> from combra import data
>>> ds = data.PobeditDataset(path=data.microstructure_class_path())
>>> ds.generate_beams(
...     save_path='./beams',
...     types_dict={'Ultra_Co11': 'medium grain',
...                 'Ultra_Co25': 'fine grain'},
...     step=4, pixel=50/1000,
...     run_meta={'family': 'real', 'resolution': 256},
... )
```

---

### `PobeditDataset.close`

```python
PobeditDataset.close() → None
```

Tear down the persistent worker pool. `generate_angles` / `generate_beams` already call this on exit; you only need to invoke it explicitly when reusing the same dataset across long-lived sessions and want to free workers between phases.

**Returns**

None. Closes and joins the persistent worker pool if one is open.

**Return type**

*None*

---

## `combra.data.PoliamidDataset`

```python
CLASS combra.data.PoliamidDataset(images_folder_path, group_size=250)
```

Polyamide fracture-image dataset. Filenames are expected to be plain integers; images are bucketed into consecutive groups of `group_size` and metrics are aggregated per group.

**Parameters**

- **images_folder_path** (*str or Path*) – Folder of integer-named image files (e.g. `0.png`, `1.png`, …).
- **group_size** (*int, optional*) – Consecutive images per group. Default: `250`.

---

### `PoliamidDataset.generate`

```python
PoliamidDataset.generate(out_path, n_jobs=20, N=500) → None
```

For each group, compute contour fractal dimension, image-level fractal dimension, contour lengths and areas. Writes one row per group to `out_path`.

**Parameters**

- **out_path** (*str or Path*) – Output parquet path.
- **n_jobs** (*int, optional*) – Worker count. Default: `20`.
- **N** (*int, optional*) – Minimum contour length below which a contour is skipped. Default: `500`.

**Returns**

None. Writes one row per group to `out_path` as a side effect.

**Return type**

*None*

**Example**

```python
>>> from combra import data
>>> pa_ds = data.PoliamidDataset('/path/to/poliamid', group_size=250)
>>> pa_ds.generate('poliamid.parquet', n_jobs=20, N=500)
```

---

## `combra.data.generate_angles_sweep`

```python
combra.data.generate_angles_sweep(h5_path, out_dir, ns, step, types_dict,
                                  tag='', run_meta=None, **gen_kwargs) → None
```

Batch helper that builds an angle-convergence sweep: for each `N` in `ns` it instantiates a `PobeditDataset` capped at `N` images per class and calls `generate_angles`, writing one `angles_n{N}.parquet` per sample size into `out_dir`. A parquet is only (re)generated when it is missing **or** exists but lacks rows at `step` (`generate_angles` overwrites, it does not append), so reruns are cheap. One status line is printed per call.

**Parameters**

- **h5_path** (*str or Path*) – Source HDF5 (or class-folder) passed to each `PobeditDataset`.
- **out_dir** (*str or Path*) – Directory for the per-N parquets.
- **ns** (*Iterable[int]*) – Sample sizes to sweep — one parquet per value.
- **step** (*float*) – Histogram step used both for generation and for the "already present" check.
- **types_dict** (*dict[str, str]*) – Class-name to display-label map, forwarded to `generate_angles`.
- **tag** (*str, optional*) – Prefix shown in brackets in the printed status lines (e.g. the generator family). Default: `''`.
- **run_meta** (*dict or None, optional*) – Provenance forwarded to each `generate_angles` call. Default: `None`.
- **gen_kwargs** – Extra keyword arguments forwarded verbatim to `PobeditDataset.generate_angles` (e.g. `workers`, `min_segment_len`, `keep_contours`).

**Returns**

None. Parquets are written to `out_dir` as a side effect.

**Return type**

*None*

**Example**

```python
>>> from combra import data
>>> data.generate_angles_sweep(
...     h5_path='./data/h5/00017-diffit-256_N10000.h5',
...     out_dir='./sweeps/diffit_256_msl5',
...     ns=[100, 250, 1000, 10000],
...     step=2.0,
...     types_dict={'Ultra_Co11': 'small grain',
...                 'Ultra_Co25': 'medium grain',
...                 'Ultra_Co6_2': 'large grain'},
...     tag='diffit', min_segment_len=5.0, workers=20,
... )
```

The output sweep folder is exactly the shape [`combra.metrics.metrics_vs_n`]({{< relref "/docs/metrics" >}}) and [`combra.metrics.find_ethalon`]({{< relref "/docs/metrics" >}}) expect.

---

## See also

- [`combra.angles`]({{< relref "/docs/angles" >}}) — the angle extractor `PobeditDataset.generate_angles` calls per image.
- [`combra.mvee`]({{< relref "/docs/mvee" >}}) — the MVEE primitive `generate_beams` calls per image, plus the parquet plotters.
- [`combra.metrics`]({{< relref "/docs/metrics" >}}) — comparing parquet outputs across runs.
