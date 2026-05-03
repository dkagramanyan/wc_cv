---
title: "Metrics"
weight: 11
---

The `combra.metrics` module bundles two families of metrics:

- **FID (Fréchet Inception Distance)** — image-level distance between a real and a generated set, on top of `pytorch_fid`.
- **Distribution comparison helpers** — for comparing per-class angle distributions stored as parquet files.

## FID

### `load_inception(dims=2048, device=None)`

Builds an `InceptionV3` feature extractor and moves it to the chosen device. If `device` is `None`, CUDA is used when available.

**Returns:** `(model, device)` — pass both to `compute_fid` to reuse a single model across calls.

### `collect_images(folder)`

Recursively walks `folder` and returns a sorted list of image paths whose extensions are recognised by `pytorch_fid` (handles per-class subdirectories).

### `compute_stats(folder, model, device, dims=2048, batch_size=50, num_workers=8)`

Computes the activation statistics `(mu, sigma, n_files)` for every image collected from `folder`.

### `compute_fid(real_folder, gen_folder, model=None, device=None, dims=2048, batch_size=50, num_workers=8)`

End-to-end FID between two folders.

**Parameters:**
- `real_folder`, `gen_folder`: paths to the real and generated image folders.
- `model`, `device`: optional pre-built InceptionV3 / device pair. If omitted, a fresh one is created on each call — pass them in when looping over many folder pairs.
- `dims`: feature dimension (`2048` is the standard choice).
- `batch_size`, `num_workers`: forwarded to the `pytorch_fid` data loader.

**Returns:** `(fid, n_real, n_gen)` — `fid` is a Python `float`.

**Example:**
```python
from combra.metrics import load_inception, compute_fid

model, device = load_inception()
fid, n_real, n_gen = compute_fid('data/real', 'data/gen', model=model, device=device)
print(f'FID = {fid:.4f}  (real={n_real}, gen={n_gen})')
```

A full multi-resolution loop is shown in the [FID example]({{< relref "/examples/fid" >}}).

## Distribution comparison

These helpers compare angle-distribution parquet files (as produced by `combra.angles`) against an "ethalon" reference.

### `load_rows(parquet_path)`

Reads a parquet file and returns a flat list of `{'meta': {..., 'step'}, 'prep': {...}}` rows. Both the legacy schema (one row per `(class, step)` with `meta.step`) and the newer per-class `prep_per_step` schema are flattened to the same uniform shape.

### `index_by_name_step(rows)`

Builds a `{(name, step): row}` lookup from the rows returned by `load_rows`. Used to align generated rows with their ethalon counterparts.

### `compute_metrics(real, fake)`

Given a matched `(real, fake)` pair, returns:

- `w_dist` — Wasserstein distance between the angle density curves.
- `mus_m`, `sig_m`, `amp_m` — relative differences `(fake - real) / real` of the bimodal Gaussian fit parameters.

### `compare_folders(folder_paths, ethalon_path, class_map=None, steps=None, coef=1000, verbose=True)`

Walks every parquet under each folder in `folder_paths`, looks each row up in `ethalon_path`, and prints/collects the metrics.

**Parameters:**
- `folder_paths`: iterable of directories, each holding generated parquets. The folder name suffix after `_kimg_` becomes the `kimg` tag in the output records (the full folder name is used if absent).
- `ethalon_path`: parquet file with reference distributions.
- `class_map`: optional `{fake_class: real_class}` mapping when the names don't match.
- `steps`: optional iterable of steps to keep — others are skipped.
- `coef`: scaling factor applied to `w_dist` in the printed table only.
- `verbose`: print a per-row table.

**Returns:** list of records `{'kimg', 'class', 'step', 'w_dist', 'mus_m', 'sig_m', 'amp_m'}`.

### `wdist_vs_n(folder, ethalon_path, class_map=None, step=5.0)`

Sweeps every parquet under `folder` (each assumed to be the same generator at a different sample size) and returns one record per `(parquet, class)` with `n_images` (read from `meta.n_images`, not the filename) and the Wasserstein distance to the ethalon.

**Returns:** list of `{'n_images', 'class', 'w_dist'}` records sorted by `(class, n_images)`.

**Example:**
```python
from combra.metrics import compare_folders, wdist_vs_n

records = compare_folders(
    ['runs/gen_kimg_1000', 'runs/gen_kimg_2000'],
    ethalon_path='ethalon.parquet',
    steps=[5.0],
)

curve = wdist_vs_n('runs/n_sweep', ethalon_path='ethalon.parquet', step=5.0)
```
