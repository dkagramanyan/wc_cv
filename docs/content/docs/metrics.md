---
title: "Metrics"
weight: 11
---

The `combra.metrics` module bundles three families of metrics:

- **FID (Fréchet Inception Distance)** — image-level distance between a real and a generated set, on top of `pytorch_fid`.
- **Distribution comparison helpers** — for comparing per-class angle distributions stored as parquet files.
- **Convergence analysis** — N-sweep aggregation, Kendall trend tests, plateau fits, and the convergence-grid / gain-distribution plots used by `3_metrics_convergence.ipynb`.

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

- **dims** (*int*, default `2048`) — Feature dimension. `2048` is the standard FID setting.
- **device** (*str or None*, default `None`) — Torch device string. When `None`, uses CUDA if available.

**Returns**

- **model** (*nn.Module*) — InceptionV3 feature extractor.
- **device** (*str*) — The device the model lives on.

Pass both to `compute_fid` to reuse the model across calls.

---

### `combra.metrics.collect_images`

```python
collect_images(folder)
```

Recursively walk `folder` and return a sorted list of image paths whose extensions are recognised by `pytorch_fid` (handles per-class subdirectories).

**Parameters**

- **folder** (*str or Path*) — Root folder to walk.

**Returns**

- **paths** (*list[Path]*) — Sorted image paths.

---

### `combra.metrics.compute_stats`

```python
compute_stats(folder, model, device, dims=2048, batch_size=50, num_workers=8)
```

Compute the activation statistics for every image collected from `folder`.

**Parameters**

- **folder** (*str or Path*) — Image folder.
- **model** (*nn.Module*) — InceptionV3 from `load_inception`.
- **device** (*str*) — Torch device.
- **dims** (*int*, default `2048`) — Feature dimension.
- **batch_size** (*int*, default `50`) — Forward-pass batch size.
- **num_workers** (*int*, default `8`) — `DataLoader` worker count.

**Returns**

- **mu** (*ndarray[dims]*) — Mean activation.
- **sigma** (*ndarray[dims, dims]*) — Covariance of activations.
- **n_files** (*int*) — Image count.

---

### `combra.metrics.compute_fid`

```python
compute_fid(real_folder, gen_folder, model=None, device=None,
            dims=2048, batch_size=50, num_workers=8)
```

End-to-end FID between two folders.

**Parameters**

- **real_folder**, **gen_folder** (*str*) — Paths to the real and generated image folders.
- **model**, **device** (*from `load_inception`*, default `None`) — Pre-built InceptionV3 / device pair. If omitted, a fresh one is created on each call — pass them in when looping over many folder pairs.
- **dims** (*int*, default `2048`) — Feature dimension.
- **batch_size**, **num_workers** (*int*, default `50`, `8`) — Forwarded to the `pytorch_fid` data loader.

**Returns**

- **fid** (*float*) — Distance value.
- **n_real** (*int*) — Number of images contributing on the real side.
- **n_gen** (*int*) — Same on the generated side.

**Examples**

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

**Parameters**

- **parquet_path** (*str or Path*) — Path to an angles parquet.

**Returns**

- **rows** (*list[dict]*) — Each entry has `meta` (with synthesised `step`) and `prep` (single step's fit + density curves).

---

### `combra.metrics.index_by_name_step`

```python
index_by_name_step(rows)
```

Build a `{(name, step): row}` lookup from rows returned by `load_rows`. Used internally to align generated rows with their ethalon counterparts.

**Parameters**

- **rows** (*list[dict]*) — Output of `load_rows`.

**Returns**

- **index** (*dict*) — `{(name, step): row}`.

---

### `combra.metrics.find_ethalon`

```python
find_ethalon(sweep_dir)
```

Locate the largest-N parquet inside a sweep folder. Used as the reference distribution for convergence comparisons (each `metrics_vs_n` curve compares smaller-N parquets against this file).

**Parameters**

- **sweep_dir** (*str or Path*) — Folder containing `angles_n*.parquet` files.

**Returns**

- **parquet** (*Path or None*) — Path of the parquet with the largest `meta.n_images`, or `None` if the folder is empty.

---

### `combra.metrics.parquet_has_step`

```python
parquet_has_step(parquet_path, step)
```

True iff the parquet exists AND contains rows tagged with `step`. Used to decide whether a previously-generated sweep parquet is still usable when `STEP` changes (`generate_angles` overwrites, does not append).

**Parameters**

- **parquet_path** (*str or Path*) — Parquet to inspect.
- **step** (*float*) — Step value to look for.

**Returns**

- **has_step** (*bool*) — Whether the file contains a row at this step.

---

### `combra.metrics.compute_metrics`

```python
compute_metrics(real, fake)
```

Given a matched `(real, fake)` row pair, compute the per-step Wasserstein distance and bimodal-Gauss parameter deltas.

**Parameters**

- **real**, **fake** (*dict*) — Single rows from `load_rows`, expected to have matching `meta.step`.

**Returns**

- **w_dist** (*float*) — Wasserstein distance between the angle density curves, in degrees.
- **mus_m** (*ndarray[2]*) — Relative differences `(fake - real) / real` on the two Gaussian means.
- **sig_m** (*ndarray[2]*) — Same for the sigmas.
- **amp_m** (*ndarray[2]*) — Same for the amplitudes.

---

### `combra.metrics.compare_folders`

```python
compare_folders(folder_paths, ethalon_path, class_map=None, steps=None,
                coef=1000, verbose=True, fid_by_kimg=None)
```

Walk every parquet under each folder in `folder_paths`, look each row up in `ethalon_path`, print/collect the metrics.

**Parameters**

- **folder_paths** (*Iterable[str]*) — Folders, each holding one generator's parquets. The folder name suffix after `_kimg_` becomes the `kimg` tag in records (full name used if absent). May also pass `.parquet` paths directly to process exactly one file.
- **ethalon_path** (*str*) — Parquet file with reference distributions.
- **class_map** (*dict[str, str] or None*, default `None`) — `{fake_class: real_class}` mapping when names don't match.
- **steps** (*Iterable[float] or None*, default `None`) — Subset of steps to keep — others are skipped.
- **coef** (*float*, default `1000`) — Multiplier applied to `w_dist` in the printed table only (records keep raw values).
- **verbose** (*bool*, default `True`) — Print a per-row table.
- **fid_by_kimg** (*dict[str, float] or None*, default `None`) — Optional `{kimg_key: fid}` map — prints a `[kimg=… FID=…]` header above each checkpoint's rows.

**Returns**

- **records** (*list[dict]*) — `{'kimg', 'class', 'step', 'w_dist', 'mus_m', 'sig_m', 'amp_m'}` per matched row.

---

### `combra.metrics.compare_pairs`

```python
compare_pairs(pairs, step, coef=1000, verbose=True, label_header='label')
```

Like `compare_folders` but accepts a list of explicit `(label, ethalon_pq, fake_pq, class_map)` tuples — one row per pair, exactly one step. Used by `4_grid_plot.ipynb` to print one row per resolution.

**Parameters**

- **pairs** (*list[tuple]*) — Each tuple is `(label, orig_pq, gen_pq, class_map)`. `class_map` is `{orig_class: gen_class}`.
- **step** (*float*) — The single step to compare at.
- **coef** (*float*, default `1000`) — Multiplier applied to `w_dist` in the printed table only.
- **verbose** (*bool*, default `True`) — Print a per-row table.
- **label_header** (*str*, default `'label'`) — Column header used in the printed table for the first column.

**Returns**

- **records** (*list[dict]*) — `{'label', 'class', 'step', 'w_dist', 'mus_m', 'sig_m', 'amp_m'}`.

---

### `combra.metrics.metrics_vs_n`

```python
metrics_vs_n(folder, ethalon_path, class_map=None, step=5.0, allowed_ns=None)
```

Walk every parquet under `folder` (each assumed to be the same generator at a different sample size) and emit one record per `(parquet, class)` with all seven metrics: `w_dist` plus the Gaussian-fit relative errors `(mu1, mu2, sigma1, sigma2, amp1, amp2)`. N is read from `meta.n_images`, so the filename convention does not matter.

**Parameters**

- **folder** (*str or Path*) — Sweep folder containing one parquet per N.
- **ethalon_path** (*str*) — Reference parquet (typically `find_ethalon(folder)` for the same generator, or a separate "real" parquet for cross-generator comparisons).
- **class_map** (*dict[str, str] or None*, default `None`) — `{fake_class: real_class}` mapping when names don't match between folders.
- **step** (*float*, default `5.0`) — Histogram step to filter rows on.
- **allowed_ns** (*set[int] or None*, default `None`) — If given, restrict N values to this set — stray parquets at other sample sizes are skipped.

**Returns**

- **records** (*list[dict]*) — Sorted by `(class, n_images)`. Each entry: `{'n_images', 'class', 'w_dist', 'mu1', 'mu2', 'sigma1', 'sigma2', 'amp1', 'amp2'}`.

**Examples**

```python
from combra.metrics import metrics_vs_n, find_ethalon

ethalon = find_ethalon('./sweeps/real_msl5')
recs = metrics_vs_n('./sweeps/diffit_msl5', str(ethalon),
                    class_map={'class_0': 'class_Ultra_Co11'},
                    step=2.0, allowed_ns={100, 250, 1000, 10000})
```

---

## Convergence analysis

Tools that aggregate `metrics_vs_n` records into per-curve statistics (Kendall trend test, endpoint relative error, plateau fit) and turn them into the tables and figures consumed by `3_metrics_convergence.ipynb`.

### `combra.metrics.convergence_stats`

```python
convergence_stats(df_metrics, metrics, endpoints_by_kind, expected_points=None)
```

Per `(kind, resolution, class)` curve, compute trend significance, endpoint relative error, and a plateau fit `|m|(N) = a + b · N^(-1/2)`.

**Parameters**

- **df_metrics** (*pd.DataFrame*) — Long-form table with columns `kind`, `resolution`, `class`, `n_images`, and one column per metric in `metrics`.
- **metrics** (*list[str]*) — Metric column names to analyse (e.g. `['w_dist', 'mu1', 'mu2', 'sigma1', 'sigma2', 'amp1', 'amp2']`).
- **endpoints_by_kind** (*dict[str, tuple[int, int]]*) — `{kind: (n_lo, n_hi)}` used for the `rel_err_abs_%` computation.
- **expected_points** (*dict[str, int] or None*, default `None`) — `{kind: expected_n_points_per_curve}`. When given, prints a warning if a curve has the wrong count.

**Returns**

- **result** (*pd.DataFrame*) — One row per `(kind, resolution, class, metric)`. Columns: `kendall_p`, `rel_err_abs_%`, `m_at_nmax`, `a_hat`, `a_se`, `gain_abs`, `gain_pct`, `vs_a_pct`.

**Notes**

- `kendall_p` — one-sided Kendall τ p-value for "|metric| decreases with N".
- `rel_err_abs_%` — `(|m|@N_lo − |m|@N_hi) / |m|@N_lo × 100`. Positive = improvement between the two N endpoints.
- `a_hat`, `a_se` — plateau (bias floor) and its standard error from `combra.approx.fit_plateau`.
- `gain_pct` — sampling-only gain from `N_hi` to infinity, in percent of `|m|@N_hi`.
- `vs_a_pct` — excess of `|m|@N_lo` over the bias floor `a_hat`, in percent.

---

### `combra.metrics.print_convergence_report`

```python
print_convergence_report(result, metrics, kinds, endpoints_by_kind,
                         alpha=0.05, step=None, kind_display=None)
```

Print three human-readable tables from a `convergence_stats` result:

- **T1 (kendall_byclass)** — per-class Kendall p-values + `rel_err_abs_%`, with Fisher's combined p across the 3 classes.
- **T2 (kendall_fisher_byres)** — per-resolution Fisher across classes; one panel per resolution.
- **T3 (asymptote)** — plateau-fit summary (`a_gen`, `gain_%`, `vs_a_%`). A trailing `*` on `a_gen` means `a_se >= a` (plateau not yet identified).

**Parameters**

- **result** (*pd.DataFrame*) — Output of `convergence_stats`.
- **metrics** (*list[str]*) — Metric order in the printed tables.
- **kinds** (*list[str]*) — Generators to include (e.g. `['san', 'diffit']`).
- **endpoints_by_kind** (*dict[str, tuple[int, int]]*) — Same mapping passed to `convergence_stats`; used to annotate each subsection with its `(N_lo, N_hi)`.
- **alpha** (*float*, default `0.05*) — Significance threshold used for the per-metric `rej/n` counts.
- **step** (*float or None*, default `None`) — If provided, prefixed to the header line for traceability.
- **kind_display** (*dict[str, str] or None*, default `None`) — Display names (e.g. `{'san': 'SAN', 'diffit': 'DiffiT'}`).

**Returns**

*None* — output goes to stdout.

---

### `combra.metrics.summarize_metric_distribution`

```python
summarize_metric_distribution(result, metric, kinds, resolutions)
```

Per `(kind, resolution)`, summarise the distribution of one column of a `convergence_stats` result. Useful as a one-line companion to `plot_metric_distribution`.

**Parameters**

- **result** (*pd.DataFrame*) — Output of `convergence_stats`.
- **metric** (*str*) — Column to summarise (typically `'gain_pct'`).
- **kinds** (*list[str]*) — Generators to include.
- **resolutions** (*list[int]*) — Resolutions to include.

**Returns**

- **summary** (*dict[str, str]*) — `{f'{kind}_{res}': 'mean=... median=... std=... n=...'}`.

---

### `combra.metrics.plot_wdist_convergence_grid`

```python
plot_wdist_convergence_grid(records_by_panel, classes,
                            kind_labels=None, grain_labels=None,
                            row_keys=None, col_label_fn=None,
                            title_fn=None, save_path=None, png_meta=None,
                            fonts=None, height_per_row=380, width_per_col=560)
```

Plotly grid of W-dist-vs-N curves. Rows are resolutions (or arbitrary `row_keys`), columns are classes. Curves on the same axes share a kind color/legend group; the legend is shown only once.

**Parameters**

- **records_by_panel** (*dict[tuple, list[dict]]*) — `{(row_key, kind): [records]}` where each record carries `n_images`, `class`, `w_dist` (as emitted by `metrics_vs_n`).
- **classes** (*list[str]*) — Column ordering.
- **kind_labels** (*dict[str, str] or None*, default `None`) — `{kind: legend_label}`.
- **grain_labels** (*dict[str, str] or None*, default `None`) — `{class: column_label}`.
- **row_keys** (*list or None*, default `None`) — Ordered row identifiers. Defaults to sorted unique keys from `records_by_panel`.
- **col_label_fn** (*callable or None*, default `None`) — Override `class → column label`.
- **title_fn** (*callable or None*, default `None`) — Override `(row_key, class) → subplot title`.
- **save_path** (*str or None*, default `None`) — PNG output path. `None` skips saving.
- **png_meta** (*dict or None*, default `None`) — `{key: value}` written as PNG tEXt chunks.
- **fonts** (*dict or None*, default `None`) — Override the `title/axis/tick/legend` font sizes.
- **height_per_row**, **width_per_col** (*int*, default `380`, `560`) — Per-cell dimensions.

**Returns**

- **fig** (*plotly.graph_objects.Figure*) — The grid figure.

---

### `combra.metrics.plot_metric_distribution`

```python
plot_metric_distribution(result, metric, kinds, resolutions,
                         kind_display=None, res_style=None, bin_step=1.5,
                         x_range=None, save_path=None, png_meta=None,
                         fonts=None, height=900, width=900)
```

Per-kind distribution of one `convergence_stats` column (one subplot per kind, colored by resolution). Companion plot to `summarize_metric_distribution`.

**Parameters**

- **result** (*pd.DataFrame*) — Output of `convergence_stats`.
- **metric** (*str*) — Column to bin (e.g. `'gain_pct'`).
- **kinds** (*list[str]*) — Generators — one subplot each.
- **resolutions** (*list[int]*) — Resolutions to overlay.
- **kind_display** (*dict[str, str] or None*, default `None`) — Subplot titles.
- **res_style** (*dict[int, dict] or None*, default `None`) — `{res: {'color': str, 'label': str}}`. Defaults to an auto-palette.
- **bin_step** (*float*, default `1.5`) — Histogram bin width on the x-axis.
- **x_range** (*tuple[float, float] or None*, default `None`) — Clamp the x-axis. `None` → data-driven.
- **save_path** (*str or None*, default `None`) — PNG output path.
- **png_meta** (*dict or None*, default `None`) — PNG tEXt metadata.
- **fonts** (*dict or None*, default `None`) — Override `title/axis/tick/legend` font sizes.
- **height**, **width** (*int*, default `900`) — Figure dimensions.

**Returns**

- **fig** (*plotly.graph_objects.Figure*) — The distribution figure.

**Examples**

End-to-end: drive `metrics_vs_n` over every sweep folder, run `convergence_stats`, then print + plot.

```python
from combra.metrics import (
    metrics_vs_n, find_ethalon, convergence_stats,
    print_convergence_report, summarize_metric_distribution,
    plot_wdist_convergence_grid, plot_metric_distribution,
)

# … walk SOURCES → records_by_panel and df_metrics …
result = convergence_stats(df_metrics, METRICS, ENDPOINTS_BY_KIND,
                           expected_points={'real': 5, 'san': 8, 'diffit': 8})
print_convergence_report(result, METRICS, KINDS, ENDPOINTS_BY_KIND,
                         step=STEP, kind_display=KIND_DISPLAY)

fig = plot_wdist_convergence_grid(records_by_panel, classes=CLASSES,
                                  kind_labels=LABELS, save_path='wdist.png')
fig = plot_metric_distribution(result, metric='gain_pct',
                               kinds=KINDS, resolutions=[256, 512],
                               save_path='gain.png')
```

A full notebook walkthrough is in `co_angles/3_metrics_convergence.ipynb`.

---

## See also

- [`combra.data.PobeditDataset.generate_angles`]({{< relref "/docs/data#pobeditdatasetgenerate_angles" >}}) — produces the angles parquets these comparators consume.
- [`combra.angles.angles_plot_grid`]({{< relref "/docs/angles" >}}) — visualise the same comparisons as overlaid grids.
- [`combra.approx.fit_plateau`]({{< relref "/docs/approx" >}}) — the plateau fitter used inside `convergence_stats`.
- [`combra.stats.inference`]({{< relref "/docs/stats" >}}) — Kendall + Fisher primitives.
- [FID example]({{< relref "/examples/fid" >}}) — multi-resolution loop using `load_inception` + `compute_fid`.
