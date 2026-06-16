---
title: "Angles"
weight: 5
---

The `combra.angles` module extracts polygon angles from preprocessed binary images and provides plotting helpers for the resulting distributions. It's the per-image primitive that `PobeditDataset.generate_angles` calls in parallel.

```python
from combra import angles
```

## Extraction

### `combra.angles.get_angles`

```python
combra.angles.get_angles(image, border_eps=5, tol=3, min_segment_len=10.0) → tuple[ndarray, list[ndarray]]
```

Extract polygon angles from a preprocessed binary image. Each contour is Douglas–Peucker simplified, short segments are pruned, and the angle at every remaining vertex is computed (signed, counter-clockwise traversal — values >180° are preserved).

**Parameters**

- **image** (*ndarray*) – Preprocessed image, shape `(H, W)` or `(H, W, 1)`. Use `combra.image.do_otsu` upstream.
- **border_eps** (*int, optional*) – Distance from image edge in pixels — contours whose bounding box sits inside this margin are dropped. Default: `5`.
- **tol** (*float, optional*) – Douglas–Peucker simplification tolerance. Default: `3`.
- **min_segment_len** (*float, optional*) – Vertices producing shorter neighbouring segments are iteratively removed before angle calculation. Higher → smoother distributions, fewer angles. Recommended: 5–20 px. Default: `10.0`.

**Returns**

- **angles_array** (*ndarray[float64]*) – All extracted vertex angles in degrees, concatenated across contours.
- **contours** (*list[ndarray]*) – The (simplified) contours that produced the angles — `(N_points, 2)` int arrays.

**Return type**

*tuple(ndarray, list[ndarray])*

**Example**

```python
>>> from combra import angles, data, image
>>> _, img = data.microstructure_images()[0]
>>> processed = image.do_otsu(img)
>>> arr, contours = angles.get_angles(processed, border_eps=5, tol=3, min_segment_len=10.0)
>>> print(f'{len(arr)} angles  mean={arr.mean():.1f}°')
```

---

### `combra.angles.angles_legend`

```python
combra.angles.angles_legend(images_amount, name, itype, step, mus, sigmas, amps, norm) → str
```

Format a multi-line legend string from bimodal-Gaussian fit parameters. Useful when overlaying text on a custom plot.

**Parameters**

- **images_amount** (*int*) – Number of images contributing to the fit (`meta.n_images`).
- **name** (*str*) – Class name (from `types_dict`).
- **itype** (*str*) – Display type (from `types_dict`).
- **step** (*float*) – Histogram bin width.
- **mus** (*list[float]*) – Length-2 bimodal-Gauss means.
- **sigmas** (*list[float]*) – Length-2 bimodal-Gauss standard deviations.
- **amps** (*list[float]*) – Length-2 bimodal-Gauss amplitudes.
- **norm** (*int*) – Total angles count (`raw.angles_count`).

**Returns**

- **label** (*str*) – Multi-line label ready to drop into a matplotlib title.

**Return type**

*str*

**Example**

```python
>>> from combra import angles
>>> label = angles.angles_legend(
...     images_amount=360, name='Ultra_Co11', itype='small grain', step=2.0,
...     mus=[90.5, 270.3], sigmas=[18.4, 22.1], amps=[0.012, 0.015], norm=4200,
... )
>>> print(label)
```

---

## Plotting

### `combra.angles.angles_plot_base`

```python
combra.angles.angles_plot_base(rows=None, save_name=None, N=20, M=20, save=False,
                               indices=None, font_size=20, scatter_size=20,
                               xlim=None, ylim=None, parquet_path=None,
                               step=None, show=True) → None
```

Plot angle density curves and bimodal Gaussian fits in an `N × M` plotly grid. Pass either `rows` (list of dicts) or `parquet_path` (string).

**Parameters**

- **rows** (*list[dict] or None, optional*) – Pre-loaded rows from `combra.metrics.load_rows` or `pq.read_table().to_pydict()`. Required if `parquet_path` is not given. Default: `None`.
- **save_name** (*str or None, optional*) – Title and filename. Derived from `parquet_path` if absent. Default: `None`.
- **N** (*int, optional*) – Grid rows. Default: `20`.
- **M** (*int, optional*) – Grid columns. Default: `20`.
- **save** (*bool, optional*) – Save the figure to `<save_name>.png` / `.html`. Default: `False`.
- **indices** (*list[int] or None, optional*) – Subset of rows to draw. Default: `None`.
- **font_size** (*int, optional*) – Plot font size. Default: `20`.
- **scatter_size** (*int, optional*) – Marker size. Default: `20`.
- **xlim** (*tuple[float, float] or None, optional*) – X-axis limits. Default: `None`.
- **ylim** (*tuple[float, float] or None, optional*) – Y-axis limits. Default: `None`.
- **parquet_path** (*str or None, optional*) – Alternative to `rows` — loads the file in place. Default: `None`.
- **step** (*float or None, optional*) – When the parquet has multiple steps under `prep_per_step`, pick this one. Default: `None`.
- **show** (*bool, optional*) – If `False`, skip `fig.show()` (useful in batch). Default: `True`.

**Returns**

None. Renders the plotly grid and, when `save=True`, writes `<save_name>.png` / `.html`.

**Return type**

*None*

**Example**

```python
>>> from combra import angles
>>> angles.angles_plot_base(
...     parquet_path='./data/angles/o_bc_left_4x_1536_1024x1024_256x256_rgb_N360_msl5/angles_n360.parquet',
...     N=2, M=2, step=2.0, save=False, font_size=18,
... )
```

---

### `combra.angles.angles_plot_grid`

```python
combra.angles.angles_plot_grid(grid, row_titles, col_titles, step, title=None,
                               save=None, show=True, scatter_size=5,
                               cell_width=320, cell_height=300, ylim=None) → plotly.graph_objects.Figure
```

Plot a 2-D grid of angle distributions where each cell overlays multiple sources (e.g. original vs SAN-GAN vs DiffiT at one resolution / class).

**Parameters**

- **grid** (*list[list[list[dict]]]*) – `grid[r][c]` lists the overlay traces for the cell at row `r`, column `c`. Each trace is a dict with keys `parquet`, `class_name`, `label`, `color`, `marker`.
- **row_titles** (*list[str]*) – Row labels.
- **col_titles** (*list[str]*) – Column labels.
- **step** (*float*) – Histogram step to filter on.
- **title** (*str or None, optional*) – Figure title. Default: `None`.
- **save** (*str or None, optional*) – If given, save the rendered HTML/PNG to this path. Default: `None`.
- **show** (*bool, optional*) – `True` to call `fig.show()`. Default: `True`.
- **scatter_size** (*int, optional*) – Marker size. Default: `5`.
- **cell_width** (*int, optional*) – Per-cell width in pixels. Default: `320`.
- **cell_height** (*int, optional*) – Per-cell height in pixels. Default: `300`.
- **ylim** (*tuple[float, float] or None, optional*) – Shared y-axis range across cells. Default: `None`.

**Returns**

- **fig** (*plotly.graph_objects.Figure*) – The assembled grid figure.

**Return type**

*plotly.graph_objects.Figure*

**Example**

Three-source 3×3 grid (real, SAN-GAN, DiffiT × small/medium/large grains):

```python
>>> from combra import angles
>>> STYLES = {'orig': dict(color='blue', marker='circle'),
...           'gan':  dict(color='orange', marker='square'),
...           'diffit': dict(color='green', marker='diamond')}
>>> grid = []
>>> for res in [256, 512, 1024]:
...     row = []
...     for cls in ['class_Ultra_Co11', 'class_Ultra_Co25', 'class_Ultra_Co6_2']:
...         cell = [{'parquet': f'.../o_bc_left_..._{res}x{res}_..._msl5/angles_n360.parquet',
...                  'class_name': cls, 'label': 'orig', **STYLES['orig']}]
...         row.append(cell)
...     grid.append(row)
>>> fig = angles.angles_plot_grid(
...     grid=grid,
...     row_titles=['256×256', '512×512', '1024×1024'],
...     col_titles=['small grain', 'medium grain', 'large grain'],
...     step=2.0,
... )
```

---

## See also

- [`combra.data.PobeditDataset.generate_angles`]({{< relref "/docs/data#pobeditdatasetgenerate_angles" >}}) — drives `get_angles` across whole class folders and writes parquet.
- [`combra.contours`]({{< relref "/docs/contours" >}}) — the contour extractor `get_angles` relies on internally.
- [`combra.metrics.load_rows`]({{< relref "/docs/metrics" >}}) — loads angles parquets into the row shape these plotters expect.
