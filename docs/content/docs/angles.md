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
get_angles(image, border_eps=5, tol=3, min_segment_len=10.0)
```

Extract polygon angles from a preprocessed binary image. Each contour is Douglas–Peucker simplified, short segments are pruned, and the angle at every remaining vertex is computed (signed, counter-clockwise traversal — values >180° are preserved).

**Parameters**

- **image** (*ndarray*) — Preprocessed image, shape `(H, W)` or `(H, W, 1)`. Use `combra.image.do_otsu` upstream.
- **border_eps** (*int*, default `5`) — Distance from image edge in pixels — contours whose bounding box sits inside this margin are dropped.
- **tol** (*float*, default `3`) — Douglas–Peucker simplification tolerance.
- **min_segment_len** (*float*, default `10.0`) — Vertices producing shorter neighbouring segments are iteratively removed before angle calculation. Higher → smoother distributions, fewer angles. Recommended: 5–20 px.

**Returns**

- **angles_array** (*ndarray[float64]*) — All extracted vertex angles in degrees, concatenated across contours.
- **contours** (*list[ndarray]*) — The (simplified) contours that produced the angles — `(N_points, 2)` int arrays.

**Examples**

```python
from combra import angles, data, image

_, img = data.microstructure_images()[0]
processed = image.do_otsu(img)

arr, contours = angles.get_angles(processed, border_eps=5, tol=3, min_segment_len=10.0)
print(f'{len(arr)} angles  mean={arr.mean():.1f}°')
```

---

### `combra.angles.angles_legend`

```python
angles_legend(images_amount, name, itype, step, mus, sigmas, amps, norm)
```

Format a multi-line legend string from bimodal-Gaussian fit parameters. Useful when overlaying text on a custom plot.

**Parameters**

- **images_amount** (*int*) — Number of images contributing to the fit (`meta.n_images`).
- **name**, **itype** (*str*) — Class name and display type (from `types_dict`).
- **step** (*float*) — Histogram bin width.
- **mus**, **sigmas**, **amps** (*list[float]*) — Length-2 bimodal-Gauss parameters.
- **norm** (*int*) — Total angles count (`raw.angles_count`).

**Returns**

- **label** (*str*) — Multi-line label ready to drop into a matplotlib title.

---

## Plotting

### `combra.angles.angles_plot_base`

```python
angles_plot_base(
    rows=None, save_name=None,
    N=20, M=20, save=False, indices=None,
    font_size=20, scatter_size=20, xlim=None, ylim=None,
    parquet_path=None, step=None, show=True,
)
```

Plot angle density curves and bimodal Gaussian fits in an `N × M` plotly grid. Pass either `rows` (list of dicts) or `parquet_path` (string).

**Parameters**

- **rows** (*list[dict] or None*, default `None`) — Pre-loaded rows from `combra.metrics.load_rows` or `pq.read_table().to_pydict()`. Required if `parquet_path` is not given.
- **save_name** (*str or None*, default `None`) — Title and filename. Derived from `parquet_path` if absent.
- **N**, **M** (*int*, default `20`) — Grid dimensions.
- **save** (*bool*, default `False`) — Save the figure to `<save_name>.png` / `.html`.
- **indices** (*list[int] or None*, default `None`) — Subset of rows to draw.
- **font_size**, **scatter_size** (*int*, default `20`) — Plot styling.
- **xlim**, **ylim** (*tuple[float, float] or None*, default `None`) — Axis limits.
- **parquet_path** (*str or None*, default `None`) — Alternative to `rows` — loads the file in place.
- **step** (*float or None*, default `None`) — When the parquet has multiple steps under `prep_per_step`, pick this one.
- **show** (*bool*, default `True`) — If `False`, skip `fig.show()` (useful in batch).

**Examples**

```python
from combra import angles

angles.angles_plot_base(
    parquet_path='./data/angles/o_bc_left_4x_1536_1024x1024_256x256_rgb_N360_msl5/angles_n360.parquet',
    N=2, M=2, step=2.0, save=False, font_size=18,
)
```

---

### `combra.angles.angles_plot_grid`

```python
angles_plot_grid(
    grid, row_titles, col_titles, step,
    title=None, save=None, show=True,
    scatter_size=5, cell_width=320, cell_height=300, ylim=None,
)
```

Plot a 2-D grid of angle distributions where each cell overlays multiple sources (e.g. original vs SAN-GAN vs DiffiT at one resolution / class).

**Parameters**

- **grid** (*list[list[list[dict]]]*) — `grid[r][c]` lists the overlay traces for the cell at row `r`, column `c`. Each trace is a dict with keys `parquet`, `class_name`, `label`, `color`, `marker`.
- **row_titles**, **col_titles** (*list[str]*) — Axis labels.
- **step** (*float*) — Histogram step to filter on.
- **title** (*str or None*, default `None`) — Figure title.
- **save** (*str or None*, default `None`) — If given, save the rendered HTML/PNG to this path.
- **show** (*bool*, default `True`) — `True` to call `fig.show()`.
- **scatter_size**, **cell_width**, **cell_height** (*int*, default `5`, `320`, `300`) — Layout knobs.
- **ylim** (*tuple[float, float] or None*, default `None`) — Shared y-axis range across cells.

**Examples**

Three-source 3×3 grid (real, SAN-GAN, DiffiT × small/medium/large grains):

```python
from combra import angles

STYLES = {'orig': dict(color='blue', marker='circle'),
          'gan':  dict(color='orange', marker='square'),
          'diffit': dict(color='green', marker='diamond')}

grid = []
for res in [256, 512, 1024]:
    row = []
    for cls in ['class_Ultra_Co11', 'class_Ultra_Co25', 'class_Ultra_Co6_2']:
        cell = [{'parquet': f'.../o_bc_left_..._{res}x{res}_..._msl5/angles_n360.parquet',
                 'class_name': cls, 'label': 'orig', **STYLES['orig']}]
        row.append(cell)
    grid.append(row)

angles.angles_plot_grid(
    grid=grid,
    row_titles=['256×256', '512×512', '1024×1024'],
    col_titles=['small grain', 'medium grain', 'large grain'],
    step=2.0,
)
```

---

## See also

- [`combra.data.PobeditDataset.generate_angles`]({{< relref "/docs/data#pobeditdatasetgenerate_angles" >}}) — drives `get_angles` across whole class folders and writes parquet.
- [`combra.contours`]({{< relref "/docs/contours" >}}) — the contour extractor `get_angles` relies on internally.
- [`combra.metrics.load_rows`]({{< relref "/docs/metrics" >}}) — loads angles parquets into the row shape these plotters expect.
