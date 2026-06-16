# combra.angles

The `combra.angles` module extracts polygon angles from preprocessed binary images and
provides plotting helpers for the resulting distributions. It is the per-image primitive
that {py:meth}`combra.data.PobeditDataset.generate_angles` calls in parallel.

```python
from combra import angles
```

## Extraction

````{py:function} combra.angles.get_angles(image, border_eps=5, tol=3, min_segment_len=10.0) -> tuple[ndarray, list[ndarray]]

Extract polygon angles from a preprocessed binary image. Each contour is
Douglas–Peucker simplified, short segments are pruned, and the angle at every
remaining vertex is computed (signed, counter-clockwise traversal — values
above 180° are preserved).

:param image: Preprocessed image, shape $(H, W)$ or $(H, W, 1)$. Use {py:func}`combra.image.do_otsu` upstream.
:type image: ndarray
:param border_eps: Distance from the image edge in pixels — contours whose bounding box sits inside this margin are dropped. Default: ``5``.
:type border_eps: int, optional
:param tol: Douglas–Peucker simplification tolerance. Default: ``3``.
:type tol: float, optional
:param min_segment_len: Vertices producing shorter neighbouring segments are iteratively removed before angle calculation. Higher values give smoother distributions and fewer angles. Recommended: 5–20 px. Default: ``10.0``.
:type min_segment_len: float, optional
:returns: **angles_array** (*ndarray[float64]*) – all extracted vertex angles in degrees, concatenated across contours; and **contours** (*list[ndarray]*) – the simplified contours that produced the angles, as $(N_{points}, 2)$ int arrays.
:rtype: tuple[ndarray, list[ndarray]]

**Example**

```python
>>> from combra import angles, data, image
>>> _, img = data.microstructure_images()[0]
>>> processed = image.do_otsu(img)
>>> arr, contours = angles.get_angles(processed, border_eps=5, tol=3, min_segment_len=10.0)
>>> print(f'{len(arr)} angles  mean={arr.mean():.1f}°')
```
````

````{py:function} combra.angles.angles_legend(images_amount, name, itype, step, mus, sigmas, amps, norm) -> str

Format a multi-line legend string from bimodal-Gaussian fit parameters. Useful when
overlaying text on a custom plot.

:param images_amount: Number of images contributing to the fit (``meta.n_images``).
:type images_amount: int
:param name: Class name (from ``types_dict``).
:type name: str
:param itype: Display type (from ``types_dict``).
:type itype: str
:param step: Histogram bin width.
:type step: float
:param mus: Length-2 bimodal-Gauss means.
:type mus: list[float]
:param sigmas: Length-2 bimodal-Gauss standard deviations.
:type sigmas: list[float]
:param amps: Length-2 bimodal-Gauss amplitudes.
:type amps: list[float]
:param norm: Total angles count (``raw.angles_count``).
:type norm: int
:returns: **label** – a multi-line label ready to drop into a matplotlib title.
:rtype: str

**Example**

```python
>>> from combra import angles
>>> label = angles.angles_legend(
...     images_amount=360, name='Ultra_Co11', itype='small grain', step=2.0,
...     mus=[90.5, 270.3], sigmas=[18.4, 22.1], amps=[0.012, 0.015], norm=4200,
... )
>>> print(label)
```
````

## Plotting

````{py:function} combra.angles.angles_plot_base(rows=None, save_name=None, N=20, M=20, save=False, indices=None, font_size=20, scatter_size=20, xlim=None, ylim=None, parquet_path=None, step=None, show=True) -> None

Plot angle density curves and bimodal Gaussian fits in an $N \times M$ plotly grid.
Pass either ``rows`` (list of dicts) or ``parquet_path`` (string).

:param rows: Pre-loaded rows from {py:func}`combra.metrics.load_rows` or ``pq.read_table().to_pydict()``. Required if ``parquet_path`` is not given. Default: ``None``.
:type rows: list[dict] or None, optional
:param save_name: Title and filename. Derived from ``parquet_path`` if absent. Default: ``None``.
:type save_name: str or None, optional
:param N: Grid rows. Default: ``20``.
:type N: int, optional
:param M: Grid columns. Default: ``20``.
:type M: int, optional
:param save: Save the figure to ``<save_name>.png`` / ``.html``. Default: ``False``.
:type save: bool, optional
:param indices: Subset of rows to draw. Default: ``None``.
:type indices: list[int] or None, optional
:param font_size: Plot font size. Default: ``20``.
:type font_size: int, optional
:param scatter_size: Marker size. Default: ``20``.
:type scatter_size: int, optional
:param xlim: X-axis limits. Default: ``None``.
:type xlim: tuple[float, float] or None, optional
:param ylim: Y-axis limits. Default: ``None``.
:type ylim: tuple[float, float] or None, optional
:param parquet_path: Alternative to ``rows`` — loads the file in place. Default: ``None``.
:type parquet_path: str or None, optional
:param step: When the parquet has multiple steps under ``prep_per_step``, pick this one. Default: ``None``.
:type step: float or None, optional
:param show: If ``False``, skip ``fig.show()`` (useful in batch). Default: ``True``.
:type show: bool, optional
:returns: Nothing. Renders the plotly grid and, when ``save=True``, writes ``<save_name>.png`` / ``.html``.
:rtype: None

**Example**

```python
>>> from combra import angles
>>> angles.angles_plot_base(
...     parquet_path='./data/angles/o_bc_left_4x_1536_1024x1024_256x256_rgb_N360_msl5/angles_n360.parquet',
...     N=2, M=2, step=2.0, save=False, font_size=18,
... )
```
````

````{py:function} combra.angles.angles_plot_grid(grid, row_titles, col_titles, step, title=None, save=None, show=True, scatter_size=5, cell_width=320, cell_height=300, ylim=None) -> plotly.graph_objects.Figure

Plot a 2-D grid of angle distributions where each cell overlays multiple sources
(e.g. original vs SAN-GAN vs DiffiT at one resolution / class).

:param grid: ``grid[r][c]`` lists the overlay traces for the cell at row ``r``, column ``c``. Each trace is a dict with keys ``parquet``, ``class_name``, ``label``, ``color``, ``marker``.
:type grid: list[list[list[dict]]]
:param row_titles: Row labels.
:type row_titles: list[str]
:param col_titles: Column labels.
:type col_titles: list[str]
:param step: Histogram step to filter on.
:type step: float
:param title: Figure title. Default: ``None``.
:type title: str or None, optional
:param save: If given, save the rendered HTML/PNG to this path. Default: ``None``.
:type save: str or None, optional
:param show: ``True`` to call ``fig.show()``. Default: ``True``.
:type show: bool, optional
:param scatter_size: Marker size. Default: ``5``.
:type scatter_size: int, optional
:param cell_width: Per-cell width in pixels. Default: ``320``.
:type cell_width: int, optional
:param cell_height: Per-cell height in pixels. Default: ``300``.
:type cell_height: int, optional
:param ylim: Shared y-axis range across cells. Default: ``None``.
:type ylim: tuple[float, float] or None, optional
:returns: **fig** – the assembled grid figure.
:rtype: plotly.graph_objects.Figure

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
````

## See also

- {py:meth}`combra.data.PobeditDataset.generate_angles` — drives `get_angles` across whole class folders and writes parquet.
- {doc}`combra.contours <contours>` — the contour extractor `get_angles` relies on internally.
- {py:func}`combra.metrics.load_rows` — loads angles parquets into the row shape these plotters expect.
