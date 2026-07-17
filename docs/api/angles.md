# combra.angles

The `combra.angles` module extracts polygon angles from preprocessed binary images and
provides plotting helpers for the resulting distributions. It is the per-image primitive
that {py:meth}`combra.data.PobeditDataset.generate_angles` calls in parallel.

```python
from combra import angles
```

## Extraction

````{py:function} combra.angles.vertex_angles(image, border_eps=5, tol=3, min_segment_len=10.0) -> tuple[ndarray, list[ndarray]]

Extract polygon angles from a preprocessed binary image. Each contour is
Douglas–Peucker simplified, short segments are pruned, and the angle at every
remaining vertex is computed (signed, counter-clockwise traversal — values
above 180° are preserved).

```{versionchanged} 0.4
Renamed from ``get_angles`` (dropping the ``get_`` prefix, scikit-image
convention). ``get_angles`` stays as a deprecated alias until 0.6.
```

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
>>> arr, contours = angles.vertex_angles(processed, border_eps=5, tol=3, min_segment_len=10.0)
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

## Output layout

````{py:function} combra.angles.angles_out_dir(out_root, src_path, msl) -> pathlib.Path

Canonical per-source folder for generated angle parquets:
`{out_root}/{stem}_msl{int(msl)}`, where `stem` is the source file/folder stem and
`msl` the `min_segment_len` used at generation. This is the exact folder
{py:func}`combra.data.generate_angles_sweep` writes into, so the generation and
plotting sides share one naming convention instead of re-deriving the `_msl` suffix.

:param out_root: Root directory holding the per-source angle folders.
:type out_root: str or pathlib.Path
:param src_path: Source image folder or h5 path; only its stem is used.
:type src_path: str or pathlib.Path
:param msl: ``min_segment_len`` used at generation; ``int(msl)`` forms the ``_msl`` suffix.
:type msl: float
:returns: **out_dir** – ``Path(out_root) / f'{Path(src_path).stem}_msl{int(msl)}'``.
:rtype: pathlib.Path

**Example**

```python
>>> from combra import angles
>>> angles.angles_out_dir('./data/angles', './data/h5/gen_san_256x256_N100_000.h5', 5.0)
PosixPath('data/angles/gen_san_256x256_N100_000_msl5')
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

````{py:function} combra.angles.build_overlay_grid(rows, classes, gen_name_for_mode, styles, orig_label='orig') -> tuple[list, dict]

Assemble the nested ``grid`` for {py:func}`combra.angles.angles_plot_grid` **and** the per-mode comparison pairs for {py:func}`combra.metrics.compare_pairs` from already-resolved per-row sources — so an "original vs N generators across resolution × class" notebook only declares its paths/availability and trace styles instead of hand-building the grid.

:param rows: One `(row_label, orig_parquet, {mode: gen_parquet})` per grid row (e.g. per resolution). The `{mode: gen_parquet}` dict lists only the generators actually available for that row.
:type rows: list[tuple[str, str, dict[str, str]]]
:param classes: Ordered class keys **without** the `class_` prefix; each cell's orig trace uses `class_name = f'class_{key}'`.
:type classes: list[str]
:param gen_name_for_mode: `{mode: {class_key: gen_class_name}}` — the generated `meta.name` per (mode, class), since generators index classes differently.
:type gen_name_for_mode: dict[str, dict[str, str]]
:param styles: `{label: {'color': ..., 'marker': ...}}` for `orig_label` and each mode; merged into every trace dict.
:type styles: dict[str, dict]
:param orig_label: Label/style key for the reference (original) trace. Default: `'orig'`.
:type orig_label: str, optional
:returns: **(grid, pairs_by_mode)** – `grid` is the nested trace structure for `angles_plot_grid`; `pairs_by_mode` is `{mode: [(row_label, orig_parquet, gen_parquet, class_map), ...]}` for `compare_pairs` (with `class_map = {f'class_{key}': gen_class_name}`).
:rtype: tuple[list, dict]

**Example**

From `co_angles/1_generation_and_plots.ipynb` — resolve availability, then let combra build both the plot grid and the metric pairs:

```python
>>> from combra import angles
>>> from combra.metrics import compare_pairs
>>> STYLES = {'orig': dict(color='blue', marker='circle'),
...           'san':  dict(color='orange', marker='square'),
...           'diffit': dict(color='green', marker='diamond')}
>>> gen_name = {'san':    {'Ultra_Co25': 'class_0', 'Ultra_Co11': 'class_1', 'Ultra_Co6_2': 'class_2'},
...             'diffit': {'Ultra_Co11': 'class_0', 'Ultra_Co25': 'class_1', 'Ultra_Co6_2': 'class_2'}}
>>> rows = [('256', './a/orig/angles_n360.parquet',
...          {'san': './a/san/angles_n10000.parquet', 'diffit': './a/diffit/angles_n10000.parquet'})]
>>> grid, pairs = angles.build_overlay_grid(
...     rows, ['Ultra_Co11', 'Ultra_Co25', 'Ultra_Co6_2'], gen_name, STYLES)
>>> compare_pairs(pairs['diffit'], step=2, coef=1, label_header='res')
>>> fig = angles.angles_plot_grid(grid, ['256×256'],
...                               ['small', 'medium', 'large'], step=2)
```
````

````{py:function} combra.angles.resolve_overlay_rows(out_root, sources, n, msl, real_family='real') -> list

Group a generation *sources manifest* into the `rows` argument of
{py:func}`combra.angles.build_overlay_grid`, so a comparison notebook declares its
sources once — the same `(src_path, n_list, family, resolution)` list it feeds
{py:func}`combra.data.generate_angles_sweep` — and the plot side cannot drift from
generation. Per resolution (in first-seen order) the single `real_family` source
becomes the reference (`orig`) parquet at its own first N; every other source is
added at `angles_n{n}.parquet` when `n` is in its `n_list` **and** the file exists
on disk. Folder names are resolved via {py:func}`combra.angles.angles_out_dir`.

:param out_root: Root holding the per-source angle folders.
:type out_root: str or pathlib.Path
:param sources: Manifest of ``(src_path, n_list, family, resolution)`` tuples; ``n_list`` is each source's available N budgets.
:type sources: list[tuple]
:param n: Generated-images-per-class budget to plot for the non-real sources.
:type n: int
:param msl: ``min_segment_len`` used at generation; selects the folder suffix.
:type msl: float
:param real_family: Family label of the reference source. Default: ``'real'``.
:type real_family: str, optional
:returns: **rows** – ``[(row_label, orig_parquet, {family: gen_parquet}), ...]`` for each resolution that has a real source, ready for {py:func}`~combra.angles.build_overlay_grid`.
:rtype: list[tuple[str, str, dict[str, str]]]

**Example**

```python
>>> from combra import angles
>>> SOURCES = [
...     ('o_bc_left_..._256x256_rgb_N360', [360],           'real',   256),
...     ('./data/h5/gen_san_256x256_N100_000.h5', [1000, 10000], 'san', 256),
...     ('./data/h5/00017-diffit-256-..._N10000.h5', [1000, 10000], 'diffit', 256),
... ]
>>> rows = angles.resolve_overlay_rows('./data/angles', SOURCES, n=10000, msl=5.0)
```
````

````{py:function} combra.angles.plot_overlay_grid_from_sources(out_root, sources, n, msl, classes, gen_name_for_mode, styles, col_titles=None, row_label_fmt='{r}×{r}', step=2, ylim=None, title=None, save=None, show=True, compare=False, compare_coef=1, real_family='real', orig_label='orig') -> tuple

End-to-end orig-vs-generators overlay grid for one N, straight from a sources
manifest. Composes {py:func}`~combra.angles.resolve_overlay_rows` →
{py:func}`~combra.angles.build_overlay_grid` → (optional
{py:func}`combra.metrics.compare_pairs` printing when ``compare=True``) →
{py:func}`~combra.angles.angles_plot_grid`, so a notebook keeps only its own config
(class maps, styles, labels, title/filename) and calls this once per ``n``.

:param out_root: Root holding the per-source angle folders.
:type out_root: str or pathlib.Path
:param sources: Manifest of ``(src_path, n_list, family, resolution)`` tuples (see {py:func}`~combra.angles.resolve_overlay_rows`).
:type sources: list[tuple]
:param n: Generated-images-per-class budget to plot for the non-real sources.
:type n: int
:param msl: ``min_segment_len`` used at generation; selects the folder suffix.
:type msl: float
:param classes: Ordered class keys **without** the ``class_`` prefix.
:type classes: list[str]
:param gen_name_for_mode: ``{mode: {class_key: gen_class_name}}`` — the generated ``meta.name`` per (mode, class).
:type gen_name_for_mode: dict[str, dict[str, str]]
:param styles: ``{label: {'color': ..., 'marker': ...}}`` for ``orig_label`` and each mode.
:type styles: dict[str, dict]
:param col_titles: Per-column labels. Defaults to ``classes``.
:type col_titles: list[str] or None, optional
:param row_label_fmt: Format string for each resolution row title (``{r}`` = resolution). Default: ``'{r}×{r}'``.
:type row_label_fmt: str, optional
:param step: Histogram step to filter on. Default: ``2``.
:type step: float, optional
:param ylim: Shared y-axis range across cells. Default: ``None``.
:type ylim: tuple[float, float] or None, optional
:param title: Figure title. Default: ``None``.
:type title: str or None, optional
:param save: If given, save the figure to this path. Default: ``None``.
:type save: str or None, optional
:param show: ``True`` to call ``fig.show()``. Default: ``True``.
:type show: bool, optional
:param compare: When ``True``, print the per-mode Wasserstein table before plotting. Default: ``False``.
:type compare: bool, optional
:param compare_coef: Multiplier passed to {py:func}`combra.metrics.compare_pairs` (``1`` prints raw degrees). Default: ``1``.
:type compare_coef: float, optional
:param real_family: Family label of the reference source. Default: ``'real'``.
:type real_family: str, optional
:param orig_label: Label/style key for the reference trace. Default: ``'orig'``.
:type orig_label: str, optional
:returns: **(fig, pairs_by_mode)** – the assembled grid figure and the per-mode comparison pairs from {py:func}`~combra.angles.build_overlay_grid`.
:rtype: tuple[plotly.graph_objects.Figure, dict]

**Example**

From `co_angles/1_generation_and_plots.ipynb` — one call per N draws the grid and prints the metric table:

```python
>>> from combra import angles
>>> for max_n in [1000, 10000]:
...     angles.plot_overlay_grid_from_sources(
...         './data/angles', SOURCES, n=max_n, msl=5.0,
...         classes=['Ultra_Co11', 'Ultra_Co25', 'Ultra_Co6_2'],
...         col_titles=['small', 'medium', 'large'],
...         gen_name_for_mode=gen_name, styles=STYLES,
...         step=2, compare=True, title=f'angles N={max_n}',
...     )
```
````

## See also

- {py:meth}`combra.data.PobeditDataset.generate_angles` — drives `vertex_angles` across whole class folders and writes parquet.
- {doc}`combra.contours <contours>` — the contour extractor `vertex_angles` relies on internally.
- {py:func}`combra.metrics.load_rows` — loads angles parquets into the row shape these plotters expect.
