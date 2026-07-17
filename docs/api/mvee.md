# combra.mvee

The `combra.mvee` module fits the **Minimum Volume Enclosing Ellipse** to each polygon in an image and provides plotting / comparison helpers. The algorithm comes from [L.N. Khachiyan](https://en.wikipedia.org/wiki/Ellipsoid_method) (implementation borrowed from [radio-beam](https://radio-beam.readthedocs.io/en/latest/api/radio_beam.commonbeam.getMinVolEllipse.html)).

![Enclosed Ellipse](https://pobedit.s3.us-east-2.amazonaws.com/docs_images/enclosed-ellipse.png)

```python
from combra import mvee
```

## Build

````{py:function} combra.mvee.fit_mvee(image, tol=0.2) -> MveeResult

Fit MVEE to every contour in a preprocessed image. This is the per-image primitive that {py:meth}`combra.data.PobeditDataset.generate_beams` calls in parallel.

```{versionchanged} 0.4
Renamed from ``get_mvee_params`` (scikit-image ``get_`` drop) and now returns a
{py:class}`~combra.mvee.MveeResult` named tuple. Field order is unchanged, so
``a, b, angles, centroids, cnts = fit_mvee(...)`` still unpacks. The old
``get_mvee_params`` name is removed (no alias).
```

:param image: Preprocessed image.
:type image: ndarray
:param tol: Convergence tolerance. Lower → tighter ellipses, slower. Default: `0.2`.
:type tol: float, optional
:returns: **result** – an {py:class}`~combra.mvee.MveeResult` ``(a, b, angle_rad, centroid, contour)``: per-contour semi-major axes, semi-minor axes, rotation angles (radians), centre coordinates, and the source contours in fit order.
:rtype: MveeResult

**Example**

```python
>>> from combra import mvee, image, data
>>> _, img = data.microstructure_images()[0]
>>> processed = image.do_otsu(img)
>>> res = mvee.fit_mvee(processed, tol=0.2)
>>> print(f'{len(res.a)} polygons   median a/b = {res.a.sum()/res.b.sum():.2f}')
```
````

````{py:class} combra.mvee.MveeResult

SciPy-style named tuple returned by {py:func}`~combra.mvee.fit_mvee` (cf.
``scipy.stats.linregress``). Unpacking-compatible with the historical 5-tuple.

:param a: Per-contour semi-major axis lengths (pixels).
:type a: ndarray
:param b: Per-contour semi-minor axis lengths (pixels).
:type b: ndarray
:param angle_rad: Per-contour ellipse orientation in radians.
:type angle_rad: ndarray
:param centroid: Per-contour ``(x, y)`` centre coordinates.
:type centroid: ndarray
:param contour: The accepted ``(N, 2)`` contours the ellipses were fit to.
:type contour: list[ndarray]
````

````{py:function} combra.mvee.beams_legend(images_amount, name, itype, norm, k, angle, b, score, dist_step, dist_mean) -> str

Format a multi-line legend string for a beam-distribution plot. Used inside {py:meth}`combra.data.PobeditDataset.generate_beams` to populate `prep.beams_legend_a` / `prep.beams_legend_b`.

:param images_amount: Number of images contributing to the fit.
:type images_amount: int
:param name: Class name.
:type name: str
:param itype: Display label.
:type itype: str
:param norm: Total beam count.
:type norm: int
:param k: Linear-fit slope of the log-density.
:type k: float
:param angle: `arctan(k)` in degrees.
:type angle: float
:param b: Linear-fit intercept of the log-density.
:type b: float
:param score: Fit R².
:type score: float
:param dist_step: Histogram bin width.
:type dist_step: float
:param dist_mean: Mean beam length.
:type dist_mean: float
:returns: **label** – Multi-line legend string.
:rtype: str

**Example**

Format a per-class legend for a beam-length log-density plot:

```python
>>> from combra import mvee
>>> label = mvee.beams_legend(
...     images_amount=360, name='Ultra_Co11', itype='medium grain',
...     norm=4200, k=-1.34, b=2.10, angle=-53.2,
...     score=0.987, dist_step=4.0, dist_mean=18.5,
... )
>>> print(label)
```
````

## Plotting

````{py:function} combra.mvee.plot_beam_base(rows, save_name, step, N, M, indices=None, save=False, scatter_size=60, font_size=20) -> None

Plot the `a_beams` and `b_beams` distributions for each class in an $N \times M$ grid.

:param rows: Rows from a beams parquet (e.g. via `pq.read_table().to_pylist()`).
:type rows: list[dict]
:param save_name: Title and filename.
:type save_name: str
:param step: Filter to this histogram step.
:type step: float
:param N: Grid rows.
:type N: int
:param M: Grid columns.
:type M: int
:param indices: Class indices to draw. Default: `None`.
:type indices: list[int] or None, optional
:param save: Write to `<save_name>.png`. Default: `False`.
:type save: bool, optional
:param scatter_size: Marker size. Default: `60`.
:type scatter_size: int, optional
:param font_size: Plot font size. Default: `20`.
:type font_size: int, optional
:returns: Nothing. Renders the grid and, when `save=True`, writes `<save_name>.png`.
:rtype: None

**Example**

```python
>>> import pyarrow.parquet as pq
>>> from combra import data, mvee
>>> ds = data.PobeditDataset(path=data.microstructure_class_path())
>>> ds.generate_beams(
...     save_path='./beams',
...     types_dict={'Ultra_Co11': 'medium grain'},
...     step=4, pixel=50/1000,
... )
>>> rows = pq.read_table('./beams/beams_n100.parquet').to_pylist()
>>> mvee.plot_beam_base(rows, save_name='beams.png', step=4, N=2, M=1, save=False)
```
````

````{py:function} combra.mvee.plot_angles(data, saved_image_name, step, N, M, indices=None, save=False) -> None

Plot the ellipse rotation-angle distributions across classes.

:param data: Rows from a beams parquet.
:type data: list[dict]
:param saved_image_name: Output filename.
:type saved_image_name: str
:param step: Histogram step to filter on.
:type step: float
:param N: Grid rows.
:type N: int
:param M: Grid columns.
:type M: int
:param indices: Class indices to draw. Default: `None`.
:type indices: list[int] or None, optional
:param save: Write the figure. Default: `False`.
:type save: bool, optional
:returns: Nothing. Renders the figure and, when `save=True`, writes it to disk.
:rtype: None

**Example**

```python
>>> import pyarrow.parquet as pq
>>> from combra import mvee
>>> rows = pq.read_table('./beams/beams_n100.parquet').to_pylist()
>>> mvee.plot_angles(rows, saved_image_name='angles.png',
...                  step=4, N=2, M=2, save=False)
```
````

````{py:function} combra.mvee.plot_beam_compare(data_1, data_2, save_name, beam_types, N, M, indices_1, indices_2, save=False, scatter_size=60, font_size=20) -> list[str]

Side-by-side comparison of two parquet datasets at the same step.

:param data_1: Rows from the first beams parquet.
:type data_1: list[dict]
:param data_2: Rows from the second beams parquet.
:type data_2: list[dict]
:param save_name: Output filename.
:type save_name: str
:param beam_types: Which fields to compare — e.g. `['a_beams', 'b_beams']`.
:type beam_types: list[str]
:param N: Grid rows.
:type N: int
:param M: Grid columns.
:type M: int
:param indices_1: Class indices from the first set to align.
:type indices_1: list[int]
:param indices_2: Class indices from the second set to align.
:type indices_2: list[int]
:param save: Write the figure. Default: `False`.
:type save: bool, optional
:param scatter_size: Marker size. Default: `60`.
:type scatter_size: int, optional
:param font_size: Plot font size. Default: `20`.
:type font_size: int, optional
:returns: **fit_metrics** (*list[str]*) – One formatted `"<Δk%> <Δb%>"` string per aligned class pair — the relative differences of the linear-fit slope `k` and intercept `b` between the two datasets, in percent.
:rtype: list[str]

**Example**

```python
>>> import pyarrow.parquet as pq
>>> from combra import mvee
>>> rows_real = pq.read_table('./beams/real_n360.parquet').to_pylist()
>>> rows_gen  = pq.read_table('./beams/gen_n10000.parquet').to_pylist()
>>> metrics = mvee.plot_beam_compare(
...     rows_real, rows_gen, save_name='compare.png',
...     beam_types=['a_beams', 'b_beams'], N=2, M=2,
...     indices_1=[0, 1], indices_2=[0, 1],
... )
```
````

````{py:function} combra.mvee.beams_heatmap(data, step, saved_names, indices=None, bin_max=30, N=7, M=7, font_size=20, save=False, scatter_size=60) -> None

2-D heatmap of `(a_beam, b_beam)` pairs per class.

:param data: Rows from a beams parquet.
:type data: list[dict]
:param step: Histogram step to filter on.
:type step: float
:param saved_names: Per-class display names (overrides `meta.name`).
:type saved_names: list[str]
:param indices: Class indices to draw. Default: `None`.
:type indices: list[int] or None, optional
:param bin_max: Upper bound on histogram axes. Default: `30`.
:type bin_max: float, optional
:param N: Heatmap bin count along rows. Default: `7`.
:type N: int, optional
:param M: Heatmap bin count along columns. Default: `7`.
:type M: int, optional
:param font_size: Plot font size. Default: `20`.
:type font_size: int, optional
:param save: Write the figure. Default: `False`.
:type save: bool, optional
:param scatter_size: Marker size. Default: `60`.
:type scatter_size: int, optional
:returns: Nothing. Renders the heatmaps and, when `save=True`, writes them to disk.
:rtype: None

**Example**

```python
>>> import pyarrow.parquet as pq
>>> from combra import mvee
>>> rows = pq.read_table('./beams/beams_n100.parquet').to_pylist()
>>> mvee.beams_heatmap(rows, step=4, saved_names=['small', 'medium', 'large'],
...                    indices=[0, 1, 2], bin_max=30, N=7, M=7)
```
````

````{py:function} combra.mvee.enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15) -> None

Plot a single polygon (index `pos`) and the ellipse fitted around it. Useful for sanity-checking `tolerance`.

:param image: Source image (raw or preprocessed).
:type image: ndarray
:param pos: Index of the contour to inspect. Default: `0`.
:type pos: int, optional
:param tolerance: MVEE tolerance. Default: `0.2`.
:type tolerance: float, optional
:param N: Number of points sampled on the rendered ellipse. Default: `15`.
:type N: int, optional
:returns: Nothing. Renders the polygon and its fitted ellipse.
:rtype: None

**Example**

```python
>>> from combra import mvee, data
>>> _, img = data.microstructure_images()[0]
>>> mvee.enclosing_ellipse_show(img, pos=0, tolerance=0.2, N=200)
```
````

## See also

- {py:meth}`combra.data.PobeditDataset.generate_beams` — drives `fit_mvee` across whole class folders and writes parquet.
