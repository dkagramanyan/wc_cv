# combra.areas

Plotting helpers for polygon-area and effective-radius distributions. Both expect rows produced by {py:meth}`combra.data.PobeditDataset.generate_beams` (parquet).

```python
from combra import areas
```

## Plotting

````{py:function} combra.areas.plot_polygons_area(data, saved_image_name, step, N, M, indices=None, save=False, start=1, end=None, pixel=50/1000, font_size=20, s=60, log_min_val=-8, min_area_num=10) -> None

Log probability distribution of polygon areas with a linear fit per class on the log-log axes.

:param data: Rows from a beams parquet.
:type data: list[dict]
:param saved_image_name: Filename for the figure.
:type saved_image_name: str
:param step: Bin size in calibrated units.
:type step: float
:param N: Grid rows when arranging multiple classes.
:type N: int
:param M: Grid columns when arranging multiple classes.
:type M: int
:param indices: Class subset to draw. Default: ``None``.
:type indices: list[int] or None, optional
:param save: Write the figure to disk. Default: ``False``.
:type save: bool, optional
:param start: Histogram slice start bound for the linear fit. Default: ``1``.
:type start: int, optional
:param end: Histogram slice end bound for the linear fit. Default: ``None``.
:type end: int, optional
:param pixel: Physical pixel size (e.g. ``50/1000`` for 50 nm pixels). Default: ``50/1000``.
:type pixel: float, optional
:param font_size: Plot font size. Default: ``20``.
:type font_size: int, optional
:param s: Marker size. Default: ``60``.
:type s: int, optional
:param log_min_val: Log-density floor below which bins are discarded before the fit. Default: ``-8``.
:type log_min_val: float, optional
:param min_area_num: Minimum count per bin to include in the fit. Default: ``10``.
:type min_area_num: int, optional
:returns: Nothing. Renders the matplotlib figure and, when ``save=True``, writes ``polygons_area_<saved_image_name>_step_<step>.png``.
:rtype: None

**Example**

```python
>>> import pyarrow.parquet as pq
>>> from combra import areas
>>> rows = pq.read_table('./beams/beams_n100.parquet').to_pylist()
>>> areas.plot_polygons_area(
...     rows, 'areas.png', step=1, N=2, M=2,
...     indices=[0, 1], save=False, log_min_val=-10, min_area_num=10,
... )
```
````

````{py:function} combra.areas.plot_polygons_effective_radius(data, saved_image_name, step, N, M, indices=None, save=False, start=1, end=None, pixel=50/1000, font_size=20, s=60, log_min_val=-8, max_area_val=35) -> None

Same idea as {py:func}`combra.areas.plot_polygons_area` but plots the effective radius $\sqrt{area / \pi}$ instead of the area directly.

:param data: Rows from a beams parquet.
:type data: list[dict]
:param saved_image_name: Filename for the figure.
:type saved_image_name: str
:param step: Bin size in calibrated units.
:type step: float
:param N: Grid rows when arranging multiple classes.
:type N: int
:param M: Grid columns when arranging multiple classes.
:type M: int
:param indices: Class subset to draw. Default: ``None``.
:type indices: list[int] or None, optional
:param save: Write the figure to disk. Default: ``False``.
:type save: bool, optional
:param start: Histogram slice start bound for the linear fit. Default: ``1``.
:type start: int, optional
:param end: Histogram slice end bound for the linear fit. Default: ``None``.
:type end: int, optional
:param pixel: Physical pixel size (e.g. ``50/1000`` for 50 nm pixels). Default: ``50/1000``.
:type pixel: float, optional
:param font_size: Plot font size. Default: ``20``.
:type font_size: int, optional
:param s: Marker size. Default: ``60``.
:type s: int, optional
:param log_min_val: Log-density floor below which bins are discarded before the fit. Default: ``-8``.
:type log_min_val: float, optional
:param max_area_val: Upper bound on radius before the fit. Default: ``35``.
:type max_area_val: float, optional
:returns: Nothing. Renders the matplotlib figure and, when ``save=True``, writes ``polygons_radius_<saved_image_name>_step_<step>.png``.
:rtype: None

**Example**

```python
>>> import pyarrow.parquet as pq
>>> from combra import data, areas
>>> ds = data.PobeditDataset(path=data.microstructure_class_path())
>>> ds.generate_beams(
...     save_path='./beams',
...     types_dict={'Ultra_Co11': 'medium', 'Ultra_Co25': 'fine'},
...     step=4, pixel=50/1000,
... )
>>> rows = pq.read_table('./beams/beams_n100.parquet').to_pylist()
>>> areas.plot_polygons_area(
...     rows, 'areas.png', step=1, N=2, M=2,
...     indices=[0, 1], save=False, log_min_val=-10, min_area_num=10,
... )
>>> areas.plot_polygons_effective_radius(
...     rows, 'radii.png', step=2, N=2, M=2,
...     indices=[0, 1], save=False,
... )
```
````

## See also

- {doc}`combra.mvee <mvee>` — the MVEE primitive whose output drives these plots.
- {py:meth}`combra.data.PobeditDataset.generate_beams` — produces the parquet these helpers consume.
