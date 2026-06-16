---
title: "MVEE"
weight: 6
---

The `combra.mvee` module fits the **Minimum Volume Enclosing Ellipse** to each polygon in an image and provides plotting / comparison helpers. The algorithm comes from [L.N. Khachiyan](https://en.wikipedia.org/wiki/Ellipsoid_method) (implementation borrowed from [radio-beam](https://radio-beam.readthedocs.io/en/latest/api/radio_beam.commonbeam.getMinVolEllipse.html)).

![Enclosed Ellipse](https://pobedit.s3.us-east-2.amazonaws.com/docs_images/enclosed-ellipse.png)

```python
from combra import mvee
```

## Build

### `combra.mvee.get_mvee_params`

```python
combra.mvee.get_mvee_params(image, tol=0.2) → tuple[ndarray, ndarray, ndarray, ndarray, list[ndarray]]
```

Fit MVEE to every contour in a preprocessed image. This is the per-image primitive that `PobeditDataset.generate_beams` calls in parallel.

**Parameters**

- **image** (*ndarray*) – Preprocessed image.
- **tol** (*float, optional*) – Convergence tolerance. Lower → tighter ellipses, slower. Default: `0.2`.

**Returns**

- **a_beams** (*ndarray*) – Semi-major axes (in pixels).
- **b_beams** (*ndarray*) – Semi-minor axes.
- **angles** (*ndarray*) – Rotation angles in radians.
- **centroids** (*ndarray*) – Centre coordinates.
- **contours** (*list[ndarray]*) – The source contours, in the order MVEEs were fit.

**Return type**

*tuple(ndarray, ndarray, ndarray, ndarray, list[ndarray])*

**Example**

```python
>>> from combra import mvee, image, data
>>> _, img = data.microstructure_images()[0]
>>> processed = image.do_otsu(img)
>>> a, b, angles, centroids, cnts = mvee.get_mvee_params(processed, tol=0.2)
>>> print(f'{len(a)} polygons   median a/b = {sum(a)/sum(b):.2f}')
```

---

### `combra.mvee.beams_legend`

```python
combra.mvee.beams_legend(images_amount, name, itype, norm, k, angle, b, score,
                         dist_step, dist_mean) → str
```

Format a multi-line legend string for a beam-distribution plot. Used inside `generate_beams` to populate `prep.beams_legend_a` / `prep.beams_legend_b`.

**Parameters**

- **images_amount** (*int*) – Number of images contributing to the fit.
- **name** (*str*) – Class name.
- **itype** (*str*) – Display label.
- **norm** (*int*) – Total beam count.
- **k** (*float*) – Linear-fit slope of the log-density.
- **angle** (*float*) – `arctan(k)` in degrees.
- **b** (*float*) – Linear-fit intercept of the log-density.
- **score** (*float*) – Fit R².
- **dist_step** (*float*) – Histogram bin width.
- **dist_mean** (*float*) – Mean beam length.

**Returns**

- **label** (*str*) – Multi-line legend string.

**Return type**

*str*

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

---

## Plotting

### `combra.mvee.plot_beam_base`

```python
combra.mvee.plot_beam_base(rows, save_name, step, N, M, indices=None, save=False,
                           scatter_size=60, font_size=20) → None
```

Plot the `a_beams` and `b_beams` distributions for each class in an `N × M` grid.

**Parameters**

- **rows** (*list[dict]*) – Rows from a beams parquet (e.g. via `pq.read_table().to_pylist()`).
- **save_name** (*str*) – Title and filename.
- **step** (*float*) – Filter to this histogram step.
- **N** (*int*) – Grid rows.
- **M** (*int*) – Grid columns.
- **indices** (*list[int] or None, optional*) – Class indices to draw. Default: `None`.
- **save** (*bool, optional*) – Write to `<save_name>.png`. Default: `False`.
- **scatter_size** (*int, optional*) – Marker size. Default: `60`.
- **font_size** (*int, optional*) – Plot font size. Default: `20`.

**Returns**

None. Renders the grid and, when `save=True`, writes `<save_name>.png`.

**Return type**

*None*

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

---

### `combra.mvee.plot_angles`

```python
combra.mvee.plot_angles(data, saved_image_name, step, N, M, indices=None, save=False) → None
```

Plot the ellipse rotation-angle distributions across classes.

**Parameters**

- **data** (*list[dict]*) – Rows from a beams parquet.
- **saved_image_name** (*str*) – Output filename.
- **step** (*float*) – Histogram step to filter on.
- **N** (*int*) – Grid rows.
- **M** (*int*) – Grid columns.
- **indices** (*list[int] or None, optional*) – Class indices to draw. Default: `None`.
- **save** (*bool, optional*) – Write the figure. Default: `False`.

**Returns**

None. Renders the figure and, when `save=True`, writes it to disk.

**Return type**

*None*

**Example**

```python
>>> import pyarrow.parquet as pq
>>> from combra import mvee
>>> rows = pq.read_table('./beams/beams_n100.parquet').to_pylist()
>>> mvee.plot_angles(rows, saved_image_name='angles.png',
...                  step=4, N=2, M=2, save=False)
```

---

### `combra.mvee.plot_beam_compare`

```python
combra.mvee.plot_beam_compare(data_1, data_2, save_name, beam_types, N, M,
                              indices_1, indices_2, save=False,
                              scatter_size=60, font_size=20) → list[str]
```

Side-by-side comparison of two parquet datasets at the same step.

**Parameters**

- **data_1** (*list[dict]*) – Rows from the first beams parquet.
- **data_2** (*list[dict]*) – Rows from the second beams parquet.
- **save_name** (*str*) – Output filename.
- **beam_types** (*list[str]*) – Which fields to compare — e.g. `['a_beams', 'b_beams']`.
- **N** (*int*) – Grid rows.
- **M** (*int*) – Grid columns.
- **indices_1** (*list[int]*) – Class indices from the first set to align.
- **indices_2** (*list[int]*) – Class indices from the second set to align.
- **save** (*bool, optional*) – Write the figure. Default: `False`.
- **scatter_size** (*int, optional*) – Marker size. Default: `60`.
- **font_size** (*int, optional*) – Plot font size. Default: `20`.

**Returns**

- **fit_metrics** (*list[str]*) – One formatted `"<Δk%> <Δb%>"` string per aligned class pair — the relative differences of the linear-fit slope `k` and intercept `b` between the two datasets, in percent.

**Return type**

*list[str]*

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

---

### `combra.mvee.beams_heatmap`

```python
combra.mvee.beams_heatmap(data, step, saved_names, indices=None, bin_max=30, N=7, M=7,
                          font_size=20, save=False, scatter_size=60) → None
```

2-D heatmap of `(a_beam, b_beam)` pairs per class.

**Parameters**

- **data** (*list[dict]*) – Rows from a beams parquet.
- **step** (*float*) – Histogram step to filter on.
- **saved_names** (*list[str]*) – Per-class display names (overrides `meta.name`).
- **indices** (*list[int] or None, optional*) – Class indices to draw. Default: `None`.
- **bin_max** (*float, optional*) – Upper bound on histogram axes. Default: `30`.
- **N** (*int, optional*) – Heatmap bin count along rows. Default: `7`.
- **M** (*int, optional*) – Heatmap bin count along columns. Default: `7`.
- **font_size** (*int, optional*) – Plot font size. Default: `20`.
- **save** (*bool, optional*) – Write the figure. Default: `False`.
- **scatter_size** (*int, optional*) – Marker size. Default: `60`.

**Returns**

None. Renders the heatmaps and, when `save=True`, writes them to disk.

**Return type**

*None*

**Example**

```python
>>> import pyarrow.parquet as pq
>>> from combra import mvee
>>> rows = pq.read_table('./beams/beams_n100.parquet').to_pylist()
>>> mvee.beams_heatmap(rows, step=4, saved_names=['small', 'medium', 'large'],
...                    indices=[0, 1, 2], bin_max=30, N=7, M=7)
```

---

### `combra.mvee.enclosing_ellipse_show`

```python
combra.mvee.enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15) → None
```

Plot a single polygon (index `pos`) and the ellipse fitted around it. Useful for sanity-checking `tolerance`.

**Parameters**

- **image** (*ndarray*) – Source image (raw or preprocessed).
- **pos** (*int, optional*) – Index of the contour to inspect. Default: `0`.
- **tolerance** (*float, optional*) – MVEE tolerance. Default: `0.2`.
- **N** (*int, optional*) – Number of points sampled on the rendered ellipse. Default: `15`.

**Returns**

None. Renders the polygon and its fitted ellipse.

**Return type**

*None*

**Example**

```python
>>> from combra import mvee, data
>>> _, img = data.microstructure_images()[0]
>>> mvee.enclosing_ellipse_show(img, pos=0, tolerance=0.2, N=200)
```

---

## See also

- [`combra.data.PobeditDataset.generate_beams`]({{< relref "/docs/data#pobeditdatasetgenerate_beams" >}}) — drives `get_mvee_params` across whole class folders and writes parquet.
- [`combra.areas`]({{< relref "/docs/areas" >}}) — area / effective-radius plots built on top of beams parquets.
