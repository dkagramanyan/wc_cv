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
get_mvee_params(image, tol=0.2)
```

Fit MVEE to every contour in a preprocessed image. This is the per-image primitive that `PobeditDataset.generate_beams` calls in parallel.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `image` | `ndarray` | — | Preprocessed image. |
| `tol` | `float` | `0.2` | Convergence tolerance. Lower → tighter ellipses, slower. |

**Returns**

| name | type | description |
| --- | --- | --- |
| `a_beams` | `list[float]` | Semi-major axes (in pixels). |
| `b_beams` | `list[float]` | Semi-minor axes. |
| `angles` | `list[float]` | Rotation angles in radians. |
| `centroids` | `list[(x, y)]` | Centre coordinates. |
| `contours` | `list[ndarray]` | The source contours, in the order MVEEs were fit. |

**Example**

```python
from combra import mvee, image, data

_, img = data.microstructure_images()[0]
processed = image.do_otsu(img)
a, b, angles, centroids, cnts = mvee.get_mvee_params(processed, tol=0.2)
print(f'{len(a)} polygons   median a/b = {sum(a)/sum(b):.2f}')
```

---

### `combra.mvee.beams_legend`

```python
beams_legend(images_amount, name, itype, norm, k, angle, b, score, dist_step, dist_mean)
```

Format a multi-line legend string for a beam-distribution plot. Used inside `generate_beams` to populate `prep.beams_legend_a` / `prep.beams_legend_b`.

---

## Plotting

### `combra.mvee.plot_beam_base`

```python
plot_beam_base(rows, save_name, step, N, M, indices=None, save=False, scatter_size=60, font_size=20)
```

Plot the `a_beams` and `b_beams` distributions for each class in an `N × M` grid.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `rows` | `list[dict]` | — | Rows from a beams parquet (e.g. via `pq.read_table().to_pylist()`). |
| `save_name` | `str` | — | Title and filename. |
| `step` | `float` | — | Filter to this histogram step. |
| `N`, `M` | `int` | — | Grid dimensions. |
| `indices` | `list[int] \| None` | `None` | Class indices to draw. |
| `save` | `bool` | `False` | Write to `<save_name>.png`. |
| `scatter_size`, `font_size` | `int` | `60`, `20` | Styling. |

**Example**

```python
import pyarrow.parquet as pq
from combra import data, mvee

ds = data.PobeditDataset(path=data.microstructure_class_path())
ds.generate_beams(
    save_path='./beams',
    types_dict={'Ultra_Co11': 'medium grain'},
    step=4, pixel=50/1000,
)

rows = pq.read_table('./beams/beams_n100.parquet').to_pylist()
mvee.plot_beam_base(rows, save_name='beams.png', step=4, N=2, M=1, save=False)
```

---

### `combra.mvee.plot_angles`

```python
plot_angles(data, saved_image_name, step, N, M, indices=None, save=False)
```

Plot the ellipse rotation-angle distributions across classes.

---

### `combra.mvee.plot_beam_compare`

```python
plot_beam_compare(data_1, data_2, save_name, beam_types, N, M, indices_1, indices_2,
                  save=False, scatter_size=60, font_size=20)
```

Side-by-side comparison of two parquet datasets at the same step.

**Parameters**

| name | type | description |
| --- | --- | --- |
| `data_1`, `data_2` | `list[dict]` | Rows from two beams parquets. |
| `beam_types` | `list[str]` | Which fields to compare — e.g. `['a_beams', 'b_beams']`. |
| `indices_1`, `indices_2` | `list[int]` | Class indices to align between the two sets. |

**Returns** `dict[str, dict]` — per-class fit metrics for the overlay.

---

### `combra.mvee.beams_heatmap`

```python
beams_heatmap(data, step, saved_names, indices=None, bin_max=30, N=7, M=7,
              font_size=20, save=False, scatter_size=60)
```

2-D heatmap of `(a_beam, b_beam)` pairs per class.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `data` | `list[dict]` | — | Rows from a beams parquet. |
| `step` | `float` | — | Histogram step to filter on. |
| `saved_names` | `list[str]` | — | Per-class display names (overrides `meta.name`). |
| `bin_max` | `float` | `30` | Upper bound on histogram axes. |
| `N`, `M` | `int` | `7` | Heatmap bin counts. |

---

### `combra.mvee.enclosing_ellipse_show`

```python
enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15)
```

Plot a single polygon (index `pos`) and the ellipse fitted around it. Useful for sanity-checking `tolerance`.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `image` | `ndarray` | — | Source image (raw or preprocessed). |
| `pos` | `int` | `0` | Index of the contour to inspect. |
| `tolerance` | `float` | `0.2` | MVEE tolerance. |
| `N` | `int` | `15` | Number of points sampled on the rendered ellipse. |

**Example**

```python
from combra import mvee, data

_, img = data.microstructure_images()[0]
mvee.enclosing_ellipse_show(img, pos=0, tolerance=0.2, N=200)
```

---

## Notes

`__all__` lists `diametr_approx_save` but it is not implemented in the current source — the parquet workflow on `PobeditDataset.generate_beams` replaces it. `plot_beam_compare` is listed twice in `__all__`; the duplicate is harmless.

## See also

- [`combra.data.PobeditDataset.generate_beams`]({{< relref "/docs/data#pobeditdatasetgenerate_beams" >}}) — drives `get_mvee_params` across whole class folders and writes parquet.
- [`combra.areas`]({{< relref "/docs/areas" >}}) — area / effective-radius plots built on top of beams parquets.
