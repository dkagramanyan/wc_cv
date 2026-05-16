---
title: "Areas"
weight: 7
---

Plotting helpers for polygon-area and effective-radius distributions. Both expect rows produced by `PobeditDataset.generate_beams` (parquet).

```python
from combra import areas
```

## `combra.areas.plot_polygons_area`

```python
plot_polygons_area(data, saved_image_name, step, N, M, indices=None, save=False,
                   start=1, end=None, pixel=50/1000, font_size=20, s=60,
                   log_min_val=-8, min_area_num=10)
```

Log probability distribution of polygon areas with a linear fit per class on the log-log axes.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `data` | `list[dict]` | — | Rows from a beams parquet. |
| `saved_image_name` | `str` | — | Filename for the figure. |
| `step` | `float` | — | Bin size in calibrated units. |
| `N`, `M` | `int` | — | Grid dimensions when arranging multiple classes. |
| `indices` | `list[int] \| None` | `None` | Class subset to draw. |
| `save` | `bool` | `False` | Write the figure to disk. |
| `start`, `end` | `int` | `1`, `None` | Histogram slice bounds for the linear fit. |
| `pixel` | `float` | `50/1000` | Physical pixel size (e.g. `50/1000` for 50 nm pixels). |
| `font_size`, `s` | `int` | `20`, `60` | Styling. |
| `log_min_val` | `float` | `-8` | Log-density floor below which bins are discarded before the fit. |
| `min_area_num` | `int` | `10` | Minimum count per bin to include in the fit. |

---

## `combra.areas.plot_polygons_effective_radius`

```python
plot_polygons_effective_radius(data, saved_image_name, step, N, M, indices=None, save=False,
                               start=1, end=None, pixel=50/1000, font_size=20, s=60,
                               log_min_val=-8, max_area_val=35)
```

Same idea as `plot_polygons_area` but plots the effective radius `√(area / π)` instead of the area directly.

**Parameters** — identical to `plot_polygons_area`, with `min_area_num` replaced by `max_area_val` (upper bound on radius before the fit).

**Example**

```python
import pyarrow.parquet as pq
from combra import data, areas

ds = data.PobeditDataset(path=data.microstructure_class_path())
ds.generate_beams(
    save_path='./beams',
    types_dict={'Ultra_Co11': 'medium', 'Ultra_Co25': 'fine'},
    step=4, pixel=50/1000,
)

rows = pq.read_table('./beams/beams_n100.parquet').to_pylist()

areas.plot_polygons_area(
    rows, 'areas.png', step=1, N=2, M=2,
    indices=[0, 1], save=False, log_min_val=-10, min_area_num=10,
)

areas.plot_polygons_effective_radius(
    rows, 'radii.png', step=2, N=2, M=2,
    indices=[0, 1], save=False,
)
```

## See also

- [`combra.mvee`]({{< relref "/docs/mvee" >}}) — the MVEE primitive whose output drives these plots.
- [`combra.data.PobeditDataset.generate_beams`]({{< relref "/docs/data#pobeditdatasetgenerate_beams" >}}) — produces the parquet these helpers consume.
