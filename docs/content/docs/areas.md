---
title: "Areas"
weight: 7
---

Plotting helpers for polygon-area and effective-radius distributions. Both expect rows produced by `PobeditDataset.generate_beams` (parquet) or the legacy JSON layout.

## Quick start

```python
from combra import data, areas

ds = data.PobeditDataset(path=data.microstructure_class_path())
ds.generate_beams(
    save_path='beams',
    types_dict={'Ultra_Co11': 'medium', 'Ultra_Co25': 'fine'},
    step=4, pixel=50/1000,
)
```

## Reference

### `plot_polygons_area(data, saved_image_name, step, N, M, indices=None, save=False, start=1, end=None, pixel=0.05, font_size=20, s=60, log_min_val=-8, min_area_num=10)`
Log probability distribution of polygon areas with a linear fit per class on the log-log axes.

- `step`: bin size (in calibrated units).
- `N`, `M`: grid dimensions when arranging multiple classes.
- `pixel`: physical size of one pixel (e.g. `50/1000` for 50 nm pixels).
- `log_min_val`, `min_area_num`: thresholds used to discard low-statistics bins before the linear fit.

### `plot_polygons_effective_radius(data, saved_image_name, step, N, M, indices=None, save=False, start=1, end=None, pixel=0.05, font_size=20, s=60, log_min_val=-8, max_area_val=35)`
Same idea but plots the effective radius `sqrt(area / π)`.

```python
areas.plot_polygons_area(rows, 'areas.png', step=1, N=2, M=2,
                         indices=[0, 1], save=False, log_min_val=-10, min_area_num=10)

areas.plot_polygons_effective_radius(rows, 'radii.png', step=2, N=2, M=2,
                                     indices=[0, 1], save=False)
```
