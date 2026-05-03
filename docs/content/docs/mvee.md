---
title: "MVEE"
weight: 6
---

The `combra.mvee` module fits the **Minimum Volume Enclosing Ellipse** to each polygon in an image and provides the matching plotting / comparison helpers. The MVEE algorithm comes from [L.N. Khachiyan](https://ru.wikipedia.org/wiki/Метод_эллипсоидов) (the implementation is borrowed from [radio-beam](https://radio-beam.readthedocs.io/en/latest/api/radio_beam.commonbeam.getMinVolEllipse.html)).

![Enclosed Ellipse](https://pobedit.s3.us-east-2.amazonaws.com/docs_images/enclosed-ellipse.png)

## Quick start

Compute beams for all classes via the dataset, then plot:

```python
from combra import data, mvee

ds = data.PobeditDataset(path=data.microstructure_class_path())
ds.generate_beams(
    save_path='beams',
    types_dict={'Ultra_Co11': 'medium', 'Ultra_Co25': 'fine'},
    step=4, pixel=50/1000,
)

# the parquet is named beams_step_4_N_<n>.parquet — load it however you like
import pyarrow.parquet as pq
rows = pq.read_table('beams_step_4_N_360.parquet').to_pylist()

mvee.plot_beam_base(rows, save_name='beams.png', step=4, N=2, M=1, save=False)
```

## Build

### `get_mvee_params(image, tol=0.2)`
Fit MVEE to every contour in a preprocessed image. Returns `(a_beams, b_beams, angles, centroids, contours)` — semi-major axes, semi-minor axes, rotation angles (radians), centroids, and the source contours. `tol` controls compactness vs runtime.

### `beams_legend(images_amount, name, itype, norm, k, angle, b, score, dist_step, dist_mean)`
Format a legend string for a beam-distribution plot.

## Plot

### `plot_beam_base(rows, save_name, step, N, M, indices=None, save=False, scatter_size=60, font_size=20)`
Plot the `a_beams` and `b_beams` distributions for each class in an `N×M` grid.

### `plot_angles(data, saved_image_name, step, N, M, indices=None, save=False)`
Plot the ellipse rotation-angle distributions.

### `plot_beam_compare(data_1, data_2, save_name, beam_types, N, M, indices_1, indices_2, save=False, scatter_size=60, font_size=20)`
Side-by-side comparison of two parquet datasets. `beam_types` selects which fields (`a_beams`, `b_beams`) to compare. Returns the per-class fit metrics.

### `beams_heatmap(data, step, saved_names, indices=None, bin_max=30, N=7, M=7, font_size=20, save=False, scatter_size=60)`
2-D heatmap of beam-length pairs (`a_beam`, `b_beam`) per class. `saved_names` overrides class display names.

### `enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15)`
Plot a single polygon (index `pos`) and the ellipse circumscribed around it — useful for sanity-checking `tolerance`.

```python
from combra import data, mvee

img = data.microstructure_images()[0][1]
mvee.enclosing_ellipse_show(img, pos=0, tolerance=0.2, N=15)
```

## Notes

`__all__` lists `diametr_approx_save` but it is not implemented in the current source — the parquet workflow on `PobeditDataset.generate_beams` replaces it. `__all__` also re-exports `plot_polygons_area` (from `combra.areas`) and lists `plot_beam_compare` twice; the duplicate is harmless.
