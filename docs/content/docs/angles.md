---
title: "Angles"
weight: 5
---

The `combra.angles` module extracts polygon angles from preprocessed binary images and provides plotting helpers for the resulting distributions.

## Quick start

Compute angles for one image:

```python
from combra import angles, image, data

img_name, img = data.microstructure_images()[0]
processed = image.do_otsu(img)

angles_array, contours = angles.get_angles(processed, border_eps=5, tol=3, min_segment_len=10.0)
print(f'{len(angles_array)} angles, mean={angles_array.mean():.2f}°')
```

For a whole class folder, drive the dataset:

```python
from combra import data

ds = data.PobeditDataset(path=data.microstructure_class_path(),
                         max_images_num_per_class=360)
ds.generate_angles('orig.parquet',
                   types_dict={'Ultra_Co11': 'medium', 'Ultra_Co25': 'fine'},
                   step=5, workers=20)
```

Then plot from the parquet:

```python
angles.angles_plot_base(parquet_path='orig_step_5_N_360.parquet',
                        N=2, M=1, save=False, scatter_size=20, font_size=20)
```

## Reference

### `get_angles(image, border_eps=5, tol=3, min_segment_len=10.0)`

Extract polygon angles from a preprocessed binary image.

- `image`: preprocessed image `(H, W)` or `(H, W, 1)`.
- `border_eps`: distance from image edge — contours closer than this are dropped.
- `tol`: Douglas–Peucker simplification tolerance.
- `min_segment_len`: vertices that produce shorter neighbouring segments are iteratively removed before angle calculation. Higher values yield smoother distributions but fewer angles (5–20 px is typical).

Returns `(angles_array, contours)`. Angles are signed and use counter-clockwise traversal so values >180° are preserved.

### `angles_plot_base(rows=None, save_name=None, N=20, M=20, save=False, indices=None, font_size=20, scatter_size=20, xlim=None, ylim=None, parquet_path=None, step=None, show=True)`

Plot angle density curves and bimodal Gaussian fits in an `N×M` grid.

Pass either `rows` (list of dicts) or `parquet_path`. When loading a parquet that contains multiple steps under `prep_per_step`, supply `step` to pick one. `indices` filters which rows to draw.

### `angles_legend(images_amount, name, itype, step, mus, sigmas, amps, norm)`

Format a multi-line legend string from Gaussian fit parameters. Useful when you want to overlay text on a custom plot.

## Notes

`__all__` lists `angles_approx_save` and `angles_approx_modes` but those names are not implemented in the current source — the parquet workflow on `PobeditDataset.generate_angles` replaces them.
