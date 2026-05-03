---
title: "Image"
weight: 3
---

The `combra.image` module bundles every pixel-level helper used elsewhere in combra: binarisation, file format conversion, geometric helpers, fractal dimension, and a few low-level numba kernels.

## Preprocessing

```
preproc_image = 1 - Otsu(median_filter(image)) + grad(Otsu(median_filter(image)))
```

Pixel value classes after the standard pipeline:
- `0` — WC grain
- `127` — Co region
- `254` — boundary of Co region adjacent to WC grains (1 px thick)

### `do_otsu(img)`
Single Otsu threshold. Returns a binary uint8 image.

### `align_figures(orig_img_padded, tol, labeled_cnts=False, labels=False)`
Run contour extraction with Douglas–Peucker tolerance `tol` on a padded binary image. Returns `(visualisation, contours)`.

### `fill_polygon(grid, corners, fill_value=1)`
Rasterize a polygon defined by `corners` into `grid` using a numba point-in-polygon test.

### `resize(input_root, output_root, target_size)`
Recursively walk an image tree, resize each image to `target_size`, and convert to RGB if needed.

### `tiff2jpg(folder_path, start_name=0, stop_name=-4, new_folder_path='resized')`
Convert every 16-bit TIFF in `folder_path` to 8-bit JPEG under `new_folder_path`.

### `combine(image, h, k=0.5)`, `imdivide(image, h, side)`
Tools for two-camera SEM images. `imdivide` returns the left or right half (`side='left'` / `'right'`); `combine` blends both halves with weight `k`. `h` is the horizontal split position.

## Fractal dimension

```python
from combra import image
import numpy as np

binary = image.do_otsu(img)
sizes, n_boxes = image.valid_box_sizes_from_shape(binary.shape, min_boxes=6)
fd = image.image_fractal_dimension(binary, sizes, max_shift=0)
```

### `valid_box_sizes_from_shape(shape, min_boxes=6)`
Return adaptive dyadic box sizes for the given image shape. Used to drive `image_fractal_dimension`.

### `image_fractal_dimension(binary, sizes, max_shift=0)`
Box-counting fractal dimension of a binary image. `max_shift>0` averages over multiple grid offsets to reduce alignment bias.

### `contour_fractal_dimension(contour, max_size_thr)`
Fractal dimension of a single contour. `max_size_thr` filters out contours too short to fit `max_size_thr` boxes.

## Geometry & line kernels

All functions in this group are numba-accelerated and zero-allocating; they're meant to be hot-loop friendly.

### `bresenham_line(x0, y0, x1, y1)`
Bresenham line rasterisation. Returns `(xs, ys)`.

### `get_bresenham_eps_pixels(img_contours_np, start_node_x, start_node_y, end_node_x, end_node_y, border_pixel=255)`
Count contour pixels that fall on the Bresenham line between two points. Used to score the "thickness" of an edge between graph nodes.

### `check_border_pixels_vectorized(img_contours_np, start_x, start_y, end_x, end_y, perp_v_x, perp_v_y, line_eps, border_pixel, border_eps)`
Count contour pixels intersected by a band of width `2*line_eps` perpendicular to a line.

### `get_perp_v(start_x, start_y, end_x, end_y, line_eps=10)`
Unit perpendicular vector scaled by `line_eps`. Pair with `check_border_pixels_vectorized`.

### `find_intersection_2d(p1, p2, p3, p4)`
Boolean: do segments `(p1,p2)` and `(p3,p4)` intersect.

### `is_point_in_polygon(x, y, corners)`
Numba point-in-polygon test.

## Notes

`mean_pixel`, `do_edt`, and `split_rotate` are in `__all__` but rely on stale helpers or hardcoded paths. Treat them as legacy.
