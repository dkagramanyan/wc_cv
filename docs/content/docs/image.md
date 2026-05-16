---
title: "Image"
weight: 3
---

The `combra.image` module bundles every pixel-level helper used elsewhere in combra: binarisation, file format conversion, geometric helpers, fractal dimension, and a few low-level numba kernels.

```python
from combra import image
```

The standard preprocessing pipeline produces an image with three pixel value classes:

```
preproc = 1 − Otsu(median(image)) + grad(Otsu(median(image)))
```

| value | meaning |
| --- | --- |
| `0` | WC grain |
| `127` | Co region |
| `254` | boundary of Co region adjacent to a WC grain (1 px thick) |

---

## Preprocessing

### `combra.image.do_otsu`

```python
do_otsu(img)
```

Single Otsu threshold. Returns a binary `uint8` image.

**Parameters**

| name | type | description |
| --- | --- | --- |
| `img` | `ndarray` | Input image, any shape Otsu can handle. |

**Returns** `ndarray[uint8]` — binary `{0, 255}` image.

**Example**

```python
from combra import image, data

_, img = data.microstructure_images()[0]
binary = image.do_otsu(img)
```

---

### `combra.image.align_figures`

```python
align_figures(orig_img_padded, tol, labeled_cnts=False, labels=False)
```

Run contour extraction with Douglas–Peucker tolerance `tol` on a padded binary image. Returns `(visualisation, contours)`. Use `labeled_cnts=True` (with `labels`) when working with hand-annotated contours and want to skip binarisation.

---

### `combra.image.fill_polygon`

```python
fill_polygon(grid, corners, fill_value=1)
```

Rasterize a polygon defined by `corners` into `grid` using a numba point-in-polygon test.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `grid` | `ndarray` | — | Output grid; mutated in place. |
| `corners` | `ndarray[N, 2]` | — | Polygon vertices. |
| `fill_value` | scalar | `1` | Value written to grid cells inside the polygon. |

---

### `combra.image.resize`

```python
resize(input_root, output_root, target_size)
```

Recursively walk an image tree, resize each image to `target_size`, and convert to RGB if needed. Mirrors the source tree structure under `output_root`.

---

### `combra.image.tiff2jpg`

```python
tiff2jpg(folder_path, start_name=0, stop_name=-4, new_folder_path='resized')
```

Convert every 16-bit TIFF in `folder_path` to 8-bit JPEG under `new_folder_path`.

---

### `combra.image.combine` / `combra.image.imdivide`

```python
combine(image, h, k=0.5)
imdivide(image, h, side)
```

Tools for dual-camera SEM images split at horizontal position `h`. `imdivide` returns the left or right half (`side='left'` or `'right'`); `combine` blends both halves with weight `k`.

---

## Fractal dimension

### `combra.image.valid_box_sizes_from_shape`

```python
valid_box_sizes_from_shape(shape, min_boxes=6)
```

Return adaptive dyadic box sizes for the given image shape. Pass the result as the `sizes` argument to `image_fractal_dimension`.

**Returns**

| name | type | description |
| --- | --- | --- |
| `sizes` | `ndarray[int]` | Valid box sizes (powers-of-2 trimmed to shape). |
| `n_boxes` | `ndarray[int]` | Box count at each size. |

---

### `combra.image.image_fractal_dimension`

```python
image_fractal_dimension(binary, sizes, max_shift=0)
```

Box-counting fractal dimension of a binary image.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `binary` | `ndarray[uint8]` | — | Binary image (any non-zero pixel counts as "filled"). |
| `sizes` | `ndarray[int]` | — | Box sizes to sweep — from `valid_box_sizes_from_shape`. |
| `max_shift` | `int` | `0` | If >0, average box counts over `max_shift` grid offsets to reduce alignment bias. |

**Returns** `float` — fractal dimension estimate.

**Example**

```python
from combra import image, data

_, img = data.microstructure_images()[0]
binary = image.do_otsu(img)

sizes, _ = image.valid_box_sizes_from_shape(binary.shape, min_boxes=6)
fd = image.image_fractal_dimension(binary, sizes, max_shift=0)
print(f'fractal dimension = {fd:.3f}')
```

---

### `combra.image.contour_fractal_dimension`

```python
contour_fractal_dimension(contour, max_size_thr)
```

Fractal dimension of a single contour. Returns `np.nan` for contours too short to span `max_size_thr` boxes.

---

## Geometry & line kernels

These functions are numba-accelerated and (where possible) zero-allocating — intended for hot-loop use inside the graph builder and angle extractor.

### `combra.image.bresenham_line`

```python
bresenham_line(x0, y0, x1, y1)
```

Bresenham line rasterisation. **Returns** `(xs, ys)` — int arrays of pixel coordinates along the line.

---

### `combra.image.get_bresenham_eps_pixels`

```python
get_bresenham_eps_pixels(img_contours_np, start_node_x, start_node_y, end_node_x, end_node_y, border_pixel=255)
```

Count contour pixels (value == `border_pixel`) that fall on the Bresenham line between two points. Used in the crack-graph builder to score the "thickness" of an edge.

**Returns** `int` — pixel count.

---

### `combra.image.check_border_pixels_vectorized`

```python
check_border_pixels_vectorized(img_contours_np,
                               start_x, start_y, end_x, end_y,
                               perp_v_x, perp_v_y,
                               line_eps, border_pixel, border_eps)
```

Count contour pixels intersected by a band of width `2*line_eps` perpendicular to a line. Pair with `get_perp_v`.

---

### `combra.image.get_perp_v`

```python
get_perp_v(start_x, start_y, end_x, end_y, line_eps=10)
```

Unit perpendicular vector to the line `(start → end)`, scaled by `line_eps`. Use with `check_border_pixels_vectorized` to define the perpendicular band.

---

### `combra.image.find_intersection_2d`

```python
find_intersection_2d(p1, p2, p3, p4)
```

**Returns** `bool` — `True` if segments `(p1, p2)` and `(p3, p4)` intersect.

---

### `combra.image.is_point_in_polygon`

```python
is_point_in_polygon(x, y, corners)
```

Numba point-in-polygon test. **Returns** `bool`.

---

## Notes

`mean_pixel`, `do_edt`, and `split_rotate` are in `__all__` but rely on stale helpers or hardcoded paths. Treat them as legacy.

## See also

- [`combra.contours`]({{< relref "/docs/contours" >}}) — polygon extraction from `do_otsu` output.
- [`combra.graph`]({{< relref "/docs/graph" >}}) — heavy user of the geometry kernels here.
