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

- **img** (*ndarray*) — Input image, any shape Otsu can handle.

**Returns**

- **binary** (*ndarray[uint8]*) — Binary `{0, 255}` image.

**Examples**

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

**Parameters**

- **orig_img_padded** (*ndarray*) — Padded binary input image.
- **tol** (*float*) — Douglas–Peucker simplification tolerance.
- **labeled_cnts** (*bool*, default `False`) — When `True`, use the supplied `labels` instead of running binarisation.
- **labels** (*ndarray or bool*, default `False`) — Pre-existing contour labels (only used when `labeled_cnts=True`).

**Returns**

- **visualisation** (*ndarray*) — Drawn-on copy of the input.
- **contours** (*list[ndarray]*) — Extracted (simplified) contours.

---

### `combra.image.fill_polygon`

```python
fill_polygon(grid, corners, fill_value=1)
```

Rasterize a polygon defined by `corners` into `grid` using a numba point-in-polygon test.

**Parameters**

- **grid** (*ndarray*) — Output grid; mutated in place.
- **corners** (*ndarray[N, 2]*) — Polygon vertices.
- **fill_value** (*scalar*, default `1`) — Value written to grid cells inside the polygon.

---

### `combra.image.resize`

```python
resize(input_root, output_root, target_size)
```

Recursively walk an image tree, resize each image to `target_size`, and convert to RGB if needed. Mirrors the source tree structure under `output_root`.

**Parameters**

- **input_root** (*str or Path*) — Source root.
- **output_root** (*str or Path*) — Destination root.
- **target_size** (*tuple[int, int]*) — `(width, height)` in pixels.

---

### `combra.image.tiff2jpg`

```python
tiff2jpg(folder_path, start_name=0, stop_name=-4, new_folder_path='resized')
```

Convert every 16-bit TIFF in `folder_path` to 8-bit JPEG under `new_folder_path`.

**Parameters**

- **folder_path** (*str or Path*) — Folder of TIFFs.
- **start_name**, **stop_name** (*int*, default `0`, `-4`) — Slice bounds applied to the source filename when deriving the JPEG name.
- **new_folder_path** (*str*, default `'resized'`) — Destination folder.

---

### `combra.image.combine` / `combra.image.imdivide`

```python
combine(image, h, k=0.5)
imdivide(image, h, side)
```

Tools for dual-camera SEM images split at horizontal position `h`. `imdivide` returns the left or right half (`side='left'` or `'right'`); `combine` blends both halves with weight `k`.

**Parameters**

- **image** (*ndarray*) — Composite SEM image.
- **h** (*int*) — Horizontal split coordinate.
- **k** (*float*, default `0.5*) — Blending weight (`combine` only).
- **side** (*str*) — `'left'` or `'right'` (`imdivide` only).

**Returns**

- *ndarray* — Combined image (`combine`) or one half (`imdivide`).

---

## Fractal dimension

### `combra.image.valid_box_sizes_from_shape`

```python
valid_box_sizes_from_shape(shape, min_boxes=6)
```

Return adaptive dyadic box sizes for the given image shape. Pass the result as the `sizes` argument to `image_fractal_dimension`.

**Parameters**

- **shape** (*tuple[int, int]*) — Image `(H, W)`.
- **min_boxes** (*int*, default `6`) — Minimum number of valid sizes the result is required to contain.

**Returns**

- **sizes** (*ndarray[int]*) — Valid box sizes (powers-of-2 trimmed to shape).
- **n_boxes** (*ndarray[int]*) — Box count at each size.

---

### `combra.image.image_fractal_dimension`

```python
image_fractal_dimension(binary, sizes, max_shift=0)
```

Box-counting fractal dimension of a binary image.

**Parameters**

- **binary** (*ndarray[uint8]*) — Binary image (any non-zero pixel counts as "filled").
- **sizes** (*ndarray[int]*) — Box sizes to sweep — from `valid_box_sizes_from_shape`.
- **max_shift** (*int*, default `0`) — If `>0`, average box counts over `max_shift` grid offsets to reduce alignment bias.

**Returns**

- **fd** (*float*) — Fractal dimension estimate.

**Examples**

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

Fractal dimension of a single contour.

**Parameters**

- **contour** (*ndarray[N, 2]*) — Contour vertices.
- **max_size_thr** (*int*) — Maximum box size considered.

**Returns**

- **fd** (*float*) — Fractal dimension, or `np.nan` for contours too short to span `max_size_thr` boxes.

---

## Geometry & line kernels

These functions are numba-accelerated and (where possible) zero-allocating — intended for hot-loop use inside the graph builder and angle extractor.

### `combra.image.bresenham_line`

```python
bresenham_line(x0, y0, x1, y1)
```

Bresenham line rasterisation.

**Parameters**

- **x0**, **y0**, **x1**, **y1** (*int*) — Line endpoints.

**Returns**

- **xs**, **ys** (*ndarray[int]*) — Pixel coordinates along the line.

---

### `combra.image.get_bresenham_eps_pixels`

```python
get_bresenham_eps_pixels(img_contours_np, start_node_x, start_node_y,
                        end_node_x, end_node_y, border_pixel=255)
```

Count contour pixels (value `== border_pixel`) that fall on the Bresenham line between two points. Used in the crack-graph builder to score the "thickness" of an edge.

**Parameters**

- **img_contours_np** (*ndarray*) — Image of drawn contours.
- **start_node_x**, **start_node_y**, **end_node_x**, **end_node_y** (*int*) — Line endpoints.
- **border_pixel** (*int*, default `255`) — Pixel value that counts as a contour hit.

**Returns**

- **count** (*int*) — Pixel count.

---

### `combra.image.check_border_pixels_vectorized`

```python
check_border_pixels_vectorized(img_contours_np,
                               start_x, start_y, end_x, end_y,
                               perp_v_x, perp_v_y,
                               line_eps, border_pixel, border_eps)
```

Count contour pixels intersected by a band of width `2*line_eps` perpendicular to a line. Pair with `get_perp_v`.

**Parameters**

- **img_contours_np** (*ndarray*) — Image of drawn contours.
- **start_x**, **start_y**, **end_x**, **end_y** (*int*) — Line endpoints.
- **perp_v_x**, **perp_v_y** (*float*) — Perpendicular vector components (from `get_perp_v`).
- **line_eps** (*int*) — Half-width of the perpendicular band.
- **border_pixel** (*int*) — Contour-pixel value.
- **border_eps** (*int*) — Margin from image edge that excludes detections.

**Returns**

- **count** (*int*) — Pixel count.

---

### `combra.image.get_perp_v`

```python
get_perp_v(start_x, start_y, end_x, end_y, line_eps=10)
```

Unit perpendicular vector to the line `(start → end)`, scaled by `line_eps`. Use with `check_border_pixels_vectorized` to define the perpendicular band.

**Parameters**

- **start_x**, **start_y**, **end_x**, **end_y** (*int*) — Line endpoints.
- **line_eps** (*int*, default `10`) — Scale of the resulting vector.

**Returns**

- **perp_v_x**, **perp_v_y** (*float*) — Components of the scaled perpendicular vector.

---

### `combra.image.find_intersection_2d`

```python
find_intersection_2d(p1, p2, p3, p4)
```

Test if two line segments intersect.

**Parameters**

- **p1**, **p2**, **p3**, **p4** (*tuple[float, float]*) — Endpoints. The first pair defines one segment, the second pair the other.

**Returns**

- **intersect** (*bool*) — `True` if segments `(p1, p2)` and `(p3, p4)` intersect.

---

### `combra.image.is_point_in_polygon`

```python
is_point_in_polygon(x, y, corners)
```

Numba point-in-polygon test.

**Parameters**

- **x**, **y** (*float*) — Query point.
- **corners** (*ndarray[N, 2]*) — Polygon vertices.

**Returns**

- **inside** (*bool*) — Whether the point is inside the polygon.

---

## Notes

`mean_pixel`, `do_edt`, and `split_rotate` are in `__all__` but rely on stale helpers or hardcoded paths. Treat them as legacy.

## See also

- [`combra.contours`]({{< relref "/docs/contours" >}}) — polygon extraction from `do_otsu` output.
- [`combra.graph`]({{< relref "/docs/graph" >}}) — heavy user of the geometry kernels here.
