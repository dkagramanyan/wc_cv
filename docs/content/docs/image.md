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
combra.image.do_otsu(img) → ndarray
```

Single Otsu threshold. Returns a binary `uint8` image.

**Parameters**

- **img** (*ndarray*) – Input image, any shape Otsu can handle.

**Returns**

- **binary** (*ndarray[uint8]*) – Binary `{0, 255}` image.

**Return type**

*ndarray*

**Example**

```python
>>> from combra import image, data
>>> _, img = data.microstructure_images()[0]
>>> binary = image.do_otsu(img)
```

---

### `combra.image.align_figures`

```python
combra.image.align_figures(orig_img_padded, tol, labeled_cnts=False, labels=False) → tuple[ndarray, list[ndarray]]
```

Run contour extraction with Douglas–Peucker tolerance `tol` on a padded binary image. Returns `(visualisation, contours)`. Use `labeled_cnts=True` (with `labels`) when working with hand-annotated contours and want to skip binarisation.

**Parameters**

- **orig_img_padded** (*ndarray*) – Padded binary input image.
- **tol** (*float*) – Douglas–Peucker simplification tolerance.
- **labeled_cnts** (*bool, optional*) – When `True`, use the supplied `labels` instead of running binarisation. Default: `False`.
- **labels** (*ndarray or bool, optional*) – Pre-existing contour labels (only used when `labeled_cnts=True`). Default: `False`.

**Returns**

- **visualisation** (*ndarray*) – Drawn-on copy of the input.
- **contours** (*list[ndarray]*) – Extracted (simplified) contours.

**Return type**

*tuple(ndarray, list[ndarray])*

**Example**

```python
>>> import numpy as np
>>> from combra import image, data
>>> _, img = data.microstructure_images()[0]
>>> padded = np.pad(image.do_otsu(img), 30)
>>> vis, cnts = image.align_figures(padded, tol=3)
>>> print(f'{len(cnts)} contours')
```

---

### `combra.image.fill_polygon`

```python
combra.image.fill_polygon(grid, corners, fill_value=1) → ndarray
```

Rasterize a polygon defined by `corners` into `grid` using a numba point-in-polygon test.

**Parameters**

- **grid** (*ndarray*) – Output grid; mutated in place.
- **corners** (*ndarray[N, 2]*) – Polygon vertices.
- **fill_value** (*scalar, optional*) – Value written to grid cells inside the polygon. Default: `1`.

**Returns**

- **grid** (*ndarray*) – The mutated grid (same array passed in).

**Return type**

*ndarray*

**Example**

```python
>>> import numpy as np
>>> from combra import image
>>> grid = np.zeros((100, 100), dtype=np.uint8)
>>> corners = np.array([[20, 20], [80, 20], [80, 80], [20, 80]])  # square
>>> image.fill_polygon(grid, corners, fill_value=255)
>>> print(grid.sum())   # ≈ 60*60*255 (interior pixels)
```

---

### `combra.image.resize`

```python
combra.image.resize(input_root, output_root, target_size) → None
```

Recursively walk an image tree, resize each image to `target_size`, and convert to RGB if needed. Mirrors the source tree structure under `output_root`.

**Parameters**

- **input_root** (*str or Path*) – Source root.
- **output_root** (*str or Path*) – Destination root.
- **target_size** (*tuple[int, int]*) – `(width, height)` in pixels.

**Returns**

None. Writes resized images under `output_root`.

**Return type**

*None*

**Example**

```python
>>> from combra import image
>>> # Downsample a folder-of-classes from 1024x1024 to 256x256, preserving subdir layout.
>>> image.resize('./data/orig_1024', './data/orig_256', target_size=(256, 256))
```

---

### `combra.image.tiff2jpg`

```python
combra.image.tiff2jpg(folder_path, start_name=0, stop_name=-4, new_folder_path='resized') → None
```

Convert every 16-bit TIFF in `folder_path` to 8-bit JPEG under `new_folder_path`.

**Parameters**

- **folder_path** (*str or Path*) – Folder of TIFFs.
- **start_name** (*int, optional*) – Slice start applied to the source filename when deriving the JPEG name. Default: `0`.
- **stop_name** (*int, optional*) – Slice stop applied to the source filename when deriving the JPEG name. Default: `-4`.
- **new_folder_path** (*str, optional*) – Destination folder. Default: `'resized'`.

**Returns**

None. Writes JPEGs under `new_folder_path`.

**Return type**

*None*

**Example**

```python
>>> from combra import image
>>> image.tiff2jpg('./raw_tiffs', new_folder_path='./jpegs')
```

---

### `combra.image.combine` / `combra.image.imdivide`

```python
combra.image.combine(image, h, k=0.5) → ndarray
combra.image.imdivide(image, h, side) → ndarray
```

Tools for dual-camera SEM images split at horizontal position `h`. `imdivide` returns the left or right half (`side='left'` or `'right'`); `combine` blends both halves with weight `k`.

**Parameters**

- **image** (*ndarray*) – Composite SEM image.
- **h** (*int*) – Horizontal split coordinate.
- **k** (*float, optional*) – Blending weight (`combine` only). Default: `0.5`.
- **side** (*str*) – `'left'` or `'right'` (`imdivide` only).

**Returns**

- *ndarray* – Combined image (`combine`) or one half (`imdivide`).

**Return type**

*ndarray*

**Example**

```python
>>> from combra import image, data
>>> _, img = data.microstructure_images()[0]
>>> h = img.shape[1] // 2
>>> left  = image.imdivide(img, h=h, side='left')
>>> right = image.imdivide(img, h=h, side='right')
>>> blended = image.combine(img, h=h, k=0.5)
>>> print(left.shape, right.shape, blended.shape)
```

---

## Fractal dimension

### `combra.image.valid_box_sizes_from_shape`

```python
combra.image.valid_box_sizes_from_shape(shape, min_boxes=6) → tuple[ndarray, int]
```

Return adaptive dyadic box sizes for the given image shape. Pass the result as the `sizes` argument to `image_fractal_dimension`.

**Parameters**

- **shape** (*tuple[int, int]*) – Image `(H, W)`.
- **min_boxes** (*int, optional*) – Minimum number of valid sizes the result is required to contain. Default: `6`.

**Returns**

- **sizes** (*ndarray[int]*) – Valid box sizes (powers-of-2 trimmed to shape). `None` when the image is too small (`min(H, W) < 4`).
- **scale** (*int*) – Adaptive integer scale factor derived from the shape (the upsampling factor applied when too few dyadic levels fit; `1` when no upscaling is needed). `None` alongside `sizes` when the image is too small.

**Return type**

*tuple(ndarray, int)*

**Example**

```python
>>> from combra import image
>>> sizes, scale = image.valid_box_sizes_from_shape((1024, 1024), min_boxes=6)
>>> print(sizes)      # [2, 4, 8, 16, 32, 64, 128, ...]
>>> print(scale)      # 1 (no upscaling needed at this size)
```

---

### `combra.image.image_fractal_dimension`

```python
combra.image.image_fractal_dimension(binary, sizes, max_shift=0) → float
```

Box-counting fractal dimension of a binary image.

**Parameters**

- **binary** (*ndarray[uint8]*) – Binary image (any non-zero pixel counts as "filled").
- **sizes** (*ndarray[int]*) – Box sizes to sweep — from `valid_box_sizes_from_shape`.
- **max_shift** (*int, optional*) – If `>0`, average box counts over `max_shift` grid offsets to reduce alignment bias. Default: `0`.

**Returns**

- **fd** (*float*) – Fractal dimension estimate.

**Return type**

*float*

**Example**

```python
>>> from combra import image, data
>>> _, img = data.microstructure_images()[0]
>>> binary = image.do_otsu(img)
>>> sizes, _ = image.valid_box_sizes_from_shape(binary.shape, min_boxes=6)
>>> fd = image.image_fractal_dimension(binary, sizes, max_shift=0)
>>> print(f'fractal dimension = {fd:.3f}')
```

---

### `combra.image.contour_fractal_dimension`

```python
combra.image.contour_fractal_dimension(contour, max_size_thr) → float
```

Fractal dimension of a single contour.

**Parameters**

- **contour** (*ndarray[N, 2]*) – Contour vertices.
- **max_size_thr** (*int*) – Maximum box size considered.

**Returns**

- **fd** (*float*) – Fractal dimension, or `np.nan` for contours too short to span `max_size_thr` boxes.

**Return type**

*float*

**Example**

```python
>>> from combra import image, contours, data
>>> _, img = data.microstructure_images()[0]
>>> processed = image.do_otsu(img)
>>> cnts = contours.get_contours(processed, tol=3)
>>> fd = image.contour_fractal_dimension(cnts[0], max_size_thr=64)
>>> print(f'contour fd = {fd:.3f}')
```

---

## Geometry & line kernels

These functions are numba-accelerated and (where possible) zero-allocating — intended for hot-loop use inside the graph builder and angle extractor.

### `combra.image.bresenham_line`

```python
combra.image.bresenham_line(x0, y0, x1, y1) → tuple[ndarray, ndarray]
```

Bresenham line rasterisation.

**Parameters**

- **x0** (*int*) – Start x-coordinate.
- **y0** (*int*) – Start y-coordinate.
- **x1** (*int*) – End x-coordinate.
- **y1** (*int*) – End y-coordinate.

**Returns**

- **xs** (*ndarray[int]*) – Pixel x-coordinates along the line.
- **ys** (*ndarray[int]*) – Pixel y-coordinates along the line.

**Return type**

*tuple(ndarray, ndarray)*

**Example**

```python
>>> from combra import image
>>> xs, ys = image.bresenham_line(0, 0, 10, 5)
>>> print(list(zip(xs, ys)))   # all pixel coordinates on the line
```

---

### `combra.image.get_bresenham_eps_pixels`

```python
combra.image.get_bresenham_eps_pixels(img_contours_np, start_node_x, start_node_y,
                                       end_node_x, end_node_y, border_pixel=255) → int
```

Count contour pixels (value `== border_pixel`) that fall on the Bresenham line between two points. Used in the crack-graph builder to score the "thickness" of an edge.

**Parameters**

- **img_contours_np** (*ndarray*) – Image of drawn contours.
- **start_node_x** (*int*) – Start x-coordinate.
- **start_node_y** (*int*) – Start y-coordinate.
- **end_node_x** (*int*) – End x-coordinate.
- **end_node_y** (*int*) – End y-coordinate.
- **border_pixel** (*int, optional*) – Pixel value that counts as a contour hit. Default: `255`.

**Returns**

- **count** (*int*) – Pixel count.

**Return type**

*int*

**Example**

```python
>>> from combra import image
>>> # img_contours: binary contour mask with contour pixels == 255
>>> n = image.get_bresenham_eps_pixels(
...     img_contours, start_node_x=10, start_node_y=20, end_node_x=80, end_node_y=60,
...     border_pixel=255,
... )
>>> print(f'{n} contour pixels lie on the line')
```

---

### `combra.image.check_border_pixels_vectorized`

```python
combra.image.check_border_pixels_vectorized(img_contours_np,
                                             start_x, start_y, end_x, end_y,
                                             perp_v_x, perp_v_y,
                                             line_eps, border_pixel, border_eps) → int
```

Count contour pixels intersected by a band of width `2*line_eps` perpendicular to a line. Pair with `get_perp_v`.

**Parameters**

- **img_contours_np** (*ndarray*) – Image of drawn contours.
- **start_x** (*int*) – Start x-coordinate.
- **start_y** (*int*) – Start y-coordinate.
- **end_x** (*int*) – End x-coordinate.
- **end_y** (*int*) – End y-coordinate.
- **perp_v_x** (*float*) – Perpendicular vector x-component (from `get_perp_v`).
- **perp_v_y** (*float*) – Perpendicular vector y-component (from `get_perp_v`).
- **line_eps** (*int*) – Half-width of the perpendicular band.
- **border_pixel** (*int*) – Contour-pixel value.
- **border_eps** (*int*) – Margin from image edge that excludes detections.

**Returns**

- **count** (*int*) – Pixel count.

**Return type**

*int*

**Example**

```python
>>> from combra import image
>>> perp_x, perp_y = image.get_perp_v(10, 20, 80, 60, line_eps=10)
>>> n = image.check_border_pixels_vectorized(
...     img_contours, 10, 20, 80, 60, perp_x, perp_y,
...     line_eps=10, border_pixel=255, border_eps=2,
... )
>>> print(n)
```

---

### `combra.image.get_perp_v`

```python
combra.image.get_perp_v(start_x, start_y, end_x, end_y, line_eps=10) → tuple[float, float]
```

Unit perpendicular vector to the line `(start → end)`, scaled by `line_eps`. Use with `check_border_pixels_vectorized` to define the perpendicular band.

**Parameters**

- **start_x** (*int*) – Start x-coordinate.
- **start_y** (*int*) – Start y-coordinate.
- **end_x** (*int*) – End x-coordinate.
- **end_y** (*int*) – End y-coordinate.
- **line_eps** (*int, optional*) – Scale of the resulting vector. Default: `10`.

**Returns**

- **perp_v_x** (*float*) – X-component of the scaled perpendicular vector.
- **perp_v_y** (*float*) – Y-component of the scaled perpendicular vector.

**Return type**

*tuple(float, float)*

**Example**

```python
>>> from combra import image
>>> # Perpendicular to a horizontal line; line_eps=10 ⇒ vector points up by 10.
>>> px, py = image.get_perp_v(0, 0, 100, 0, line_eps=10)
>>> print(px, py)   # ≈ (0, 10) or (0, -10)
```

---

### `combra.image.find_intersection_2d`

```python
combra.image.find_intersection_2d(p1, p2, p3, p4) → bool
```

Test if two line segments intersect.

**Parameters**

- **p1** (*tuple[float, float]*) – First endpoint of the first segment.
- **p2** (*tuple[float, float]*) – Second endpoint of the first segment.
- **p3** (*tuple[float, float]*) – First endpoint of the second segment.
- **p4** (*tuple[float, float]*) – Second endpoint of the second segment.

**Returns**

- **intersect** (*bool*) – `True` if segments `(p1, p2)` and `(p3, p4)` intersect.

**Return type**

*bool*

**Example**

```python
>>> from combra import image
>>> # Crossing 'X' shape — segments (0,0)→(10,10) and (0,10)→(10,0) cross at (5,5).
>>> print(image.find_intersection_2d((0, 0), (10, 10), (0, 10), (10, 0)))   # True
>>> print(image.find_intersection_2d((0, 0), (1, 1), (5, 5), (6, 6)))       # False
```

---

### `combra.image.is_point_in_polygon`

```python
combra.image.is_point_in_polygon(x, y, corners) → bool
```

Numba point-in-polygon test.

**Parameters**

- **x** (*float*) – Query point x-coordinate.
- **y** (*float*) – Query point y-coordinate.
- **corners** (*ndarray[N, 2]*) – Polygon vertices.

**Returns**

- **inside** (*bool*) – Whether the point is inside the polygon.

**Return type**

*bool*

**Example**

```python
>>> import numpy as np
>>> from combra import image
>>> square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
>>> print(image.is_point_in_polygon(5, 5, square))    # True
>>> print(image.is_point_in_polygon(15, 5, square))   # False
```

---

## Notes

`do_edt` and `split_rotate` are in `__all__` but rely on stale helpers or hardcoded paths. Treat them as legacy.

## See also

- [`combra.contours`]({{< relref "/docs/contours" >}}) — polygon extraction from `do_otsu` output.
- [`combra.graph`]({{< relref "/docs/graph" >}}) — heavy user of the geometry kernels here.
</content>
</invoke>
