# combra.image

The `combra.image` module bundles every pixel-level helper used elsewhere in combra: binarisation, file format conversion, geometric helpers, fractal dimension, and a few low-level numba kernels.

```python
from combra import image
```

The standard preprocessing pipeline produces an image with three pixel value classes:

$$
\text{preproc} = 1 - \mathrm{Otsu}(\mathrm{median}(\text{image})) + \mathrm{grad}(\mathrm{Otsu}(\mathrm{median}(\text{image})))
$$

| value | meaning |
| --- | --- |
| `0` | WC grain |
| `127` | Co region |
| `254` | boundary of Co region adjacent to a WC grain (1 px thick) |

## Preprocessing

````{py:function} combra.image.do_otsu(img) -> ndarray

Single Otsu threshold. Returns a binary `uint8` image.

:param img: Input image, any shape Otsu can handle.
:type img: ndarray
:returns: **binary** (*ndarray[uint8]*) – Binary `{0, 255}` image.
:rtype: ndarray

**Example**

```python
>>> from combra import image, data
>>> _, img = data.microstructure_images()[0]
>>> binary = image.do_otsu(img)
```
````

````{py:function} combra.image.align_figures(orig_img_padded, tol, labeled_cnts=False, labels=False) -> tuple[ndarray, list[ndarray]]

Run contour extraction with Douglas–Peucker tolerance `tol` on a padded binary image. Returns `(visualisation, contours)`. Use `labeled_cnts=True` (with `labels`) when working with hand-annotated contours and want to skip binarisation.

:param orig_img_padded: Padded binary input image.
:type orig_img_padded: ndarray
:param tol: Douglas–Peucker simplification tolerance.
:type tol: float
:param labeled_cnts: When `True`, use the supplied `labels` instead of running binarisation. Default: `False`.
:type labeled_cnts: bool, optional
:param labels: Pre-existing contour labels (only used when `labeled_cnts=True`). Default: `False`.
:type labels: ndarray or bool, optional
:returns: **visualisation** (*ndarray*) – Drawn-on copy of the input; and **contours** (*list[ndarray]*) – Extracted (simplified) contours.
:rtype: tuple(ndarray, list[ndarray])

**Example**

```python
>>> import numpy as np
>>> from combra import image, data
>>> _, img = data.microstructure_images()[0]
>>> padded = np.pad(image.do_otsu(img), 30)
>>> vis, cnts = image.align_figures(padded, tol=3)
>>> print(f'{len(cnts)} contours')
```
````

````{py:function} combra.image.fill_polygon(grid, corners, fill_value=1) -> ndarray

Rasterize a polygon defined by `corners` into `grid` using a numba point-in-polygon test.

:param grid: Output grid; mutated in place.
:type grid: ndarray
:param corners: Polygon vertices.
:type corners: ndarray[N, 2]
:param fill_value: Value written to grid cells inside the polygon. Default: `1`.
:type fill_value: scalar, optional
:returns: **grid** (*ndarray*) – The mutated grid (same array passed in).
:rtype: ndarray

**Example**

```python
>>> import numpy as np
>>> from combra import image
>>> grid = np.zeros((100, 100), dtype=np.uint8)
>>> corners = np.array([[20, 20], [80, 20], [80, 80], [20, 80]])  # square
>>> image.fill_polygon(grid, corners, fill_value=255)
>>> print(grid.sum())   # ≈ 60*60*255 (interior pixels)
```
````

````{py:function} combra.image.resize(input_root, output_root, target_size) -> None

Recursively walk an image tree, resize each image to `target_size`, and convert to RGB if needed. Mirrors the source tree structure under `output_root`.

:param input_root: Source root.
:type input_root: str or Path
:param output_root: Destination root.
:type output_root: str or Path
:param target_size: `(width, height)` in pixels.
:type target_size: tuple[int, int]
:returns: Nothing. Writes resized images under `output_root`.
:rtype: None

**Example**

```python
>>> from combra import image
>>> # Downsample a folder-of-classes from 1024x1024 to 256x256, preserving subdir layout.
>>> image.resize('./data/orig_1024', './data/orig_256', target_size=(256, 256))
```
````

````{py:function} combra.image.tiff2jpg(folder_path, start_name=0, stop_name=-4, new_folder_path='resized') -> None

Convert every 16-bit TIFF in `folder_path` to 8-bit JPEG under `new_folder_path`.

:param folder_path: Folder of TIFFs.
:type folder_path: str or Path
:param start_name: Slice start applied to the source filename when deriving the JPEG name. Default: `0`.
:type start_name: int, optional
:param stop_name: Slice stop applied to the source filename when deriving the JPEG name. Default: `-4`.
:type stop_name: int, optional
:param new_folder_path: Destination folder. Default: `'resized'`.
:type new_folder_path: str, optional
:returns: Nothing. Writes JPEGs under `new_folder_path`.
:rtype: None

**Example**

```python
>>> from combra import image
>>> image.tiff2jpg('./raw_tiffs', new_folder_path='./jpegs')
```
````

````{py:function} combra.image.combine(image, h, k=0.5) -> ndarray

Tools for dual-camera SEM images split at horizontal position `h`. `combine` blends both halves with weight `k`.

:param image: Composite SEM image.
:type image: ndarray
:param h: Horizontal split coordinate.
:type h: int
:param k: Blending weight. Default: `0.5`.
:type k: float, optional
:returns: *ndarray* – Combined image.
:rtype: ndarray

**Example**

```python
>>> from combra import image, data
>>> _, img = data.microstructure_images()[0]
>>> h = img.shape[1] // 2
>>> blended = image.combine(img, h=h, k=0.5)
>>> print(blended.shape)
```
````

````{py:function} combra.image.imdivide(image, h, side) -> ndarray

Tools for dual-camera SEM images split at horizontal position `h`. `imdivide` returns the left or right half (`side='left'` or `'right'`).

:param image: Composite SEM image.
:type image: ndarray
:param h: Horizontal split coordinate.
:type h: int
:param side: `'left'` or `'right'`.
:type side: str
:returns: *ndarray* – One half of the image.
:rtype: ndarray

**Example**

```python
>>> from combra import image, data
>>> _, img = data.microstructure_images()[0]
>>> h = img.shape[1] // 2
>>> left  = image.imdivide(img, h=h, side='left')
>>> right = image.imdivide(img, h=h, side='right')
>>> print(left.shape, right.shape)
```
````

## Fractal dimension

````{py:function} combra.image.valid_box_sizes_from_shape(shape, min_boxes=6) -> tuple[ndarray, int]

Return adaptive dyadic box sizes for the given image shape. Pass the result as the `sizes` argument to `image_fractal_dimension`.

:param shape: Image `(H, W)`.
:type shape: tuple[int, int]
:param min_boxes: Minimum number of valid sizes the result is required to contain. Default: `6`.
:type min_boxes: int, optional
:returns: **sizes** (*ndarray[int]*) – Valid box sizes (powers-of-2 trimmed to shape). `None` when the image is too small (`min(H, W) < 4`); and **scale** (*int*) – Adaptive integer scale factor derived from the shape (the upsampling factor applied when too few dyadic levels fit; `1` when no upscaling is needed). `None` alongside `sizes` when the image is too small.
:rtype: tuple(ndarray, int)

**Example**

```python
>>> from combra import image
>>> sizes, scale = image.valid_box_sizes_from_shape((1024, 1024), min_boxes=6)
>>> print(sizes)      # [2, 4, 8, 16, 32, 64, 128, ...]
>>> print(scale)      # 1 (no upscaling needed at this size)
```
````

````{py:function} combra.image.image_fractal_dimension(binary, sizes, max_shift=0) -> float

Box-counting fractal dimension of a binary image.

:param binary: Binary image (any non-zero pixel counts as "filled").
:type binary: ndarray[uint8]
:param sizes: Box sizes to sweep — from `valid_box_sizes_from_shape`.
:type sizes: ndarray[int]
:param max_shift: If `>0`, average box counts over `max_shift` grid offsets to reduce alignment bias. Default: `0`.
:type max_shift: int, optional
:returns: **fd** (*float*) – Fractal dimension estimate.
:rtype: float

**Example**

```python
>>> from combra import image, data
>>> _, img = data.microstructure_images()[0]
>>> binary = image.do_otsu(img)
>>> sizes, _ = image.valid_box_sizes_from_shape(binary.shape, min_boxes=6)
>>> fd = image.image_fractal_dimension(binary, sizes, max_shift=0)
>>> print(f'fractal dimension = {fd:.3f}')
```
````

````{py:function} combra.image.contour_fractal_dimension(contour, max_size_thr) -> float

Fractal dimension of a single contour.

:param contour: Contour vertices.
:type contour: ndarray[N, 2]
:param max_size_thr: Maximum box size considered.
:type max_size_thr: int
:returns: **fd** (*float*) – Fractal dimension, or `np.nan` for contours too short to span `max_size_thr` boxes.
:rtype: float

**Example**

```python
>>> from combra import image, contours, data
>>> _, img = data.microstructure_images()[0]
>>> processed = image.do_otsu(img)
>>> cnts = contours.get_contours(processed, tol=3)
>>> fd = image.contour_fractal_dimension(cnts[0], max_size_thr=64)
>>> print(f'contour fd = {fd:.3f}')
```
````

## Geometry & line kernels

These functions are numba-accelerated and (where possible) zero-allocating — intended for hot-loop use inside the graph builder and angle extractor.

````{py:function} combra.image.bresenham_line(x0, y0, x1, y1) -> tuple[ndarray, ndarray]

Bresenham line rasterisation.

:param x0: Start x-coordinate.
:type x0: int
:param y0: Start y-coordinate.
:type y0: int
:param x1: End x-coordinate.
:type x1: int
:param y1: End y-coordinate.
:type y1: int
:returns: **xs** (*ndarray[int]*) – Pixel x-coordinates along the line; and **ys** (*ndarray[int]*) – Pixel y-coordinates along the line.
:rtype: tuple(ndarray, ndarray)

**Example**

```python
>>> from combra import image
>>> xs, ys = image.bresenham_line(0, 0, 10, 5)
>>> print(list(zip(xs, ys)))   # all pixel coordinates on the line
```
````

````{py:function} combra.image.get_bresenham_eps_pixels(img_contours_np, start_node_x, start_node_y, end_node_x, end_node_y, border_pixel=255) -> int

Count contour pixels (value `== border_pixel`) that fall on the Bresenham line between two points. Used in the crack-graph builder to score the "thickness" of an edge.

:param img_contours_np: Image of drawn contours.
:type img_contours_np: ndarray
:param start_node_x: Start x-coordinate.
:type start_node_x: int
:param start_node_y: Start y-coordinate.
:type start_node_y: int
:param end_node_x: End x-coordinate.
:type end_node_x: int
:param end_node_y: End y-coordinate.
:type end_node_y: int
:param border_pixel: Pixel value that counts as a contour hit. Default: `255`.
:type border_pixel: int, optional
:returns: **count** (*int*) – Pixel count.
:rtype: int

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
````

````{py:function} combra.image.check_border_pixels_vectorized(img_contours_np, start_x, start_y, end_x, end_y, perp_v_x, perp_v_y, line_eps, border_pixel, border_eps) -> int

Count contour pixels intersected by a band of width `2*line_eps` perpendicular to a line. Pair with `get_perp_v`.

:param img_contours_np: Image of drawn contours.
:type img_contours_np: ndarray
:param start_x: Start x-coordinate.
:type start_x: int
:param start_y: Start y-coordinate.
:type start_y: int
:param end_x: End x-coordinate.
:type end_x: int
:param end_y: End y-coordinate.
:type end_y: int
:param perp_v_x: Perpendicular vector x-component (from `get_perp_v`).
:type perp_v_x: float
:param perp_v_y: Perpendicular vector y-component (from `get_perp_v`).
:type perp_v_y: float
:param line_eps: Half-width of the perpendicular band.
:type line_eps: int
:param border_pixel: Contour-pixel value.
:type border_pixel: int
:param border_eps: Margin from image edge that excludes detections.
:type border_eps: int
:returns: **count** (*int*) – Pixel count.
:rtype: int

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
````

````{py:function} combra.image.get_perp_v(start_x, start_y, end_x, end_y, line_eps=10) -> tuple[float, float]

Unit perpendicular vector to the line `(start → end)`, scaled by `line_eps`. Use with `check_border_pixels_vectorized` to define the perpendicular band.

:param start_x: Start x-coordinate.
:type start_x: int
:param start_y: Start y-coordinate.
:type start_y: int
:param end_x: End x-coordinate.
:type end_x: int
:param end_y: End y-coordinate.
:type end_y: int
:param line_eps: Scale of the resulting vector. Default: `10`.
:type line_eps: int, optional
:returns: **perp_v_x** (*float*) – X-component of the scaled perpendicular vector; and **perp_v_y** (*float*) – Y-component of the scaled perpendicular vector.
:rtype: tuple(float, float)

**Example**

```python
>>> from combra import image
>>> # Perpendicular to a horizontal line; line_eps=10 ⇒ vector points up by 10.
>>> px, py = image.get_perp_v(0, 0, 100, 0, line_eps=10)
>>> print(px, py)   # ≈ (0, 10) or (0, -10)
```
````

````{py:function} combra.image.find_intersection_2d(p1, p2, p3, p4) -> bool

Test if two line segments intersect.

:param p1: First endpoint of the first segment.
:type p1: tuple[float, float]
:param p2: Second endpoint of the first segment.
:type p2: tuple[float, float]
:param p3: First endpoint of the second segment.
:type p3: tuple[float, float]
:param p4: Second endpoint of the second segment.
:type p4: tuple[float, float]
:returns: **intersect** (*bool*) – `True` if segments `(p1, p2)` and `(p3, p4)` intersect.
:rtype: bool

**Example**

```python
>>> from combra import image
>>> # Crossing 'X' shape — segments (0,0)→(10,10) and (0,10)→(10,0) cross at (5,5).
>>> print(image.find_intersection_2d((0, 0), (10, 10), (0, 10), (10, 0)))   # True
>>> print(image.find_intersection_2d((0, 0), (1, 1), (5, 5), (6, 6)))       # False
```
````

````{py:function} combra.image.is_point_in_polygon(x, y, corners) -> bool

Numba point-in-polygon test.

:param x: Query point x-coordinate.
:type x: float
:param y: Query point y-coordinate.
:type y: float
:param corners: Polygon vertices.
:type corners: ndarray[N, 2]
:returns: **inside** (*bool*) – Whether the point is inside the polygon.
:rtype: bool

**Example**

```python
>>> import numpy as np
>>> from combra import image
>>> square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
>>> print(image.is_point_in_polygon(5, 5, square))    # True
>>> print(image.is_point_in_polygon(15, 5, square))   # False
```
````

## Notes

:::{note}
`do_edt` and `split_rotate` are in `__all__` but rely on stale helpers or hardcoded paths. Treat them as legacy.
:::

## See also

- {doc}`combra.contours <contours>` — polygon extraction from `do_otsu` output.
- {doc}`combra.graph <graph>` — heavy user of the geometry kernels here.
