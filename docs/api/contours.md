# combra.contours

The `combra.contours` module extracts polygon contours from preprocessed binary images and renders them back onto images for visualisation. Used internally by {py:func}`combra.angles.vertex_angles`, {py:func}`combra.mvee.fit_mvee`, and the crack-graph builder.

```python
from combra import contours
```

## Extraction

````{py:function} combra.contours.get_row_contours(image) -> list[ndarray]

Extract raw contours via Canny edges + Suzuki contour finding. No simplification.

:param image: Preprocessed binary image.
:type image: ndarray
:returns: **contours** (*list[ndarray]*) – One `(N_points, 2)` array per region, every boundary pixel.
:rtype: list[ndarray]

**Example**

```python
>>> from combra import contours, image, data
>>> _, img = data.microstructure_images()[0]
>>> processed = image.do_otsu(img)
>>> raw = contours.get_row_contours(processed)
>>> print(f'{len(raw)} contours; first has {len(raw[0])} vertices')
```
````

````{py:function} combra.contours.find_contours(image, tol=3) -> list[ndarray]

Same as `get_row_contours` but applies Douglas–Peucker simplification with tolerance `tol` to every contour. This is what most downstream code calls.

```{versionchanged} 0.4
Renamed from ``find_contours`` (scikit-image ``find_contours`` convention) and
the missing ``cv2.approxPolyDP`` ``closed`` argument fixed (it previously raised
``TypeError``). The old ``get_contours`` name is removed (no alias).
```

:param image: Preprocessed binary image.
:type image: ndarray
:param tol: Simplification tolerance in pixels — higher → fewer vertices. Default: `3`.
:type tol: float, optional
:returns: **contours** (*list[ndarray]*) – Simplified contours.
:rtype: list[ndarray]

**Example**

```python
>>> from combra import contours, image, data
>>> _, img = data.microstructure_images()[0]
>>> processed = image.do_otsu(img)
>>> raw = contours.get_row_contours(processed)             # ~thousands of vertices per region
>>> simplified = contours.find_contours(processed, tol=3)  # ~tens of vertices per region
>>> print(f'raw[0]: {len(raw[0])} pts   simplified[0]: {len(simplified[0])} pts')
```
````

````{py:function} combra.contours.skeletons_coords(image) -> list[ndarray]

Morphological skeletonisation + per-component split via `scipy.ndimage.label`. One coordinate array per skeleton component.

:param image: Binary image.
:type image: ndarray
:returns: **coords** (*list[ndarray]*) – `(N_pixels, 2)` int arrays, one per connected component.
:rtype: list[ndarray]

**Example**

```python
>>> from combra import contours, image, data
>>> _, img = data.microstructure_images()[0]
>>> binary = image.do_otsu(img)
>>> skels = contours.skeletons_coords(binary)
>>> print(f'{len(skels)} skeleton components')
```
````

````{py:function} combra.contours.mark_corners_and_classes(image, max_num=100000, sens=0.1, max_dist=1) -> tuple[ndarray, ndarray, int]

```{warning}
Experimental — no working guarantee. Detects corner candidates with OpenCV `goodFeaturesToTrack` and labels gradient-magnitude clusters with `scipy.ndimage.label`.
```

:param image: Single-channel image.
:type image: ndarray
:param max_num: Maximum number of corners to return. Default: `100000`.
:type max_num: int, optional
:param sens: Quality level passed to `goodFeaturesToTrack`. Default: `0.1`.
:type sens: float, optional
:param max_dist: Minimum allowed distance between corners. Default: `1`.
:type max_dist: int, optional
:returns: **corners** (*ndarray*) – `(N, 1, 2)` integer corner coordinates; and **classes** (*ndarray*) – Labelled gradient-cluster image; and **num** (*int*) – Number of labelled clusters.
:rtype: tuple(ndarray, ndarray, int)

**Example**

```python
>>> from combra import contours, image, data
>>> _, img = data.microstructure_images()[0]
>>> binary = image.do_otsu(img)
>>> corners, classes, num = contours.mark_corners_and_classes(binary)
>>> print(f'{len(corners)} corners, {num} gradient clusters')
```
````

````{py:function} combra.contours.contour_to_binary_mask(contour, eps=1, thickness=1, pad=2) -> ndarray

Render a single contour into a small binary mask.

:param contour: Polygon vertices in raw OpenCV layout (the shape returned by `cv2.findContours`). Reshape `(N, 2)` arrays to `(N, 1, 2)` first.
:type contour: ndarray[N, 1, 2]
:param eps: Mask quantisation factor (downsamples the bounding box by `eps`). Default: `1`.
:type eps: int, optional
:param thickness: Drawing line thickness. Default: `1`.
:type thickness: int, optional
:param pad: Margin added around the contour. Default: `2`.
:type pad: int, optional
:returns: **mask** (*ndarray[uint8]*) – Small `{0, 1}` mask sized to the contour's bounding box + padding.
:rtype: ndarray

**Example**

From `poliamid/fractals.ipynb`:

```python
>>> from combra.contours import contour_to_binary_mask
>>> mask = contour_to_binary_mask(cnt, eps=1, thickness=1, pad=2)
>>> print(mask.shape, mask.dtype)
```
````

````{py:function} combra.contours.scale_contour(contour, factor) -> ndarray

Multiply contour coordinates by `factor`. Use when rescaling contours for a resized image.

:param contour: Polygon vertices in raw OpenCV layout (as returned by `cv2.findContours`).
:type contour: ndarray[N, 1, 2]
:param factor: Multiplicative scale.
:type factor: float
:returns: **scaled** (*ndarray[N, 1, 2]*) – Rescaled vertices.
:rtype: ndarray

**Example**

```python
>>> from combra import contours
>>> # Contour was extracted at 256x256; rescale up to 1024x1024 coordinates.
>>> upscaled = contours.scale_contour(cnt, factor=4.0)
```
````

## Drawing

````{py:function} combra.contours.draw_contours(image, cnts, color_corner=(0, 139, 139), color_line=(255, 140, 0), r=2, e_width=5, l_width=2, corners=False) -> PIL.Image

Draw simplified contours onto a `PIL.Image`. When `corners=True`, also draws filled circles of radius `r` at each vertex.

:param image: Background to draw on.
:type image: PIL.Image
:param cnts: Contours from `find_contours`.
:type cnts: list[ndarray]
:param color_corner: Vertex marker colour `(R, G, B)`. Default: `(0, 139, 139)`.
:type color_corner: tuple[int, int, int], optional
:param color_line: Edge colour `(R, G, B)`. Default: `(255, 140, 0)`.
:type color_line: tuple[int, int, int], optional
:param r: Vertex marker radius (if `corners=True`). Default: `2`.
:type r: int, optional
:param e_width: Vertex outline width. Default: `5`.
:type e_width: int, optional
:param l_width: Line width. Default: `2`.
:type l_width: int, optional
:param corners: Draw filled circles at vertices. Default: `False`.
:type corners: bool, optional
:returns: **image** (*PIL.Image*) – Modified in place and returned.
:rtype: PIL.Image

**Example**

```python
>>> from PIL import Image
>>> from skimage import color
>>> from combra import contours, data, image
>>> _, img = data.microstructure_images()[0]
>>> simplified = contours.find_contours(image.do_otsu(img), tol=3)
>>> pil = Image.fromarray(color.gray2rgb(image.do_otsu(img)))
>>> overlay = contours.draw_contours(pil, simplified, corners=True, r=2)
```
````

````{py:function} combra.contours.draw_edges(image, cnts, color=(0, 139, 139), r=4, e_width=5, l_width=4) -> ndarray

Numpy version of `draw_contours` — operates on an `ndarray` instead of `PIL.Image`. Use when the surrounding pipeline already works in numpy.

:param image: Background image.
:type image: ndarray
:param cnts: Contours.
:type cnts: list[ndarray]
:param color: Edge colour. Default: `(0, 139, 139)`.
:type color: tuple[int, int, int], optional
:param r: Vertex marker radius. Default: `4`.
:type r: int, optional
:param e_width: Vertex outline width. Default: `5`.
:type e_width: int, optional
:param l_width: Line width. Default: `4`.
:type l_width: int, optional
:returns: **image** (*ndarray*) – Modified image.
:rtype: ndarray

**Example**

```python
>>> from PIL import Image
>>> from skimage import color
>>> from combra import contours, data, image
>>> _, img = data.microstructure_images()[0]
>>> processed = image.do_otsu(img)
>>> simplified = contours.find_contours(processed, tol=3)
>>> rgb = color.gray2rgb(processed)
>>> overlay = contours.draw_edges(rgb, simplified, color=(255, 140, 0), l_width=2)
>>> pil = Image.fromarray(rgb)
>>> overlay_pil = contours.draw_contours(pil, simplified, corners=True, r=2)
```
````

## Notes

:::{note}
`mark_corners_and_classes` is in `__all__` but is incomplete in the current source (depends on a removed helper). Treat it as deprecated.
:::

## See also

- {py:func}`combra.image.do_otsu` — the binariser upstream of all contour calls.
- {py:func}`combra.angles.vertex_angles` — uses `find_contours` internally.
- {py:func}`combra.mvee.fit_mvee` — fits an MVEE to each `find_contours` output.
