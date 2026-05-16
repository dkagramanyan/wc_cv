---
title: "Contours"
weight: 4
---

The `combra.contours` module extracts polygon contours from preprocessed binary images and renders them back onto images for visualisation. Used internally by `combra.angles.get_angles`, `combra.mvee.get_mvee_params`, and the crack-graph builder.

```python
from combra import contours
```

## Extraction

### `combra.contours.get_row_contours`

```python
get_row_contours(image)
```

Extract raw contours via Canny edges + Suzuki contour finding. No simplification.

**Parameters**

| name | type | description |
| --- | --- | --- |
| `image` | `ndarray` | Preprocessed binary image. |

**Returns** `list[ndarray]` — one `(N_points, 2)` array per region, every boundary pixel.

---

### `combra.contours.get_contours`

```python
get_contours(image, tol=3)
```

Same as `get_row_contours` but applies Douglas–Peucker simplification with tolerance `tol` to every contour. This is what most downstream code calls.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `image` | `ndarray` | — | Preprocessed binary image. |
| `tol` | `float` | `3` | Simplification tolerance in pixels — higher → fewer vertices. |

**Returns** `list[ndarray]` — simplified contours.

**Example**

```python
from combra import contours, image, data

_, img = data.microstructure_images()[0]
processed = image.do_otsu(img)

raw = contours.get_row_contours(processed)        # ~thousands of vertices per region
simplified = contours.get_contours(processed, tol=3)  # ~tens of vertices per region
print(f'raw[0]: {len(raw[0])} pts   simplified[0]: {len(simplified[0])} pts')
```

---

### `combra.contours.skeletons_coords`

```python
skeletons_coords(image)
```

Morphological skeletonisation + per-component split via `scipy.ndimage.label`. One coordinate array per skeleton component.

**Returns** `list[ndarray]` — `(N_pixels, 2)` int arrays.

---

### `combra.contours.contour_to_binary_mask`

```python
contour_to_binary_mask(contour, eps=1, thickness=1, pad=2)
```

Render a single contour into a small binary mask.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `contour` | `ndarray[N, 2]` | — | Polygon vertices. |
| `eps` | `int` | `1` | Mask quantisation factor (downsamples the bounding box by `eps`). |
| `thickness` | `int` | `1` | Drawing line thickness. |
| `pad` | `int` | `2` | Margin added around the contour. |

**Returns** `ndarray[bool]` — small mask sized to the contour's bounding box + padding.

---

### `combra.contours.scale_contour`

```python
scale_contour(contour, factor)
```

Multiply contour coordinates by `factor`. Use when rescaling contours for a resized image.

---

## Drawing

### `combra.contours.draw_contours`

```python
draw_contours(image, cnts,
              color_corner=(0, 139, 139), color_line=(255, 140, 0),
              r=2, e_width=5, l_width=2, corners=False)
```

Draw simplified contours onto a `PIL.Image`. When `corners=True`, also draws filled circles of radius `r` at each vertex.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `image` | `PIL.Image` | — | Background to draw on. |
| `cnts` | `list[ndarray]` | — | Contours from `get_contours`. |
| `color_corner` | `(R, G, B)` | `(0, 139, 139)` | Vertex marker colour. |
| `color_line` | `(R, G, B)` | `(255, 140, 0)` | Edge colour. |
| `r` | `int` | `2` | Vertex marker radius (if `corners=True`). |
| `e_width`, `l_width` | `int` | `5, 2` | Vertex outline / line widths. |
| `corners` | `bool` | `False` | Draw filled circles at vertices. |

**Returns** `PIL.Image` — modified in place and returned.

---

### `combra.contours.draw_edges`

```python
draw_edges(image, cnts, color=(0, 139, 139), r=4, e_width=5, l_width=4)
```

Numpy version of `draw_contours` — operates on an `ndarray` instead of `PIL.Image`. Use when the surrounding pipeline already works in numpy.

**Returns** `ndarray` — the modified image.

**Example**

```python
from PIL import Image
from skimage import color
from combra import contours, data, image

_, img = data.microstructure_images()[0]
processed = image.do_otsu(img)
simplified = contours.get_contours(processed, tol=3)

rgb = color.gray2rgb(processed)
overlay = contours.draw_edges(rgb, simplified, color=(255, 140, 0), l_width=2)

pil = Image.fromarray(rgb)
overlay_pil = contours.draw_contours(pil, simplified, corners=True, r=2)
```

---

## Notes

`mark_corners_and_classes` is in `__all__` but is incomplete in the current source (depends on a removed helper). Treat it as deprecated.

## See also

- [`combra.image.do_otsu`]({{< relref "/docs/image" >}}) — the binariser upstream of all contour calls.
- [`combra.angles.get_angles`]({{< relref "/docs/angles" >}}) — uses `get_contours` internally.
- [`combra.mvee.get_mvee_params`]({{< relref "/docs/mvee" >}}) — fits an MVEE to each `get_contours` output.
