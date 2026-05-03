---
title: "Contours"
weight: 4
---

The `combra.contours` module extracts polygon contours from preprocessed binary images and renders them back onto images for visualisation.

## Quick start

```python
from combra import contours, image, data
from PIL import Image
from skimage import color
import numpy as np

img = data.microstructure_images()[0][1]
processed = image.do_otsu(img)

raw = contours.get_row_contours(processed)              # all contour pixels
simplified = contours.get_contours(processed, tol=3)    # Douglas-Peucker

rgb = color.gray2rgb(processed)
overlay = contours.draw_edges(rgb, simplified, color=(255, 140, 0), l_width=2)
```

## Extraction

### `get_row_contours(image)`
Extract raw contours via Canny edges + Suzuki contour finding. Returns a list of `(N_points, 2)` arrays — one per region, with all boundary pixels.

### `get_contours(image, tol=3)`
Same as `get_row_contours` but applies Douglas–Peucker simplification with tolerance `tol` to every contour.

### `skeletons_coords(image)`
Run morphological skeletonisation on a binary image and return a per-component list of skeleton coordinates (one component per `scipy.ndimage.label` region).

### `contour_to_binary_mask(contour, eps=1, thickness=1, pad=2)`
Render a single contour into a small binary mask. `pad` adds margin around the contour; `thickness` controls line thickness.

### `scale_contour(contour, factor)`
Multiply contour coordinates by `factor`. Handy for rescaling contours when an image has been resized.

## Drawing

### `draw_contours(image, cnts, color_corner=(0,139,139), color_line=(255,140,0), r=2, e_width=5, l_width=2, corners=False)`
Draw simplified contours onto a `PIL.Image`. When `corners=True`, also draws filled circles of radius `r` at each vertex. Returns the modified `PIL.Image`.

### `draw_edges(image, cnts, color=(0,139,139), r=4, e_width=5, l_width=4)`
Numpy version that draws line segments along contour vertices on an ndarray image. Returns the modified ndarray.

```python
from PIL import Image
from combra import contours

pil = Image.fromarray(rgb)
overlay = contours.draw_contours(pil, simplified,
                                 color_line=(255, 140, 0),
                                 corners=True, r=2)
```

## Notes

`mark_corners_and_classes` is in `__all__` but is incomplete in the current source (depends on a removed helper). Treat it as deprecated.
