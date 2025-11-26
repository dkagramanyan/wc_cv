---
title: "Show"
weight: 13
---

## img_show(image, N=20, cmap=plt.cm.nipy_spectral)

Displays the image

**Parameters:**
- `image`: ndarray (height,width,channels)
- `N`: int
- `cmap`: plt cmap

**Returns:** None

## enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15)

Displays polygon points at position pos and the ellipse circumscribed around it

**Parameters:**
- `image`: ndarray (height,width,channels)
- `pos`: int
- `tolerance`: float, coefficient of ellipse compactness
- `N`: int

**Returns:** None
