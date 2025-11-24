---
title: "Show"
weight: 13
---

## img_show(image, N=20, cmap=plt.cm.nipy_spectral)

Выводит изображение image

**Параметры:**
- `image`: ndarray (height,width,channels)
- `N`: int
- `cmap`: plt cmap

**Возвращает:** None

## enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15)

Выводит точки многоугольника с позиции pos и описанного вокруг него эллипса

**Параметры:**
- `image`: ndarray (height,width,channels)
- `pos`: int
- `tolerance`: foat, koef of ellipse compactness
- `N`: int

**Возвращает:** None

