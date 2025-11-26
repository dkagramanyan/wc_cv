---
title: "Fig"
weight: 11
---

## line(point1, point2)

Returns raster coordinates of a line between two points using [Bresenham's algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm)

**Parameters:**
- `point1`: tuple (int, int)
- `point2`: tuple (int, int)

**Returns:** ndarray (n_points,(x,y))

## rect(point1, point2)

Returns raster coordinates of a rectangle of width 2r, constructed between two points. It is unclear why the dimension is multiplied by 2.

**Parameters:**
- `point1`: tuple (int, int)
- `point2`: tuple (int, int)
- `r`: int

**Returns:** tuple (n_points, rect_diag*2,2)
