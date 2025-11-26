---
title: "Draw"
weight: 14
---

## draw_corners(image, corners, color=255)

Draws points on the image at locations where there are corners from the corners list

**Parameters:**
- `image`: ndarray (width, height, channels)
- `corners`: list (n_corners,2)
- `color`: int

**Returns:** ndarray (width, height, channels)

## draw_edges(image, cnts, color=(50, 50, 50))

Draws lines on the image along contour points cnts, lines in style x^1->x^2,x^2->x^3, etc.

**Parameters:**
- `image`: ndarray (width, height, channels)
- `cnts`: ndarray (n_cnts,n,2)
- `color`: tuple (3,)

**Returns:** ndarray (width, height, channels)

## draw_tree(image, centres=False, leafs=False, nodes=False, bones=False)

Takes a binarized image as input. Draws a skeleton on the inverted image: points of centers, leaves, nodes and skeleton pixels

**Parameters:**
- `img`: ndarray (width, height)
- `centres`: Bool
- `leafs`: Bool
- `nodes`: Bool
- `bones`: Bool

**Returns:** ndarray (width, height, channels)
