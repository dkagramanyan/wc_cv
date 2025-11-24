---
title: "Draw"
weight: 14
---

## draw_corners(image, corners, color=255)

Наносит на изображение точки в местах, где есть углы списка corners

**Параметры:**
- `image`: ndarray (width, height, channels)
- `corners`: list (n_corners,2)
- `color`: int

**Возвращает:** ndarray (width, height, channels)

## draw_edges(image, cnts, color=(50, 50, 50))

Рисует на изображении линии по точкам контура cnts, линии в стиле x^1->x^2,x^2->x^3 и тд

**Параметры:**
- `image`: ndarray (width, height, channels)
- `cnts`: ndarray (n_cnts,n,2)
- `color`: tuple (3,)

**Возвращает:** ndarray (width, height, channels)

## draw_tree(image, centres=False, leafs=False, nodes=False, bones=False)

На вход подается бинаризованное изображение. Рисует на инвертированном изображении скелет: точки их центров, листьев, узлов и пикселей скелета

**Параметры:**
- `img`: ndarray (width, height)
- `centres`: Bool
- `leafs`: Bool
- `nodes`: Bool
- `bones`: Bool

**Возвращает:** ndarray (width, height, channels)

