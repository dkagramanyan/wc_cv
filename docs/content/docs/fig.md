---
title: "Fig"
weight: 11
---

## line(point1, point2)

Возвращает растровые координаты прямой между двумя точками при помощи алгоритма [Брезенхема](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm)

**Параметры:**
- `point1`: tuple (int, int)
- `point2`: tuple (int, int)

**Возвращает:** ndarray (n_points,(x,y))

## rect(point1, point2)

Возвращает растровые координаты прямоугольника ширины 2r, построеного между двумя точками. Не ясно зачем умножать в размерность на 2.

**Параметры:**
- `point1`: tuple (int, int)
- `point2`: tuple (int, int)
- `r`: int

**Возвращает:** tuple (n_points, rect_diag*2,2)

