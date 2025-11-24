---
title: "Stats"
weight: 15
---

## kernel_points(image, point, step=1)

Возвращает координаты пикселей квадратной матрицы шириной 2*step, центр которой это point

**Параметры:**
- `image`: ndarray (width, height)
- `point`: tuple (2,)
- `step`: int

**Возвращает:** tuple (n_points,2)

## stats_preprocess(array, step)

Приведение углов к кратости, например 0,step,2*step и тд

**Параметры:**
- `array`: list, ndarray (n,)
- `step`: int

**Возвращает:** array_copy, array_copy_set, dens_curve

## gaussian(x, mu, sigma, amp=1)

Наносит на изображение точки в местах, где есть углы списка corners

**Параметры:**
- `x`: list (n,)
- `mu`: float
- `sigma`: float
- `amp`: float

**Возвращает:** list (n,)

## gaussian_bimodal(x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1)

Возвращает бимодальную нормальную фунцию по заданным параметрам

**Параметры:**
- `x`: list (n,)
- `mu1`: float
- `mu2`: float
- `sigma1`: float
- `sigma2`: float
- `amp1`: float
- `amp2`: float

**Возвращает:** list (n,)

## gaussian_termodal(x, mu1, mu2, mu3, sigma1, sigma2, sigma3, amp1=1, amp2=1, amp3=1)

Возвращает термодальную нормальную фунцию по заданным параметрам

**Параметры:**
- `x`: list (n,)
- `mu1`: float
- `mu2`: float
- `mu3`: float
- `sigma1`: float
- `sigma2`: float
- `sigma3`: float
- `amp1`: float
- `amp2`: float
- `amp3`: float

**Возвращает:** list (n,)

## ellipse(a, b, angle, xc=0, yc=0, num=50)

Возвращает координаты эллипса, построенного по заданным параметрам. По умолчанию центр (0,0). Угол в радианах, уменьшение угла обозначает поворот эллипса по часовой стрелке

**Параметры:**
- `a`: float
- `b`: float
- `angle`: float, rad
- `xc`: float, center coord x
- `yc`: float, center coord y
- `num`: int, number of ellipse points

**Возвращает:** tuple (num, 2)

