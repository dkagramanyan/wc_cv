---
title: "Stats"
weight: 15
---

## kernel_points(image, point, step=1)

Returns coordinates of pixels in a square matrix of width 2*step, whose center is point

**Parameters:**
- `image`: ndarray (width, height)
- `point`: tuple (2,)
- `step`: int

**Returns:** tuple (n_points,2)

## stats_preprocess(array, step)

Rounding angles to multiples, e.g. 0, step, 2*step, etc.

**Parameters:**
- `array`: list, ndarray (n,)
- `step`: int

**Returns:** array_copy, array_copy_set, dens_curve

## gaussian(x, mu, sigma, amp=1)

Draws points on the image at locations where there are corners from the corners list

**Parameters:**
- `x`: list (n,)
- `mu`: float
- `sigma`: float
- `amp`: float

**Returns:** list (n,)

## gaussian_bimodal(x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1)

Returns a bimodal normal function with given parameters

**Parameters:**
- `x`: list (n,)
- `mu1`: float
- `mu2`: float
- `sigma1`: float
- `sigma2`: float
- `amp1`: float
- `amp2`: float

**Returns:** list (n,)

## gaussian_termodal(x, mu1, mu2, mu3, sigma1, sigma2, sigma3, amp1=1, amp2=1, amp3=1)

Returns a trimodal normal function with given parameters

**Parameters:**
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

**Returns:** list (n,)

## ellipse(a, b, angle, xc=0, yc=0, num=50)

Returns coordinates of an ellipse constructed from given parameters. Center is (0,0) by default. Angle in radians, decreasing the angle denotes clockwise rotation of the ellipse

**Parameters:**
- `a`: float
- `b`: float
- `angle`: float, rad
- `xc`: float, center coord x
- `yc`: float, center coord y
- `num`: int, number of ellipse points

**Returns:** tuple (num, 2)
