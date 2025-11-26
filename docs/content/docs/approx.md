---
title: "Approximation"
weight: 16
---

## gaussian_fit(x, y, mu=1, sigma=1, amp=1)

Approximates given points with a normal function

**Parameters:**
- `x`: list (n,)
- `y`: list (n,)
- `mu`: float
- `sigma`: float
- `amp`: float

**Returns:** mus, sigmas, amps

## gaussian_fit_bimodal(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)

Approximates given points with a bimodal normal function

**Parameters:**
- `x`: list (n,)
- `y`: list (n,)
- `mu1`: float
- `mu2`: float
- `sigma1`: float
- `sigma2`: float
- `amp1`: float
- `amp2`: float

**Returns:** mus, sigmas, amps

## gaussian_fit_termodal(x, y, mu1=10, mu2=100, mu3=240, sigma1=10, sigma2=30, sigma3=30, amp1=1, amp2=1, amp3=1)

Approximates given points with a trimodal normal function

**Parameters:**
- `x`: list (n,)
- `y`: list (n,)
- `mu1`: float
- `mu2`: float
- `mu3`: float
- `sigma1`: float
- `sigma2`: float
- `sigma3`: float
- `amp1`: float
- `amp2`: float
- `amp3`: float

**Returns:** mus, sigmas, amps

## lin_regr_approx(x, y)

Approximates distribution with a linear function and creates a plot from distribution parameters

**Parameters:**
- `x`: list (n,)
- `y`: list (n,)

**Returns:** (x_pred, y_pred), k, b, angle, score

## bimodal_gauss_approx(x, y)

Approximates distribution with a bimodal Gaussian

**Parameters:**
- `x`: list (n,)
- `y`: list (n,)

**Returns:** (x_gauss, y_gauss), mus, sigmas, amps
