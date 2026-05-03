---
title: "Approx"
weight: 10
---

The `combra.approx` module fits parametric distributions and lines to 1-D data. The `*_fit` functions return only the optimised parameters; the `*_approx` functions return both the parameters and a sampled curve ready for plotting.

## Example

```python
import numpy as np
from combra import angles, approx, stats

angles_array, _ = angles.get_angles(img, border_eps=5, tol=3)
x, y = stats.stats_preprocess(angles_array, step=5)

(x_gauss, y_gauss), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)
```

## Gaussian fits

### `gaussian_fit(x, y, mu=1, sigma=1, amp=1)`
Single Gaussian via `scipy.optimize.curve_fit`. Returns `(mu, sigma, amp)`.

### `gaussian_fit_bimodal(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)`
Bimodal Gaussian. Returns `(mus, sigmas, amps)` as length-2 lists.

### `gaussian_fit_termodal(x, y, mu1=10, mu2=100, mu3=240, sigma1=10, sigma2=30, sigma3=30, amp1=1, amp2=1, amp3=1)`
Trimodal Gaussian. Returns `(mus, sigmas, amps)` as length-3 lists.

### `gauss_approx(x, y, mu=1, sigma=1, amp=1, x_lim=None, N=100)`
Single Gaussian fit + sampled curve. Returns `((x_gauss, y_gauss), mu, sigma, amp)`. `x_lim` overrides the curve x-range.

### `bimodal_gauss_approx(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)`
Bimodal Gaussian fit + sampled curve. Returns `((x_gauss, y_gauss), mus, sigmas, amps)`.

## Other distributions

### `binomial_approx(x, y, n=10, p=0.5, amp=1, x_lim=None, N=100)`
Binomial fit + sampled curve. Returns `((x_pred, y_pred), n, p, amp)`.

### `poisson_approx(x, y, lam=1, amp=1, x_lim=None, N=100)`
Poisson fit + sampled curve. Returns `((x_pred, y_pred), lam, amp)`.

### `exponential_approx(x, y, a=1, amp=1, x_lim=None, N=100)`
Exponential decay `amp * exp(-a*x)` fit + sampled curve. Returns `((x_pred, y_pred), a, amp)`.

## Linear fits

### `linear_approx(x, y)`
Least-squares line `y = k*x + b`. Returns `((x_pred, y_pred), k, b, angle, score)` — `angle` in degrees, `score` is the R².

### `polyfit_deg1(x, y, i0, i1)`
Fast numba-compiled line fit on the slice `[i0:i1]`. Returns `(a, b, r_squared)`. Use this in tight loops where `linear_approx` overhead matters.
