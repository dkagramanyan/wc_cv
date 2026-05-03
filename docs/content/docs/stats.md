---
title: "Stats"
weight: 9
---

The `combra.stats` module exposes parametric distribution functions and the histogram preprocessor used by every distribution-fitting routine in combra. The functions here are pure (no I/O); fit them with `combra.approx`.

## Distributions

### `gaussian(x, mu, sigma, amp=1)`
Standard Gaussian — useful as a target function for `scipy.optimize.curve_fit`.

### `gaussian_bimodal(x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1)`
Sum of two Gaussians.

### `gaussian_termodal(x, mu1, mu2, mu3, sigma1, sigma2, sigma3, amp1=1, amp2=1, amp3=1)`
Sum of three Gaussians.

### `ellipse(a, b, angle, xc=0, yc=0, num=50)`
Sample `num` points on the ellipse with semi-axes `(a, b)`, rotation `angle` (radians) and centre `(xc, yc)`. Returns an `(num, 2)` array — handy for overlaying MVEE results on a plot.

## Histogram preprocessing

### `stats_preprocess(array, step)`
Quantize `array` to multiples of `step`, count occurrences via `np.bincount`, and normalise to a probability distribution. Returns `(x_bins, y_density)`.

```python
import numpy as np
from combra import stats, approx

angles_array = np.array([12, 13, 87, 90, 92, 178, 180])
x, y = stats.stats_preprocess(angles_array, step=5)
(x_g, y_g), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)
```

### `legacy_stats_preprocess(array, step)`
Older variant that also returns the per-bin index dictionary and a non-normalised count. Kept for backwards compatibility.

## Notes

`calculate_density` is in `__all__` but is deprecated — it walks hardcoded folder paths from the original notebooks. Don't use it in new code.
