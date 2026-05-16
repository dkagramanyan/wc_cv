---
title: "Stats"
weight: 9
---

The `combra.stats` module exposes the parametric distribution functions used as targets by `combra.approx`, plus the histogram preprocessor every distribution fit calls first. The functions here are pure (no I/O); they're target functions for `scipy.optimize.curve_fit` or sampling helpers.

```python
from combra import stats
```

## Histogram preprocessing

### `combra.stats.stats_preprocess`

```python
stats_preprocess(array, step)
```

Quantize `array` to multiples of `step`, count occurrences via `np.bincount`, and normalise to a probability distribution. This is the first thing every distribution-fit in combra calls.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `array` | `array_like` | — | 1-D input (angles, beam lengths, etc.). |
| `step` | `float` | — | Bin width in input units. |

**Returns**

| name | type | description |
| --- | --- | --- |
| `x_bins` | `ndarray[float32]` | Bin centres (only non-empty bins). |
| `y_density` | `ndarray[float32]` | Normalised counts, `sum(y) == 1`. |

**Example**

```python
import numpy as np
from combra import stats, approx

angles_array = np.array([12, 13, 87, 90, 92, 178, 180])
x, y = stats.stats_preprocess(angles_array, step=5)
(x_g, y_g), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)
```

---

### `combra.stats.legacy_stats_preprocess`

```python
legacy_stats_preprocess(array, step)
```

Older variant that also returns the per-bin index dictionary and a non-normalised count. Kept for backwards compatibility with pre-parquet code paths; prefer `stats_preprocess` in new code.

---

## Distributions

These functions are designed to be passed to `scipy.optimize.curve_fit` by `combra.approx`. They return ndarrays — feed them an x grid and parameters, get back y values.

### `combra.stats.gaussian`

```python
gaussian(x, mu, sigma, amp=1)
```

Single Gaussian, `amp * 𝒩(x | mu, sigma)`.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `x` | `array_like` | — | Evaluation points. |
| `mu` | `float` | — | Mean. |
| `sigma` | `float` | — | Standard deviation. |
| `amp` | `float` | `1` | Multiplicative amplitude. |

**Returns** `ndarray` — function values at `x`.

---

### `combra.stats.gaussian_bimodal`

```python
gaussian_bimodal(x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1)
```

Sum of two Gaussians. Use `combra.approx.bimodal_gauss_approx` to fit it to a histogram.

---

### `combra.stats.gaussian_termodal`

```python
gaussian_termodal(x, mu1, mu2, mu3, sigma1, sigma2, sigma3, amp1=1, amp2=1, amp3=1)
```

Sum of three Gaussians.

---

### `combra.stats.ellipse`

```python
ellipse(a, b, angle, xc=0, yc=0, num=50)
```

Sample `num` points on the ellipse with semi-axes `(a, b)`, rotation `angle` (radians, decreasing → clockwise), and centre `(xc, yc)`. Handy for overlaying MVEE results.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `a`, `b` | `float` | — | Semi-major / semi-minor axes. |
| `angle` | `float` | — | Rotation in radians. |
| `xc`, `yc` | `float` | `0` | Centre coordinates. |
| `num` | `int` | `50` | Number of sample points. |

**Returns** `ndarray[num, 2]` — `(x, y)` points along the ellipse.

**Example**

```python
import matplotlib.pyplot as plt
from combra import stats

xy = stats.ellipse(a=20, b=8, angle=0.4, xc=0, yc=0, num=200)
plt.plot(xy[:, 0], xy[:, 1])
plt.gca().set_aspect('equal'); plt.show()
```

---

## Notes

`calculate_density` is in `__all__` but is deprecated — it walks hardcoded folder paths from the original notebooks. Don't use it in new code.

## See also

- [`combra.approx`]({{< relref "/docs/approx" >}}) — fits these distributions to data.
- [`combra.angles`]({{< relref "/docs/angles" >}}) — uses `stats_preprocess` + `gaussian_bimodal` for angle histograms.
