---
title: "Stats"
weight: 9
---

The `combra.stats` module exposes the parametric distribution functions used as targets by `combra.approx`, plus the histogram preprocessor every distribution fit calls first, plus the inference helpers (Kendall, Fisher) used by the convergence-analysis pipeline.

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

- **array** (*array_like*) — 1-D input (angles, beam lengths, etc.).
- **step** (*float*) — Bin width in input units.

**Returns**

- **x_bins** (*ndarray[float32]*) — Bin centres (only non-empty bins).
- **y_density** (*ndarray[float32]*) — Normalised counts; `sum(y) == 1`.

**Examples**

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

**Parameters**

- **array** (*array_like*) — 1-D input.
- **step** (*float*) — Bin width.

**Returns**

- **x_bins**, **y_counts** (*ndarray*) — Bin centres and raw (non-normalised) counts.
- **index_map** (*dict*) — `{bin_centre: [original_indices…]}`.

---

## Distributions

These functions are designed to be passed to `scipy.optimize.curve_fit` by `combra.approx`. They return ndarrays — feed them an x grid and parameters, get back y values.

### `combra.stats.gaussian`

```python
gaussian(x, mu, sigma, amp=1)
```

Single Gaussian, `amp * 𝒩(x | mu, sigma)`.

**Parameters**

- **x** (*array_like*) — Evaluation points.
- **mu** (*float*) — Mean.
- **sigma** (*float*) — Standard deviation.
- **amp** (*float*, default `1`) — Multiplicative amplitude.

**Returns**

- **y** (*ndarray*) — Function values at `x`.

---

### `combra.stats.gaussian_bimodal`

```python
gaussian_bimodal(x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1)
```

Sum of two Gaussians. Use `combra.approx.bimodal_gauss_approx` to fit it to a histogram.

**Parameters**

- **x** (*array_like*) — Evaluation points.
- **mu1**, **mu2** (*float*) — Means.
- **sigma1**, **sigma2** (*float*) — Standard deviations.
- **amp1**, **amp2** (*float*, default `1`) — Amplitudes.

**Returns**

- **y** (*ndarray*) — Function values at `x`.

---

### `combra.stats.gaussian_termodal`

```python
gaussian_termodal(x, mu1, mu2, mu3, sigma1, sigma2, sigma3, amp1=1, amp2=1, amp3=1)
```

Sum of three Gaussians.

**Parameters**

- **x** (*array_like*) — Evaluation points.
- **mu1**, **mu2**, **mu3** (*float*) — Means.
- **sigma1**, **sigma2**, **sigma3** (*float*) — Standard deviations.
- **amp1**, **amp2**, **amp3** (*float*, default `1`) — Amplitudes.

**Returns**

- **y** (*ndarray*) — Function values at `x`.

---

### `combra.stats.ellipse`

```python
ellipse(a, b, angle, xc=0, yc=0, num=50)
```

Sample `num` points on the ellipse with semi-axes `(a, b)`, rotation `angle` (radians, decreasing → clockwise), and centre `(xc, yc)`. Handy for overlaying MVEE results.

**Parameters**

- **a**, **b** (*float*) — Semi-major / semi-minor axes.
- **angle** (*float*) — Rotation in radians.
- **xc**, **yc** (*float*, default `0`) — Centre coordinates.
- **num** (*int*, default `50`) — Number of sample points.

**Returns**

- **points** (*ndarray[num, 2]*) — `(x, y)` points along the ellipse.

**Examples**

```python
import matplotlib.pyplot as plt
from combra import stats

xy = stats.ellipse(a=20, b=8, angle=0.4, xc=0, yc=0, num=200)
plt.plot(xy[:, 0], xy[:, 1])
plt.gca().set_aspect('equal'); plt.show()
```

---

## Inference

Hypothesis-testing primitives used by `combra.metrics.convergence_stats` / `print_convergence_report`. Both are pure and tiny — exposed so notebooks can call them directly on ad-hoc curves.

### `combra.stats.kendall_decreasing_p`

```python
kendall_decreasing_p(ns, vals)
```

One-sided Kendall τ p-value for the null "`|vals|` does not decrease as `ns` grows" against the alternative "decreases" (`scipy.stats.kendalltau(..., alternative='less')`). Rank-based, handles ties cleanly.

**Parameters**

- **ns** (*array_like*) — Sample-size axis (typically integer Ns).
- **vals** (*array_like*) — Metric values at each `N`. Sign is ignored — the test runs on `|vals|`.

**Returns**

- **p** (*float*) — One-sided p-value. NaN when the input has fewer than 3 points or is perfectly flat.

---

### `combra.stats.fisher_combine`

```python
fisher_combine(ps)
```

Combine independent one-sided p-values via Fisher's method (`χ²` = `-2 · Σ log p` with `2k` degrees of freedom). Entries that are NaN, ≤ 0, or > 1 are silently filtered out (only `0 < p ≤ 1` is meaningful for `log p`).

**Parameters**

- **ps** (*array_like*) — Individual p-values.

**Returns**

- **combined_p** (*float*) — Fisher's combined p-value. NaN if no valid input remained after filtering.
- **k** (*int*) — Count of p-values that survived filtering and contributed to `combined_p`.

---

## Notes

`calculate_density` is in `__all__` but is deprecated — it walks hardcoded folder paths from the original notebooks. Don't use it in new code.

## See also

- [`combra.approx`]({{< relref "/docs/approx" >}}) — fits these distributions to data.
- [`combra.angles`]({{< relref "/docs/angles" >}}) — uses `stats_preprocess` + `gaussian_bimodal` for angle histograms.
- [`combra.metrics.convergence_stats`]({{< relref "/docs/metrics" >}}) — drives `kendall_decreasing_p` + `fisher_combine` across whole convergence tables.
