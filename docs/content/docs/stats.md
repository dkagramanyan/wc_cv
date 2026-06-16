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
combra.stats.stats_preprocess(array, step) → tuple[ndarray, ndarray]
```

Quantize `array` to multiples of `step`, count occurrences via `np.bincount`, and normalise to a probability distribution. This is the first thing every distribution-fit in combra calls.

**Parameters**

- **array** (*array_like*) – 1-D input (angles, beam lengths, etc.).
- **step** (*float*) – Bin width in input units.

**Returns**

- **x_bins** (*ndarray[float32]*) – Bin centres (only non-empty bins).
- **y_density** (*ndarray[float32]*) – Normalised counts; `sum(y) == 1`.

**Return type**

*tuple(ndarray, ndarray)*

**Example**

```python
>>> import numpy as np
>>> from combra import stats, approx
>>> angles_array = np.array([12, 13, 87, 90, 92, 178, 180])
>>> x, y = stats.stats_preprocess(angles_array, step=5)
>>> (x_g, y_g), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)
```

---

### `combra.stats.legacy_stats_preprocess`

```python
combra.stats.legacy_stats_preprocess(array, step) → tuple[ndarray, ndarray, ndarray, dict]
```

Older variant that returns the snapped array, the unique bin centres, the raw counts, and a per-bin index dictionary. Kept for backwards compatibility with pre-parquet code paths; prefer `stats_preprocess` in new code.

**Parameters**

- **array** (*array_like*) – 1-D input.
- **step** (*int*) – Bin width. Must be non-zero.

**Returns**

- **new_array** (*ndarray*) – Original entries snapped to multiples of `step`.
- **bin_centres** (*ndarray*) – Sorted unique snapped values.
- **counts** (*ndarray*) – Per-bin counts (non-normalised).
- **index_map** (*dict*) – `{bin_centre: [original_values_in_bin…]}`.

**Return type**

*tuple(ndarray, ndarray, ndarray, dict)*

**Example**

```python
>>> import numpy as np
>>> from combra import stats
>>> arr = np.array([1.0, 1.2, 5.1, 5.3, 9.8, 10.2])
>>> new_arr, centres, counts, idx = stats.legacy_stats_preprocess(arr, step=1)
>>> print(f'centres={centres}  counts={counts}')
```

---

## Distributions

These functions are designed to be passed to `scipy.optimize.curve_fit` by `combra.approx`. They return ndarrays — feed them an x grid and parameters, get back y values.

### `combra.stats.gaussian`

```python
combra.stats.gaussian(x, mu, sigma, amp=1) → ndarray
```

Single Gaussian, `amp * 𝒩(x | mu, sigma)`.

**Parameters**

- **x** (*array_like*) – Evaluation points.
- **mu** (*float*) – Mean.
- **sigma** (*float*) – Standard deviation.
- **amp** (*float, optional*) – Multiplicative amplitude. Default: `1`.

**Returns**

- **y** (*ndarray*) – Function values at `x`.

**Return type**

*ndarray*

**Example**

```python
>>> import numpy as np
>>> from combra import stats
>>> x = np.linspace(0, 360, 200)
>>> y = stats.gaussian(x, mu=180, sigma=30, amp=1.0)
```

---

### `combra.stats.gaussian_bimodal`

```python
combra.stats.gaussian_bimodal(x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1) → ndarray
```

Sum of two Gaussians. Use `combra.approx.bimodal_gauss_approx` to fit it to a histogram.

**Parameters**

- **x** (*array_like*) – Evaluation points.
- **mu1** (*float*) – First mean.
- **mu2** (*float*) – Second mean.
- **sigma1** (*float*) – First standard deviation.
- **sigma2** (*float*) – Second standard deviation.
- **amp1** (*float, optional*) – First amplitude. Default: `1`.
- **amp2** (*float, optional*) – Second amplitude. Default: `1`.

**Returns**

- **y** (*ndarray*) – Function values at `x`.

**Return type**

*ndarray*

**Example**

```python
>>> import numpy as np
>>> from combra import stats
>>> x = np.linspace(0, 360, 200)
>>> y = stats.gaussian_bimodal(x, mu1=90, mu2=270, sigma1=20, sigma2=25, amp1=1.0, amp2=0.8)
```

---

### `combra.stats.gaussian_termodal`

```python
combra.stats.gaussian_termodal(x, mu1, mu2, mu3, sigma1, sigma2, sigma3,
                               amp1=1, amp2=1, amp3=1) → ndarray
```

Sum of three Gaussians.

**Parameters**

- **x** (*array_like*) – Evaluation points.
- **mu1** (*float*) – First mean.
- **mu2** (*float*) – Second mean.
- **mu3** (*float*) – Third mean.
- **sigma1** (*float*) – First standard deviation.
- **sigma2** (*float*) – Second standard deviation.
- **sigma3** (*float*) – Third standard deviation.
- **amp1** (*float, optional*) – First amplitude. Default: `1`.
- **amp2** (*float, optional*) – Second amplitude. Default: `1`.
- **amp3** (*float, optional*) – Third amplitude. Default: `1`.

**Returns**

- **y** (*ndarray*) – Function values at `x`.

**Return type**

*ndarray*

**Example**

```python
>>> import numpy as np
>>> from combra import stats
>>> x = np.linspace(0, 360, 200)
>>> y = stats.gaussian_termodal(
...     x, mu1=30, mu2=180, mu3=300,
...     sigma1=15, sigma2=25, sigma3=20,
...     amp1=0.6, amp2=1.0, amp3=0.5,
... )
```

---

### `combra.stats.ellipse`

```python
combra.stats.ellipse(a, b, angle, xc=0, yc=0, num=50) → tuple[ndarray, ndarray]
```

Sample `num` points on the ellipse with semi-axes `(a, b)`, rotation `angle` (radians, decreasing → clockwise), and centre `(xc, yc)`. Handy for overlaying MVEE results.

**Parameters**

- **a** (*float*) – Semi-major axis.
- **b** (*float*) – Semi-minor axis.
- **angle** (*float*) – Rotation in radians.
- **xc** (*float, optional*) – Centre x-coordinate. Default: `0`.
- **yc** (*float, optional*) – Centre y-coordinate. Default: `0`.
- **num** (*int, optional*) – Number of sample points. Default: `50`.

**Returns**

- **x** (*ndarray*) – Length-`num` 1-D array of the `x` coordinates along the ellipse.
- **y** (*ndarray*) – Length-`num` 1-D array of the `y` coordinates along the ellipse.

**Return type**

*tuple(ndarray, ndarray)*

**Example**

```python
>>> import matplotlib.pyplot as plt
>>> from combra import stats
>>> x, y = stats.ellipse(a=20, b=8, angle=0.4, xc=0, yc=0, num=200)
>>> plt.plot(x, y)
>>> plt.gca().set_aspect('equal'); plt.show()
```

---

## Inference

Hypothesis-testing primitives used by `combra.metrics.convergence_stats` / `print_convergence_report`. Both are pure and tiny — exposed so notebooks can call them directly on ad-hoc curves.

### `combra.stats.kendall_decreasing_p`

```python
combra.stats.kendall_decreasing_p(ns, vals) → float
```

One-sided Kendall τ p-value for the null "`|vals|` does not decrease as `ns` grows" against the alternative "decreases" (`scipy.stats.kendalltau(..., alternative='less')`). Rank-based, handles ties cleanly.

**Parameters**

- **ns** (*array_like*) – Sample-size axis (typically integer Ns).
- **vals** (*array_like*) – Metric values at each `N`. Sign is ignored — the test runs on `|vals|`.

**Returns**

- **p** (*float*) – One-sided p-value. NaN when the input has fewer than 3 points or is perfectly flat.

**Return type**

*float*

**Example**

```python
>>> import numpy as np
>>> from combra import stats
>>> ns   = np.array([100, 250, 500, 1000, 2500, 5000])
>>> vals = np.array([0.40, 0.28, 0.21, 0.15, 0.11, 0.09])   # monotonically shrinking
>>> p = stats.kendall_decreasing_p(ns, vals)
>>> print(f'p={p:.4f}  (small p ⇒ |vals| really does decrease with N)')
```

---

### `combra.stats.fisher_combine`

```python
combra.stats.fisher_combine(ps) → tuple[float, int]
```

Combine independent one-sided p-values via Fisher's method (`χ²` = `-2 · Σ log p` with `2k` degrees of freedom). Entries that are NaN, ≤ 0, or > 1 are silently filtered out (only `0 < p ≤ 1` is meaningful for `log p`).

**Parameters**

- **ps** (*array_like*) – Individual p-values.

**Returns**

- **combined_p** (*float*) – Fisher's combined p-value. NaN if no valid input remained after filtering.
- **k** (*int*) – Count of p-values that survived filtering and contributed to `combined_p`.

**Return type**

*tuple(float, int)*

**Example**

```python
>>> from combra import stats
>>> per_class_ps = [0.012, 0.041, 0.087, float('nan')]   # nan is silently dropped
>>> combined_p, k = stats.fisher_combine(per_class_ps)
>>> print(f'combined p={combined_p:.4f}   (k={k} of {len(per_class_ps)} curves contributed)')
```

---

## See also

- [`combra.approx`]({{< relref "/docs/approx" >}}) — fits these distributions to data.
- [`combra.angles`]({{< relref "/docs/angles" >}}) — uses `stats_preprocess` + `gaussian_bimodal` for angle histograms.
- [`combra.metrics.convergence_stats`]({{< relref "/docs/metrics" >}}) — drives `kendall_decreasing_p` + `fisher_combine` across whole convergence tables.
