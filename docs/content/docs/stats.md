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

Older variant that returns the snapped array, the unique bin centres, the raw counts, and a per-bin index dictionary. Kept for backwards compatibility with pre-parquet code paths; prefer `stats_preprocess` in new code.

**Parameters**

- **array** (*array_like*) — 1-D input.
- **step** (*int*) — Bin width. Must be non-zero.

**Returns**

- **new_array** (*ndarray*) — Original entries snapped to multiples of `step`.
- **bin_centres** (*ndarray*) — Sorted unique snapped values.
- **counts** (*ndarray*) — Per-bin counts (non-normalised).
- **index_map** (*dict*) — `{bin_centre: [original_values_in_bin…]}`.

**Examples**

```python
import numpy as np
from combra import stats

arr = np.array([1.0, 1.2, 5.1, 5.3, 9.8, 10.2])
new_arr, centres, counts, idx = stats.legacy_stats_preprocess(arr, step=1)
print(f'centres={centres}  counts={counts}')
```

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

**Examples**

```python
import numpy as np
from combra import stats

x = np.linspace(0, 360, 200)
y = stats.gaussian(x, mu=180, sigma=30, amp=1.0)
```

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

**Examples**

```python
import numpy as np
from combra import stats

x = np.linspace(0, 360, 200)
y = stats.gaussian_bimodal(x, mu1=90, mu2=270, sigma1=20, sigma2=25, amp1=1.0, amp2=0.8)
```

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

**Examples**

```python
import numpy as np
from combra import stats

x = np.linspace(0, 360, 200)
y = stats.gaussian_termodal(
    x, mu1=30, mu2=180, mu3=300,
    sigma1=15, sigma2=25, sigma3=20,
    amp1=0.6, amp2=1.0, amp3=0.5,
)
```

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

- **x**, **y** (*tuple[ndarray, ndarray]*) — Two length-`num` 1-D arrays of the `x` and `y` coordinates along the ellipse.

**Examples**

```python
import matplotlib.pyplot as plt
from combra import stats

x, y = stats.ellipse(a=20, b=8, angle=0.4, xc=0, yc=0, num=200)
plt.plot(x, y)
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

**Examples**

```python
import numpy as np
from combra import stats

ns   = np.array([100, 250, 500, 1000, 2500, 5000])
vals = np.array([0.40, 0.28, 0.21, 0.15, 0.11, 0.09])   # monotonically shrinking
p = stats.kendall_decreasing_p(ns, vals)
print(f'p={p:.4f}  (small p ⇒ |vals| really does decrease with N)')
```

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

**Examples**

```python
from combra import stats

per_class_ps = [0.012, 0.041, 0.087, float('nan')]   # nan is silently dropped
combined_p, k = stats.fisher_combine(per_class_ps)
print(f'combined p={combined_p:.4f}   (k={k} of {len(per_class_ps)} curves contributed)')
```

---

## See also

- [`combra.approx`]({{< relref "/docs/approx" >}}) — fits these distributions to data.
- [`combra.angles`]({{< relref "/docs/angles" >}}) — uses `stats_preprocess` + `gaussian_bimodal` for angle histograms.
- [`combra.metrics.convergence_stats`]({{< relref "/docs/metrics" >}}) — drives `kendall_decreasing_p` + `fisher_combine` across whole convergence tables.
