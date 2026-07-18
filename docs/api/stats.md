# combra.stats

The `combra.stats` module exposes the parametric distribution functions used as targets by {doc}`combra.approx <approx>`, plus the histogram preprocessor every distribution fit calls first, plus the inference helpers (Kendall, Fisher) used by the convergence-analysis pipeline.

```python
from combra import stats
```

## Histogram preprocessing

````{py:function} combra.stats.stats_preprocess(array, step) -> tuple[ndarray, ndarray]

Quantize `array` to multiples of `step`, count occurrences via `np.bincount`, and normalise to a probability distribution. This is the first thing every distribution-fit in combra calls.

:param array: 1-D input (angles, beam lengths, etc.).
:type array: array_like
:param step: Bin width in input units.
:type step: float
:returns: **x_bins** (*ndarray[float32]*) – Bin centres (only non-empty bins); and **y_density** (*ndarray[float32]*) – Normalised counts; `sum(y) == 1`.
:rtype: tuple(ndarray, ndarray)

**Example**

```python
>>> import numpy as np
>>> from combra import stats, approx
>>> angles_array = np.array([12, 13, 87, 90, 92, 178, 180])
>>> x, y = stats.stats_preprocess(angles_array, step=5)
>>> (x_g, y_g), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)
```
````

## Distributions

These functions are designed to be passed to `scipy.optimize.curve_fit` by {doc}`combra.approx <approx>`. They return ndarrays — feed them an x grid and parameters, get back y values.

````{py:function} combra.stats.gaussian(x, mu, sigma, amp=1) -> ndarray

Single Gaussian, $amp \cdot \mathcal{N}(x \mid \mu, \sigma)$.

:param x: Evaluation points.
:type x: array_like
:param mu: Mean.
:type mu: float
:param sigma: Standard deviation.
:type sigma: float
:param amp: Multiplicative amplitude. Default: `1`.
:type amp: float, optional
:returns: **y** (*ndarray*) – Function values at `x`.
:rtype: ndarray

**Example**

```python
>>> import numpy as np
>>> from combra import stats
>>> x = np.linspace(0, 360, 200)
>>> y = stats.gaussian(x, mu=180, sigma=30, amp=1.0)
```
````

````{py:function} combra.stats.gaussian_bimodal(x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1) -> ndarray

Sum of two Gaussians. Use {py:func}`combra.approx.bimodal_gauss_approx` to fit it to a histogram.

:param x: Evaluation points.
:type x: array_like
:param mu1: First mean.
:type mu1: float
:param mu2: Second mean.
:type mu2: float
:param sigma1: First standard deviation.
:type sigma1: float
:param sigma2: Second standard deviation.
:type sigma2: float
:param amp1: First amplitude. Default: `1`.
:type amp1: float, optional
:param amp2: Second amplitude. Default: `1`.
:type amp2: float, optional
:returns: **y** (*ndarray*) – Function values at `x`.
:rtype: ndarray

**Example**

```python
>>> import numpy as np
>>> from combra import stats
>>> x = np.linspace(0, 360, 200)
>>> y = stats.gaussian_bimodal(x, mu1=90, mu2=270, sigma1=20, sigma2=25, amp1=1.0, amp2=0.8)
```
````

````{py:function} combra.stats.ellipse(a, b, angle, xc=0, yc=0, num=50) -> tuple[ndarray, ndarray]

Sample `num` points on the ellipse with semi-axes `(a, b)`, rotation `angle` (radians, decreasing → clockwise), and centre `(xc, yc)`. Handy for overlaying MVEE results.

:param a: Semi-major axis.
:type a: float
:param b: Semi-minor axis.
:type b: float
:param angle: Rotation in radians.
:type angle: float
:param xc: Centre x-coordinate. Default: `0`.
:type xc: float, optional
:param yc: Centre y-coordinate. Default: `0`.
:type yc: float, optional
:param num: Number of sample points. Default: `50`.
:type num: int, optional
:returns: **x** (*ndarray*) – Length-`num` 1-D array of the `x` coordinates along the ellipse; and **y** (*ndarray*) – Length-`num` 1-D array of the `y` coordinates along the ellipse.
:rtype: tuple(ndarray, ndarray)

**Example**

```python
>>> import matplotlib.pyplot as plt
>>> from combra import stats
>>> x, y = stats.ellipse(a=20, b=8, angle=0.4, xc=0, yc=0, num=200)
>>> plt.plot(x, y)
>>> plt.gca().set_aspect('equal'); plt.show()
```
````

## Inference

Hypothesis-testing primitives used by {py:func}`combra.metrics.convergence_stats` / {py:func}`combra.metrics.print_convergence_report`. Both are pure and tiny — exposed so notebooks can call them directly on ad-hoc curves.

````{py:function} combra.stats.kendall_decreasing_p(ns, vals) -> float

One-sided Kendall τ p-value for the null "`|vals|` does not decrease as `ns` grows" against the alternative "decreases" (`scipy.stats.kendalltau(..., alternative='less')`). Rank-based, handles ties cleanly.

:param ns: Sample-size axis (typically integer Ns).
:type ns: array_like
:param vals: Metric values at each `N`. Sign is ignored — the test runs on `|vals|`.
:type vals: array_like
:returns: **p** (*float*) – One-sided p-value. NaN when the input has fewer than 3 points or is perfectly flat.
:rtype: float

**Example**

```python
>>> import numpy as np
>>> from combra import stats
>>> ns   = np.array([100, 250, 500, 1000, 2500, 5000])
>>> vals = np.array([0.40, 0.28, 0.21, 0.15, 0.11, 0.09])   # monotonically shrinking
>>> p = stats.kendall_decreasing_p(ns, vals)
>>> print(f'p={p:.4f}  (small p ⇒ |vals| really does decrease with N)')
```
````

````{py:function} combra.stats.fisher_combine(ps) -> tuple[float, int]

Combine independent one-sided p-values via Fisher's method ($\chi^2 = -2 \sum \log p$ with `2k` degrees of freedom). Entries that are NaN, ≤ 0, or > 1 are silently filtered out (only `0 < p ≤ 1` is meaningful for `log p`).

:param ps: Individual p-values.
:type ps: array_like
:returns: **combined_p** (*float*) – Fisher's combined p-value. NaN if no valid input remained after filtering; and **k** (*int*) – Count of p-values that survived filtering and contributed to `combined_p`.
:rtype: tuple(float, int)

**Example**

```python
>>> from combra import stats
>>> per_class_ps = [0.012, 0.041, 0.087, float('nan')]   # nan is silently dropped
>>> combined_p, k = stats.fisher_combine(per_class_ps)
>>> print(f'combined p={combined_p:.4f}   (k={k} of {len(per_class_ps)} curves contributed)')
```
````

## See also

- {doc}`combra.approx <approx>` — fits these distributions to data.
- {doc}`combra.angles <angles>` — uses `stats_preprocess` + `gaussian_bimodal` for angle histograms.
- {py:func}`combra.metrics.convergence_stats` — drives `kendall_decreasing_p` + `fisher_combine` across whole convergence tables.
