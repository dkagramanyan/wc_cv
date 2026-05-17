---
title: "Approx"
weight: 10
---

The `combra.approx` module fits parametric distributions and lines to 1-D data. Two flavours:

- **`*_fit`** — return only the optimised parameters. Useful when you just want the numbers.
- **`*_approx`** — return both the parameters *and* a sampled curve ready for plotting.

```python
from combra import approx
```

## Gaussian fits

### `combra.approx.gaussian_fit`

```python
gaussian_fit(x, y, mu=1, sigma=1, amp=1)
```

Fit a single Gaussian to `(x, y)` via `scipy.optimize.curve_fit`.

**Parameters**

- **x**, **y** (*array_like*) — Histogram bin centres and densities (typically from `combra.stats.stats_preprocess`).
- **mu**, **sigma**, **amp** (*float*, default `1`) — Initial guesses.

**Returns**

- **mu**, **sigma**, **amp** (*float*) — Fitted parameters.

**Examples**

```python
import numpy as np
from combra import stats, approx

samples = np.random.normal(loc=90, scale=15, size=2000)
x, y = stats.stats_preprocess(samples, step=2)
mu, sigma, amp = approx.gaussian_fit(x, y, mu=90, sigma=10, amp=1)
print(f'mu={mu:.2f}  sigma={sigma:.2f}  amp={amp:.4f}')
```

---

### `combra.approx.gaussian_fit_bimodal`

```python
gaussian_fit_bimodal(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)
```

Bimodal Gaussian fit. Bounded: `sigma > 0`, `amp ≥ 0` (prevents the silent sign-flip that older builds produced).

**Parameters**

- **x**, **y** (*array_like*) — Histogram bin centres and densities.
- **mu1**, **mu2** (*float*, default `100`, `240`) — Initial guesses for the two means.
- **sigma1**, **sigma2** (*float*, default `30`) — Initial sigmas.
- **amp1**, **amp2** (*float*, default `1`) — Initial amplitudes.

**Returns**

- **mus** (*list[float, float]*) — Fitted `[mu1, mu2]`.
- **sigmas** (*list[float, float]*) — Fitted `[sigma1, sigma2]`.
- **amps** (*list[float, float]*) — Fitted `[amp1, amp2]`.

**Examples**

```python
import numpy as np
from combra import stats, approx

arr = np.concatenate([np.random.normal(90, 15, 1000),
                      np.random.normal(270, 20, 1500)])
x, y = stats.stats_preprocess(arr, step=2)
mus, sigmas, amps = approx.gaussian_fit_bimodal(x, y)
print(f'mus={mus}  sigmas={sigmas}  amps={amps}')
```

---

### `combra.approx.gaussian_fit_termodal`

```python
gaussian_fit_termodal(x, y, mu1=10, mu2=100, mu3=240,
                     sigma1=10, sigma2=30, sigma3=30,
                     amp1=1, amp2=1, amp3=1)
```

Trimodal Gaussian fit.

**Parameters**

- **x**, **y** (*array_like*) — Histogram bin centres and densities.
- **mu1**, **mu2**, **mu3** (*float*) — Initial means.
- **sigma1**, **sigma2**, **sigma3** (*float*) — Initial sigmas.
- **amp1**, **amp2**, **amp3** (*float*) — Initial amplitudes.

**Returns**

- **mus**, **sigmas**, **amps** (*list[float, float, float]*) — Fitted length-3 lists.

**Examples**

```python
import numpy as np
from combra import stats, approx

arr = np.concatenate([np.random.normal(20, 5, 800),
                      np.random.normal(120, 25, 1500),
                      np.random.normal(260, 20, 1000)])
x, y = stats.stats_preprocess(arr, step=2)
mus, sigmas, amps = approx.gaussian_fit_termodal(x, y)
print(f'mus={mus}')
```

---

### `combra.approx.gauss_approx`

```python
gauss_approx(x, y, mu=1, sigma=1, amp=1, x_lim=None, N=100)
```

Single Gaussian fit + sampled curve.

**Parameters**

- **x**, **y** (*array_like*) — Input histogram.
- **mu**, **sigma**, **amp** (*float*, default `1`) — Initial guesses.
- **x_lim** (*tuple[float, float] or None*, default `None`) — `(x_min, x_max)` for the sampled curve. Defaults to `(x.min(), x.max())`.
- **N** (*int*, default `100`) — Number of sample points on the returned curve.

**Returns**

- **curve** (*tuple[ndarray, ndarray]*) — `(x_gauss, y_gauss)` sampled curve.
- **mu**, **sigma**, **amp** (*float*) — Fitted parameters.

**Examples**

Adapted from `poliamid/data_viz.ipynb` (per-group Gaussian fit on contour-length histograms):

```python
from combra import stats, approx

x_orig, y_orig = stats.stats_preprocess(len_list, step=1)
(x_fit, y_fit), mu, sigma, amp = approx.gauss_approx(
    x_orig, y_orig, mu=3, sigma=3, amp=1, x_lim=[0, 25], N=100,
)
```

---

### `combra.approx.bimodal_gauss_approx`

```python
bimodal_gauss_approx(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)
```

Bimodal Gaussian fit + sampled curve. Used inside `PobeditDataset.generate_angles` to populate `prep_per_step.angles_gauss_*`.

**Parameters**

- **x**, **y** (*array_like*) — Input histogram.
- **mu1**, **mu2**, **sigma1**, **sigma2**, **amp1**, **amp2** (*float*) — Initial guesses.

**Returns**

- **curve** (*tuple[ndarray, ndarray]*) — `(x_gauss, y_gauss)` sampled curve.
- **mus** (*list[float, float]*) — Fitted means.
- **sigmas** (*list[float, float]*) — Fitted sigmas.
- **amps** (*list[float, float]*) — Fitted amplitudes.

**Examples**

```python
import numpy as np
from combra import angles, approx, stats

# Suppose `arr` is the angles array from combra.angles.get_angles
arr = np.concatenate([np.random.normal(90, 20, 1000),
                      np.random.normal(270, 25, 1500)])
x, y = stats.stats_preprocess(arr, step=2)
(x_g, y_g), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)
print(f'mu = {mus},  sigma = {sigmas},  amp = {amps}')
```

---

## Other distributions

### `combra.approx.binomial_approx`

```python
binomial_approx(x, y, n=10, p=0.5, amp=1, x_lim=None, N=100)
```

Binomial fit + sampled curve.

**Parameters**

- **x**, **y** (*array_like*) — Input histogram.
- **n** (*int*, default `10`) — Initial trial count.
- **p** (*float*, default `0.5`) — Initial success probability.
- **amp** (*float*, default `1`) — Initial amplitude.
- **x_lim** (*tuple[float, float] or None*, default `None`) — Range for the sampled curve.
- **N** (*int*, default `100`) — Number of sample points.

**Returns**

- **curve** (*tuple[ndarray, ndarray]*) — `(x_pred, y_pred)`.
- **n** (*int*) — Fitted.
- **p**, **amp** (*float*) — Fitted.

**Examples**

From `poliamid/data_viz.ipynb`:

```python
from combra import stats, approx

x, y = stats.stats_preprocess(len_list, step=1)
(x_fit, y_fit), n_fit, p_fit, amp = approx.binomial_approx(
    x, y, n=25, p=0.2, x_lim=[0, 25], N=100,
)
```

---

### `combra.approx.poisson_approx`

```python
poisson_approx(x, y, lam=1, amp=1, x_lim=None, N=100)
```

Poisson fit + sampled curve.

**Parameters**

- **x**, **y** (*array_like*) — Input histogram.
- **lam** (*float*, default `1`) — Initial rate.
- **amp** (*float*, default `1`) — Initial amplitude.
- **x_lim** (*tuple[float, float] or None*, default `None`) — Range for the sampled curve.
- **N** (*int*, default `100`) — Number of sample points.

**Returns**

- **curve** (*tuple[ndarray, ndarray]*) — `(x_pred, y_pred)`.
- **lam**, **amp** (*float*) — Fitted parameters.

**Examples**

From `poliamid/data_viz.ipynb`:

```python
from combra import stats, approx

x, y = stats.stats_preprocess(len_list, step=1)
(x_fit, y_fit), lam, amp = approx.poisson_approx(x, y, x_lim=[-5, 25], N=100)
```

---

### `combra.approx.exponential_approx`

```python
exponential_approx(x, y, a=1, amp=1, x_lim=None, N=100)
```

Exponential decay `amp * exp(-x / a)` fit + sampled curve.

**Parameters**

- **x**, **y** (*array_like*) — Input histogram.
- **a** (*float*, default `1`) — Initial decay constant.
- **amp** (*float*, default `1`) — Initial amplitude.
- **x_lim** (*tuple[float, float] or None*, default `None`) — Range for the sampled curve.
- **N** (*int*, default `100`) — Number of sample points.

**Returns**

- **curve** (*tuple[ndarray, ndarray]*) — `(x_pred, y_pred)`.
- **a**, **amp** (*float*) — Fitted parameters.

**Examples**

From `poliamid/data_viz.ipynb`:

```python
from combra import stats, approx

x, y = stats.stats_preprocess(len_list, step=1)
(x_fit, y_fit), a, amp = approx.exponential_approx(
    x, y, a=5, amp=1, x_lim=[0, 25], N=100,
)
```

---

## Linear fits

### `combra.approx.linear_approx`

```python
linear_approx(x, y)
```

Least-squares line `y = k*x + b`. Used in `PobeditDataset.generate_beams` to populate `prep.a_*` and `prep.b_*` fit fields.

**Parameters**

- **x**, **y** (*array_like*) — Input series.

**Returns**

- **curve** (*tuple[ndarray, ndarray]*) — `(x_pred, y_pred)` sampled line.
- **k** (*float*) — Slope.
- **b** (*float*) — Intercept.
- **angle_deg** (*float*) — `arctan(k)` in degrees.
- **score** (*float*) — R².

**Examples**

```python
import numpy as np
from combra import approx

x = np.linspace(0, 10, 50)
y = 2.5 * x + 1.0 + np.random.normal(scale=0.5, size=50)
(x_pred, y_pred), k, b, angle_deg, score = approx.linear_approx(x, y)
print(f'k={k:.3f}  b={b:.3f}  angle={angle_deg:.2f}°  R²={score:.3f}')
```

---

### `combra.approx.polyfit_deg1`

```python
polyfit_deg1(x, y, i0, i1)
```

Fast numba-compiled line fit on the slice `[i0:i1]`. Use this in tight loops where `linear_approx`'s scipy overhead matters.

**Parameters**

- **x**, **y** (*ndarray*) — Input series.
- **i0**, **i1** (*int*) — Slice bounds.

**Returns**

- **a** (*float*) — Slope.
- **b** (*float*) — Intercept.
- **r_squared** (*float*) — Goodness of fit.

**Examples**

```python
import numpy as np
from combra import approx

x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(scale=0.5, size=100)
k, b, r2 = approx.polyfit_deg1(x, y, i0=10, i1=90)
print(f'slope={k:.3f}  intercept={b:.3f}  R²={r2:.3f}')
```

---

## Plateau / asymptote fits

### `combra.approx.fit_plateau`

```python
fit_plateau(ns, vals)
```

Fit `|m|(N) = a + b · N^(-1/2)` with `a, b ≥ 0`. The asymptote `a` is the irreducible `|m|` as `N → ∞` (e.g. a generator's bias floor); `b` captures the Monte-Carlo sampling-noise term (theoretical `N^(-1/2)` decay for Wasserstein / Gaussian-fit moment errors).

Used by `combra.metrics.convergence_stats` to estimate per-curve plateaus and the standard error around them.

**Parameters**

- **ns** (*array_like[int]*) — Sample sizes (N values along the convergence curve).
- **vals** (*array_like[float]*) — Metric values at each `N`. Sign is ignored; the fit is on `|vals|`.

**Returns**

- **a_hat** (*float*) — Plateau (irreducible `|m|` at infinite N).
- **a_se** (*float*) — Standard error on `a_hat` from the covariance matrix. NaN if the fit fails or is degenerate.
- **b_hat** (*float*) — Sampling-noise coefficient.

NaN for all three when the fit fails, when fewer than 3 points are supplied, or when all `vals` are identical.

**Examples**

Driven inside `combra.metrics.convergence_stats` over a W-dist curve. Standalone:

```python
import numpy as np
from combra import approx

ns = np.array([100, 250, 500, 1000, 2500, 5000, 10000])
# True asymptote a=0.05; sampling-noise b=0.30; small additive jitter
vals = 0.05 + 0.30 / np.sqrt(ns) + np.random.normal(scale=0.005, size=len(ns))
a, a_se, b = approx.fit_plateau(ns, vals)
print(f'plateau a={a:.4f} ± {a_se:.4f}    b={b:.4f}')
```

---

## See also

- [`combra.stats`]({{< relref "/docs/stats" >}}) — the distribution functions these fits target.
- [`combra.angles`]({{< relref "/docs/angles" >}}) — uses `bimodal_gauss_approx` for angle histograms.
- [`combra.mvee`]({{< relref "/docs/mvee" >}}) — uses `linear_approx` for beam-length log-density fits.
- [`combra.metrics.convergence_stats`]({{< relref "/docs/metrics" >}}) — uses `fit_plateau` to estimate per-curve bias floors.
