# combra.approx

The `combra.approx` module fits parametric distributions and lines to 1-D data. Two flavours:

- **`*_fit`** — return only the optimised parameters. Useful when you just want the numbers.
- **`*_approx`** — return both the parameters *and* a sampled curve ready for plotting.

```python
from combra import approx
```

## Gaussian fits

````{py:function} combra.approx.gaussian_fit(x, y, mu=1, sigma=1, amp=1) -> tuple[float, float, float]

Fit a single Gaussian to `(x, y)` via `scipy.optimize.curve_fit`.

:param x: Histogram bin centres (typically from `combra.stats.stats_preprocess`).
:type x: array_like
:param y: Histogram densities (typically from `combra.stats.stats_preprocess`).
:type y: array_like
:param mu: Initial guess for the mean. Default: `1`.
:type mu: float, optional
:param sigma: Initial guess for the standard deviation. Default: `1`.
:type sigma: float, optional
:param amp: Initial guess for the amplitude. Default: `1`.
:type amp: float, optional
:returns: **mu** (*float*) – Fitted mean; and **sigma** (*float*) – Fitted standard deviation; and **amp** (*float*) – Fitted amplitude.
:rtype: tuple(float, float, float)

**Example**

```python
>>> import numpy as np
>>> from combra import stats, approx
>>> samples = np.random.normal(loc=90, scale=15, size=2000)
>>> x, y = stats.stats_preprocess(samples, step=2)
>>> mu, sigma, amp = approx.gaussian_fit(x, y, mu=90, sigma=10, amp=1)
>>> print(f'mu={mu:.2f}  sigma={sigma:.2f}  amp={amp:.4f}')
```
````

````{py:function} combra.approx.gaussian_fit_bimodal(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1) -> tuple[list[float], list[float], list[float]]

Bimodal Gaussian fit. Bounded: `sigma > 0`, `amp ≥ 0` (prevents the silent sign-flip that older builds produced).

:param x: Histogram bin centres.
:type x: array_like
:param y: Histogram densities.
:type y: array_like
:param mu1: Initial guess for the first mean. Default: `100`.
:type mu1: float, optional
:param mu2: Initial guess for the second mean. Default: `240`.
:type mu2: float, optional
:param sigma1: Initial guess for the first sigma. Default: `30`.
:type sigma1: float, optional
:param sigma2: Initial guess for the second sigma. Default: `30`.
:type sigma2: float, optional
:param amp1: Initial guess for the first amplitude. Default: `1`.
:type amp1: float, optional
:param amp2: Initial guess for the second amplitude. Default: `1`.
:type amp2: float, optional
:returns: **mus** (*list[float, float]*) – Fitted `[mu1, mu2]`; and **sigmas** (*list[float, float]*) – Fitted `[sigma1, sigma2]`; and **amps** (*list[float, float]*) – Fitted `[amp1, amp2]`.
:rtype: tuple(list[float], list[float], list[float])

**Example**

```python
>>> import numpy as np
>>> from combra import stats, approx
>>> arr = np.concatenate([np.random.normal(90, 15, 1000),
...                       np.random.normal(270, 20, 1500)])
>>> x, y = stats.stats_preprocess(arr, step=2)
>>> mus, sigmas, amps = approx.gaussian_fit_bimodal(x, y)
>>> print(f'mus={mus}  sigmas={sigmas}  amps={amps}')
```
````

````{py:function} combra.approx.gaussian_fit_termodal(x, y, mu1=10, mu2=100, mu3=240, sigma1=10, sigma2=30, sigma3=30, amp1=1, amp2=1, amp3=1) -> tuple[list[float], list[float], list[float]]

Trimodal Gaussian fit.

:param x: Histogram bin centres.
:type x: array_like
:param y: Histogram densities.
:type y: array_like
:param mu1: Initial guess for the first mean. Default: `10`.
:type mu1: float, optional
:param mu2: Initial guess for the second mean. Default: `100`.
:type mu2: float, optional
:param mu3: Initial guess for the third mean. Default: `240`.
:type mu3: float, optional
:param sigma1: Initial guess for the first sigma. Default: `10`.
:type sigma1: float, optional
:param sigma2: Initial guess for the second sigma. Default: `30`.
:type sigma2: float, optional
:param sigma3: Initial guess for the third sigma. Default: `30`.
:type sigma3: float, optional
:param amp1: Initial guess for the first amplitude. Default: `1`.
:type amp1: float, optional
:param amp2: Initial guess for the second amplitude. Default: `1`.
:type amp2: float, optional
:param amp3: Initial guess for the third amplitude. Default: `1`.
:type amp3: float, optional
:returns: **mus** (*list[float, float, float]*) – Fitted length-3 means; and **sigmas** (*list[float, float, float]*) – Fitted length-3 sigmas; and **amps** (*list[float, float, float]*) – Fitted length-3 amplitudes.
:rtype: tuple(list[float], list[float], list[float])

**Example**

```python
>>> import numpy as np
>>> from combra import stats, approx
>>> arr = np.concatenate([np.random.normal(20, 5, 800),
...                       np.random.normal(120, 25, 1500),
...                       np.random.normal(260, 20, 1000)])
>>> x, y = stats.stats_preprocess(arr, step=2)
>>> mus, sigmas, amps = approx.gaussian_fit_termodal(x, y)
>>> print(f'mus={mus}')
```
````

````{py:function} combra.approx.gauss_approx(x, y, mu=1, sigma=1, amp=1, x_lim=None, N=100) -> tuple[tuple[ndarray, ndarray], float, float, float]

Single Gaussian fit + sampled curve.

:param x: Input histogram bin centres.
:type x: array_like
:param y: Input histogram densities.
:type y: array_like
:param mu: Initial guess for the mean. Default: `1`.
:type mu: float, optional
:param sigma: Initial guess for the standard deviation. Default: `1`.
:type sigma: float, optional
:param amp: Initial guess for the amplitude. Default: `1`.
:type amp: float, optional
:param x_lim: `(x_min, x_max)` for the sampled curve. Defaults to `(x.min(), x.max())`. Default: `None`.
:type x_lim: tuple[float, float] or None, optional
:param N: Number of sample points on the returned curve. Default: `100`.
:type N: int, optional
:returns: **curve** (*tuple[ndarray, ndarray]*) – `(x_gauss, y_gauss)` sampled curve; and **mu** (*float*) – Fitted mean; and **sigma** (*float*) – Fitted standard deviation; and **amp** (*float*) – Fitted amplitude.
:rtype: tuple(tuple(ndarray, ndarray), float, float, float)

**Example**

Adapted from `poliamid/data_viz.ipynb` (per-group Gaussian fit on contour-length histograms):

```python
>>> from combra import stats, approx
>>> x_orig, y_orig = stats.stats_preprocess(len_list, step=1)
>>> (x_fit, y_fit), mu, sigma, amp = approx.gauss_approx(
...     x_orig, y_orig, mu=3, sigma=3, amp=1, x_lim=[0, 25], N=100,
... )
```
````

````{py:function} combra.approx.bimodal_gauss_approx(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1) -> tuple[tuple[ndarray, ndarray], list[float], list[float], list[float]]

Bimodal Gaussian fit + sampled curve. Used inside {py:meth}`combra.data.PobeditDataset.generate_angles` to populate `prep_per_step.angles_gauss_*`.

```{versionchanged} 0.4
Returns a {py:class}`~combra.approx.BimodalGaussFit` named tuple
``(curve, mus, sigmas, amps)``. Field order is unchanged, so
``(x, y), mus, sigmas, amps = bimodal_gauss_approx(...)`` still unpacks; the
fields are also reachable as ``.mus`` / ``.sigmas`` / ``.amps``.
```

:param x: Input histogram bin centres.
:type x: array_like
:param y: Input histogram densities.
:type y: array_like
:param mu1: Initial guess for the first mean. Default: `100`.
:type mu1: float, optional
:param mu2: Initial guess for the second mean. Default: `240`.
:type mu2: float, optional
:param sigma1: Initial guess for the first sigma. Default: `30`.
:type sigma1: float, optional
:param sigma2: Initial guess for the second sigma. Default: `30`.
:type sigma2: float, optional
:param amp1: Initial guess for the first amplitude. Default: `1`.
:type amp1: float, optional
:param amp2: Initial guess for the second amplitude. Default: `1`.
:type amp2: float, optional
:returns: **result** – a {py:class}`~combra.approx.BimodalGaussFit` ``(curve, mus, sigmas, amps)``: the `(x_gauss, y_gauss)` sampled curve, and the fitted per-mode means, sigmas and amplitudes.
:rtype: BimodalGaussFit

**Example**

```python
>>> import numpy as np
>>> from combra import angles, approx, stats
>>> # Suppose `arr` is the angles array from combra.angles.vertex_angles
>>> arr = np.concatenate([np.random.normal(90, 20, 1000),
...                       np.random.normal(270, 25, 1500)])
>>> x, y = stats.stats_preprocess(arr, step=2)
>>> (x_g, y_g), mus, sigmas, amps = approx.bimodal_gauss_approx(x, y)
>>> print(f'mu = {mus},  sigma = {sigmas},  amp = {amps}')
```
````

## Other distributions

````{py:function} combra.approx.binomial_approx(x, y, n=10, p=0.5, amp=1, x_lim=None, N=100) -> tuple[tuple[ndarray, ndarray], float, float, float]

Binomial fit + sampled curve.

:param x: Input histogram bin centres.
:type x: array_like
:param y: Input histogram densities.
:type y: array_like
:param n: Initial trial count. Default: `10`.
:type n: int, optional
:param p: Initial success probability. Default: `0.5`.
:type p: float, optional
:param amp: Initial amplitude. Default: `1`.
:type amp: float, optional
:param x_lim: Range for the sampled curve. Default: `None`.
:type x_lim: tuple[float, float] or None, optional
:param N: Number of sample points. Default: `100`.
:type N: int, optional
:returns: **curve** (*tuple[ndarray, ndarray]*) – `(x_pred, y_pred)`; and **n** (*float*) – Fitted trial count; and **p** (*float*) – Fitted success probability; and **amp** (*float*) – Fitted amplitude.
:rtype: tuple(tuple(ndarray, ndarray), float, float, float)

**Example**

From `poliamid/data_viz.ipynb`:

```python
>>> from combra import stats, approx
>>> x, y = stats.stats_preprocess(len_list, step=1)
>>> (x_fit, y_fit), n_fit, p_fit, amp = approx.binomial_approx(
...     x, y, n=25, p=0.2, x_lim=[0, 25], N=100,
... )
```
````

````{py:function} combra.approx.poisson_approx(x, y, lam=1, amp=1, x_lim=None, N=100) -> tuple[tuple[ndarray, ndarray], float, float]

Poisson fit + sampled curve.

:param x: Input histogram bin centres.
:type x: array_like
:param y: Input histogram densities.
:type y: array_like
:param lam: Initial rate. Default: `1`.
:type lam: float, optional
:param amp: Initial amplitude. Default: `1`.
:type amp: float, optional
:param x_lim: Range for the sampled curve. Default: `None`.
:type x_lim: tuple[float, float] or None, optional
:param N: Number of sample points. Default: `100`.
:type N: int, optional
:returns: **curve** (*tuple[ndarray, ndarray]*) – `(x_pred, y_pred)`; and **lam** (*float*) – Fitted rate; and **amp** (*float*) – Fitted amplitude.
:rtype: tuple(tuple(ndarray, ndarray), float, float)

**Example**

From `poliamid/data_viz.ipynb`:

```python
>>> from combra import stats, approx
>>> x, y = stats.stats_preprocess(len_list, step=1)
>>> (x_fit, y_fit), lam, amp = approx.poisson_approx(x, y, x_lim=[-5, 25], N=100)
```
````

````{py:function} combra.approx.exponential_approx(x, y, a=1, amp=1, x_lim=None, N=100) -> tuple[tuple[ndarray, ndarray], float, float]

Exponential decay $amp \cdot e^{-x/a}$ fit + sampled curve.

:param x: Input histogram bin centres.
:type x: array_like
:param y: Input histogram densities.
:type y: array_like
:param a: Initial decay constant. Default: `1`.
:type a: float, optional
:param amp: Initial amplitude. Default: `1`.
:type amp: float, optional
:param x_lim: Range for the sampled curve. Default: `None`.
:type x_lim: tuple[float, float] or None, optional
:param N: Number of sample points. Default: `100`.
:type N: int, optional
:returns: **curve** (*tuple[ndarray, ndarray]*) – `(x_pred, y_pred)`; and **a** (*float*) – Fitted decay constant; and **amp** (*float*) – Fitted amplitude.
:rtype: tuple(tuple(ndarray, ndarray), float, float)

**Example**

From `poliamid/data_viz.ipynb`:

```python
>>> from combra import stats, approx
>>> x, y = stats.stats_preprocess(len_list, step=1)
>>> (x_fit, y_fit), a, amp = approx.exponential_approx(
...     x, y, a=5, amp=1, x_lim=[0, 25], N=100,
... )
```
````

## Linear fits

````{py:function} combra.approx.linear_approx(x, y) -> tuple[tuple[ndarray, ndarray], float, float, float, float]

Least-squares line $y = kx + b$. Used in {py:meth}`combra.data.PobeditDataset.generate_beams` to populate `prep.a_*` and `prep.b_*` fit fields.

```{versionchanged} 0.4
Returns a {py:class}`~combra.approx.LinearFit` named tuple
``(curve, slope, intercept, angle_deg, r2)``. Field order is unchanged, so
``(x_pred, y_pred), k, b, angle, score = linear_approx(...)`` still unpacks; the
fields are also reachable as ``.slope`` / ``.intercept`` / ``.angle_deg`` / ``.r2``.
```

:param x: Input series.
:type x: array_like
:param y: Input series.
:type y: array_like
:returns: **result** – a {py:class}`~combra.approx.LinearFit` ``(curve, slope, intercept, angle_deg, r2)``: the `(x_pred, y_pred)` sampled line, the slope, the intercept, `arctan(slope)` in degrees, and the R².
:rtype: LinearFit

**Example**

```python
>>> import numpy as np
>>> from combra import approx
>>> x = np.linspace(0, 10, 50)
>>> y = 2.5 * x + 1.0 + np.random.normal(scale=0.5, size=50)
>>> fit = approx.linear_approx(x, y)
>>> print(f'k={fit.slope:.3f}  b={fit.intercept:.3f}  angle={fit.angle_deg:.2f}°  R²={fit.r2:.3f}')
>>> (x_pred, y_pred), k, b, angle_deg, score = fit  # positional unpacking still works
```
````

````{py:function} combra.approx.polyfit_deg1(x, y, i0, i1) -> tuple[float, float, float]

Fast numba-compiled line fit on the slice `[i0:i1]`. Use this in tight loops where `linear_approx`'s scipy overhead matters.

:param x: Input series.
:type x: ndarray
:param y: Input series.
:type y: ndarray
:param i0: Slice start bound.
:type i0: int
:param i1: Slice end bound.
:type i1: int
:returns: **a** (*float*) – Slope; and **b** (*float*) – Intercept; and **r_squared** (*float*) – Goodness of fit.
:rtype: tuple(float, float, float)

**Example**

```python
>>> import numpy as np
>>> from combra import approx
>>> x = np.linspace(0, 10, 100)
>>> y = 2.0 * x + 1.0 + np.random.normal(scale=0.5, size=100)
>>> k, b, r2 = approx.polyfit_deg1(x, y, i0=10, i1=90)
>>> print(f'slope={k:.3f}  intercept={b:.3f}  R²={r2:.3f}')
```
````

## Plateau / asymptote fits

````{py:function} combra.approx.fit_plateau(ns, vals) -> tuple[float, float, float]

Fit $|m|(N) = a + b \cdot N^{-1/2}$ with `a, b ≥ 0`. The asymptote `a` is the irreducible `|m|` as `N → ∞` (e.g. a generator's bias floor); `b` captures the Monte-Carlo sampling-noise term (theoretical `N^(-1/2)` decay for Wasserstein / Gaussian-fit moment errors).

Used by {py:func}`combra.metrics.convergence_stats` to estimate per-curve plateaus and the standard error around them.

:param ns: Sample sizes (N values along the convergence curve).
:type ns: array_like[int]
:param vals: Metric values at each `N`. Sign is ignored; the fit is on `|vals|`.
:type vals: array_like[float]
:returns: **a_hat** (*float*) – Plateau (irreducible `|m|` at infinite N); and **a_se** (*float*) – Standard error on `a_hat` from the covariance matrix. NaN if the fit fails or is degenerate; and **b_hat** (*float*) – Sampling-noise coefficient. NaN for all three when the fit fails, when fewer than 3 points are supplied, or when all `vals` are identical.
:rtype: tuple(float, float, float)

**Example**

Driven inside {py:func}`combra.metrics.convergence_stats` over a W-dist curve. Standalone:

```python
>>> import numpy as np
>>> from combra import approx
>>> ns = np.array([100, 250, 500, 1000, 2500, 5000, 10000])
>>> # True asymptote a=0.05; sampling-noise b=0.30; small additive jitter
>>> vals = 0.05 + 0.30 / np.sqrt(ns) + np.random.normal(scale=0.005, size=len(ns))
>>> a, a_se, b = approx.fit_plateau(ns, vals)
>>> print(f'plateau a={a:.4f} ± {a_se:.4f}    b={b:.4f}')
```
````

## Result types

The fit functions return SciPy-style named tuples (cf. `scipy.stats.linregress`),
so results carry attribute names while staying unpacking-compatible with the
historical plain tuples.

````{py:class} combra.approx.LinearFit

Result of {py:func}`~combra.approx.linear_approx`.

:param curve: `(x_pred, y_pred)` of the fitted line sampled across the data span.
:type curve: tuple[ndarray, ndarray]
:param slope: Line slope `k`.
:type slope: float
:param intercept: Line intercept `b` (value at `x = 0`).
:type intercept: float
:param angle_deg: Slope as an angle in degrees, `atan(k)`.
:type angle_deg: float
:param r2: Coefficient of determination on the input points.
:type r2: float
````

````{py:class} combra.approx.BimodalGaussFit

Result of {py:func}`~combra.approx.bimodal_gauss_approx`.

:param curve: `(x, y)` of the fitted bimodal-Gaussian density.
:type curve: tuple[ndarray, ndarray]
:param mus: The two per-mode means.
:type mus: list[float]
:param sigmas: The two per-mode standard deviations.
:type sigmas: list[float]
:param amps: The two per-mode amplitudes.
:type amps: list[float]
````

## See also

- {doc}`combra.stats <stats>` — the distribution functions these fits target.
- {doc}`combra.angles <angles>` — uses `bimodal_gauss_approx` for angle histograms.
- {doc}`combra.mvee <mvee>` — uses `linear_approx` for beam-length log-density fits.
- {py:func}`combra.metrics.convergence_stats` — uses `fit_plateau` to estimate per-curve bias floors.
