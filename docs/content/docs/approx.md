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

| name | type | default | description |
| --- | --- | --- | --- |
| `x`, `y` | `array_like` | — | Histogram bin centres and densities (typically from `combra.stats.stats_preprocess`). |
| `mu`, `sigma`, `amp` | `float` | `1, 1, 1` | Initial guesses. |

**Returns** `(mu, sigma, amp)` — fitted parameters.

---

### `combra.approx.gaussian_fit_bimodal`

```python
gaussian_fit_bimodal(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)
```

Bimodal Gaussian fit. Bounded: `sigma > 0`, `amp ≥ 0` (prevents the silent sign-flip that older builds produced).

**Returns** `(mus, sigmas, amps)` — each is a length-2 list `[component1, component2]`.

---

### `combra.approx.gaussian_fit_termodal`

```python
gaussian_fit_termodal(x, y, mu1=10, mu2=100, mu3=240, sigma1=10, sigma2=30, sigma3=30, amp1=1, amp2=1, amp3=1)
```

Trimodal Gaussian fit. Returns `(mus, sigmas, amps)` as length-3 lists.

---

### `combra.approx.gauss_approx`

```python
gauss_approx(x, y, mu=1, sigma=1, amp=1, x_lim=None, N=100)
```

Single Gaussian fit + sampled curve.

**Parameters**

| name | type | default | description |
| --- | --- | --- | --- |
| `x`, `y` | `array_like` | — | Input histogram. |
| `mu`, `sigma`, `amp` | `float` | — | Initial guesses. |
| `x_lim` | `tuple[float, float] \| None` | `None` | `(x_min, x_max)` for the sampled curve. Defaults to `(x.min(), x.max())`. |
| `N` | `int` | `100` | Number of sample points on the returned curve. |

**Returns** `((x_gauss, y_gauss), mu, sigma, amp)`.

---

### `combra.approx.bimodal_gauss_approx`

```python
bimodal_gauss_approx(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)
```

Bimodal Gaussian fit + sampled curve. Used inside `PobeditDataset.generate_angles` to populate `prep_per_step.angles_gauss_*`.

**Returns** `((x_gauss, y_gauss), mus, sigmas, amps)`.

**Example**

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

**Returns** `((x_pred, y_pred), n, p, amp)`.

---

### `combra.approx.poisson_approx`

```python
poisson_approx(x, y, lam=1, amp=1, x_lim=None, N=100)
```

Poisson fit + sampled curve.

**Returns** `((x_pred, y_pred), lam, amp)`.

---

### `combra.approx.exponential_approx`

```python
exponential_approx(x, y, a=1, amp=1, x_lim=None, N=100)
```

Exponential decay `amp * exp(-x / a)` fit + sampled curve.

**Returns** `((x_pred, y_pred), a, amp)`.

---

## Linear fits

### `combra.approx.linear_approx`

```python
linear_approx(x, y)
```

Least-squares line `y = k*x + b`.

**Returns** `((x_pred, y_pred), k, b, angle_deg, score)` — `angle_deg` is `arctan(k)` in degrees, `score` is the R².

Used in `PobeditDataset.generate_beams` to populate `prep.a_*` and `prep.b_*` fit fields.

---

### `combra.approx.polyfit_deg1`

```python
polyfit_deg1(x, y, i0, i1)
```

Fast numba-compiled line fit on the slice `[i0:i1]`. Use this in tight loops where `linear_approx`'s scipy overhead matters.

**Returns** `(a, b, r_squared)`.

**Example**

```python
import numpy as np
from combra import approx

x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(scale=0.5, size=100)
k, b, r2 = approx.polyfit_deg1(x, y, i0=10, i1=90)
print(f'slope={k:.3f}  intercept={b:.3f}  R²={r2:.3f}')
```

---

## See also

- [`combra.stats`]({{< relref "/docs/stats" >}}) — the distribution functions these fits target.
- [`combra.angles`]({{< relref "/docs/angles" >}}) — uses `bimodal_gauss_approx` for angle histograms.
- [`combra.mvee`]({{< relref "/docs/mvee" >}}) — uses `linear_approx` for beam-length log-density fits.
