# combra.tests

Self-validation helpers shipped with combra. They build shapes with a known
answer and check combra's estimators against it — handy as a post-install
sanity check.

```python
from combra import tests
```

````{py:function} combra.tests.test_fractal_dimensions(sizes) -> None

Validate the box-counting fractal-dimension estimator on canonical fractals with known Hausdorff dimensions — Cantor dust ($\log 4/\log 3 \approx 1.262$), the Sierpiński carpet ($\log 8/\log 3 \approx 1.893$), and the Sierpiński triangle ($\log 3/\log 2 \approx 1.585$) — and compares {py:func}`combra.image.image_fractal_dimension` against each reference value.

:param sizes: Box sizes to sweep (passed through to {py:func}`combra.image.image_fractal_dimension`).
:type sizes: ndarray or list[int]
:returns: Nothing. Prints the estimated-vs-expected dimension for each reference fractal.
:rtype: None

**Example**

```python
>>> import numpy as np
>>> from combra import tests
>>> tests.test_fractal_dimensions(sizes=np.array([2, 3, 4, 6, 8, 12, 16, 24, 32]))
```
````
