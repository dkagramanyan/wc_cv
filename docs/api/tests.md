# combra.tests

Self-validation helpers shipped with combra. They build shapes with a known
answer and check combra's estimators against it — handy as a post-install
sanity check.

```python
from combra import tests
```

````{py:function} combra.tests.test_fractal_dimensions(sizes) -> None

Validate the box-counting fractal-dimension estimator on seven reference shapes with known dimensions and compare {py:func}`combra.image.image_fractal_dimension` against each:

- a single **point** ($D = 0$),
- a straight **line** ($D = 1$),
- a **filled square** ($D = 2$),
- **Cantor dust** ($\log 4/\log 3 \approx 1.262$),
- the **Sierpiński carpet** ($\log 8/\log 3 \approx 1.893$),
- the **Sierpiński triangle** ($\log 3/\log 2 \approx 1.585$),
- the **Koch curve** ($\log 4/\log 3 \approx 1.262$).

For each shape it prints the expected vs. estimated dimension and the relative error, then a final **SUMMARY** line with the average relative error across the non-degenerate shapes (the zero-dimension point is excluded from the average).

:param sizes: Box sizes to sweep (passed through to {py:func}`combra.image.image_fractal_dimension`).
:type sizes: ndarray or list[int]
:returns: Nothing. Prints the estimated-vs-expected dimension and relative error for each reference shape, followed by the average relative error.
:rtype: None

**Example**

```python
>>> import numpy as np
>>> from combra import tests
>>> tests.test_fractal_dimensions(sizes=np.array([2, 3, 4, 6, 8, 12, 16, 24, 32]))
```
````
