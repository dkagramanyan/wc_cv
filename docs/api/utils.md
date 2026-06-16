# combra.utils

Small helpers shared across combra.

```python
from combra import utils
```

## NumpyEncoder

````{py:class} combra.utils.NumpyEncoder(json.JSONEncoder)

A `json.JSONEncoder` subclass that handles numpy scalars and arrays. Use it whenever you need to dump combra outputs to JSON without converting types by hand.

**Handles**

- **`np.float32`, `np.float64`** — serialised as Python `float`.
- **`np.int32`, `np.int64`** — serialised as Python `int`.
- **`np.ndarray`** — serialised as nested `list` (via `tolist()`).
- **anything else** — falls through to the default `json.JSONEncoder` (raises `TypeError` for unsupported types).

**Example**

```python
>>> import json
>>> import numpy as np
>>> from combra.utils import NumpyEncoder
>>> payload = {
...     'mus': np.array([90.5, 270.3]),
...     'count': np.int64(42),
...     'score': np.float32(0.97),
... }
>>> with open('out.json', 'w') as f:
...     json.dump(payload, f, cls=NumpyEncoder)
```
````
