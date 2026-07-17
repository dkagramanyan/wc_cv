# combra.utils

Small helpers shared across combra.

```python
from combra import utils
```

## Bunch

````{py:class} combra.utils.Bunch(**kwargs)

A `dict` that also exposes its keys as attributes — the scikit-learn container
convention (`sklearn.utils.Bunch`). Dataset loaders such as
{py:func}`combra.data.load_microstructure` return one so results are
self-describing (`ds.images`, `ds.class_names`) while still behaving as a plain
`dict`. Assigning a new attribute adds a key.

```{versionadded} 0.5
```

**Example**

```python
>>> from combra.utils import Bunch
>>> b = Bunch(images=[...], class_names=['Ultra_Co11', 'Ultra_Co25'])
>>> b.class_names[0]
'Ultra_Co11'
>>> b['class_names'] is b.class_names   # dict access too
True
```
````

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
