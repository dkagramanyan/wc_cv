---
title: "Utils"
weight: 13
---

Miscellaneous helpers used across combra.

## `NumpyEncoder`

A `json.JSONEncoder` subclass that handles numpy scalars and arrays. Use it whenever you need to dump combra outputs to JSON without converting types by hand.

```python
import json
import numpy as np
from combra.utils import NumpyEncoder

payload = {
    'mus': np.array([90.5, 270.3]),
    'count': np.int64(42),
    'score': np.float32(0.97),
}

with open('out.json', 'w') as f:
    json.dump(payload, f, cls=NumpyEncoder)
```

Handles `np.float32`, `np.float64`, `np.int32`, `np.int64`, and any `np.ndarray`. Anything else falls through to the default JSON encoder (which raises for unsupported types).
