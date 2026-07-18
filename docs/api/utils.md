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

```{versionadded} 0.4
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
