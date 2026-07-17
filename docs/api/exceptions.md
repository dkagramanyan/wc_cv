# combra.exceptions

```{versionadded} 0.4
```

combra's typed error + warning hierarchy. Every library-specific error derives from
{py:class}`~combra.exceptions.CombraError`, so a caller can catch all of combra's
failures with one `except`. Each concrete error also subclasses the built-in it
logically *is* (e.g. `ValueError`), so existing `except ValueError` handlers keep
working.

```python
from combra import exceptions
```

````{py:exception} combra.exceptions.CombraError

Base class for all combra-specific errors (subclasses `Exception`).
````

````{py:exception} combra.exceptions.SchemaError

A parquet / HDF5 file does not match the expected combra schema — e.g.
{py:func}`combra.io.load_rows` on a non-current-schema parquet. Subclasses
{py:class}`~combra.exceptions.CombraError` and `ValueError`.
````

````{py:exception} combra.exceptions.IncompleteShardError

A generated-image HDF5 shard is incomplete (a nonzero `missing_count` or unwritten
slots in the `written` mask). Raised by {py:class}`combra.data.PobeditDataset` when
it refuses to consume a crashed generation run. Subclasses
{py:class}`~combra.exceptions.CombraError` and `ValueError`.
````

````{py:exception} combra.exceptions.CombraWarning

Base class for all combra-specific warnings (subclasses `UserWarning`).
````

````{py:exception} combra.exceptions.UnknownFormatWarning

An HDF5 file carries an unrecognized `format` attribute. Emitted (not raised) by
{py:class}`combra.data.PobeditDataset`. Subclasses
{py:class}`~combra.exceptions.CombraWarning`.
````

**Example**

```python
>>> from combra import exceptions, io
>>> try:
...     io.load_rows('old_schema.parquet')
... except exceptions.CombraError as e:   # catches SchemaError and any other combra error
...     print('combra rejected the file:', e)
```
