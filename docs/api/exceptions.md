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

```{eval-rst}
.. autoexception:: combra.exceptions.CombraError
.. autoexception:: combra.exceptions.SchemaError
.. autoexception:: combra.exceptions.IncompleteShardError
.. autoexception:: combra.exceptions.CombraWarning
.. autoexception:: combra.exceptions.UnknownFormatWarning
```

`SchemaError` and `IncompleteShardError` also subclass `ValueError`;
`UnknownFormatWarning` subclasses `CombraWarning` (itself a `UserWarning`).
`SchemaError` is what {py:func}`combra.io.load_rows` raises on a non-current-schema
parquet, and `IncompleteShardError` is raised by {py:class}`combra.data.PobeditDataset`
when it refuses to consume a crashed generation run (a nonzero `missing_count` or
unwritten slots in the `written` mask).

**Example**

```python
>>> from combra import exceptions, io
>>> try:
...     io.load_rows('old_schema.parquet')
... except exceptions.CombraError as e:   # catches SchemaError and any other combra error
...     print('combra rejected the file:', e)
```
