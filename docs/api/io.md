# combra.io

```{versionadded} 0.4
```

The `combra.io` module is the single home for reading and writing combra's on-disk
artifacts — the angle/beam **parquet** files and the image **HDF5** containers —
plus the parquet **schemas** and run **provenance**. It follows the `skimage.io`
convention: one loader, one place.

```python
from combra import io
```

## Parquet loading

```{eval-rst}
.. autofunction:: combra.io.load_rows
.. autofunction:: combra.io.load_angles
.. autofunction:: combra.io.load_beams
```

The comparison layer shares this reader — {py:func}`combra.metrics.load_rows` is
the same function.

**Example**

```python
>>> from combra import io
>>> rows = io.load_rows('./data/angles/o_bc_..._msl5/angles_n360.parquet')
>>> rows[0]['meta']['name'], rows[0]['meta']['step']
('Ultra_Co11', 2.0)
```

## Schemas & provenance

```{eval-rst}
.. autodata:: combra.io.ANGLES_SCHEMA
   :annotation:
.. autodata:: combra.io.BEAMS_SCHEMA
   :annotation:
.. autofunction:: combra.io.build_run_meta
```

`ANGLES_SCHEMA` / `BEAMS_SCHEMA` are the pyarrow schemas that
{py:meth}`combra.data.PobeditDataset.generate_angles` /
{py:meth}`~combra.data.PobeditDataset.generate_beams` write — columns `meta`,
`raw`, `prep_per_step`, `run_meta`. `build_run_meta` resolves `code_commit` from
**combra's own package directory**, so it records combra's commit rather than
whatever repo the calling notebook lives in.

## HDF5 conversion

```{eval-rst}
.. autofunction:: combra.io.convert_folder_to_hdf5
.. autofunction:: combra.io.repack_hdf5_uncompressed
```

## See also

- {py:class}`combra.data.PobeditDataset` — writes these parquets via `generate_angles` / `generate_beams`.
- {py:func}`combra.metrics.load_rows` — the same reader, re-exported under `combra.metrics`.
