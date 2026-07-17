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

````{py:function} combra.io.load_rows(parquet_path) -> list[dict]

Load an angle/beam parquet as a flat list of `{'meta', 'prep'}` records — one per
`(class, step)`. Each class-level `meta` is copied into each record with the
step's `step` merged in, and `prep` is that step's entry from `prep_per_step`.
This is the one reader the comparison layer ({py:func}`combra.metrics.load_rows`
is the same function) and the angle plotters share.

:param parquet_path: Path to a combra angle/beam parquet.
:type parquet_path: str or pathlib.Path
:returns: **rows** – one `{'meta': {..., 'step'}, 'prep': {...}}` dict per (class, step).
:rtype: list[dict]

**Example**

```python
>>> from combra import io
>>> rows = io.load_rows('./data/angles/o_bc_..._msl5/angles_n360.parquet')
>>> rows[0]['meta']['name'], rows[0]['meta']['step']
('Ultra_Co11', 2.0)
```
````

````{py:function} combra.io.load_angles(path) -> list[dict]

Alias of {py:func}`~combra.io.load_rows` for angle parquets.
````

````{py:function} combra.io.load_beams(path) -> list[dict]

Alias of {py:func}`~combra.io.load_rows` for beam parquets (angle and beam files
share the `meta` / `prep_per_step` layout).
````

## Schemas & provenance

````{py:data} combra.io.ANGLES_SCHEMA

The pyarrow schema for angle parquets ({py:meth}`combra.data.PobeditDataset.generate_angles`
writes it): columns `meta`, `raw`, `prep_per_step`, `run_meta`. Lifting it here
(out of the dataset class) gives writers and readers one shared definition.
````

````{py:data} combra.io.BEAMS_SCHEMA

The pyarrow schema for beam parquets ({py:meth}`combra.data.PobeditDataset.generate_beams`).
Same top-level columns as {py:data}`~combra.io.ANGLES_SCHEMA`, beam-specific
`raw` / `prep_per_step` fields.
````

````{py:function} combra.io.build_run_meta(h5_path, n_requested, extraction_params, user) -> dict

Assemble the `run_meta` provenance dict written on every class-row: `model_tag`,
`kimg` (parsed from the h5 stem), `source_h5`, `code_commit`, `generated_at`, and
the exact `extraction_params`. `code_commit` is resolved from **combra's own
package directory**, so it records combra's commit rather than whatever repo the
calling notebook lives in.

:param h5_path: Source HDF5 path; its stem seeds `model_tag` and `kimg`.
:type h5_path: str
:param n_requested: Requested images-per-class for this run.
:type n_requested: int
:param extraction_params: Task-specific params matching the schema's `extraction_params` struct.
:type extraction_params: dict
:param user: Optional overrides — `family`, `resolution`, `tags`, `notes`.
:type user: dict or None
:returns: **run_meta** – the provenance dict.
:rtype: dict
````

## HDF5 conversion

````{py:function} combra.io.convert_folder_to_hdf5(folder_path, output_path, max_images_per_class=None, compression=None, chunk_images=32) -> pathlib.Path

Convert a folder of `class_*/` image subdirectories into a combra `pobedit_images`
HDF5 file (`class_<name>/images` uint8 datasets, class balanced to the smallest
class). Written to a temp file and renamed on full success. This is what
{py:class}`combra.data.PobeditDataset` calls when handed a folder.

:param folder_path: Root with one image subfolder per class.
:type folder_path: str or pathlib.Path
:param output_path: Destination `.h5` path.
:type output_path: str or pathlib.Path
:param max_images_per_class: Cap per class (`None` = all).
:type max_images_per_class: int or None, optional
:param compression: h5 compression (`None` = uncompressed; `'lzf'` for smaller files).
:type compression: str or None, optional
:param chunk_images: Images per HDF5 chunk. Default: `32`.
:type chunk_images: int, optional
:returns: **output_path** – the written file.
:rtype: pathlib.Path
````

````{py:function} combra.io.repack_hdf5_uncompressed(src_path, dst_path=None, chunk_images=32, overwrite=False) -> pathlib.Path

Re-pack an existing pobedit HDF5 to uncompressed, preserving all attributes. Use
once on an LZF cache to eliminate the source-decompression bottleneck.
````

## See also

- {py:class}`combra.data.PobeditDataset` — writes these parquets via `generate_angles` / `generate_beams`.
- {py:func}`combra.metrics.load_rows` — the same reader, re-exported under `combra.metrics`.
