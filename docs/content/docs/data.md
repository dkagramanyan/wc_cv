---
title: "Data"
weight: 2
---

The `combra.data` module bundles example images shipped with the package and two dataset classes that orchestrate preprocessing + per-class metric generation into parquet.

## Bundled fetchers

These are zero-argument helpers that read images/JSON shipped under `combra/data/`.

### `microstructure_images()`
Five SEM example images (`Ultra_Co{6_2, 8, 11, 15, 25}-001_part_1.jpeg`). Returns `[(filename, ndarray), ...]`.

### `microstructure_class_path()`
Path to the bundled 5-class WC/Co directory tree — pass it to `PobeditDataset(path=...)` for end-to-end smoke tests.

### `crack_images()`, `crack_labeled_json()`
One example crack image and the matching label-studio JSON.

```python
from combra import data

name, img = data.microstructure_images()[0]
class_root = data.microstructure_class_path()
```

## `PobeditDataset`

Manages WC/Co microstructure images. The first call to a `generate_*` method preprocesses every image and caches the result in an HDF5 file alongside the dataset; subsequent calls reuse the cache.

### `PobeditDataset(path=None, max_images_num_per_class=100, compression='lzf', print_depth=3)`
- `path`: either a folder of class subfolders (`Ultra_Co11/...`) or an existing `*.h5` cache.
- `max_images_num_per_class`: cap per class (`None` = use everything).
- `compression`: HDF5 compression for the preprocessed-image cache (`'lzf'`, `'gzip'`, or `None`).

### `generate_angles(save_path, types_dict, step, workers=20, angles_tol=3, min_segment_len=10.0, queue_size=32, keep_contours=True, chunksize=64)`
Compute angle distributions for every image and write them to `<save_path>_step_<step>_N_<n>.parquet`. `step` may be a list to fit several histogram steps in one pass — they are stored under `prep_per_step` in the same row.

### `generate_beams(save_path, types_dict, step, pixel, start=2, end=-3, workers=20, mvee_tol=0.2, queue_size=32, keep_contours=True)`
Compute MVEE beam parameters (`a_beams`, `b_beams`, rotation angles, areas) per image. `pixel` is the physical pixel size for area calibration; `start`/`end` clip the histogram extremes before the linear fit.

### `close()`
Tear down the worker pool. Optional — runs at GC.

```python
from combra import data

ds = data.PobeditDataset(path=data.microstructure_class_path(),
                         max_images_num_per_class=360)
ds.generate_angles(
    save_path='orig',
    types_dict={'Ultra_Co11': 'medium', 'Ultra_Co25': 'fine'},
    step=[1, 5, 10],
    workers=20,
)
ds.generate_beams(
    save_path='beams',
    types_dict={'Ultra_Co11': 'medium', 'Ultra_Co25': 'fine'},
    step=4, pixel=50/1000,
)
```

## `PoliamidDataset`

Polyamide fracture-image dataset. Filenames are expected to be plain integers; images are bucketed into groups of `group_size` consecutive numbers and metrics are aggregated per group.

### `PoliamidDataset(images_folder_path, group_size=250)`
### `generate(out_path, n_jobs=20, N=500)`
For each group, computes contour fractal dimension, image-level fractal dimension, contour lengths and areas, and writes one row per group to a parquet file. `N` is the minimum contour length below which a contour is skipped.

```python
from combra import data

pa_ds = data.PoliamidDataset('/path/to/poliamid', group_size=250)
pa_ds.generate('poliamid.parquet', n_jobs=20, N=500)
```
