# Get started

## Installation

Combra is a standard Python package — all dependencies install automatically. From a clone of the repository:

```bash
pip install -e .
```

On Linux, OpenCV also needs the system OpenGL libraries:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

FID metrics work out of the box. `combra.metrics.fid` delegates to [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [torch-fidelity](https://github.com/toshas/torch-fidelity), which ship as core dependencies and download/cache their own InceptionV3 weights on first use — no manual model setup. `compute_fid` runs on CUDA when available and falls back to CPU; see {doc}`combra.metrics <api/metrics>`.

## Smoke test

Five lines that exercise the full pipeline — load a bundled image, preprocess it, extract angles:

```python
>>> from combra import data, image, angles
>>> _, img = data.microstructure_images()[0]
>>> processed = image.do_otsu(img)
>>> arr, contours = angles.get_angles(processed, border_eps=5, tol=3, min_segment_len=10.0)
>>> print(f'{len(arr)} angles, mean={arr.mean():.2f}°')
```

## First parquet

End-to-end: run angle extraction on every class in the bundled dataset, write one parquet with full provenance.

```python
>>> from combra import data
>>> ds = data.PobeditDataset(
...     path=data.microstructure_class_path(),
...     max_images_num_per_class=50,
... )
>>> ds.generate_angles(
...     save_path='./smoke_test',
...     types_dict={'Ultra_Co11': 'medium', 'Ultra_Co25': 'fine'},
...     step=[1, 5, 10],
...     workers=8, min_segment_len=5.0, keep_contours=False,
...     run_meta={'family': 'real', 'resolution': 1024, 'notes': 'smoke'},
... )
```

The output file's `run_meta` column records who/when/what — including the git commit and exact extraction params — so the parquet is fully self-describing.

## Module map

| module | what it does |
| --- | --- |
| {doc}`combra.data <api/data>` | Datasets, bundled fetchers, parquet writers (`generate_angles`, `generate_beams`). |
| {doc}`combra.image <api/image>` | Pixel-level preprocessing, fractal dimension, numba geometry kernels. |
| {doc}`combra.contours <api/contours>` | Polygon extraction + drawing. |
| {doc}`combra.angles <api/angles>` | Per-image angle extraction and grid plots. |
| {doc}`combra.mvee <api/mvee>` | Minimum-volume enclosing ellipses, beam distributions. |
| {doc}`combra.areas <api/areas>` | Polygon-area and effective-radius distribution plots. |
| {doc}`combra.stats <api/stats>` | Parametric distributions + histogram preprocessor. |
| {doc}`combra.approx <api/approx>` | Fits Gaussian/binomial/poisson/exponential/linear models. |
| {doc}`combra.metrics <api/metrics>` | FID, per-class Wasserstein comparison, and convergence-vs-N analysis (Kendall trend, plateau fit, gain-distribution plot). |
| {doc}`combra.graph <api/graph>` | Crack-image → graph → shortest-energy-path search. |
| {doc}`combra.utils <api/utils>` | `NumpyEncoder` for JSON dumps. |
