# Get started

## Installation

Combra is a standard Python package — all dependencies are pure-pip and install automatically, with no system packages required. combra uses the headless OpenCV build (`opencv-python-headless`), so no `libGL`/`libglib` system libraries are needed; it runs out of the box in minimal containers and HPC environments where you can only install Python packages.

combra is not on PyPI yet, so install it from source:

```bash
git clone https://github.com/dkagramanyan/combra.git
cd combra
pip install .          # or:  pip install -e .   for an editable install
```

### Optional extras

| Extra         | Install                          | Adds                                          |
| ------------- | -------------------------------- | --------------------------------------------- |
| `tests`       | `pip install ".[tests]"`         | pytest + pytest-cov                           |
| `docs`        | `pip install ".[docs]"`          | Sphinx docs toolchain                         |
| `gen-metrics` | `pip install ".[gen-metrics]"`   | `open-clip-torch` for `compute_cmmd` / `compute_fd_dinov2` |
| `dev`         | `pip install -e ".[dev]"`        | the `tests` extra + ruff                      |

FID metrics work out of the box. `combra.metrics.fid` delegates to [pytorch-fid](https://github.com/mseitzer/pytorch-fid), which ships as a core dependency and downloads/caches its own InceptionV3 weights on first use — no manual model setup. `compute_fid` runs on CUDA when available and falls back to CPU; see {doc}`combra.metrics <api/metrics>`.

The angle-Wasserstein training metrics use [POT](https://pythonot.github.io/) (`pot`), a core dependency. The CLIP-MMD and DINOv2 metrics (`compute_cmmd`, `compute_fd_dinov2`) additionally need the `gen-metrics` extra; the DINOv2 backbone is fetched from `torch.hub` on first use.

## Testing

After a development install you can run the full test suite, the linter, and the formatter:

```bash
pip install -e ".[dev]"

pytest                           # run the test suite
ruff check combra tests          # lint
ruff format combra tests         # format
```

CI (GitHub Actions) runs the ruff lint + format checks and `pytest` on Python 3.10, 3.11, and 3.12.

For a quick post-install sanity check that doesn't need the dev tools, run the bundled self-validation helper — it estimates the fractal dimension of reference shapes with known answers (see {doc}`combra.tests <api/tests>`):

```python
>>> import numpy as np
>>> from combra import tests
>>> tests.test_fractal_dimensions(np.array([2, 3, 4, 6, 8, 12, 16, 24, 32]))
```

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
| {doc}`combra.metrics <api/metrics>` | FID, batch generative-quality metrics (CMMD, FD-DINOv2, angle-Wasserstein), per-class Wasserstein comparison, and convergence-vs-N analysis (Kendall trend, plateau fit, gain-distribution plot). |
| {doc}`combra.graph <api/graph>` | Crack-image → graph → shortest-energy-path search. |
| {doc}`combra.utils <api/utils>` | `NumpyEncoder` for JSON dumps. |
