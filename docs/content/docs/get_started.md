---
title: "Get started"
weight: 1
---

## Installation

Combra ships as an editable Python package. Install in two steps because `crdp` (Douglas–Peucker accelerator) requires `cython` to be available before its own build kicks in.

```bash
pip install cython
pip install --no-build-isolation 'crdp>=0.0.2'
pip install -e .
```

## Smoke test

Five lines that exercise the full pipeline — load a bundled image, preprocess it, extract angles:

```python
from combra import data, image, angles

_, img = data.microstructure_images()[0]
processed = image.do_otsu(img)
arr, contours = angles.get_angles(processed, border_eps=5, tol=3, min_segment_len=10.0)
print(f'{len(arr)} angles, mean={arr.mean():.2f}°')
```

## First parquet

End-to-end: run angle extraction on every class in the bundled dataset, write one parquet with full provenance.

```python
from combra import data

ds = data.PobeditDataset(
    path=data.microstructure_class_path(),
    max_images_num_per_class=50,
)
ds.generate_angles(
    save_path='./smoke_test',
    types_dict={'Ultra_Co11': 'medium', 'Ultra_Co25': 'fine'},
    step=[1, 5, 10],
    workers=8, min_segment_len=5.0, keep_contours=False,
    run_meta={'family': 'real', 'resolution': 1024, 'notes': 'smoke'},
)
# Saved: ./smoke_test/angles_n50.parquet
```

The output file's `run_meta` column records who/when/what — including the git commit and exact extraction params — so the parquet is fully self-describing.

## Module map

| module | what it does |
| --- | --- |
| [`combra.data`]({{< relref "/docs/data" >}}) | Datasets, bundled fetchers, parquet writers (`generate_angles`, `generate_beams`). |
| [`combra.image`]({{< relref "/docs/image" >}}) | Pixel-level preprocessing, fractal dimension, numba geometry kernels. |
| [`combra.contours`]({{< relref "/docs/contours" >}}) | Polygon extraction + drawing. |
| [`combra.angles`]({{< relref "/docs/angles" >}}) | Per-image angle extraction and grid plots. |
| [`combra.mvee`]({{< relref "/docs/mvee" >}}) | Minimum-volume enclosing ellipses, beam distributions. |
| [`combra.areas`]({{< relref "/docs/areas" >}}) | Polygon-area and effective-radius distribution plots. |
| [`combra.stats`]({{< relref "/docs/stats" >}}) | Parametric distributions + histogram preprocessor. |
| [`combra.approx`]({{< relref "/docs/approx" >}}) | Fits Gaussian/binomial/poisson/exponential/linear models. |
| [`combra.metrics`]({{< relref "/docs/metrics" >}}) | FID, per-class Wasserstein comparison, and convergence-vs-N analysis (Kendall trend, plateau fit, gain-distribution plot). |
| [`combra.graph`]({{< relref "/docs/graph" >}}) | Crack-image → graph → shortest-energy-path search. |
| [`combra.utils`]({{< relref "/docs/utils" >}}) | `NumpyEncoder` for JSON dumps. |
