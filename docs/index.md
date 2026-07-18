# combra

Computer-vision tools for analysis of **WC-Co composite-alloy microstructure SEM images** —
contour/angle extraction, MVEE beam fitting, fractal dimension, crack graphs, and
distribution metrics.

```python
import combra
```

```{toctree}
:maxdepth: 1
:caption: Getting started

get_started
```

```{toctree}
:maxdepth: 1
:caption: Python API

api/data
api/image
api/contours
api/angles
api/mvee
api/stats
api/approx
api/metrics
api/graph
api/io
api/viz
api/exceptions
api/utils
api/tests
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/angles
examples/fid
examples/models_api
examples/models_api_proposal
examples/san_v2
examples/styleswin
examples/diffit
examples/edm2
examples/sampler_comparison
```

## Module map

| module | what it does |
| --- | --- |
| {doc}`combra.data <api/data>` | Datasets, bundled fetchers, parquet writers (`generate_angles`, `generate_beams`). |
| {doc}`combra.image <api/image>` | Pixel-level preprocessing, fractal dimension, numba geometry kernels. |
| {doc}`combra.contours <api/contours>` | Polygon extraction + drawing. |
| {doc}`combra.angles <api/angles>` | Per-image angle extraction and grid plots. |
| {doc}`combra.mvee <api/mvee>` | Minimum-volume enclosing ellipses, beam distributions. |
| {doc}`combra.stats <api/stats>` | Parametric distributions + histogram preprocessor. |
| {doc}`combra.approx <api/approx>` | Fits Gaussian/binomial/poisson/exponential/linear models. |
| {doc}`combra.metrics <api/metrics>` | FID, batch generative-quality metrics (CMMD, FD-DINOv2, angle-Wasserstein), per-class Wasserstein comparison, sampler-vs-steps comparison, and convergence-vs-N analysis. |
| {doc}`combra.graph <api/graph>` | Crack-image → graph → shortest-energy-path search. |
| {doc}`combra.io <api/io>` | One loader (`load_rows`/`load_angles`) + parquet schemas + HDF5 conversion. |
| {doc}`combra.viz <api/viz>` | Shared plotting theme: palette, axis style, PNG export. |
| {doc}`combra.exceptions <api/exceptions>` | Typed error/warning hierarchy (`CombraError`, `SchemaError`, …). |
| {doc}`combra.utils <api/utils>` | `Bunch` attribute-accessible dict container. |
| {doc}`combra.tests <api/tests>` | Self-validation helpers (fractal-dimension sanity check). |
