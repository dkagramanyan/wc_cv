# combra — proposed library standardization ("v0.4 convention")

```{note}
**Status: proposal.** Nothing here is implemented yet. This page is the
combra-side companion to {doc}`models_api_proposal` (which standardizes the
four generative-model repos): **Part 1** specifies the target library API,
**Part 2** is the prioritized inventory of defects and changes, with adoption
order. All findings below were verified against the code and against the
on-disk artifacts in `co_angles/data/` (66 angle parquets, ~300 GB of derived
data) on 2026-07-17.
```

The scope is deliberately narrow: the **active path** today is
`PobeditDataset` with `generate_angles` / `generate_beams`, plus the
`combra.metrics` comparison layer that consumes their parquets. Everything
else (areas, graph, poliamid, legacy stats) is stabilized or explicitly
marked legacy — not redesigned.

The design rule for the target API: **when a convention already exists in
NumPy, scikit-image, or PyTorch, adopt it instead of inventing one.**
Concretely: scikit-image's explicit dtype/range contract and flat functional
subpackages, SciPy's named-tuple results, torchvision's dataset attributes
(`classes` / `class_to_idx`), and NumPy's deprecation policy.

---

# Part 1 — The target API

## 1. Package layout & naming

The flat scikit-image-style subpackage layout stays (`combra.angles`,
`combra.mvee`, `combra.stats`, …) — it already mirrors `skimage.*`. Three
changes:

| change | rationale |
|---|---|
| new `combra.io` subpackage: `convert_folder_to_hdf5`, `repack_hdf5_uncompressed` (from `data._hdf5`) + **one** parquet loader `load_angles(path)` / `load_beams(path)` | mirrors `skimage.io`; today the old-vs-new parquet schema sniffing is duplicated in `metrics/compare.py:15-24` (pyarrow) and `angles/plot.py:21-34` (polars) — two copies to keep in sync |
| `combra.tests` → `combra.diagnostics` | no mature library ships a public `tests` subpackage; `combra.tests.fractal.test_fractal_dimensions(sizes)` is a `test_`-prefixed function with a required positional arg — pytest collects it and errors the moment collection is widened (`--pyargs combra`) |
| drop the `get_` prefix on extraction functions: `get_angles` → `vertex_angles`, `get_mvee_params` → `fit_mvee`, `get_contours` → `find_contours` | skimage convention (`measure.find_contours`, `feature.canny` — no `get_`); old names remain as deprecated aliases for one minor release (§6) |

## 2. Input contract — explicit dtype & range, no guessing

**uint8 grayscale/RGB at every public boundary; float ranges are declared,
never inferred.** This is the library-side half of the normalization contract
in {doc}`models_api_proposal` §5.

- Image functions accept `uint8` arrays of shape `(H, W)` or `(H, W, C)`
  (NumPy/skimage axis order). Float input is accepted **only** with an
  explicit range parameter.
- `metrics._to_uint8` (`metrics/training.py:315-328`) loses its per-image
  min/max guessing. Replacement signature, skimage-style:

  ```python
  to_uint8(a, data_range=None)
  # data_range: None    -> a must already be uint8, else ValueError
  #             (lo, hi) -> linear map [lo, hi] -> [0, 255]
  #             "infer"  -> today's heuristic, kept one release, warns
  ```

  Today the heuristic is applied **per image** (`training.py:119`), so an
  all-positive `[-1, 1]` image in a batch is silently scaled ×255 instead of
  ×127.5 — a content-dependent metric bias with no way for the caller to pin
  the range.
- Contours are `(N, 2)` float64 in **(x, y)** order — the cv2 convention the
  extraction already uses — stated once in the docs and asserted at module
  boundaries. Today `contours/preprocess.py` mixes `(N, 2)` and `(N, 1, 2)`
  contracts within one file, and `angles/generate.py:210-211` swaps the
  x/y columns before the angle kernel (winding-dependent: the swap flips the
  convex-vs-reflex decision — it must be either fixed or documented as
  intentional with a test pinning the behavior).

## 3. Result objects — SciPy-style named tuples

Positional multi-tuples are replaced by `NamedTuple` results (compare
`scipy.stats.linregress`):

| today | target |
|---|---|
| `linear_approx` → `((x_pred, y_pred), k, b, angle, score)` 5-tuple (`approx/fit.py:180`) | `LinearFit(slope, intercept, angle_deg, r2, x_pred, y_pred)` |
| `bimodal_gauss_approx` → `((x, y), mus, sigmas, amps)` (`approx/fit.py:106`) | `BimodalGaussFit(mus, sigmas, amps, x, y)` |
| `get_mvee_params` → `(a, b, angles, centroids, cnts)` (`mvee/build.py`) | `MveeResult(a, b, angle_rad, centroid, contour)` with **`a >= b` guaranteed** (today the primary path takes `axes[0], axes[1]` unordered while the cv2 fallback uses `max/min` — the a/b series can be silently mixed) |
| `_class_meta` 9-positional tuple with dummy args (`pobedit_dataset.py:536-550`) | plain dict / dataclass |

The two independent degree-1 fit implementations (`linear_approx` via
sklearn, `polyfit_deg1` via numba, different return shapes and R²
conventions) collapse into one.

## 4. Dataset API — torchvision-style class handling

`PobeditDataset` keeps its role (HDF5-backed, prep-cache, parquet writers)
but adopts the `torchvision.datasets.ImageFolder` attribute contract, which
fixes the class-identity defects in one move:

```python
ds = PobeditDataset(path, max_per_class=None)
ds.classes        # ['Ultra_Co11', 'Ultra_Co25', ...]  — BARE names, sorted
ds.class_to_idx   # {'Ultra_Co11': 0, ...}
len(ds)           # n_classes * n_per_class
ds[i]             # (image_uint8, class_idx)
```

- **Name resolution order**: h5 `class_names` root attr → per-group
  `class_name` attr → group-name suffix (`class_` stripped) as fallback.
  Today **no h5 attribute is ever read back** even though
  `convert_folder_to_hdf5` writes them (`data/_hdf5.py:123`); identity is
  the literal group-name string, so generated files yield classes
  `"0"/"1"/"2"` and real files `"class_Ultra_Co*"`.
- **The `class_` prefix never escapes the data layer.** Today
  `_class_meta` keeps it, so `types_dict.get(name)` (keyed by bare names in
  every notebook and doc example) misses for every class — **verified:
  `meta.type == ""` in all 66 existing parquets**.
- `class_*` groups are sorted **numerically when the suffix is an integer**
  (today's plain string sort orders `class_10` before `class_2`).
- `CLASS_MAP` (legacy generated-artifact remapping) is implemented as a real
  module-level registry in `combra.data` — today it exists **only in the
  docs** ({doc}`models_api_proposal` §12); the code has per-call `class_map`
  dict arguments.

Deprecated constructor params (`images_folder_path`, `image_hdf_path`, the
unused `hdf_images_threshold`, `*args/**kwargs`) are removed in the same
release.

## 5. Extraction pipeline — angles/beams parity

`generate_angles` and `generate_beams` become `extract_angles` /
`extract_beams` (old names deprecated-aliased) with **mirrored signatures**:

- both take `force_rebuild_cache` and `chunksize` (today only
  `generate_angles` has them; beams hardcodes `chunksize=4`);
- both take `types_dict` keyed by **bare** class names (§4);
- shared keyword set: `save_path, types_dict, step, workers, keep_contours,
  run_meta, force_rebuild_cache, chunksize` + method-specific params
  (`angles_tol, min_segment_len` / `pixel, start, end, mvee_tol`);
- the two legend builders get one parameter order (today `angles_legend`
  and `beams_legend` disagree, and `beams_legend` is called with literal
  `images_amount=0, dist_mean=0.0` — dead parameters that render as "0");
- edge guards: `stats_preprocess(step=0)` raises `ValueError` instead of
  print-and-return-`None` (every caller unpacks the result → `TypeError`
  today); `bimodal_gauss_approx` and `linear_approx` handle empty input
  (an empty `slice(2, -3)` histogram slice currently crashes a beams run
  after the full cache build).

## 6. Parquet contract

The angle/beam parquets are combra's primary scientific artifact and get the
same treatment {doc}`models_api_proposal` §4 gives the generated-image h5:

| rule | detail |
|---|---|
| `schema_version` | stamped in `run_meta` (start at `2` for the current 4-column layout); every writer writes it, every reader checks it |
| `run_meta` on every file | today **48 of the 66 on-disk parquets are old-schema** (`meta/raw/prep_per_step`, no `run_meta`) and 18 are new-schema — readers branch by column sniffing |
| **pyarrow floor: `pyarrow>=22`** declared in `pyproject.toml` | files written by Arrow 22 are **unreadable** by pyarrow 19 on every nested column (`OSError: Repetition level histogram size mismatch`) — verified: the base anaconda env cannot read any current parquet; the data is silently welded to the `wc-cv` env |
| one loader | `combra.io.load_angles/load_beams` owns the old-vs-new branch; the duplicated sniffing in `metrics/compare.py` and `angles/plot.py` delegates to it |
| provenance fixes | `run_meta.code_commit` currently runs `git rev-parse` in the **CWD** — it records the commit of whatever repo the calling notebook lives in; resolve from combra's own package directory or drop it. `source_h5_sha` is always `None` — populate or remove |
| filename honesty | `angles_n{N}.parquet` / cache `*_n{N}.npy` use the **requested** N; when the source has fewer images the file is mislabeled and `generate_angles_sweep`'s skip logic trusts the wrong name — `N` becomes the actual per-class count |

A one-shot migration script (read old-schema parquet → rewrite with
`run_meta` + `schema_version`) is optional tooling, not a contract item —
regeneration is usually cheaper.

## 7. Logging

Library convention (NumPy/skimage do not print):

- one logger tree, `logging.getLogger("combra.<module>")`, `NullHandler`
  attached at the package root;
- every `print()` in the pipeline (16 in `pobedit_dataset.py` alone, plus
  `data/_hdf5.py`, `metrics/compare.py`, `metrics/convergence.py`) becomes
  `logger.info` / `logger.warning`, keeping the `[HDF5]` / `[CACHE]`
  prefixes as message tags;
- `tqdm` progress bars stay (they are UI, not logs);
- errors raise; warnings use `warnings.warn` — never a print that scrolls
  away.

## 8. Errors & deprecation policy

NumPy's policy, scaled down:

- invalid input → `ValueError`/`TypeError` with the offending value in the
  message — never print-and-return-`None`;
- every renamed symbol keeps its old name for **one minor release** as a
  wrapper that emits `DeprecationWarning`; the removal release is named in
  the warning text;
- behavior changes that alter numbers (a/b ordering, border-median fix,
  `data_range` explicitness) are called out in the changelog under a
  **"results change"** heading, since downstream science must know to
  regenerate.

## 9. Cache & data lifecycle

The prep cache is 71 GB and the h5 mirror 213 GB with no tooling:

- cache filename gains a **preprocessing-version tag**
  (`angles_n360_p2.npy`) so `_cache_ok`'s shape/dtype check can no longer
  silently reuse a cache built by older preprocessing (today the only
  protection is remembering `force_rebuild_cache=True`);
- one documented layout — `<data>/h5/`, `<data>/cache/<h5-stem>/`,
  `<data>/angles/` — and the legacy sibling layout
  (`*.prep_cache_*.npy` next to source h5s, still present under
  `datasets/san/` and `co_angles/data/original_images/`) is retired;
- a small `python -m combra.cache {ls,prune}` utility lists caches with
  sizes and deletes those whose source h5 or version tag no longer matches.

## 10. Packaging & CI

| change | rationale |
|---|---|
| torch stack (`torch`, `torchvision`, `pytorch-fid`, `open-clip-torch`, `pot`) moves from core `dependencies` to a `[metrics]` extra | the code already guards these imports as optional (`training.py:391-398`, `504-508`) and `tests/test_public_api.py` lists them as `OPTIONAL_DEPS`; today `pip install combra` pulls the full CUDA stack to compute vertex angles |
| `requires-python >= 3.10`, CI matrix matches exactly | today 3.9–3.13 is claimed, 3.10–3.12 is tested |
| `pyarrow>=22` floor | §6 |
| audit `numba` / `kaleido` core pins | `kaleido` is plot-export-only; verify each is genuinely core |
| stale `build/lib/combra` shadow copy and `combra.egg-info` cleaned from working trees; the pip editable metadata pointing at the dead `…/python/phd/…` path refreshed by reinstall | hygiene; the shadow copy already diverges from the source |

---

# Part 2 — Prioritized inventory

Severity reflects impact on the **active path** (PobeditDataset →
angles/beams parquets → metrics). "Results change" = regenerated artifacts
will differ numerically after the fix.

## P0 — silent wrong output today

| # | defect | where | evidence / failure mode | results change |
|---|---|---|---|---|
| 1 | `types_dict` lookup misses for every class → `meta.type=""` | `data/pobedit_dataset.py:536-539,739,888` | **all 66 on-disk parquets** carry empty `type`; legends lose the grain-type text | metadata only |
| 2 | parquets unreadable by pyarrow < 22; no floor declared | writer `pobedit_dataset.py:663`; env: wc-cv=22, base=19 | `OSError: Repetition level histogram size mismatch` on `meta`/`raw`/`prep_per_step` | no |
| 3 | mixed parquet schemas, no `schema_version` | 48 old / 18 new files in `co_angles/data/angles/` | any `run_meta` consumer breaks on 73 % of files | no |
| 4 | MVEE a/b unordered in primary path, `max/min` in fallback | `mvee/build.py:102-124` | a-vs-b beam series mixed per contour; the two linear fits in `extract_beams` fit contaminated series | **yes** |
| 5 | median-disk border bug: `half` from full footprint, corners can never reach it → `np.empty` garbage at borders | `image/utils.py:18-39` | garbage border pixels feed contour extraction for **both** pipelines | **yes** (border-adjacent) |
| 6 | x/y column swap before angle kernel flips convex/reflex decision | `angles/generate.py:210-211` | undocumented orientation hazard at the heart of the angle distributions | if unintentional, **yes** |
| 7 | `_to_uint8` per-image range guessing | `metrics/training.py:315-328` | all-positive `[-1,1]` image scaled ×255; biases FID/CMMD/angle metrics content-dependently | **yes** (metrics) |
| 8 | missing edge guards (`step==0` → `None` unpack; empty-input fits; empty histogram slice) | `stats/preprocess.py:8-17`, `approx/fit.py:106,180` | crash after hours of cache building on sparse classes | no |
| 9 | `n{N}` filenames use requested-not-actual count; sweep skip logic trusts the name | `pobedit_dataset.py:439-448,570,985` | mislabeled provenance; sweeps skip work they never did | metadata only |
| 10 | `cv2.approxPolyDP` missing required `closed` arg → `TypeError` always | `contours/preprocess.py:30` | `get_contours` dead on arrival (not in active path) | no |

## P1 — API standardization (Part 1 items)

| # | change | part-1 § | breaking? |
|---|---|---|---|
| 1 | `extract_angles`/`extract_beams` parity + rename; old names deprecated | §5 | no (aliases) |
| 2 | torchvision-style `classes`/`class_to_idx`; bare names; h5-attr resolution; numeric group sort | §4 | **yes** — `meta.name` loses the `class_` prefix in new parquets |
| 3 | `combra.io` + single parquet loader + `schema_version` | §1, §6 | no |
| 4 | named-tuple results (`LinearFit`, `BimodalGaussFit`, `MveeResult`) | §3 | **yes** — return shapes |
| 5 | explicit `data_range`, guessing deprecated | §2 | **yes** — float callers must declare |
| 6 | logging instead of print | §7 | no |
| 7 | `ValueError` on invalid input instead of print/`None` | §8 | technically yes (was silent) |
| 8 | implement `CLASS_MAP` registry (today docs-only) | §4 | no (additive) |
| 9 | `combra.tests` → `combra.diagnostics` | §1 | **yes** — import path |
| 10 | legend signature harmonization; drop dead `images_amount`/`dist_mean` params | §5 | **yes** — legend text |
| 11 | remove deprecated `PobeditDataset` ctor params | §4 | **yes** |
| 12 | collapse duplicate linear-fit implementations | §3 | internal |

## P2 — hygiene

| # | item | § |
|---|---|---|
| 1 | torch stack → `[metrics]` extra | §10 |
| 2 | CI/`requires-python` alignment (3.10 floor, matrix parity) | §10 |
| 3 | cache lifecycle tooling + preprocessing-version tag; retire legacy `.prep_cache_*` siblings | §9 |
| 4 | provenance fixes (`code_commit` CWD bug, `source_h5_sha`) | §6 |
| 5 | stale `build/` shadow copy, `egg-info`, stale editable metadata | §10 |
| 6 | `areas/plot.py` reads a schema no dataset emits (`contours_series`, `linear_approx_plot`) + magic `pow(1535, 2)` — mark legacy or port to `prep_per_step` | — |
| 7 | doc drift: `docs/api/data.md:163-164` still describes the pre-`prep_per_step` beams layout; example pages present `CLASS_MAP`/`combra_smoke_test` as existing API | — |
| 8 | hardcoded angle-domain constants (`np.arange(0, 361)` fit grid, warm-start `mu=100/240`, unexposed `border_eps=5`) — document or parameterize | §5 |

## Conformance checks

Extend the existing `tests/` suite (no model execution needed):

1. **round-trip** — `extract_angles` on the bundled fetcher images →
   `combra.io.load_angles` → assert `schema_version`, bare class names,
   non-empty `type` for a supplied `types_dict`;
2. **property tests** — `MveeResult.a >= b` on random ellipses; median
   filter border row equals a scipy reference on a constant image; angle
   kernel orientation pinned against a hand-computed square/L-shape;
3. **range contract** — `to_uint8` raises on undeclared float input;
4. **public API** — `test_public_api.SUBMODULES` covers every exported
   submodule (today it omits `tests`, the one with the oddest surface);
5. **reader floor** — CI job reads a freshly written parquet with the
   minimum supported pyarrow.

## Adoption order

Step 0 mirrors {doc}`models_api_proposal` §14: cheap, independent fixes for
things that corrupt data **today**, ahead of any renaming.

0. **Hotfixes** (each standalone, land in any order):
   P0-1 (`types_dict` prefix), P0-2 (pyarrow floor), P0-3
   (`schema_version` + always-write `run_meta`), P0-4 (a/b ordering),
   P0-8 (edge guards). After P0-1/4 land, regenerate the angle/beam
   parquets — every existing file has empty `type`, and beams series may
   be a/b-mixed.
1. **Input & result contracts** (§2, §3) — `data_range`, named tuples,
   x/y-swap decision (P0-5/6/7 ride along, since they change numbers and
   should trigger exactly one regeneration wave together with step 0).
2. **Dataset & class contract** (§4) — torchvision attributes, h5-attr
   resolution, `CLASS_MAP`; coordinated with {doc}`models_api_proposal` §5
   so producer and consumer switch together.
3. **`combra.io` + loaders + extraction parity** (§1, §5, §6).
4. **Logging, errors, deprecation shims** (§7, §8).
5. **Packaging, CI, cache tooling** (§9, §10).
6. **Conformance checks** — last, locking in the finished state.
