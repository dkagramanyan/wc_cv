# Generative models — proposed standard API ("v2 convention")

```{note}
**Status: implemented.** All four repos — san-v2 (v0.2.0), StyleSwin-v2,
DiffiT-v2 and EDM2-v2 — now expose the Part 1 API; the converged current state is
documented in {doc}`models_api`. This page is retained as the specification of
record: **Part 1** specifies the target API every repo exposes; **Part 2** lists
the per-repo deltas that got each there (and the dataset-rebuild rails that remain
before the legacy `CLASS_MAP` warnings can be dropped).
```

The goal: any command, flag, checkpoint name, or generated artifact learned on
one repo transfers to the other three unchanged, and every model's generated
output feeds the wc_cv angle pipeline
(`co_angles/generate_class_samples.py`,
{py:meth}`combra.data.PobeditDataset.generate_angles`) with zero conversion.

The four repos: [san-v2](https://github.com/dkagramanyan/san-v2) (GAN —
StyleGAN3 + Projected GAN + SAN),
[StyleSwin-v2](https://github.com/dkagramanyan/StyleSwin-v2) (GAN — Swin
transformer), [DiffiT-v2](https://github.com/dkagramanyan/DiffiT-v2) (latent
diffusion — transformer, DDPM schedule) and
[EDM2-v2](https://github.com/dkagramanyan/edm2-v2) (latent diffusion — EDM
σ-space U-Net).

---

# Part 1 — The standard API

## 1. Packaging & entry points

Every repo is pip-installable (`pip install -e .`) and exposes the same
console-script family:

| command | purpose | applies to |
|---|---|---|
| `<model>-train` | training | all |
| `<model>-gen-images` | per-class generation | all |
| `<model>-eval` | standalone metrics | all |
| `<model>-prepare-data` | dataset zip builder | all |
| `<model>-download-models` | backbone / weight prefetch | all |
| `<model>-compare-samplers` | sampler-vs-steps sweep | diffusion only |

- **`pyproject.toml` is the only dependency declaration** — there is no
  `requirements.txt`, and `pip install -e .` is the one install path.
  torch / ninja stay out of `pyproject.toml` (installed from the CUDA wheel
  index / conda, as now).
- combra from **one** source everywhere: optional extra `[combra]` →
  `git+https` private repo.
- CUDA story per repo class: JIT-op repos (san-v2, StyleSwin-v2) need `nvcc`
  + `ninja`; pure-torch repos (DiffiT-v2, EDM2-v2) do not.
- **`<model>-prepare-data` is a click group** with a `convert` subcommand
  (EDM2-v2's shape), sharing one transform set everywhere: `center-crop`,
  `center-crop-wide`, `center-crop-dhariwal`. EDM2-v2 additionally keeps its
  `encode` / `decode` subcommands for VAE latents.
- Bulk generation is `<model>-gen-images --save-mode hdf5`; scoring is the
  combra eval and `<model>-eval`.

## 2. Training CLI

```bash
<model>-train --outdir <dir> --cfg <preset> --data <zip> --gpus N --batch-gpu B
```

- **The click CLI is the only interface** — there is no Hydra entry point and
  no `configs/` directory.
- **`--cfg` is the preset flag name everywhere.**
- **Progress is counted in kimg and ticks** (`--kimg`, `--tick`, `--snap`) —
  never raw image counts or `Ki/Mi` suffixes.
- **One batch formula**: `total = batch_gpu × gpus × grad_accum`, with
  `--grad-accum` explicit (default 1). There is no total-batch flag.
- **One precision scheme**: `--precision {fp32,fp16,bf16}` (each repo's
  default documented in its presets; GradScaler used only for fp16), plus
  `--tf32 True/False` (default `True`) and `--bench True/False` (default
  `True`, cuDNN autotune).
- **Boolean flags are `--flag True/False`** (click `type=bool`) — no
  `--x/--no-x` pairs.
- **`--mirror True/False`** (default `False`) means one thing everywhere: a
  stochastic per-item horizontal flip in the **training** loader. Eval and
  combra-reference loaders never flip, and datasets are never flip-doubled.
- Shared optional flags — identical names *and semantics* in all four:

  | group | flags |
  |---|---|
  | run control | `--kimg --tick --snap --seed --desc -n/--dry-run --workers` (workers default 3) |
  | data | `--cond --mirror` |
  | batch / precision | `--grad-accum --precision --tf32 --bench` |
  | checkpointing | `--snapshot-keep-last` |
  | combra eval | `--combra-metrics --num-fid-samples --combra-ref-count` |
  | diffusion eval | `--eval-sampler --eval-sampling-steps` |

- **Progressive-training flags** (model-specific, but following the same
  conventions — kebab-case names, `--flag True/False` booleans):
  - **san-v2** keeps its StyleGAN-XL progressive stack, renamed to
    kebab-case: `--stem`, `--superres`, `--up-factor`, `--head-layers`,
    `--syn-layers`, `--cls-weight`, and `--path-stem <snapshot>` (weights-only
    warm start of the frozen lower-resolution stem).
  - **DiffiT-v2** gets `--init-weights <snapshot>` — a weights-only warm
    start for higher-resolution finetuning (loads EMA weights from a previous
    stage's snapshot, fresh optimizer; replaces the removed `--resume`-based
    flow).
  - **StyleSwin-v2** and **EDM2-v2** train each resolution independently and
    have no progressive flags.
- **Self-spawning multi-GPU**: `--gpus N` spawns one worker per GPU via
  `torch.multiprocessing`; `torchrun` is never required for training.
- **`--seed` means the same thing everywhere**: it seeds weight
  initialisation, data shuffling and the eval-latent draws in all four repos,
  so two runs with the same command and seed produce the same snapshots (up
  to hardware nondeterminism from cuDNN / `torch.compile`). The eval-latent
  and grid-latent draws derive from `--seed` alone — never scaled by the GPU
  count (today the GAN loops seed them with `seed × gpus`), so the same seed
  draws the same latents at any `--gpus`. "Data shuffling" includes the
  distributed sampler: its epoch seed derives from `--seed` (today DiffiT-v2
  never seeds its `DistributedSampler`, so `--seed` does not change
  multi-GPU data order at all). Paired draws come from **one** seeded
  generator: today san-v2 draws eval latents from torch's RNG but the
  paired class labels from numpy's global RNG, so a specific fake batch is
  not reproducible even at a fixed GPU count. Generation-side determinism
  is the §4 seed rule.
- Run directory: `<outdir>/<id:05d>-<cfg>-gpus<G>-batch<B>[-desc]`, where
  `B` is the **total** batch and the name after the id is exactly
  `<cfg>-gpus<G>-batch<B>` — no dataset name spliced in (today san-v2 and
  StyleSwin-v2 embed the dataset basename). A **fresh id is always
  allocated**: existing directories are never reused (today EDM2-v2 and
  san-v2 re-enter a matching directory to resume — san-v2's reuse branch is
  permanently active because `--restart_every` defaults to 999999999, and
  each re-entry truncates `stats.jsonl` **and overwrites
  `training_options.json`**, losing config provenance; DiffiT-v2 re-enters
  a matching directory only when `--resume` is passed — all of that goes
  away with §3). The directory contents are fixed by the §7 log contract.

## 3. Checkpoint contract

Exactly one artifact kind — no resume, no best-model tracking, no separate
final checkpoint:

| artifact | rule |
|---|---|
| `<model>-snapshot-<kimg:06d>-inference.pt` | EMA-only weights; written every snapshot tick **and always at the last tick**, so the newest snapshot *is* the final model; history pruned to `--snapshot-keep-last` (default 3, `0` = keep all) |

- **No resume.** There is no `--resume` flag, no rolling `latest` checkpoint
  and no auto-restart: training runs start-to-finish.
- **Writes are atomic — MUST.** Every snapshot is written to a temp file in
  the run directory and moved into place with `os.replace`, so a snapshot
  that exists under its final name is always complete (today **no repo does
  this fully**: EDM2-v2 is atomic for its `.pt` checkpoints only — the
  per-tick inference `.pkl` is a plain in-place `pickle.dump`; DiffiT-v2
  writes its `-inference.pt` files — exactly what generation loads — in
  place, and its pruning can then delete the last good snapshot while
  keeping a corrupt newest; san-v2 and StyleSwin-v2 stream every checkpoint
  in place, so a walltime kill mid-dump corrupts the very file resume
  depends on).
- **The last tick always snapshots — MUST**, even when `--kimg` is not a
  multiple of the snapshot cadence (today EDM2-v2 never snapshots the final
  partial interval), so the newest snapshot is always the final model.
- **Only EMA weights ever touch disk** — raw (non-EMA) model weights,
  discriminators and optimizer state are never saved.
- **No `best_model.*`.** Pick the best checkpoint post-hoc from `stats.jsonl`
  against the snapshot history (set `--snapshot-keep-last 0` on runs where
  you want the full history to choose from). This depends on the §6 rule
  that combra metrics are mirrored into `stats.jsonl` — today san-v2 and
  StyleSwin-v2 write them to TensorBoard only, so post-hoc selection is
  impossible for their existing runs.
- **EMA stays per-family** (like samplers): classic half-life EMA in the
  GANs, `--ema-rate` in DiffiT-v2, PowerFunctionEMA in EDM2-v2 — the
  algorithms are genuinely different and are documented, not unified.
- **Progressive stages still work**: san-v2's `--path-stem` and DiffiT-v2's
  `--init-weights` are **weights-only warm starts** from a previous stage's
  snapshot (§2) — initialization, not resume.
- **Format: `.pt` state dicts only** — a state dict stores only weight
  tensors keyed by parameter name, so loading rebuilds the model from current
  code instead of unpickling stored classes. No pickled-module saving.
- **No `timm` in artifacts.** Both artifact kinds hold **generator-side
  weights only** — no discriminator and therefore no `timm` feature-network
  modules or weights ever enter a checkpoint. Loading a checkpoint never
  requires `timm` (or any particular `timm` version); `timm` remains a
  train-time-only dependency of the GAN discriminators.
- **Self-describing metadata** in every checkpoint:
  `{n_classes, resolution, class_names, cur_nimg}` — downstream code reads
  grain-class *names* from the checkpoint instead of guessing integer
  conventions (the full label contract is §5).

```{warning}
Runs are unrecoverable by design: a crash or SLURM walltime kill cannot be
resumed. Size `--kimg` (or split stages) so a run fits its job's time limit —
the training sbatch scripts allow 3–4 days. Because there is no resume, the
two MUSTs above are load-bearing — atomic writes guarantee a kill never
corrupts an already-written snapshot, and the last-tick snapshot guarantees a
completed run always ends in a usable model — and both are verified by the
§13 conformance suite.
```

## 4. Generation contract

```bash
<model>-gen-images \
    --network <checkpoint> --outdir <dir> \
    --classes 0,1,4-6 --samples-per-class N \
    --seed 42 --gpus 2 --batch-gpu 32 \
    --save-mode {hdf5,dir} \
    [GAN: --trunc 0.7] \
    [diffusion: --sampler <name> --steps K --cfg-scale S | --guidance G]
```

- One checkpoint flag: **`--network`**.
- **`--batch-gpu` is the only generation batching flag**, and `--gpus N`
  self-spawns per-GPU worker processes — the same launch model as training
  (no `torchrun`, no thread pools).
- `--classes` accepts indices **or class names**
  (`--classes Ultra_Co11,Ultra_Co6_2`) — see the label contract in §5 — and
  every entry is validated against the checkpoint's `n_classes` /
  `class_names` metadata (today san-v2 does no validation: an out-of-range
  index dies with a bare `IndexError` deep in `w_avg` indexing, and an
  in-range but untrained or swapped row generates silently).
- **Determinism rule** in all four: `seed = base + class·samples_per_class + idx`
  — any subset of the output is reproducible in isolation.
- **Identical outputs across repos**:
  - `hdf5` (default): per-rank shards `shards/rank_NNN.h5` in the
    **`RankH5Writer` layout** (`class_<c>/images|seeds`, images stored as
    **uint8 NHWC** — see the §5 normalization contract), merged by rank 0
    into **`<desc>.h5`** — exactly what the wc_cv angle pipeline consumes.
  - `dir`: `class_<c>/idx_<i:06d>_seed_<s>.png`.
- **One h5 signature**: every shard and merged file carries the attributes
  `format = "generated_images_shard"` (one value for all four repos) and
  `schema_version = 1`, so downstream code sniffs any model's output
  identically.
- **The merge hard-fails on incomplete shards.** Every shard records a
  per-sample `written` mask and a `missing_count` attribute; rank 0 refuses
  to produce the merged `<desc>.h5` while any `missing_count` is nonzero
  (today the san-v2 and DiffiT-v2 mergers record `missing_count` but merge
  anyway — and the combra consumer never reads it, so a crashed generation
  run's zero-filled slots are consumed downstream as black images).

## 5. Class-label & dataset contract

The integer label is an implementation detail; the grain-class *name* is the
identity. Two rules:

**Rule 1 — canonical ordering.** The integer label is the index of the class
folder in `sorted()` (alphabetical) order. Every `dataset_tool*.py` derives
labels this way.

```{warning}
**The legacy artifacts do not follow Rule 1.** The on-disk `imagenet_9to4_*`
archives consumed by the real DiffiT-v2 and StyleSwin-v2 runs carry the same
swapped `1, 0, 2` label order as the san-v2 zips (`Ultra_Co11 → 1`,
`Ultra_Co25 → 0`), and every repo takes zip labels **verbatim** at train time
— so the existing checkpoints of all four models most likely share san-v2's
convention, whatever the model pages' class tables say. The zips themselves
record no class names or original filenames, so class identity is
unrecoverable from the artifact — exactly what Rule 2 fixes. Classify every
existing run by the dataset path in its `training_options.json` before
applying any `CLASS_MAP` remap (details: the san-v2 rebuild rails in §12).
```

**Rule 2 — names travel with every artifact.**

| stage | requirement |
|---|---|
| dataset zip | `dataset.json` carries `"class_names": ["Ultra_Co11", "Ultra_Co25", "Ultra_Co6_2"]`, index-aligned, written automatically from the folder names |
| checkpoint | `class_names` copied into every checkpoint (part of the §3 metadata) |
| generated h5 | `class_names` stamped as a root attribute + per-`class_<c>` group attribute |
| generated dir | a `classes.json` manifest next to the `class_<c>/` folders |
| CLI | `--classes` accepts **names as well as indices** |
| downstream | combra matches by **name**; `CLASS_MAP` remains only as the fallback for legacy artifacts that lack `class_names` — note `CLASS_MAP` exists **only in the docs today** (combra's code has just per-call `class_map` dict arguments), so it must first be implemented: see the combra deltas in §12 |

### Dataset item contract

What a dataset yields is part of the API, identical in all four repos:

- **Item = uint8 CHW at the zip's resolution; label = one-hot float32.**
  Any normalization (`[-1,1]`, ImageNet mean/std) happens in the training
  loop, never inside the dataset class.
- **RGB everywhere.** The pipelines are explicitly 3-channel end-to-end:
  grayscale sources are converted **once, at dataset build time**
  (`<model>-prepare-data`); dataset classes and generation writers *assert*
  3 channels instead of silently converting at runtime.
- Horizontal flip is the loader-level `--mirror` augmentation defined in §2 —
  datasets are never flip-doubled.

### Normalization contract

**uint8 `[0, 255]` RGB at every boundary; float only inside the process.**

- On disk and at every artifact boundary the image format is uint8 RGB:
  dataset zips, generated PNGs, the h5 `images` datasets, and every batch
  handed to combra for scoring. Never pass float batches to combra and rely
  on its range inference — an unusual batch (e.g. an all-positive `[-1,1]`
  batch) can be misread; the training loops denormalize to uint8 first.
- The **float training space is per-family** (like EMA and samplers) because
  it is baked into the trained weights, and it stays *inside* the
  training/generation process: `[-1, 1]` for san-v2 and DiffiT-v2, the
  Stable-Diffusion VAE latent space for EDM2-v2, ImageNet mean/std for
  StyleSwin-v2. Each repo normalizes immediately after loading and
  denormalizes with the **exact inverse** immediately before writing or
  scoring — one normalize/denormalize pair per repo, defined in one place,
  asserted to round-trip.

## 6. Evaluation contract

- **In-training combra eval** (all four): every snapshot tick, fakes generated
  **sharded across all ranks**; reference = whole training set — **raw,
  unflipped uint8 dataset pixels, never VAE round-trips** — with features
  precomputed once before the loop; `combra_smoke_test` at startup (one
  shared implementation in `combra.metrics` — today only DiffiT-v2 and
  EDM2-v2 carry private copies, the GANs have none).
- **Uniform knobs**: `--num-fid-samples` (default 10000, `0` disables eval)
  and `--combra-ref-count` (cap the reference side). A capped reference is a
  **seeded random subset** — never the first N: dataset zips are
  class-sorted, so a first-N slice is class-biased (today EDM2-v2 takes the
  first N while its fakes draw classes uniformly — the two sides of the FID
  that drives `best_model.pt` don't even share a class distribution).
- **Uniform keys**: `Metrics/combra_fid10k`, `Metrics/combra_cmmd10k`,
  `Metrics/combra_fd_dinov2_10k`, the angle-density metrics, and
  `Metrics/combra_fid10k_best` — all mirrored to `stats.jsonl` (today san-v2
  and StyleSwin-v2 write `Metrics/combra_*` to TensorBoard **only**: lose the
  tfevents file and the run's entire metric history is gone). The `10k`
  suffix is **literal**: the key names never change with `--num-fid-samples`
  (as in EDM2-v2 today), so a run evaluated at a non-default count carries
  the same keys — its count is recorded only in `training_options.json`.
- `<model>-eval` standalone evaluator in all four.

### How the combra metrics are computed

One eval pass per snapshot tick, identical in all four repos:

1. **Reference (once, before the loop).** Every rank extracts features from
   its deterministic slice of the real training set — **raw dataset pixels
   as uint8, never flip-augmented and never VAE round-trips** (today EDM2-v2
   scores against encoder-decoded reals, which hides the VAE quality gap and
   breaks cross-repo comparability, and StyleSwin-v2's `--mirror`
   flip-doubles the reference) — InceptionV3 features
   (FID), CLIP embeddings (CMMD), DINOv2 features (FD-DINOv2) and pooled
   vertex angles — via combra's split APIs
   ({py:func}`combra.metrics.fid_features`, `cmmd_features`,
   `fd_dinov2_features`, {py:func}`combra.metrics.images_to_pooled_angles`).
   The gathered reference features are cached on rank 0 for the whole run.
2. **Fakes (each tick).** Every rank generates its shard of the
   `--num-fid-samples` fakes from the EMA generator — GANs by a direct
   `G_ema` forward, diffusion models by running `--eval-sampler` for
   `--eval-sampling-steps` in VAE latent space and decoding to pixels — and
   extracts the same features from its own shard.
3. **Distances (rank 0).** Only the small feature rows and pooled-angle
   arrays are gathered; rank 0 computes the Fréchet distances (`fid`,
   `fd_dinov2`), the CLIP MMD (`cmmd`) and the angle-density metrics
   (Wasserstein `w1`/`w2`/`circular_*` + bimodal-Gaussian fit errors) against
   the cached reference
   ({py:func}`combra.metrics.frechet_from_features` + analogues,
   {py:func}`combra.metrics.images_to_pooled_angles`).
4. **Logging.** Results land in TensorBoard as `Metrics/combra_*` and in
   `stats.jsonl`. A metric whose backend is unavailable (e.g. no network for
   DINOv2 weights) records `nan`; a missing combra package prints a startup
   warning and training continues without eval.

Because feature extraction is per-image and angle pooling is concatenation,
the sharded result is **exact** — numerically identical to a single-GPU
{py:func}`combra.metrics.compute_all_metrics` call over the full batches.

## 7. Logging & TensorBoard contract

Every run directory contains the same five artifacts — no more, no fewer.
One console log, one scalar stream, one TensorBoard event file:

| file | contents |
|---|---|
| `training_options.json` | the resolved launch config |
| `<id:05d>-<cfg>-gpus<G>-batch<B>[-desc].log` | rank-0 console transcript, named after the run directory; every line prefixed `[YYYY-MM-DD HH:MM:SS]` |
| `stats.jsonl` | **the machine-readable source of truth**: one JSON line per tick holding every scalar (same keys as the TensorBoard tags) plus `wall_time` / `datetime` columns — **scalar rows only**, no text/log records (today DiffiT-v2's vendored logger interleaves `{"kind": "text", ...}` console records among the scalar rows: archived run 00018 holds 2 607 text lines among 2 395 scalar rows, so any reader must shape-filter first) |
| `events.out.tfevents.*.<id:05d>-<cfg>-gpus<G>-batch<B>[-desc]` | TensorBoard scalars, image grids and text — written by rank 0 only; the run name is appended via `SummaryWriter(filename_suffix=...)` |
| `reals.png`, `fakes_init.png`, `fakes<kimg>.png` | sample grids (see below) |

The event file is written **directly in the run directory** — never in a
`tb/` or `logs/` subfolder — so the TensorBoard run name is exactly the run
directory name (`<id:05d>-<cfg>-gpus<G>-batch<B>[-desc]`). Point TensorBoard
at the parent: `tensorboard --logdir <outdir>` and every run appears under
its training-folder name. The event file also carries the run name as a
**`filename_suffix`**: TensorBoard's writer generates the
`events.out.tfevents.<time>.<host>.<pid>.<n>` prefix and the file cannot be
renamed outright, but the suffix makes a copied event file self-identifying
— the `wc_cv/ml/` analysis folders hold bare tfevents files copied out of
their run dirs, identifiable today only by the folder they were copied
into. The `.log` file carries the same name, so a run's log, its folder,
its event file and its TensorBoard curve are all found by one string.

**What the run log does and does not capture.** The `.log` file starts when
the training process installs its logger — which the trainer must do **first
thing in `main()`**, and immediately write a startup header (torch / CUDA
versions, GPU names, the relevant env vars) so the log is self-sufficient.
Anything printed *before* that is not in it: the `sh/` launch script's own
output (conda activation, env setup) and early failures (import errors, CUDA
init crashes). Under SLURM that output lands in `slurm-<jobid>.out` — keep it
as the debugging fallback for launches that die before the run directory
exists.

**Sample grids follow one scheme** (san-v2's implementation is the
reference): fixed latents seeded once at startup, class-sorted rows for
labeled data, resolution-adaptive grid size; `reals.png` built once from raw
dataset samples, `fakes_init.png` at start, `fakes<kimg>.png` every snapshot
tick, and the same grid logged to TensorBoard under the `Fakes` tag.

TensorBoard tag schema — identical namespaces in all four, with the global
step = `cur_nimg` everywhere, so curves are directly comparable across batch
sizes, GPU counts and repos:

| namespace | contents | cadence |
|---|---|---|
| `Loss/*` | model-family losses (G / D / R1 for the GANs, denoiser loss for diffusion) | every tick |
| `LearningRate/*` | effective learning rates (`G`/`D`, or `lr`) | every tick |
| `Timing/*` | sec/tick, sec/kimg, eval time | every tick |
| `Resources/*` | GPU / CPU memory | every tick |
| `Metrics/combra_*` | the §6 combra metrics | every snapshot tick, step-held between |
| `Fakes` | EMA sample grid (image) | every snapshot tick |

`stats.jsonl` keys are part of the contract, not an implementation detail:
the wc_cv analysis layer reads them directly (e.g.
`combra.metrics.load_fid_by_kimg` — which must itself be updated to the
contract keys `Metrics/combra_fid10k` + `Progress/kimg`: today it parses
the legacy bare `FID` / `kimg` pair, which no standardized run will emit;
see the combra deltas in §12), so renaming a key is a breaking change for
the analysis notebooks.

## 8. Samplers (diffusion models)

Sampler **algorithms stay per-family** — DiffiT-v2's `dpm++/unipc/ddim/ddpm`
(DDPM 1000-step schedule) and EDM2-v2's `dpm++/edm/euler/ddim` (σ-space) are
genuinely different integrators and are *not* unified. Only the **flag names**
are standardized:

| context | flags |
|---|---|
| training-time eval | `--eval-sampler` / `--eval-sampling-steps` |
| generation | `--sampler` / `--steps` |

```{note}
The overlapping names are **not interchangeable**: `ddim` and `dpm++`
integrate different parameterisations of the reverse process in the two
repos, so step counts and quality do not transfer. Calibrate per repo with
its own `<model>-compare-samplers` ({doc}`sampler_comparison`).
```

## 9. Launch scripts

Cluster launches are plain shell scripts — **no `.sbatch` files in the
repos**. Each repo ships the same set under `sh/`:

```
sh/train_256.sh     sh/train_512.sh     sh/train_1024.sh
sh/generate_256.sh  sh/generate_512.sh  sh/generate_1024.sh
```

Each script contains exactly two things:

1. **The environment** — everything a compute node needs, in one place:
   conda activation (env name = repo name), `CUDA_HOME` /
   `TORCH_CUDA_ARCH_LIST`, and the **offline-cluster contract**:
   `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` set in every script, with
   backbones prefetched once on a login node via `<model>-download-models`.
2. **One console-command call** — `<model>-train …` or
   `<model>-gen-images …` with the §2 / §4 flags.

Rules:

- **No hardcoded user homes, `--nodelist`, or account IDs** inside the
  scripts. The repo root is self-located (walk up to `pyproject.toml`);
  SLURM specifics are supplied at submission time:
  `sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_256.sh`.
- The same script runs unmodified on a workstation:
  `bash sh/train_256.sh`.

## 10. Repository infrastructure

- **Tests**: every repo ships `tests/test_smoke.py` (CPU: forward contracts,
  CLI parsing) plus CUDA-op tests where custom kernels exist (san-v2,
  StyleSwin-v2); pytest config lives in `pyproject.toml`
  (`testpaths=["tests"]`).
- **CI**: one identical `ci.yml` in all four — a ruff lint job + the CPU
  smoke tests, one Python version (3.11), modern action pins, all-branch
  push triggers, concurrency cancel, pip cache, deps via
  `pip install -e ".[dev]"`.
- **Lint**: the same ruff block in every `pyproject.toml` — `select=["F","I"]`,
  line-length 120, `combine-as-imports`.
- **Python floor**: `requires-python = ">=3.10"` everywhere; no upper caps.
- **Versioning**: semver in `pyproject.toml` + Keep-a-Changelog with real
  versioned releases (EDM2-v2's current style) — no more dated
  "Unreleased"-only changelogs.
- **`.gitignore`**: one template — `runs/`, `training-runs/`, `datasets/`,
  `generated/`, `logs/`, `*.log`, `__pycache__/`, `*.py[cod]`,
  `.pytest_cache/`, `*.egg-info/`, `build/`, `dist/`.
- **Vendored code**: one `dnnlib/` lineage (the 2024 copy + the timestamp
  patch) shared by the three repos that vendor it. `torch_utils/` is
  intentionally **not** unified — san-v2 genuinely needs its CUDA `ops/`
  tree and EDM2-v2 its distributed helpers.
- **No fork leftovers**: upstream demo images, Microsoft template meta files
  and other inherited clutter are removed (kept: the actual licenses).

---

# Part 2 — Proposed changes in the four repos

## 11. Changes common to all four

- **Checkpoint scheme (§3)**: drop `--resume` / auto-restart, `best_model.*`
  (+ `best_nimg.txt`), the rolling `latest` checkpoint,
  `--save-inference-only`, san-v2's `--save-weights-only` and the final full
  checkpoints (DiffiT-v2's `network-final*.pt`); rename snapshot files to
  `<model>-snapshot-*`; save `.pt` state dicts only and remove the legacy
  `.pkl` loaders — tag the last pickle-capable commit `legacy-pkl` so old
  artifacts stay readable by checking out old code. With resume gone, every
  launch allocates a **fresh run id** — the desc-matching run-dir reuse in
  DiffiT-v2, EDM2-v2 and san-v2 is removed. Dropping resume also removes
  four observed failure modes: the `--save-inference-only`-plus-combra-off
  trap in StyleSwin-v2 / EDM2-v2 (no resumable checkpoint is ever written —
  days of training unrecoverable after a crash), EDM2-v2's `best_fid` reset
  on resume (the first post-resume eval overwrites `best_model.pt` even if
  worse), its bare `assert` when re-launching a completed run, and san-v2's
  exit-code-3 restart machinery (dead today — nothing consumes exit code 3).
- **Training-CLI unification (§2)**: `--precision` / `--tf32` / `--bench`
  replace the per-repo precision flags; every boolean becomes
  `--flag True/False` (no `--x/--no-x` pairs); `--grad-accum` is explicit
  everywhere; `--mirror` becomes the single loader-level flip.
- **Generation (§4)**: `--batch-gpu` + self-spawning `--gpus` in every
  gen script; unified h5 signature (`format = "generated_images_shard"`,
  `schema_version = 1`) and merged filename `<desc>.h5`.
- **Dataset contract (§5)**: items are uint8 + one-hot; normalization moves
  into the training loops; grayscale→RGB conversion happens only at dataset
  build time, with runtime 3-channel asserts.
- **Normalization contract (§5)**: each repo consolidates its
  normalize/denormalize pair in one place with a round-trip assert; every
  artifact and every combra batch crosses the boundary as uint8 (no reliance
  on combra's float-range inference). The per-family float spaces themselves
  are unchanged — they are baked into the trained weights.
- **`<model>-prepare-data` becomes a click group** with `convert` (san-v2,
  StyleSwin-v2 and DiffiT-v2 wrap their single command — breaking for old
  invocations) and the shared transform set including `center-crop-dhariwal`.
- **Grid contract (§7)**: all four adopt the san-v2-style fixed-latent,
  class-sorted, resolution-adaptive grids with `reals.png` / `fakes_init.png`
  / `fakes<kimg>.png`.
- **Log contract (§7)**: both diffusion repos drop the vendored
  OpenAI-baselines logger — EDM2-v2's `progress.csv` / `progress.json`
  (duplicates of `stats.jsonl`) and DiffiT-v2's `progress.csv`. The logger's
  per-rank `log-rankNNN.txt` scheme never engages under the self-spawning
  launch: `mp.spawn` sets no `RANK` env var, so in DiffiT-v2 **every rank
  writes the same `log.txt` / `progress.csv`** — archived run 00018's
  `progress.csv` holds two ranks' rows interleaved (6 values under a
  3-column header). The `.log` transcript becomes rank-0-only (today
  EDM2-v2 appends every rank's stdout into one interleaved `log.txt`), and
  the event file gains the §7 run-name `filename_suffix`.
- **Label contract (§5)**: every `dataset_tool*.py` writes `class_names`;
  gen-images stamps names into h5 / `classes.json`; `--classes` accepts
  names; `class_names` lands in every checkpoint.
- **Infrastructure (§10)**: identical `ci.yml`, ruff block, pytest config,
  `.gitignore` template, `requires-python >= 3.10`, semver + versioned
  changelogs; one shared `dnnlib/` lineage; fork-leftover cleanup.
- **Remove Hydra** (`train_hydra.py`, `configs/`, `hydra-core` dep) from
  DiffiT-v2, EDM2-v2 and san-v2 — no launch script ever calls the Hydra
  path, the configs are single flat files with no groups, and no composition
  or multirun feature is used. StyleSwin-v2 already has none.
- **Remove `requirements.txt`** from DiffiT-v2, EDM2-v2 and san-v2 —
  maintaining two hand-written dependency lists is exactly what produced
  today's drift (stale caps, dead deps, missing entries). StyleSwin-v2
  already has none.
- **Bulk-`.npz` samplers become legacy**: `scripts/sample.py` (DiffiT-v2) and
  `sample_images.py` (EDM2-v2) fall outside the contract — their only
  consumer is the upstream-paper FID protocol, which the WC-Co workflow does
  not use. They may remain in the repos but carry no contract guarantees.
- **Launch scripts (§9)**: every `sbatch/` collection — plus DiffiT-v2's root
  `train_*prod.sbatch` / `generate_*.sbatch` / `run_train.sh` — is replaced
  by the `sh/` set. The hardcoded user homes (`/home/dgkagramanyan`),
  nodelists (`cn-049`…`cn-051`) and account IDs (`proj_1661` / `proj_1793`)
  move to submission-time SLURM options. (Every shipped generation launch is
  already broken today: DiffiT-v2's root scripts invoke a root `train.py` /
  `gen_images.py` that does not exist — the code lives under `scripts/` —
  and its `generate_*.sbatch` passes a nonexistent `--use-ddim`; EDM2-v2's
  README multi-GPU `torchrun` launch spawns N independent single-GPU
  trainers racing one run directory.)

**Breaking everywhere:** interrupted runs can no longer be continued (see the
§3 warning), old `.pkl` artifacts need the `legacy-pkl` tag, and old commands
using the removed or renamed flags fail.

## 12. Per-model deltas

### DiffiT-v2

| change | breaking? |
|---|---|
| `--steps` alias for `--num-sampling-steps` in generation | no |
| `--network` alias for `--model-path` | no |
| add `--init-weights` — weights-only warm start for progressive finetuning (the removed `--resume` covered this) | no (additive) |
| `--precision` replaces `--fp32` + `--amp-dtype`; the flag pairs (`--grad-ckpt/--no-grad-ckpt`, `--tf32/--no-tf32`, `--cache-in-ram/--no-cache-in-ram`) become `--flag True/False` | **yes** — flag renames |
| gen-images: one `--batch-gpu` replaces `--batch-size` / `--batch-sz`; ThreadPool over GPU ids → per-GPU `mp.spawn` with `--gpus N` | **yes** — flag removal |
| dataset returns uint8 + one-hot (normalization moves into the loop); the silent `convert("RGB")` is removed — conversion happens at build time; the hardcoded `random_flip=True` becomes `--mirror` | no (internal + new flag) |
| `--workers` default 4 → 3; merged h5 `generated.h5` → `<desc>.h5`; h5 `format` attr → unified value | no |
| `--combra-ref-count` knob; `--num-fid-samples` governs the combra fake count | no |
| class count & names read from `dataset.json` / `class_names` — replaces the startup probe (first ~50 shuffled batches ≈ 3200 samples) and the dir-mode `filename.split("_")[0]` class heuristic (yields `"Ultra"` for `Ultra_Co11_*.png` names, collapsing every class into one). A probe-missed class maps to −1, which is **out of range** for the label `nn.Embedding` — CPU raises `IndexError`, CUDA device-asserts or reads out-of-bounds memory — and if the probe misses the *largest* raw label the remap table itself is under-sized (`IndexError` on the first batch containing that class). Not, as previously described here, a silent null-token remap: `token_drop` never touches the −1 entries | no (internal) |
| gen-images adopts the §4 **per-image** seed formula — today one generator is seeded per batch (`base + class·10⁶ + first-batch-index`), so outputs depend on `--batch-size` **and the GPU count** (and the 10⁶ stride collides across classes above a million samples per class) | **yes** — generated outputs change |
| seed the `DistributedSampler` from `--seed` (§2) — today it is never seeded, so `--seed` does not change multi-GPU data order; the in-training eval FID also shifts with the GPU count | no (internal) |
| atomic writes for the `-inference.pt` snapshots (§3 MUST) — today only `latest` / `best` / full `final` go through `atomic_save`; the per-tick inference snapshots (exactly what `gen_images` loads) are plain `torch.save`, and `prune_inference_snapshots` can then delete the last good snapshot while keeping a corrupt newest | no (internal) |
| persist `best_fid` / `cur_tick` across restarts — today both reset on resume (`best_fid = inf`, `cur_tick = 0`), so the first post-resume eval overwrites `best_model.pt` even when worse and the snapshot cadence restarts | with §3 (resume and best-model both removed) |
| `stats.jsonl` becomes scalar-rows-only (§7) — today the vendored logger's TB-text mirror interleaves `{"kind": "text"}` records among the scalar rows | with §7 |
| TB tags and `stats.jsonl` keys renamed to the §7 schema (today tags are `Train/<name>` and stats keys are `kimg`, `sec/tick`, `sec/kimg`) | **yes** — key renames |
| combra extra → `git+https` source | no |
| dead-code cleanup: remove the unused `--metrics` train flag and the duplicate `diffit/resample.py` module (functionally identical to `timestep_sampler.py` — only the attribution header differs — and never imported); `scripts/sample.py`'s ImageNet `--num-classes 1000` default noted as legacy | no |

### EDM2-v2

| change | breaking? |
|---|---|
| drop the `--preset` alias — `--cfg` **already exists** (the two are aliases of one option today), it just becomes the only name | **yes** — commands using `--preset` must switch |
| `--duration/--status/--snapshot` (+ `Ki/Mi` parsing and the `--snap` compat shim) → `--kimg/--tick/--snap` | **yes** |
| `--batch` (total) dropped: `--batch-gpu` + explicit `--grad-accum`, like the others | **yes** |
| `--fp16` → `--precision`; `--tf32` gains a flag with default `True` (today TF32 is hardcoded **off**, no flag exists); `--latent/--pixel` → `--latent True/False`; `--bench` keeps its name | **yes** — flags + numerics default |
| add `--desc`, `--workers`, `--mirror` (today: desc auto-generated, `num_workers=2` hardcoded, no flip support) | no |
| `--network` alias for `--net` | no |
| **`edm2-gen-images`: add `--classes` + `--samples-per-class` class-batch mode and `--save-mode hdf5` (RankH5Writer layout)**; `--batch` → `--batch-gpu`; torchrun → self-spawning `--gpus` | **yes** — launch + flags (`--seeds` mode kept) |
| snapshot grids: replace the unshuffled 8×8 scheme with the class-sorted adaptive grid; reals from raw dataset samples (not encoder round-trips) | no |
| combra **reference features from raw dataset pixels** — today the reference is VAE round-tripped reals (`encode`→`decode`), hiding the VAE quality gap and breaking cross-repo FID comparability (§6): EDM2-v2 scores VAE'd reals vs VAE'd fakes while DiffiT-v2 scores raw reals vs VAE-decoded fakes | **yes** — metric values shift |
| `--combra-ref-count` takes a **seeded random slice** (§6) — today it takes the first N of a class-sorted dataset, so the capped reference is class-biased while the fakes draw classes uniformly: the two sides of the FID (and the best-model selection it drives) don't even share a class distribution | **yes** — metric values shift |
| fresh run id on every launch — the desc-matching run-dir reuse (silent auto-resume) goes away | with §3 |
| resume-machinery bugs that §3 removes outright: `best_fid` resets to `inf` on resume, so the first post-resume eval overwrites `best_model.pt` even if worse; re-launching a completed run — the documented resume method — dies on a bare `assert` | with §3 |
| the final partial interval snapshots (§3's last-tick MUST) — today no snapshot is written after the last full `--snap` interval when `--kimg` is not a multiple of the cadence | no |
| the atomic-write MUST extends to the inference artifacts — today `CheckpointIO` is atomic for `.pt` only; the per-tick inference `.pkl` is a plain in-place `pickle.dump` | no (internal) |
| `dataset_tool.py encode` / `decode` chain repaired or declared legacy — today `decode` filters its `--source` through `is_image_ext`, which rejects the `.npy` latents `encode` writes, so the two subcommands cannot chain | no |
| remove the dead suspend/stop stubs — `should_stop` / `should_suspend` / `update_progress` are no-ops, so the loop's graceful-stop block is unreachable and the only early exit is a kill (which the non-atomic `.pkl` writes make hazardous) | no (dead code) |
| fix the startup error hint that references the nonexistent `--no-combra-metrics` flag (the real spelling is `--combra-metrics False`) | no |
| drop the README / comment references to `network-snapshot-final.pkl` (nothing ever writes it) and the README's multi-GPU `torchrun` example (it spawns N independent single-GPU trainers racing one run dir — §2's self-spawning `--gpus` is the launch model) | no (docs) |
| inference snapshots keep the EMA-std suffix: `edm2-snapshot-<kimg>[-<ema_std>]-inference.pt` | no (naming detail) |
| untrack the committed `.hydra/` dir and `train_hydra.log`; adopt the full `.gitignore` | no |
| delete the committed `docs/*-help.txt` dumps (already stale once) — help text lives only in the CLI | no |

### StyleSwin-v2

| change | breaking? |
|---|---|
| console scripts `styleswin-train / -gen-images / -eval / -prepare-data` | no |
| add `--precision` (the repo currently trains pure fp32 — fp16/bf16 via autocast is new capability) | no (additive) |
| `--use-flip` merged into `--mirror`, and the flip becomes the §2 loader-level augmentation — today the two flags are OR-ed into dataset x-flip **doubling**, which also flip-doubles the combra reference (the reference is the same dataset object) | **yes** — flag removed + train/eval behavior change |
| combra metrics mirrored into `stats.jsonl` — today they are **TensorBoard-only**, so `stats.jsonl` never contains `Metrics/combra_*` and the §3 post-hoc best-snapshot selection is impossible | no (additive) |
| clear `stats_metrics` when an eval tick fails — today the dict persists across ticks, so a failed combra eval re-logs the *previous* tick's values at the new step and best-model tracking compares against a stale FID | no (bugfix) |
| resume stops silently discarding state — today D and both optimizers load inside a bare `except` that just prints "We don't load D." (a shape mismatch silently reinitializes them mid-run) while the unguarded G load hard-crashes | with §3 (resume removed) |
| checkpoint metadata gains the §3 keys **plus the architecture hyperparameters** — today `gen_images.py` hardcodes `style_dim=512` / `n_mlp=8` etc., so a checkpoint from a non-default architecture loads wrong or fails | no (additive) |
| §3 removes the `--save-inference-only` trap: with the flag set and combra eval off (or failed), **no resumable checkpoint is ever written** — a crash loses the whole run; resuming from an inference snapshot dies with `KeyError: 'g'` | with §3 |
| §5 build-time grayscale→RGB + runtime 3-channel asserts — today a grayscale zip does **not** crash (as previously described here): the `(1,3,1,1)` ImageNet mean/std broadcasts the 1-channel batch to 3 channels with *different* stats per channel — a silently tinted image trained as RGB — while the combra reference fails on the 1-channel array and eval is silently disabled | with §5 |
| `dataset_tool.py` derives labels alphabetically and writes `class_names` — today it copies labels verbatim from the source `dataset.json` (only `dataset_tool_for_imagenet.py` sorts) | no for new zips |
| run-dir desc uses `--cfg` (today it splices in the dataset basename: `styleswin-<dataset>-gpusN-batchB`) | no (naming) |
| **`gen_images.py`: add `--save-mode hdf5` (RankH5Writer layout)** | no (additive) |
| add `styleswin-eval` standalone evaluator + startup `combra_smoke_test` | no |
| `--num-fid-samples` / `--combra-ref-count` knobs (replacing fixed 10 000) | no (defaults unchanged) |
| snapshot grids: add `reals.png` + `fakes_init.png` and the class-sorted adaptive grid (today: 16-sample square, no reals); grid latents become fixed unconditionally (today they are fixed only when combra eval is active) | no |
| add `tests/` + `ci.yml` + ruff + pytest config (today: none of the four exist) | no |
| full `.gitignore` (today: one line); remove the Microsoft template meta files (`CODE_OF_CONDUCT/SECURITY/SUPPORT`) and upstream demo `imgs/` | no |
| `requires-python`: drop the `<3.14` cap, floor `>=3.10` | no |
| dead-flag cleanup: remove the reserved `--metrics` stub and the never-consumed `restart_every` config key; remove the legacy upstream `train_styleswin.py` entry point (unconditional, torchvision-transform pipeline, unpruned per-iteration checkpoints — superseded by the `train.py` harness) | no |

### san-v2 — dataset rebuild + format change

| change | breaking? |
|---|---|
| saves become `.pt` state dicts (pickled-module saving removed; `legacy.py` deleted) | **yes** — old `.pkl` artifacts readable only via the `legacy-pkl` tag |
| `--fp32` → `--precision`; `--nobench` → `--bench` (positive sense) | **yes** — flag renames |
| `--mirror` becomes the loader-level random flip (was dataset-doubling xflip) | **yes** — training-behavior change |
| `--restart_every` + the exit-code-3 auto-restart protocol removed (falls out of the §3 no-resume scheme) — the protocol is already dead: `exit(3)` is emitted but nothing anywhere requeues on it, while the `--restart_every` default (999999999) keeps the desc-matching run-dir reuse permanently active, and each re-entry **truncates `stats.jsonl`** (opened `'wt'`) so the three log sinks disagree | with §3 |
| combra metrics mirrored into `stats.jsonl` (§6) — today TensorBoard-only, like StyleSwin-v2: lose the tfevents file and the run's FID/CMMD history is gone, and §3's post-hoc best-snapshot selection is impossible | no (additive) |
| atomic snapshot writes (tmp + `os.replace`, §3) — today `latest.pt` / `best_model.pkl` are streamed in place, so a walltime kill mid-dump corrupts the file resume depends on | no (internal) |
| progressive flags renamed to kebab-case: `--up_factor`→`--up-factor`, `--path_stem`→`--path-stem`, `--syn_layers`→`--syn-layers`, `--head_layers`→`--head-layers`, `--cls_weight`→`--cls-weight` | **yes** — flag renames |
| `dataset_tool.py` derives labels alphabetically (stops copying them verbatim from the source `dataset.json`) | no for new zips |
| `dataset_tool.py` errors on missing labels — today a single unlabeled image silently drops the **entire** label set (`labels if all(...) else None`), converting the dataset to unconditional with no warning | no (bugfix) |
| class count read from `dataset.json` `class_names` — today `label_shape` is `max(label)+1`, so a split that happens to miss the highest class silently shrinks `c_dim` and shifts every class index | no (internal) |
| re-enable a DDP weight-consistency check before saving — today the stock `check_ddp_consistency` block is commented out, so silently diverged ranks save rank 0's weights undetected | no (internal) |
| `--num-fid-samples` / `--combra-ref-count` knobs | no |
| implement `--snapshot-keep-last` pruning — today **no pruning exists** in san-v2: snapshots accumulate unbounded (DiffiT-v2's archived run 00018, 48 unpruned snapshots, shows what that looks like in practice) | no (additive) |
| run-dir desc drops the dataset name (today `<cfg>-<dataset>-gpusN-batchB`) to match the §2 pattern | no (naming) |
| gen-images: merged file becomes `<desc>.h5` in `--outdir` — today it is `<network-stem>_trunc_<t>[-desc].h5` inside a self-created numbered run dir | **yes** — output paths change |
| add the startup `combra_smoke_test` (today only a try/except around the reference precompute) | no (additive) |
| one normalize/denormalize pair with a round-trip assert (§5) — today two denorm formulas coexist: `rint(x·127.5+128)` in the eval path vs `(x+1)·255/2 = x·127.5+127.5` in `gen_utils`, a constant 0.5 apart. Worse, inside the eval itself the **reals** enter as raw uint8 while the **fakes** get the `+128` formula, so every combra FID/CMMD/angle value san-v2 has ever reported carries a systematic real-vs-fake intensity offset — and on-disk generated images are not byte-identical to what combra scored during training | **yes** — metric values shift |
| the native `--metrics` registry (`fid50k_full`, …) and its per-metric jsonl files fall outside the contract — combra is the metric path, and the per-metric jsonls violate the §7 five-artifact rule | no (legacy) |
| dead-code cleanup: unused `training/networks_stylegan3.py` (only the `_resetting` variant is imported), the misnamed `Dataset.has_onehot_labels` (returns `True` for **integer** labels), the dead `set_classes` API (with its one-hot dtype bug), `debug.txt` written unguarded by every rank, and the near-duplicate `dataset_tool_for_imagenet.py` | no |
| combra install via `[combra]` extra (`git+https`) — replaces the editable `-e ../wc_cv/combra`; pyproject.toml cleaned of the dead GUI deps | no |
| CI gains a ruff job + all-branch triggers; ruff config added; `requires-python` `>=3.11` → `>=3.10` | no |
| h5 `format` attr `"stylegan_generated_images_shard"` → unified value | no (old readers via `legacy-pkl` tag era) |
| **rebuild the `imagenet_9to4_*` dataset zips with alphabetical labels + `class_names`** | **yes** — see below |

**Dataset rebuild safety rails**, because a trained model's class embedding
is welded to its training-time indices:

- **The swap is not san-v2-only.** The on-disk `imagenet_9to4_*` zips
  (`datasets/preprocessed/`) carry the **same swapped 1, 0, 2 label order**
  as the san zips (`Ultra_Co11→1`, `Ultra_Co25→0` — verified in their
  `dataset.json`), and all four repos take zip labels **verbatim** at train
  time. The "alphabetical" convention attributed to StyleSwin-v2 / DiffiT-v2
  / EDM2-v2 holds only for zips their own tools built from class folders —
  so **every existing run must be audited** via the dataset path in its
  `training_options.json` before assuming either convention.
- **The 16×16 stem zip is unlabeled**:
  `datasets/san/o_bc_left_4x_1536_1024x1024_rgb_16x16.zip` contains no
  `dataset.json` at all, so a conditional stem trained on it silently loses
  class conditioning (compounded by `dataset_tool`'s silent label-drop
  above). Rebuild it with labels + `class_names` along with the rest.
- Rebuilt zips are for **new runs only** — never finetune a pre-rebuild
  checkpoint against a rebuilt zip: the label semantics of classes 0 and 1
  silently swap.
- Pre-rebuild checkpoints and everything generated from them stay under the
  legacy `CLASS_MAP` until retrained.
- Once san-v2 is retrained on the rebuilt zips, all four models share one
  convention and the class-map warnings on the model pages become historical
  notes about legacy artifacts.

```{note}
The combra-library changes these contracts require of the downstream consumer
(the `CLASS_MAP` registry, `combra_smoke_test`, the `load_fid_by_kimg` contract
keys, the `PobeditDataset` h5-attribute / `written` / `missing_count`
validation, the single gray-conversion path, the prep-cache version tag, the
comparison class-coverage check and the strict-uint8 input mode) have been
**implemented** in combra and are documented in the {doc}`API reference
<../api/data>`. The two remaining wc_cv-repo consumer edits —
`co_angles/generate_class_samples.py`'s checkpoint glob and `ml/logs.ipynb`'s
tfevents paths — are tracked with the producer work.
```

## 13. Conformance checks

A small conformance suite in wc_cv keeps the convention from drifting again
(today's `--use-ddim` / `network-snapshot-final.pkl` / inverted-flag rot all
grew silently). No model execution needed:

1. **CLI contract** — parse `<model>-train --help` and
   `<model>-gen-images --help` per repo; assert the §2/§4 flags exist with the
   specified defaults.
2. **Run-dir contract** — `--dry-run` a training launch; assert the resolved
   option names and the run-dir naming pattern.
3. **Artifact contract** — validate a sample `.h5` against the RankH5Writer
   schema (`class_<c>/images|seeds` + the §4 `format`/`schema_version`
   attrs), assert its `written` mask is complete and `missing_count` is
   zero (the §4 merge hard-fail), and validate a checkpoint against the §3
   metadata keys; assert the snapshot writer goes through tmp +
   `os.replace`, so a `.pt` present under its final name is always complete
   (the §3 atomic-write MUST); assert the repo's normalize/denormalize pair
   round-trips exactly (§5).
4. **Label contract** — assert `class_names` present and index-aligned in
   `dataset.json`, in checkpoint metadata and in a generated h5; assert the
   alphabetical ordering rule on a fresh dataset build (§5).
5. **Launch scripts** — grep `sh/` for hardcoded home paths, `--nodelist`
   and account IDs (must be none, §9).
6. **Infrastructure** — assert `ci.yml`, the ruff block, pytest config and
   the `.gitignore` template are present and identical to the reference
   (§10).

Wired into each repo's CI next to the existing smoke tests.

## 14. Adoption order

Step 0 comes first because its items are cheap, independently landable, and
fix things that corrupt data or mislead science **today**; the contracts
follow.

0. **Hotfixes ahead of the contracts**:
   - *Label audit + docs hotfix* — classify every existing run by the
     dataset path in its `training_options.json` and correct the class
     tables on the model pages (the §5 warning): zero code risk, and the
     tables are actively misleading today.
   - *DiffiT-v2 reads class count & names from `dataset.json`* (§12) — ends
     the silent null-token mistraining risk on the very next run.
   - *Mirror `Metrics/combra_*` into `stats.jsonl`* in san-v2 and
     StyleSwin-v2 (§6) — protects the primary quality signal from tfevents
     loss and is the prerequisite for §3's post-hoc best-snapshot selection.
   - *Fix DiffiT-v2's rank-colliding logging* (or land the §7
     vendored-logger removal early) — until then every multi-GPU run
     corrupts its own `log.txt` / `progress.csv`.
   - *san-v2 denorm hotfix* — change the eval-path `+128` to `+127.5`
     (§12): every combra metric today compares `+128`-denormalized fakes
     against raw-uint8 reals; a one-line fix, though it shifts the metric
     history.
   - *Atomic-write helper in all four repos* — hoisted out of §3 because it
     is independent of the checkpoint redesign: ~10 lines of tmp +
     `os.replace` per repo; today san-v2 and StyleSwin-v2 stream every
     checkpoint in place, and DiffiT-v2 / EDM2-v2 leave exactly their
     inference artifacts unprotected.
   - *combra consumes only complete h5 slots* — read `written` /
     `missing_count` (§12); until then a crashed generation run feeds black
     images into the angle pipeline.
   - *Rebuild the unlabeled 16×16 stem zip and make san-v2's
     `dataset_tool` error on missing labels* (§12) — a conditional stem run
     today silently trains unconditional.
   - *Persist `best_fid` on resume in DiffiT-v2 and EDM2-v2* (or skip
     best-model overwrites after resume) — until §3 lands, every resumed
     run risks overwriting its true best checkpoint.
1. **Generation contract (§4)** — the only change with direct scientific
   payoff: EDM2-v2 and StyleSwin-v2 outputs become consumable by the wc_cv
   angle pipeline; DiffiT-v2's per-image seeds end batch-size/GPU-dependent
   outputs.
2. **Class-label & dataset contract (§5)** — dataset tools write
   `class_names`, RGB fixed at build time, the san-v2 zips are rebuilt
   (after the step-0 audit, so legacy artifacts are classified first);
   paired with (1) so every new artifact is self-describing from day one.
3. **Checkpoint scheme (§3)** — all four repos: EMA-snapshot-only saving as
   `.pt` state dicts with the atomic-write and last-tick MUSTs; resume /
   best-model machinery and pickle loaders removed (ends the flag
   divergence, the `timm` unpickling fragility, and the four resume-era
   failure modes listed in §11).
4. **Training-CLI unification (§2)** — progress units, batch formula,
   precision scheme, boolean style, `--mirror` (the edm2 flag migration is
   the bulk of it).
5. **StyleSwin-v2 parity kit (§12)**.
6. **Aliases, eval knobs, the log & grid contracts and launch scripts
   (§2, §6, §7, §9)**.
7. **Repository infrastructure (§10)** — CI/lint/gitignore/versioning
   parity, `dnnlib` sync, leftover cleanup.
8. **Conformance CI (§13)** — last, so it locks in the finished state.
