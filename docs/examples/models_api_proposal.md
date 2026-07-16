# Generative models — proposed standard API ("v2 convention")

```{note}
**Status: proposal.** Nothing here is implemented yet — the **current** state
of each repo (including where they diverge) is documented in
{doc}`models_api`. This page has two parts: **Part 1** specifies the target
API every repo exposes; **Part 2** lists the changes each repo needs to get
there.
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
  to hardware nondeterminism from cuDNN / `torch.compile`). Generation-side
  determinism is the §4 seed rule.
- Run directory: `<outdir>/<id:05d>-<cfg>-gpus<G>-batch<B>[-desc]`; its
  contents are fixed by the §7 log contract.

## 3. Checkpoint contract

Exactly one artifact kind — no resume, no best-model tracking, no separate
final checkpoint:

| artifact | rule |
|---|---|
| `<model>-snapshot-<kimg:06d>-inference.pt` | EMA-only weights; written every snapshot tick **and always at the last tick**, so the newest snapshot *is* the final model; history pruned to `--snapshot-keep-last` (default 3, `0` = keep all) |

- **No resume.** There is no `--resume` flag, no rolling `latest` checkpoint
  and no auto-restart: training runs start-to-finish.
- **Only EMA weights ever touch disk** — raw (non-EMA) model weights,
  discriminators and optimizer state are never saved.
- **No `best_model.*`.** Pick the best checkpoint post-hoc from `stats.jsonl`
  against the snapshot history (set `--snapshot-keep-last 0` on runs where
  you want the full history to choose from).
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
the training sbatch scripts allow 3–4 days.
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
  (`--classes Ultra_Co11,Ultra_Co6_2`) — see the label contract in §5.
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

## 5. Class-label & dataset contract

The integer label is an implementation detail; the grain-class *name* is the
identity. Two rules:

**Rule 1 — canonical ordering.** The integer label is the index of the class
folder in `sorted()` (alphabetical) order. Every `dataset_tool*.py` derives
labels this way.

**Rule 2 — names travel with every artifact.**

| stage | requirement |
|---|---|
| dataset zip | `dataset.json` carries `"class_names": ["Ultra_Co11", "Ultra_Co25", "Ultra_Co6_2"]`, index-aligned, written automatically from the folder names |
| checkpoint | `class_names` copied into every checkpoint (part of the §3 metadata) |
| generated h5 | `class_names` stamped as a root attribute + per-`class_<c>` group attribute |
| generated dir | a `classes.json` manifest next to the `class_<c>/` folders |
| CLI | `--classes` accepts **names as well as indices** |
| downstream | combra matches by **name**; `CLASS_MAP` remains only as the fallback for legacy artifacts that lack `class_names` |

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
  **sharded across all ranks**; reference = whole training set with features
  precomputed once before the loop; `combra_smoke_test` at startup.
- **Uniform knobs**: `--num-fid-samples` (default 10000, `0` disables eval)
  and `--combra-ref-count` (cap the reference side).
- **Uniform keys**: `Metrics/combra_fid10k`, `Metrics/combra_cmmd10k`,
  `Metrics/combra_fd_dinov2_10k`, the angle-density metrics, and
  `Metrics/combra_fid10k_best` — all mirrored to `stats.jsonl`.
- `<model>-eval` standalone evaluator in all four.

### How the combra metrics are computed

One eval pass per snapshot tick, identical in all four repos:

1. **Reference (once, before the loop).** Every rank extracts features from
   its deterministic slice of the real training set — InceptionV3 features
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
   ({py:func}`combra.metrics.fid_from_features` + analogues,
   `angle_density_metrics_from_pooled`).
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
| `stats.jsonl` | **the machine-readable source of truth**: one JSON line per tick holding every scalar (same keys as the TensorBoard tags) plus `wall_time` / `datetime` columns |
| `events.out.tfevents.*` | TensorBoard scalars, image grids and text — written by rank 0 only |
| `reals.png`, `fakes_init.png`, `fakes<kimg>.png` | sample grids (see below) |

The event file is written **directly in the run directory** — never in a
`tb/` or `logs/` subfolder — so the TensorBoard run name is exactly the run
directory name (`<id:05d>-<cfg>-gpus<G>-batch<B>[-desc]`). Point TensorBoard
at the parent: `tensorboard --logdir <outdir>` and every run appears under
its training-folder name. The `.log` file carries the same name, so a run's
log, its folder and its TensorBoard curve are all found by one string.

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
`combra.metrics.load_fid_by_kimg` over `Metrics/combra_fid10k`), so renaming
a key is a breaking change for the analysis notebooks.

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

- **Checkpoint scheme (§3)**: drop `--resume` / auto-restart, `best_model.*`,
  the rolling `latest` checkpoint, `--save-inference-only` and the final full
  checkpoints (DiffiT-v2's `network-final*.pt`); rename snapshot files to
  `<model>-snapshot-*`; save `.pt` state dicts only and remove the legacy
  `.pkl` loaders — tag the last pickle-capable commit `legacy-pkl` so old
  artifacts stay readable by checking out old code.
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
- **Log contract (§7)**: EDM2-v2 drops `progress.csv` / `progress.json`
  (duplicates of `stats.jsonl`) and its per-rank `log-rankNNN.txt`; DiffiT-v2
  drops its extra OpenAI-baselines logger formats.
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
  nodelists (`cn-049`…`cn-051`) and account IDs (`proj_1631` / `proj_1661`)
  move to submission-time SLURM options.

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
| combra extra → `git+https` source | no |
| dead-code cleanup: remove the unused `--metrics` train flag and the duplicate `diffit/resample.py` module | no |

### EDM2-v2

| change | breaking? |
|---|---|
| rename `--preset` → `--cfg` (no alias kept) | **yes** — existing commands must switch |
| `--duration/--status/--snapshot` (+ `Ki/Mi` parsing and the `--snap` compat shim) → `--kimg/--tick/--snap` | **yes** |
| `--batch` (total) dropped: `--batch-gpu` + explicit `--grad-accum`, like the others | **yes** |
| `--fp16` → `--precision`; TF32 default flips off → on; `--latent/--pixel` → `--latent True/False`; `--bench` keeps its name | **yes** — flags + numerics default |
| add `--desc`, `--workers`, `--mirror` (today: desc auto-generated, `num_workers=2` hardcoded, no flip support) | no |
| `--network` alias for `--net` | no |
| **`edm2-gen-images`: add `--classes` + `--samples-per-class` class-batch mode and `--save-mode hdf5` (RankH5Writer layout)**; `--batch` → `--batch-gpu`; torchrun → self-spawning `--gpus` | **yes** — launch + flags (`--seeds` mode kept) |
| snapshot grids: replace the unshuffled 8×8 scheme with the class-sorted adaptive grid; reals from raw dataset samples (not encoder round-trips) | no |
| inference snapshots keep the EMA-std suffix: `edm2-snapshot-<kimg>[-<ema_std>]-inference.pt` | no (naming detail) |
| untrack the committed `.hydra/` dir and `train_hydra.log`; adopt the full `.gitignore` | no |
| delete the committed `docs/*-help.txt` dumps (already stale once) — help text lives only in the CLI | no |

### StyleSwin-v2

| change | breaking? |
|---|---|
| console scripts `styleswin-train / -gen-images / -eval / -prepare-data` | no |
| add `--precision` (the repo currently trains pure fp32 — fp16/bf16 via autocast is new capability) | no (additive) |
| `--use-flip` merged into `--mirror` (one flip flag) | **yes** — flag removed |
| **`gen_images.py`: add `--save-mode hdf5` (RankH5Writer layout)** | no (additive) |
| add `styleswin-eval` standalone evaluator + startup `combra_smoke_test` | no |
| `--num-fid-samples` / `--combra-ref-count` knobs (replacing fixed 10 000) | no (defaults unchanged) |
| snapshot grids: add `reals.png` + `fakes_init.png` and the class-sorted adaptive grid (today: 16-sample square, no reals) | no |
| add `tests/` + `ci.yml` + ruff + pytest config (today: none of the four exist) | no |
| full `.gitignore` (today: one line); remove the Microsoft template meta files (`CODE_OF_CONDUCT/SECURITY/SUPPORT`) and upstream demo `imgs/` | no |
| `requires-python`: drop the `<3.14` cap, floor `>=3.10` | no |
| dead-flag cleanup: remove the reserved `--metrics` stub | no |

### san-v2 — dataset rebuild + format change

| change | breaking? |
|---|---|
| saves become `.pt` state dicts (pickled-module saving removed; `legacy.py` deleted) | **yes** — old `.pkl` artifacts readable only via the `legacy-pkl` tag |
| `--fp32` → `--precision`; `--nobench` → `--bench` (positive sense) | **yes** — flag renames |
| `--mirror` becomes the loader-level random flip (was dataset-doubling xflip) | **yes** — training-behavior change |
| `--restart_every` + the exit-code-3 auto-restart protocol removed (falls out of the §3 no-resume scheme) | with §3 |
| progressive flags renamed to kebab-case: `--up_factor`→`--up-factor`, `--path_stem`→`--path-stem`, `--syn_layers`→`--syn-layers`, `--head_layers`→`--head-layers`, `--cls_weight`→`--cls-weight` | **yes** — flag renames |
| `dataset_tool.py` derives labels alphabetically (stops copying them verbatim from the source `dataset.json`) | no for new zips |
| `--num-fid-samples` / `--combra-ref-count` knobs | no |
| combra install via `[combra]` extra (`git+https`) — replaces the editable `-e ../wc_cv/combra`; pyproject.toml cleaned of the dead GUI deps | no |
| CI gains a ruff job + all-branch triggers; ruff config added; `requires-python` `>=3.11` → `>=3.10` | no |
| h5 `format` attr `"stylegan_generated_images_shard"` → unified value | no (old readers via `legacy-pkl` tag era) |
| **rebuild the `imagenet_9to4_*` dataset zips with alphabetical labels + `class_names`** | **yes** — see below |

**Dataset rebuild safety rails**, because a trained model's class embedding
is welded to its training-time indices:

- Rebuilt zips are for **new runs only** — never finetune a pre-rebuild
  checkpoint against a rebuilt zip: the label semantics of classes 0 and 1
  silently swap.
- Pre-rebuild checkpoints and everything generated from them stay under the
  legacy `CLASS_MAP` until retrained.
- Once san-v2 is retrained on the rebuilt zips, all four models share one
  convention and the class-map warnings on the model pages become historical
  notes about legacy artifacts.

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
   attrs) and a checkpoint against the §3 metadata keys.
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

1. **Generation contract (§4)** — the only change with direct scientific
   payoff: EDM2-v2 and StyleSwin-v2 outputs become consumable by the wc_cv
   angle pipeline.
2. **Class-label & dataset contract (§5)** — dataset tools write
   `class_names`, RGB fixed at build time, the san-v2 zips are rebuilt;
   paired with (1) so every new artifact is self-describing from day one.
3. **Checkpoint scheme (§3)** — all four repos: EMA-snapshot-only saving as
   `.pt` state dicts; resume / best-model machinery and pickle loaders
   removed (ends the flag divergence and the `timm` unpickling fragility).
4. **Training-CLI unification (§2)** — progress units, batch formula,
   precision scheme, boolean style, `--mirror` (the edm2 flag migration is
   the bulk of it).
5. **StyleSwin-v2 parity kit (§12)**.
6. **Aliases, eval knobs, the log & grid contracts and launch scripts
   (§2, §6, §7, §9)**.
7. **Repository infrastructure (§10)** — CI/lint/gitignore/versioning
   parity, `dnnlib` sync, leftover cleanup.
8. **Conformance CI (§13)** — last, so it locks in the finished state.
