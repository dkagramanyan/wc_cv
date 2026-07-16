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
- Bulk generation is `<model>-gen-images --save-mode hdf5`; scoring is the
  combra eval and `<model>-eval`.

## 2. Training CLI

```bash
<model>-train --outdir <dir> --cfg <preset> --data <zip> --gpus N --batch-gpu B
```

- **The click CLI is the only interface** — there is no Hydra entry point and
  no `configs/` directory.
- **`--cfg` is the preset flag name everywhere.**
- Shared optional flags — identical names *and semantics* in all four:

  | group | flags |
  |---|---|
  | run control | `--kimg --tick --snap --seed --desc -n/--dry-run --workers` |
  | data | `--cond --mirror` |
  | checkpointing | `--snapshot-keep-last` |
  | combra eval | `--combra-metrics --num-fid-samples --combra-ref-count` |
  | diffusion eval | `--eval-sampler --eval-sampling-steps` |

- **Self-spawning multi-GPU**: `--gpus N` spawns one worker per GPU via
  `torch.multiprocessing`; `torchrun` is never required for training.
- Run directory: `<outdir>/<id:05d>-<cfg>-gpus<G>-batch<B>[-desc]`; its
  contents are fixed by the §7 log contract.

## 3. Checkpoint contract

Exactly two artifact kinds — no resume, no best-model tracking:

| artifact | rule |
|---|---|
| `<model>-snapshot-<kimg:06d>-inference.pt` | EMA-only weights; written every snapshot tick; history pruned to `--snapshot-keep-last` (default 3, `0` = keep all) |
| `<model>-final.pt` | model + EMA weights (no optimizer state), written once at the end of training |

- **No resume.** There is no `--resume` flag, no rolling `latest` checkpoint
  and no auto-restart: training runs start-to-finish.
- **No `best_model.*`.** Pick the best checkpoint post-hoc from `stats.jsonl`
  against the snapshot history (set `--snapshot-keep-last 0` on runs where
  you want the full history to choose from).
- **Progressive stages still work**: san-v2's `--path_stem` and DiffiT's
  higher-resolution finetuning are **weights-only warm starts** from a
  previous stage's snapshot — initialization, not resume.
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
- `--classes` accepts indices **or class names**
  (`--classes Ultra_Co11,Ultra_Co6_2`) — see the label contract in §5.
- **Determinism rule** in all four: `seed = base + class·samples_per_class + idx`
  — any subset of the output is reproducible in isolation.
- **Identical outputs across repos**:
  - `hdf5` (default): per-rank shards `shards/rank_NNN.h5` in the
    **`RankH5Writer` layout** (`class_<c>/images|seeds`), merged by rank 0 —
    exactly what the wc_cv angle pipeline consumes.
  - `dir`: `class_<c>/idx_<i:06d>_seed_<s>.png`.

## 5. Class-label contract

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
| `log.txt` | full rank-0 console transcript, every line prefixed `[YYYY-MM-DD HH:MM:SS]` |
| `stats.jsonl` | **the machine-readable source of truth**: one JSON line per tick holding every scalar (same keys as the TensorBoard tags) plus `wall_time` / `datetime` columns |
| `events.out.tfevents.*` | TensorBoard scalars, image grids and text — written by rank 0 only |
| `reals.png`, `fakes<kimg>.png` | sample grids: reals once at start, fakes every snapshot tick |

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

---

# Part 2 — Proposed changes in the four repos

## 9. Changes common to all four

- **Checkpoint scheme (§3)**: drop `--resume` / auto-restart, `best_model.*`,
  the rolling `latest` checkpoint and `--save-inference-only`; rename
  snapshot files to `<model>-snapshot-*`; save `.pt` state dicts only and
  remove the legacy `.pkl` loaders — tag the last pickle-capable commit
  `legacy-pkl` so old artifacts stay readable by checking out old code.
- **Log contract (§7)**: EDM2-v2 drops `progress.csv` / `progress.json`
  (duplicates of `stats.jsonl`) and its per-rank `log-rankNNN.txt`; DiffiT-v2
  drops its extra OpenAI-baselines logger formats.
- **Label contract (§5)**: every `dataset_tool*.py` writes `class_names`;
  gen-images stamps names into h5 / `classes.json`; `--classes` accepts
  names; `class_names` lands in every checkpoint.
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

**Breaking everywhere:** interrupted runs can no longer be continued (see the
§3 warning), old `.pkl` artifacts need the `legacy-pkl` tag, and old commands
using the removed flags fail.

## 10. Per-model deltas

### DiffiT-v2 — nearly compliant

| change | breaking? |
|---|---|
| `--steps` alias for `--num-sampling-steps` in generation | no |
| `--network` alias for `--model-path` | no |
| `--combra-ref-count` knob; `--num-fid-samples` governs the combra fake count | no |
| combra extra → `git+https` source | no |

### EDM2-v2 — generation is the gap

| change | breaking? |
|---|---|
| rename `--preset` → `--cfg` (no alias kept) | **yes** — existing commands/sbatch must switch |
| `--network` alias for `--net` | no |
| **`edm2-gen-images`: add `--classes` + `--samples-per-class` class-batch mode and `--save-mode hdf5` (RankH5Writer layout)** | no (additive; `--seeds` mode kept) |
| inference snapshots keep the EMA-std suffix: `edm2-snapshot-<kimg>[-<ema_std>]-inference.pt` | no (naming detail) |

### StyleSwin-v2 — parity kit

| change | breaking? |
|---|---|
| console scripts `styleswin-train / -gen-images / -eval / -prepare-data` | no |
| **`gen_images.py`: add `--save-mode hdf5` (RankH5Writer layout)** | no (additive) |
| add `styleswin-eval` standalone evaluator + startup `combra_smoke_test` | no |
| `--num-fid-samples` / `--combra-ref-count` knobs (replacing fixed 10 000) | no (defaults unchanged) |

### san-v2 — dataset rebuild + format change

| change | breaking? |
|---|---|
| saves become `.pt` state dicts (pickled-module saving removed; `legacy.py` deleted) | **yes** — old `.pkl` artifacts readable only via the `legacy-pkl` tag |
| `dataset_tool.py` derives labels alphabetically (stops copying them verbatim from the source `dataset.json`) | no for new zips |
| `--num-fid-samples` / `--combra-ref-count` knobs | no |
| combra install via `[combra]` extra (`git+https`) — replaces the editable `-e ../wc_cv/combra`; pyproject.toml cleaned of the dead GUI deps | no |
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

## 11. Conformance checks

A small conformance suite in wc_cv keeps the convention from drifting again
(today's `--use-ddim` / `network-snapshot-final.pkl` / inverted-flag rot all
grew silently). No model execution needed:

1. **CLI contract** — parse `<model>-train --help` and
   `<model>-gen-images --help` per repo; assert the §2/§4 flags exist with the
   specified defaults.
2. **Run-dir contract** — `--dry-run` a training launch; assert the resolved
   option names and the run-dir naming pattern.
3. **Artifact contract** — validate a sample `.h5` against the RankH5Writer
   schema (`class_<c>/images|seeds`) and a checkpoint against the §3 metadata
   keys.
4. **Label contract** — assert `class_names` present and index-aligned in
   `dataset.json`, in checkpoint metadata and in a generated h5; assert the
   alphabetical ordering rule on a fresh dataset build (§5).

Wired into each repo's CI next to the existing smoke tests.

## 12. Adoption order

1. **Generation contract (§4)** — the only change with direct scientific
   payoff: EDM2-v2 and StyleSwin-v2 outputs become consumable by the wc_cv
   angle pipeline.
2. **Class-label contract (§5)** — dataset tools write `class_names`, the
   san-v2 zips are rebuilt; paired with (1) so every new artifact is
   self-describing from day one.
3. **Checkpoint scheme (§3)** — all four repos: EMA-snapshot-only saving as
   `.pt` state dicts; resume / best-model machinery and pickle loaders
   removed (ends the flag divergence and the `timm` unpickling fragility).
4. **StyleSwin-v2 parity kit (§10)**.
5. **Aliases, eval knobs and the log contract (§2, §6, §7, §8)**.
6. **Conformance CI (§11)** — last, so it locks in the finished state.
