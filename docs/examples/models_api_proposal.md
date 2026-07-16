# Generative models — proposed standard API ("v2 convention")

```{note}
**Status: proposal.** This page specifies a single API convention for the four
model repos. Nothing here is implemented yet — the **current** state of each
repo (including where they already diverge) is documented in
{doc}`models_api`. Each axis of the spec adopts the variant that already has
majority adoption, so the migration cost is minimal.
```

The goal: any command, flag, checkpoint name, or generated artifact learned on
one repo transfers to the other three unchanged, and every model's generated
output feeds the wc_cv angle pipeline
(`co_angles/generate_class_samples.py`,
{py:meth}`combra.data.PobeditDataset.generate_angles`) with zero conversion.

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

The upstream bulk-`.npz` samplers (`scripts/sample.py` in DiffiT-v2,
`sample_images.py` in EDM2-v2) are **outside the convention**: their only
consumer is the upstream-paper FID protocol (ADM-style `.npz` reference
batches), which the WC-Co workflow does not use. Bulk generation is
`<model>-gen-images --save-mode hdf5`; scoring is the combra eval and
`<model>-eval`. The scripts may remain in the repos as legacy tools but are
not part of the contract.

- **`pyproject.toml` is the single dependency source — no `requirements.txt`.**
  Repos that have one remove it: maintaining two hand-written lists is exactly
  what produced today's dependency drift (stale caps, dead deps, missing
  entries). `pip install -e .` is the one install path. torch / ninja stay out
  of `pyproject.toml` as now (installed from the CUDA wheel index / conda).
- combra from **one** source everywhere: optional extra `[combra]` →
  `git+https` private repo (replaces san-v2's editable `-e ../wc_cv/combra`
  and DiffiT's bare `combra` extra).
- CUDA story documented per repo class: JIT-op repos (san-v2, StyleSwin-v2)
  need `nvcc` + `ninja`; pure-torch repos (DiffiT-v2, EDM2-v2) do not.

## 2. Training CLI

```bash
<model>-train --outdir <dir> --cfg <preset> --data <zip> --gpus N --batch-gpu B
```

- **The click CLI is the only interface** — no Hydra. `train_hydra.py`,
  `configs/`, and the `hydra-core` dependency are removed from the repos that
  have them: no launch script ever calls the Hydra path, the configs are
  single flat files with no groups, and no composition or multirun feature is
  used — it is a second launch path to keep in sync for zero benefit.
- **`--cfg` is the preset flag name everywhere** (EDM2 renames `--preset` →
  `--cfg`; no alias is kept).
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
- Run directory: `<outdir>/<id:05d>-<cfg>-gpus<G>-batch<B>[-desc]` containing
  `training_options.json`, `log.txt`, `stats.jsonl`, TensorBoard events and
  `fakes<kimg>.png` grids.

## 3. Checkpoint contract

Exactly two artifact kinds — no resume, no best-model tracking:

| artifact | rule |
|---|---|
| `<model>-snapshot-<kimg:06d>-inference.pt` | EMA-only weights; written every snapshot tick; history pruned to `--snapshot-keep-last` (default 3, `0` = keep all) |
| `<model>-final.pt` | model + EMA weights (no optimizer state), written once at the end of training |

- **No resume.** There is no `--resume` flag, no rolling `latest` checkpoint
  and no auto-restart: training runs start-to-finish. This also deletes
  `--save-inference-only` — and with it san-v2's inverted-flag hazard —
  because there is no full rolling checkpoint left to skip.
- **No `best_model.*`.** Per-tick best-FID bookkeeping goes away; pick the
  best checkpoint post-hoc from `stats.jsonl` against the snapshot history
  (set `--snapshot-keep-last 0` on runs where you want the full history to
  choose from).
- **Progressive stages still work**: san-v2's `--path_stem` and DiffiT's
  higher-resolution finetuning are **weights-only warm starts** from a
  previous stage's snapshot — initialization, not resume.
- **Format: `.pt` state dicts only** — a state dict stores only weight
  tensors keyed by parameter name, so loading rebuilds the model from current
  code instead of unpickling stored classes. Pickled-module saving is removed
  everywhere.
- **No `timm` in artifacts.** Both artifact kinds hold **generator-side
  weights only** — no discriminator and therefore no `timm` feature-network
  modules or weights ever enter a checkpoint. Loading a checkpoint never
  requires `timm` (or any particular `timm` version); `timm` remains a
  train-time-only dependency of the GAN discriminators.
- **No legacy `.pkl` loaders.** `legacy.py` and the pickle-reading paths are
  removed; old pickled artifacts remain readable only by checking out the
  `legacy-pkl` git tag (the last pickle-capable commit in each repo).
- **Self-describing metadata** in every checkpoint:
  `{n_classes, resolution, class_names, cur_nimg}`. `class_names` is the new
  requirement — it removes the SAN-vs-rest index-swap hazard at the source,
  because downstream code can read grain-class *names* instead of guessing
  index conventions (the full label contract is §5).

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

- One checkpoint flag: **`--network`** (deprecated aliases kept:
  `--model-path` in DiffiT-v2, `--net` in EDM2-v2).
- `--classes` accepts indices **or class names**
  (`--classes Ultra_Co11,Ultra_Co6_2`) — see the label contract in §5.
- **Determinism rule** in all four: `seed = base + class·samples_per_class + idx`
  — any subset of the output is reproducible in isolation.
- **Identical outputs across repos**:
  - `hdf5` (default): per-rank shards `shards/rank_NNN.h5` in the
    **`RankH5Writer` layout** (`class_<c>/images|seeds`), merged by rank 0 —
    exactly what the wc_cv angle pipeline consumes.
  - `dir`: `class_<c>/idx_<i:06d>_seed_<s>.png`.
- EDM2's seed-oriented mode (`--seeds 0-999 --class <idx>`) survives as a
  power-user mode; the class-batch mode above becomes the default interface.

## 5. Class-label contract

Today the artifacts carry only integers (`class_0/1/2`), and the meaning of
the integer depends on which dataset tool wrote the zip — san-v2's
`dataset.json` was once written non-alphabetically, and that decision now
lives in every SAN checkpoint and generated h5, recoverable only through
combra's hand-pinned `CLASS_MAP` (see {doc}`models_api` "Class index → grain
class"). Two rules end this:

**Rule 1 — canonical ordering.** The integer label is the index of the class
folder in `sorted()` (alphabetical) order. Every `dataset_tool*.py` derives
labels this way; san-v2's tool stops copying labels verbatim from the source
`dataset.json`.

**Rule 2 — names travel with every artifact.** The integer becomes an
implementation detail; the grain-class *name* is the identity:

| stage | requirement |
|---|---|
| dataset zip | `dataset.json` carries `"class_names": ["Ultra_Co11", "Ultra_Co25", "Ultra_Co6_2"]`, index-aligned, written automatically from the folder names |
| checkpoint | `class_names` copied into every checkpoint (part of the §3 metadata) |
| generated h5 | `class_names` stamped as a root attribute + per-`class_<c>` group attribute |
| generated dir | a `classes.json` manifest next to the `class_<c>/` folders |
| CLI | `--classes` accepts **names as well as indices**: `--classes Ultra_Co11,Ultra_Co6_2` |
| downstream | combra matches by **name**; `CLASS_MAP` remains only as the fallback for legacy artifacts that lack `class_names` |

### san-v2 dataset rebuild

The existing san-v2 `imagenet_9to4_*` zips are **rebuilt** with alphabetical
labels + `class_names`, ending the SAN exception at the data level. Safety
rails, because a trained model's class embedding is welded to its
training-time indices:

- Rebuilt zips are for **new runs only**. **Never resume or finetune a
  pre-rebuild checkpoint against a rebuilt zip** — the label semantics of
  classes 0 and 1 silently swap.
- Pre-rebuild checkpoints and everything generated from them stay under the
  legacy `CLASS_MAP` until retrained.
- Once san-v2 is retrained on the rebuilt zips, all four models share one
  convention and the class-map warnings on the model pages become historical
  notes about legacy artifacts.

## 6. Evaluation contract

- **In-training combra eval** (all four): every snapshot tick, fakes generated
  **sharded across all ranks**; reference = whole training set with features
  precomputed once before the loop; `combra_smoke_test` at startup (today only
  the diffusion repos run it).
- **Uniform knobs** (promote EDM2's flags, drop hardcoded `COMBRA_NUM_GEN`):
  `--num-fid-samples` (default 10000, `0` disables eval) and
  `--combra-ref-count` (cap the reference side).
- **Uniform keys**: `Metrics/combra_fid10k`, `Metrics/combra_cmmd10k`,
  `Metrics/combra_fd_dinov2_10k`, the angle-density metrics, and
  `Metrics/combra_fid10k_best` — all mirrored to `stats.jsonl`.
- `<model>-eval` standalone evaluator in all four (StyleSwin-v2 currently has
  none; its shim can wrap the combra split APIs over an image folder / h5).

## 7. Samplers (diffusion repos)

Sampler **algorithms stay per-family** — DiffiT-v2's `dpm++/unipc/ddim/ddpm`
(DDPM 1000-step schedule) and EDM2-v2's `dpm++/edm/euler/ddim` (σ-space) are
genuinely different integrators and are *not* unified. Only the **flag names**
are standardized:

| context | flags |
|---|---|
| training-time eval | `--eval-sampler` / `--eval-sampling-steps` |
| generation | `--sampler` / `--steps` (DiffiT-v2 renames `--num-sampling-steps` → `--steps`, old name kept as alias) |

Both repos ship `<model>-compare-samplers` producing
`sampler_comparison.parquet` + `sampler_comparison.png`
(see {doc}`sampler_comparison`).

## 8. Per-model migration deltas

**Common to all four** — the §3 checkpoint scheme: drop `--resume` /
auto-restart, `best_model.*`, the rolling `latest` checkpoint and
`--save-inference-only`; rename snapshot files to `<model>-snapshot-*`;
remove legacy `.pkl` loaders (tag the last pickle-capable commit
`legacy-pkl`). **Breaking everywhere:** interrupted runs can no longer be
continued (see the §3 warning) and old commands using the removed flags fail.

### DiffiT-v2 — nearly compliant

| change | breaking? |
|---|---|
| `--steps` alias for `--num-sampling-steps` in generation | no |
| `--network` alias for `--model-path` | no |
| `--combra-ref-count` knob; `--num-fid-samples` governs the combra fake count | no |
| combra extra → `git+https` source | no |
| remove `requirements.txt` (pyproject.toml is the single dependency source) | no |
| remove `train_hydra.py` + `configs/` + `hydra-core` dep | no (unused) |
| label contract (§5): dataset tool writes `class_names`; gen-images stamps names into h5 / `classes.json`; `--classes` accepts names; `class_names` in checkpoints | no |

### EDM2-v2 — generation is the gap

| change | breaking? |
|---|---|
| rename `--preset` → `--cfg` (no alias kept) | **yes** — existing commands/sbatch must switch |
| `--network` alias for `--net` | no |
| inference snapshots become `.pt` state dicts: `edm2-snapshot-<kimg>[-<ema_std>]-inference.pt` | **yes** — old `.pkl` snapshots readable only via the `legacy-pkl` tag |
| **`edm2-gen-images`: add `--classes` + `--samples-per-class` class-batch mode and `--save-mode hdf5` (RankH5Writer layout)** | no (additive; `--seeds` mode kept) |
| remove `requirements.txt` (pyproject.toml is the single dependency source) | no |
| remove `train_hydra.py` + `configs/` + `hydra-core` dep | no (unused) |
| label contract (§5): dataset tool writes `class_names`; gen-images stamps names into h5 / `classes.json`; `--classes` accepts names; `class_names` in inference pickles / checkpoints | no |

### StyleSwin-v2 — parity kit

| change | breaking? |
|---|---|
| console scripts `styleswin-train / -gen-images / -eval / -prepare-data` | no |
| **`gen_images.py`: add `--save-mode hdf5` (RankH5Writer layout)** | no (additive) |
| add `styleswin-eval` standalone evaluator + startup `combra_smoke_test` | no |
| `--num-fid-samples` / `--combra-ref-count` knobs (replacing fixed 10 000) | no (defaults unchanged) |
| label contract (§5): dataset tool writes `class_names`; gen-images stamps names into h5 / `classes.json`; `--classes` accepts names; `class_names` in checkpoints | no |

### san-v2 — dataset rebuild + format change

| change | breaking? |
|---|---|
| saves become `.pt` state dicts (pickled-module saving removed; `legacy.py` deleted) | **yes** — old `.pkl` artifacts readable only via the `legacy-pkl` tag |
| `--num-fid-samples` / `--combra-ref-count` knobs | no |
| combra install via `[combra]` extra (`git+https`) | no |
| remove `requirements.txt` — its editable `-e ../wc_cv/combra` moves to the `[combra]` extra; pyproject.toml (cleaned of the dead GUI deps) becomes the single dependency source | no |
| remove `train_hydra.py` + `configs/` + `hydra-core` dep | no (unused) |
| label contract (§5): `dataset_tool.py` derives labels alphabetically + writes `class_names`; gen-images stamps names; `--classes` accepts names | no |
| **rebuild the `imagenet_9to4_*` dataset zips with alphabetical labels + `class_names`** | **yes** — pre-rebuild checkpoints must never be finetuned on rebuilt zips; they and their artifacts stay under the legacy `CLASS_MAP` until retrained (§5) |

## 9. Conformance checks

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

## 10. Adoption order

1. **Generation contract (§4)** — the only change with direct scientific
   payoff: EDM2-v2 and StyleSwin-v2 outputs become consumable by the wc_cv
   angle pipeline.
2. **Class-label contract (§5)** — dataset tools write `class_names`, the
   san-v2 zips are rebuilt; paired with (1) so every new artifact is
   self-describing from day one.
3. **Checkpoint scheme (§3)** — all four repos: EMA-snapshot-only saving as
   `.pt` state dicts; resume / best-model machinery and pickle loaders
   removed (ends the flag divergence and the `timm` unpickling fragility).
4. **StyleSwin-v2 parity kit (§8)**.
5. **Aliases and eval knobs (§2, §6, §7)**.
6. **Conformance CI (§9)** — last, so it locks in the finished state.
