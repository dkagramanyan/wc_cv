# Generative models — API scheme

Four source-only repos generate WC-Co microstructure SEM images. They are
**separate forks that deliberately converge on one "san-v2-style" tooling
convention** — not a single package. This page is the cross-model map: what is
identical everywhere (the contract) and the few model-family details that stay
per-repo (samplers, EMA, float training space).

```{seealso}
This page documents the **current** state. The full specification — including the
per-repo migration deltas — is in {doc}`models_api_proposal`.
```

```{note}
**All four repos now implement the v2 convention** ({doc}`models_api_proposal`).
san-v2 (v0.2.0), StyleSwin-v2, DiffiT-v2 and EDM2-v2 each expose the shared API:
console scripts, the unified training CLI (`--precision`/`--tf32`/`--bench`,
`True/False` booleans, kimg/tick progress), EMA-only `.pt` inference snapshots
(atomic, no resume/best/latest/final), the unified HDF5 generation signature
(`format="generated_images_shard"`, `schema_version=1`, merged `<desc>.h5`),
`class_names` on every artifact, combra metrics mirrored into `stats.jsonl`, and
no Hydra / `requirements.txt`. The cells below therefore read as the target API in
every column; only the genuinely model-specific rows (samplers, EMA algorithm,
float training space) still differ, and are documented as such.
```

| repo | family | upstream | docs |
|---|---|---|---|
| [san-v2](https://github.com/dkagramanyan/san-v2) | GAN — StyleGAN3 + Projected GAN + SAN | Sony StyleSAN-XL | {doc}`san_v2` |
| [StyleSwin-v2](https://github.com/dkagramanyan/StyleSwin-v2) | GAN — Swin transformer | Microsoft StyleSwin | {doc}`styleswin` |
| [DiffiT-v2](https://github.com/dkagramanyan/DiffiT-v2) | latent diffusion — transformer, DDPM 1000-step schedule | NVlabs DiffiT | {doc}`diffit` |
| [EDM2-v2](https://github.com/dkagramanyan/edm2-v2) | latent diffusion — EDM σ-space U-Net | NVlabs EDM2 | {doc}`edm2` |

## The shared contract

Every repo follows the same conventions:

- **click CLI as the single source of truth** for options and defaults; the Hydra
  wrappers have been removed everywhere.
- **Dataset format**: StyleGAN-style `.zip` of images + `dataset.json`, which now
  carries an index-aligned `class_names` list, built with the repo's
  `<model>-prepare-data convert` tool.
- **Run directories**: auto-numbered `<outdir>/<id:05d>-<cfg>-gpus<N>-batch<B>[-desc]`
  with `training_options.json`, the rank-0 `.log`, `stats.jsonl`, TensorBoard events
  and `fakes<kimg>.png` grids; training is accounted in **kimg/ticks** with `--snap`
  controlling the snapshot cadence.
- **Multi-GPU is self-spawning**: both training and generation launch one worker per
  GPU via `torch.multiprocessing` — no `torchrun`.
- **combra evaluation** (optional, guarded import, `--combra-metrics` default
  `true`): every snapshot tick, fakes are generated **sharded across all ranks**
  and scored against the training set (reference features precomputed once) using
  combra's split APIs — {py:func}`combra.metrics.fid_features` /
  `fid_from_features`, the `cmmd_*` / `fd_dinov2_*` analogues, and
  {py:func}`combra.metrics.images_to_pooled_angles` +
  `angle_density_metrics_from_pooled` — numerically equivalent to a single-GPU
  {py:func}`combra.metrics.compute_all_metrics` call. Results are mirrored into
  **both** TensorBoard (`Metrics/combra_fid10k`, `Metrics/combra_cmmd10k`,
  `Metrics/combra_fd_dinov2_10k` + the angle metrics) and `stats.jsonl` in all
  four repos, so post-hoc snapshot selection survives loss of the tfevents file.
- **One checkpoint kind**: the EMA-only `.pt` inference snapshot, written
  atomically every snapshot tick **and always at the last tick**, pruned to
  `--snapshot-keep-last` (default 3, `0` = keep all). No resume, no rolling
  `latest`, no `best_model.*`, no separate final artifact — the best snapshot is
  chosen post-hoc from `stats.jsonl`.

## Training

| | san-v2 | StyleSwin-v2 | DiffiT-v2 | EDM2-v2 |
|---|---|---|---|---|
| entry point | `san-train` | `styleswin-train` | `diffit-train` | `edm2-train` |
| Hydra wrapper | **none** | **none** | **none** | **none** |
| presets | `--cfg` = architecture (`stylegan3-r`, …) | `--cfg styleswin-{256,512,1024}` | `--cfg diffit-{256,512,1024}` | `--cfg edm2-img{256,512,1024}-s` (+ more sizes) |
| required flags | `--outdir --cfg --data --gpus --batch-gpu` | `--outdir --data --gpus` | `--outdir --cfg --data --gpus --batch-gpu` | `--outdir --data` (`--cfg`/`--gpus`/`--batch-gpu` default) |
| precision | `--precision {fp32,fp16,bf16}` + `--tf32`/`--bench` (unified across all four; per-repo default) | | | |
| resolution strategy | **progressive**: 16² stem, then superres stages (`--superres --up-factor 2 --path-stem`, a weights-only warm start) | independent run per resolution | finetune upward via `--init-weights` (RoPE-2D) | independent preset per resolution (shared VAE latent space); no resume |

## Evaluation

| | san-v2 | StyleSwin-v2 | DiffiT-v2 | EDM2-v2 |
|---|---|---|---|---|
| in-training combra | ✔ (`--num-fid-samples`, default 10 000; `--combra-ref-count` caps the reference) | ✔ (`--num-fid-samples` / `--combra-ref-count`) | ✔ (`--num-fid-samples` / `--combra-ref-count`) | ✔ (`--num-fid-samples` / `--combra-ref-count`) |
| native metric suite | `--metrics` registry (legacy) | none (`--metrics` is a reserved stub) | Inception IS/FID/sFID/P/R via `--combra-metrics=false` | offline FID + FD-DINOv2 |
| standalone evaluator | `san-eval` | `styleswin-eval` | `diffit-eval` | `edm2-eval calc/gen/ref` |
| sampler-vs-steps sweep | n/a (GAN) | n/a (GAN) | `diffit-compare-samplers` | `edm2-compare-samplers` (see {doc}`sampler_comparison`) |

## Checkpoints

The checkpoint contract is now identical across all four (EMA-only `.pt` state
dicts, atomic, pruned, no resume/best/latest/final); only the snapshot filename
prefix and the EMA algorithm differ.

| | san-v2 | StyleSwin-v2 | DiffiT-v2 | EDM2-v2 |
|---|---|---|---|---|
| format | **`.pt` state dicts** — EMA-only + `{n_classes, resolution, class_names, cur_nimg}` metadata, atomic writes, no pickled modules | same | same | same |
| rolling full checkpoint | **none** (no resume) | **none** | **none** | **none** |
| inference snapshot (pruned) | `san-snapshot-<kimg>-inference.pt` — every tick **+ last tick**, `--snapshot-keep-last` | `network-snapshot-<kimg>-inference.pt` | `diffit-snapshot-<kimg>-inference.pt` | `edm2-snapshot-<kimg>[-<ema_std>]-inference.pt` |
| best model | **none** (post-hoc from `stats.jsonl`) | **none** | **none** | **none** |
| final artifact | **none** (newest snapshot is final) | **none** | **none** | **none** |
| EMA | classic `G_ema` (rampup) | classic `g_ema` | classic EMA (`--ema-rate`) | **PowerFunctionEMA** (one snapshot per EMA std) |

## Samplers (diffusion models only)

| | DiffiT-v2 | EDM2-v2 |
|---|---|---|
| available | `dpm++`, `unipc`, `ddim`, `ddpm` | `dpm++`, `edm` (Heun), `euler`, `ddim` (≡ `euler`) |
| space | DDPM 1000-step discrete schedule | EDM σ-space (Karras schedule) |
| eval default | `ddim` @ 100 steps | `dpm++` @ 25 steps |
| flags | `--eval-sampler` / `--eval-sampling-steps` (training), `--sampler` / `--steps` (generation) | `--eval-sampler` / `--eval-sampling-steps` (training), `--sampler` / `--steps` (generation) |

```{note}
The overlapping names are **not interchangeable**: `ddim` and `dpm++` integrate
different parameterisations of the reverse process in the two repos, so step
counts and quality do not transfer. Calibrate per repo with its own
compare-samplers tool ({doc}`sampler_comparison`).
```

## Generation

| | san-v2 | StyleSwin-v2 | DiffiT-v2 | EDM2-v2 |
|---|---|---|---|---|
| script | `san-gen-images` | `styleswin-gen-images` | `diffit-gen-images` | `edm2-gen-images` |
| checkpoint flag | `--network` (`.pt`) | `--network` (`.pt`) | `--network` (alias `--model-path`) (`.pt`) | `--net`/`--network` (`.pt`) |
| class selection | `--classes` (indices/ranges **or names**, validated) + `--samples-per-class` | same | same | `--classes 0,1,4-6` **or names** + `--samples-per-class` (`--seeds` legacy) |
| output | **HDF5** (unified sig, merged `<desc>.h5`, merge hard-fails on gaps) or PNG dirs + `classes.json` (`--save-mode`); self-spawning `--gpus` | same | same | same |
| quality knobs | `--trunc`, `--centroids-path` | `--trunc` | `--cfg-scale`, `--sampler`, `--steps` | `--sampler`, `--steps`, `--guidance --gnet` |

```{note}
**The generation artifacts are now equivalent.** All four repos emit the per-class
`RankH5Writer` HDF5 layout (`class_<c>/images|seeds`, uint8 NHWC) with the unified
`format="generated_images_shard"` / `schema_version=1` signature and `class_names`,
merged to `<desc>.h5` — directly consumable by the downstream angle pipeline
(`co_angles/generate_class_samples.py`,
{py:meth}`combra.data.PobeditDataset.generate_angles`) with no conversion step, and
the merge hard-fails on incomplete shards.
```

## Class index → grain class

All four dataset tools now derive the integer label from the **alphabetical** class
folder sort (`0 → Ultra_Co11`, `1 → Ultra_Co25`, `2 → Ultra_Co6_2`) and stamp an
index-aligned `class_names` list into `dataset.json`, every checkpoint and every
generated artifact — so new runs are self-describing by name and combra matches by
name rather than a fragile integer convention.

```{warning}
**Existing pre-migration checkpoints still use the old swapped order.** The on-disk
`imagenet_9to4_*` archives the existing runs of all four models trained on carry the
non-alphabetical order `0 → Ultra_Co25`, `1 → Ultra_Co11`, `2 → Ultra_Co6_2` (the
`Co11`↔`Co25` swap) and record no `class_names`. Those checkpoints and everything
generated from them stay under combra's legacy `CLASS_MAP` until retrained on rebuilt
zips — classify each run by the dataset path in its `training_options.json` before
remapping. See the {doc}`models_api_proposal` label contract (§5).
```

## Other known divergences

Now that all four repos share the contract, the remaining differences are
model-family details, not tooling drift:

1. **combra install** is uniform: all four pull the private repo over `git+https`
   via the `[combra]` extra, and none ship a `requirements.txt` (`pip install -e .`).
2. **CUDA toolchain**: san-v2 and StyleSwin-v2 build custom CUDA ops (san-v2 against
   conda's `nvcc` with `CUDA_HOME=$CONDA_PREFIX`; StyleSwin-v2 via the system CUDA
   module); DiffiT-v2 and EDM2-v2 are pure-torch and need no custom ops.
3. **Model-family internals stay per-repo** (documented, not unified): the samplers
   (above), the EMA algorithm (classic half-life vs `--ema-rate` vs PowerFunctionEMA),
   and the float training space (`[-1, 1]` for san-v2/DiffiT-v2, ImageNet mean/std for
   StyleSwin-v2, the VAE latent space for EDM2-v2). Every artifact still crosses the
   boundary as uint8, so cross-repo comparisons remain valid.
