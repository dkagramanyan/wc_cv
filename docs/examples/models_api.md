# Generative models — API scheme

Four source-only repos generate WC-Co microstructure SEM images. They are
**separate forks that deliberately converge on one "san-v2-style" tooling
convention** — not a single package. This page is the cross-model map: what is
identical everywhere (the contract), and where the repos still diverge.

```{seealso}
This page documents the **current** state. A specification that unifies the
remaining divergences — including per-repo migration deltas — is proposed in
{doc}`models_api_proposal`.
```

| repo | family | upstream | docs |
|---|---|---|---|
| [san-v2](https://github.com/dkagramanyan/san-v2) | GAN — StyleGAN3 + Projected GAN + SAN | Sony StyleSAN-XL | {doc}`san_v2` |
| [StyleSwin-v2](https://github.com/dkagramanyan/StyleSwin-v2) | GAN — Swin transformer | Microsoft StyleSwin | {doc}`styleswin` |
| [DiffiT-v2](https://github.com/dkagramanyan/DiffiT-v2) | latent diffusion — transformer, DDPM 1000-step schedule | NVlabs DiffiT | {doc}`diffit` |
| [EDM2-v2](https://github.com/dkagramanyan/edm2-v2) | latent diffusion — EDM σ-space U-Net | NVlabs EDM2 | {doc}`edm2` |

## The shared contract

Every repo follows the same conventions:

- **click CLI as the single source of truth** for options and defaults; where a
  Hydra wrapper exists (`train_hydra.py`), it introspects the click options and
  calls the same launch path, so both entry points produce identical runs.
- **Dataset format**: StyleGAN-style `.zip` of images + `dataset.json`
  (`{"labels": [[path, classID], ...]}`), built with the repo's `dataset_tool*.py`.
- **Run directories**: auto-numbered `<outdir>/<id:05d>-<cfg>-gpus<N>-batch<B>[-desc]`
  with `training_options.json`, `log.txt`, `stats.jsonl`, TensorBoard events and
  `fakes<kimg>.png` grids; training is accounted in **kimg/ticks** with `--snap`
  controlling the snapshot cadence.
- **Multi-GPU is self-spawning**: training launches one worker per GPU via
  `torch.multiprocessing` — no `torchrun` for training (some *generation* scripts
  do use `torchrun`).
- **combra evaluation** (optional, guarded import, `--combra-metrics` default
  `true`): every snapshot tick, fakes are generated **sharded across all ranks**
  and scored against the **whole training set** (reference features precomputed
  once) using combra's split APIs — {py:func}`combra.metrics.fid_features` /
  `fid_from_features`, the `cmmd_*` / `fd_dinov2_*` analogues, and
  {py:func}`combra.metrics.images_to_pooled_angles` +
  `angle_density_metrics_from_pooled` — numerically equivalent to a single-GPU
  {py:func}`combra.metrics.compute_all_metrics` call. Results land in TensorBoard
  as `Metrics/combra_fid10k`, `Metrics/combra_cmmd10k`,
  `Metrics/combra_fd_dinov2_10k` + the angle metrics. DiffiT-v2 and EDM2-v2
  also mirror them into `stats.jsonl`; **san-v2 and StyleSwin-v2 write them
  to TensorBoard only** — the tfevents file is the sole record of those
  runs' metric history.
- **Best-model checkpoint** selected by `combra_fid10k`, plus a rolling full
  `network-snapshot-latest.pt` overwritten in place and a history of small
  inference snapshots, pruned to `--snapshot-keep-last` (default 3) —
  except in **san-v2**, where no pruning exists and snapshots accumulate
  unbounded.

## Training

| | san-v2 | StyleSwin-v2 | DiffiT-v2 | EDM2-v2 |
|---|---|---|---|---|
| entry point | `train.py` | `train.py` | `scripts/train.py` (`diffit-train`) | `train_edm2.py` (`edm2-train`) |
| Hydra wrapper | `train_hydra.py` | **none** | `train_hydra.py` | `train_hydra.py` |
| presets | `--cfg` = architecture (`stylegan3-r`, …) | `--cfg styleswin-{256,512,1024}` | `--cfg diffit-{256,512,1024}` | `--preset edm2-img{256,512,1024}-s` (+ more sizes) |
| required flags | `--outdir --cfg --data --gpus --batch-gpu` | `--outdir --data --gpus` | `--outdir --cfg --data --gpus --batch-gpu` | `--outdir --data` |
| resolution strategy | **progressive**: 16² stem, then superres stages (`--superres --up_factor 2 --path_stem`) | independent run per resolution | finetune upward via `--resume` (RoPE-2D) | independent preset per resolution (shared VAE latent space) |

## Evaluation

| | san-v2 | StyleSwin-v2 | DiffiT-v2 | EDM2-v2 |
|---|---|---|---|---|
| in-training combra | ✔ (10 000 fakes, fixed) | ✔ (10 000 fakes, fixed) | ✔ (fakes match the whole-dataset reference) | ✔ (`--num-fid-samples`, default 10 000; `--combra-ref-count` caps the reference) |
| native metric suite | `--metrics` registry (`fid50k_full`, `is50k`, …) | none (`--metrics` is a reserved stub) | Inception IS/FID/sFID/P/R via `--combra-metrics=false` | offline FID + FD-DINOv2 |
| standalone evaluator | `calc_metrics.py` | **none** | `diffit-eval` | `edm2-eval calc/gen/ref` |
| sampler-vs-steps sweep | n/a (GAN) | n/a (GAN) | `diffit-compare-samplers` | `edm2-compare-samplers` (see {doc}`sampler_comparison`) |

## Checkpoints

| | san-v2 | StyleSwin-v2 | DiffiT-v2 | EDM2-v2 |
|---|---|---|---|---|
| format | **pickled modules** (`.pkl`, loaded via `legacy.py`) | `.pt` state dicts | `.pt` state dicts | mixed: inference `.pkl` + resume `.pt` |
| rolling full checkpoint | `network-snapshot-latest.pt` (`G`/`D`/`G_ema` + progress; **no optimizer state**) | `network-snapshot-latest.pt` (G, D, G_ema, both optimizers) | `network-snapshot-latest.pt` (model, ema, opt, scaler) | `network-snapshot-latest.pt` (state, net, loss, optimizer, ema) |
| inference history (pruned) | `network-snapshot-<kimg>-inference.pkl` — only with `--save-inference-only 1` | `network-snapshot-<kimg>-inference.pt` — always | `network-snapshot-<kimg>-inference.pt` — always | `network-snapshot-<kimg>-<ema_std>.pkl` — always, one per EMA std |
| best model | `best_model.pkl` | `best_model.pt` | `best_model.pt` | `best_model.pt` |
| final artifact | — | — | `network-final.pt` / `network-final-inference.pt` | — |
| EMA | classic `G_ema` (rampup) | classic `g_ema` | classic EMA (`--ema-rate`) | **PowerFunctionEMA** (post-hoc reconstructable via `reconstruct_phema.py`) |

```{warning}
**`--save-inference-only` means opposite things.** In DiffiT-v2, EDM2-v2 and
StyleSwin-v2, inference snapshots are always written and the flag **skips** the
rolling full `network-snapshot-latest.pt`. In **san-v2** the rolling full
checkpoint is written regardless, and `--save-inference-only 1` **adds** the
per-tick inference `.pkl`s. Check the model's own page before copying the flag
between repos.
```

## Samplers (diffusion models only)

| | DiffiT-v2 | EDM2-v2 |
|---|---|---|
| available | `dpm++`, `unipc`, `ddim`, `ddpm` | `dpm++`, `edm` (Heun), `euler`, `ddim` (≡ `euler`) |
| space | DDPM 1000-step discrete schedule | EDM σ-space (Karras schedule) |
| eval default | `ddim` @ 100 steps | `dpm++` @ 25 steps |
| flags | `--eval-sampler` / `--eval-sampling-steps` (training), `--sampler` / `--num-sampling-steps` (generation) | `--eval-sampler` / `--eval-sampling-steps` (training), `--sampler` / `--steps` (generation) |

```{note}
The overlapping names are **not interchangeable**: `ddim` and `dpm++` integrate
different parameterisations of the reverse process in the two repos, so step
counts and quality do not transfer. Calibrate per repo with its own
compare-samplers tool ({doc}`sampler_comparison`).
```

## Generation

| | san-v2 | StyleSwin-v2 | DiffiT-v2 | EDM2-v2 |
|---|---|---|---|---|
| script | `gen_images.py` | `gen_images.py` | `diffit-gen-images` | `edm2-gen-images` |
| checkpoint flag | `--network` (`.pkl`) | `--network` (`.pt`) | `--model-path` (`.pt`) | `--net` (`.pkl`) |
| class selection | `--classes 0,1,2` + `--samples-per-class` | `--classes` + `--samples-per-class` | `--classes` + `--samples-per-class` (or `--seeds`) | **`--class <idx>` + `--seeds 0-999`** — one class per run |
| output | **HDF5 (default)** or per-class PNG dirs (`--save-mode`) | per-class PNG dirs only | HDF5 shards → merged `.h5`, or dirs (`--save-mode`) | flat `<seed>.png` only |
| quality knobs | `--trunc`, `--centroids-path` | `--trunc` | `--cfg-scale`, `--sampler`, `--num-sampling-steps` | `--sampler`, `--steps`, `--guidance --gnet` |

```{warning}
**The generation artifacts are not equivalent.** The downstream angle pipeline
(`co_angles/generate_class_samples.py`, {py:meth}`combra.data.PobeditDataset.generate_angles`)
consumes the per-class HDF5 layout of san-v2's `RankH5Writer`, which DiffiT-v2
replicates. **StyleSwin-v2 and EDM2-v2 emit PNGs only** — their outputs need a
conversion step before entering that pipeline.
```

## Class index → grain class

The mapping is a property of the **zip a run trained on**, not of the repo:
the zips store only integer labels, every repo takes them **verbatim** at
train time, and no artifact records the `Ultra_Co*` names. The dataset
*tools* differ — san-v2's copies labels from a source `dataset.json` written
in the non-alphabetical order `0 → Ultra_Co25`, `1 → Ultra_Co11`,
`2 → Ultra_Co6_2`, while the others' derive them from the alphabetical
folder sort (`0 → Ultra_Co11`, `1 → Ultra_Co25`) — but the on-disk
`imagenet_9to4_*` archives that the real DiffiT-v2 and StyleSwin-v2 runs
consumed carry the **same swapped order as the san-v2 zips**.

```{warning}
The existing checkpoints of all four models therefore most likely share
san-v2's swapped convention. Before comparing across models or remapping
with combra's `CLASS_MAP`, classify each run by the dataset path in its
`training_options.json` — remapping a checkpoint that already uses san-v2's
order introduces the very swap the remap is meant to fix. Details on each
model page and in the {doc}`models_api_proposal` label contract (§5).
```

## Other known divergences

1. **combra install source differs per repo**: san-v2's `requirements.txt` uses
   an editable local checkout (`-e ../wc_cv/combra`); EDM2-v2 and StyleSwin-v2
   pull the private repo over `git+https` via the `[combra]` extra; DiffiT-v2
   expects combra installed first from a local checkout.
2. **StyleSwin-v2 has no `requirements.txt`** — dependencies live only in
   `pyproject.toml` (`pip install -e .`), unlike the other three.
3. **CUDA toolchain**: san-v2 compiles its ops against conda's `nvcc`
   (`CUDA_HOME=$CONDA_PREFIX`); StyleSwin-v2 uses the system module
   (`module load CUDA/13.1`). DiffiT-v2 and EDM2-v2 need no custom CUDA ops.
4. **combra eval sample count** is configurable only in EDM2-v2
   (`--num-fid-samples`); the other three use a fixed 10 000. Cross-model
   `combra_fid10k` comparisons are valid only at the default count.
