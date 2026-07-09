# Comparing samplers — how many steps for good quality?

A trained diffusion model can be sampled with different reverse-diffusion
samplers, and each needs a different number of steps `k` to reach good quality.
This example sweeps every sampler over a range of `k` and plots the combra
metrics against `k`, so you can read off the cheapest step count that still
produces good images.

The workflow has two halves:

- **combra** provides the generic, codebase-agnostic sweep + plot:
  {py:func}`combra.metrics.compare_samplers` and
  {py:func}`combra.metrics.plot_sampler_comparison`.
- **EDM2-v2** provides the wiring that turns a trained checkpoint and its four
  samplers (`edm`, `euler`, `ddim`, `dpm++`) into the generator callbacks that
  combra needs — the `edm2-compare-samplers` script.

## The idea

For each sampler and each `k`, generate a batch of samples, score it against a
fixed batch of **real** reference images with
{py:func}`combra.metrics.compute_all_metrics` (`image_metrics=True` adds FID /
CMMD / FD-DINOv2 on top of the angle-Wasserstein and bimodal-Gaussian metrics),
and collect one row per `(sampler, k)`. The plot then tiles one subplot per
metric with **Y = metric** and **X = k**, one curve per sampler.

`compare_samplers` never imports any diffusion code — the caller passes a
`{name: fn(k)}` mapping where `fn(k)` returns a generated batch — so the same
function works for any generator.

## Running it on EDM2-v2

{doc}`EDM2-v2 <edm2>` provides this wiring via `edm2-compare-samplers`, over its
EDM σ-space samplers (`edm`, `euler`, `ddim`, `dpm++`). It loads a checkpoint,
pulls `--num-samples` real reference images from `--data`, builds a `fn(k)` per
sampler, and calls `compare_samplers`. It writes a tidy
`sampler_comparison.parquet` and a `sampler_comparison.png`:

```bash
python compare_samplers.py \
    --net ./training-runs/00000-*/network-snapshot-final.pkl \
    --data ./datasets/wc_co_256x256.zip \
    --samplers edm,euler,ddim,dpm++ \
    --k-values 5,10,20,50,100,250 \
    --num-samples 512 --outdir ./sampler-comparison/256
# or: sbatch sbatch/compare_samplers_256x256.sbatch
```

## Calling combra directly

If you already have generator callbacks (from any model), skip the EDM2 script
and call combra directly:

```python
>>> from combra.metrics import compare_samplers, plot_sampler_comparison
>>>
>>> # fn(k) -> a batch of generated images produced with k sampling steps
>>> samplers = {'dpm++': dpmpp_fn, 'edm': edm_fn, 'ddim': ddim_fn}
>>> df = compare_samplers(
...     real_batch, samplers, k_values=[5, 10, 20, 50, 100, 250],
...     image_metrics=True,
... )
>>> df.head()
  sampler    k     fid    cmmd  fd_dinov2      w1   ...
0   dpm++    5   38.10   0.512      412.3   6.204   ...
1   dpm++   10   22.47   0.331      280.1   4.118   ...
...
>>> plot_sampler_comparison(df, save_path='sampler_comparison.png')
```

Read the answer off the curves: the `k` where each sampler's FID (and the other
metrics) flattens out is the step budget it needs — 2nd-order samplers like
`dpm++` and `edm` typically plateau far earlier than the 1st-order `ddim` /
`euler`.
