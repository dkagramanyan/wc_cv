# FID

Compute FID between generated diffusion images and real images for several resolutions.
See {py:func}`combra.metrics.compute_fid` and {py:func}`combra.metrics.calculate_metrics`.

combra does not implement FID itself — `combra.metrics.fid` delegates to the reference libraries [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [torch-fidelity](https://github.com/toshas/torch-fidelity). Both ship as core dependencies and download/cache their own InceptionV3 weights on first use, so there is no manual model setup. `compute_fid` runs on CUDA when it is available and falls back to CPU otherwise.

A multi-resolution sweep is just a loop over folder pairs:

```python
>>> from combra.metrics import compute_fid
>>> BASE = 'data/separeted'
>>> pairs = [
...     ('256x256', f'{BASE}/o_bc_left_4x_768_256x256_N360', f'{BASE}/gen_diff_256x256_N500'),
...     ('512x512', f'{BASE}/o_bc_left_4x_768_512x512_N360', f'{BASE}/gen_diff_512x512_N500'),
...     ('768x768', f'{BASE}/o_bc_left_4x_768_768x768_N360', f'{BASE}/gen_diff_768x768_N5_000'),
... ]
>>> results = {}
>>> for name, real, gen in pairs:
...     fid = compute_fid(real, gen, batch_size=50)
...     results[name] = fid
...     print(f'{name}: FID = {fid:.4f}')
```

The InceptionV3 weights are downloaded once and cached by pytorch-fid, so subsequent calls reuse them.

For the full generative-quality suite (FID/KID/IS/PRC) in a single call, use `calculate_metrics`, re-exported from torch-fidelity:

```python
>>> from combra.metrics import calculate_metrics
>>> res = calculate_metrics(input1='data/real', input2='data/gen',
...                         fid=True, kid=True, cuda=False, batch_size=50)
>>> print(f"FID = {res['frechet_inception_distance']:.4f}")
```
