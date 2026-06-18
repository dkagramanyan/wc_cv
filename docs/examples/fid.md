# FID

Compute FID between generated diffusion images and real images for several resolutions.
See {py:func}`combra.metrics.compute_fid`.

combra does not implement FID itself — `combra.metrics.fid` delegates to the reference library [pytorch-fid](https://github.com/mseitzer/pytorch-fid). It ships as a core dependency and downloads/caches its own InceptionV3 weights on first use, so there is no manual model setup. `compute_fid` runs on CUDA when it is available and falls back to CPU otherwise.

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
