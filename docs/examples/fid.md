# FID

Compute FID between generated diffusion images and real images for several resolutions.
See {py:func}`combra.metrics.compute_fid`, {py:func}`combra.metrics.compute_stats`, and
{py:func}`combra.metrics.frechet_distance`.

FID is **PyTorch-free** — InceptionV3 features run on `onnxruntime`, which ships with the default install. You only need an InceptionV3 ONNX model that outputs the 2048-d pooled features; point combra at it with the `COMBRA_INCEPTION_ONNX` environment variable (or `COMBRA_INCEPTION_URL` to download and cache it):

```bash
export COMBRA_INCEPTION_ONNX=/path/to/inception_v3.onnx
```

Building one `InceptionExtractor` and reusing it across folders runs the ONNX model only once per folder:

```python
>>> from combra.metrics import compute_stats, frechet_distance
>>> from combra.metrics.fid import InceptionExtractor
>>> extractor = InceptionExtractor()  # resolves the model via COMBRA_INCEPTION_ONNX
>>> BASE = 'data/separeted'
>>> pairs = [
...     ('256x256', f'{BASE}/o_bc_left_4x_768_256x256_N360', f'{BASE}/gen_diff_256x256_N500'),
...     ('512x512', f'{BASE}/o_bc_left_4x_768_512x512_N360', f'{BASE}/gen_diff_512x512_N500'),
...     ('768x768', f'{BASE}/o_bc_left_4x_768_768x768_N360', f'{BASE}/gen_diff_768x768_N5_000'),
... ]
>>> results = {}
>>> for name, real, gen in pairs:
...     mu_r, sig_r, n_r = compute_stats(real, extractor)
...     mu_g, sig_g, n_g = compute_stats(gen, extractor)
...     fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)
...     results[name] = {'fid': fid, 'n_real': n_r, 'n_gen': n_g}
...     print(f'{name}: FID = {fid:.4f}  (real={n_r}, gen={n_g})')
```

The InceptionV3 backbone is loaded once into the `InceptionExtractor` and reused across calls, so each `compute_stats` only runs the forward pass.

For a one-shot comparison of a single folder pair (the extractor is built internally), use `compute_fid` instead:

```python
>>> from combra.metrics import compute_fid
>>> fid, n_real, n_gen = compute_fid('data/real', 'data/gen')  # model_path=... or COMBRA_INCEPTION_ONNX
>>> print(f'FID = {fid:.4f}  (real={n_real}, gen={n_gen})')
```
