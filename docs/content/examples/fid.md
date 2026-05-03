---
title: "FID"
weight: 2
---

## Usage

Compute FID between generated diffusion images and real images for several resolutions.

Run with an environment that has `torch`, `torchvision`, and `pytorch_fid` installed.

```python
from combra.metrics import load_inception, compute_fid

model, device = load_inception()
print('device:', device)

BASE = 'data/separeted'

pairs = [
    ('256x256', f'{BASE}/o_bc_left_4x_768_256x256_N360', f'{BASE}/gen_diff_256x256_N500'),
    ('512x512', f'{BASE}/o_bc_left_4x_768_512x512_N360', f'{BASE}/gen_diff_512x512_N500'),
    ('768x768', f'{BASE}/o_bc_left_4x_768_768x768_N360', f'{BASE}/gen_diff_768x768_N5_000'),
]

results = {}
for name, real, gen in pairs:
    print(f'\n=== {name} ===')
    print(f'Real: {real}')
    print(f'Gen:  {gen}')
    fid, n_r, n_g = compute_fid(real, gen, model=model, device=device)
    results[name] = {'fid': fid, 'n_real': n_r, 'n_gen': n_g}
    print(f'  FID = {fid:.4f}  (real={n_r}, gen={n_g})')

print('\n===== Summary =====')
for name, r in results.items():
    print(f'{name}: FID = {r["fid"]:.4f}  (real={r["n_real"]}, gen={r["n_gen"]})')
```

The InceptionV3 backbone is loaded once with `load_inception()` and reused across calls so each `compute_fid` only runs the forward pass.
