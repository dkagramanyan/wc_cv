# FID

Compute FID between generated diffusion images and real images for several resolutions.
See {py:func}`combra.metrics.compute_fid`.

combra does not implement FID itself — `compute_fid` delegates to the reference library [pytorch-fid](https://github.com/mseitzer/pytorch-fid). It ships as a core dependency and downloads/caches its own InceptionV3 weights on first use, so there is no manual model setup. `compute_fid` takes **in-memory image batches** (a numpy array or torch tensor), not a folder, and runs on CUDA when it is available, falling back to CPU otherwise.

Load each resolution's images into a batch array, then loop over the pairs:

```python
>>> import numpy as np
>>> from PIL import Image
>>> from pathlib import Path
>>> from combra.metrics import compute_fid
>>>
>>> def load_batch(folder):
...     return np.stack([np.asarray(Image.open(p).convert('RGB'))
...                      for p in sorted(Path(folder).glob('*.png'))])
>>>
>>> BASE = 'data/separeted'
>>> pairs = [
...     ('256x256', f'{BASE}/o_bc_left_4x_768_256x256_N360', f'{BASE}/gen_diff_256x256_N500'),
...     ('512x512', f'{BASE}/o_bc_left_4x_768_512x512_N360', f'{BASE}/gen_diff_512x512_N500'),
...     ('768x768', f'{BASE}/o_bc_left_4x_768_768x768_N360', f'{BASE}/gen_diff_768x768_N5_000'),
... ]
>>> results = {}
>>> for name, real, gen in pairs:
...     fid = compute_fid(load_batch(real), load_batch(gen), batch_size=50)
...     results[name] = fid
...     print(f'{name}: FID = {fid:.4f}')
```

The InceptionV3 weights are downloaded once and cached by pytorch-fid, so subsequent calls reuse them.
