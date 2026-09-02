"""Extraction-parameter sweep on the 1024 px reference set.

Regenerates the angle parquet for every (min_segment_len, angles_tol) pair so
the sensitivity of the fitted modes, the grid peaks, the 180-degree trough and
the edge tails to the contour simplification can be tabulated. Reuses the
preprocessing cache next to the source (built 2026-08-27), so each run costs
only the angle extraction. Same source, classes and n as
angles_generation_plot.ipynb; the msl=5, tol=3 cell reproduces
data/angles/<stem>_msl5/angles_n360.parquet.
"""

import itertools
import logging
import time
from pathlib import Path

from combra import data

ROOT = Path("/home/david/mnt/ssd_2_sata/phd/wc_cv")
SRC = ROOT / "datasets/san/o_bc_left_4x_1536_1024x1024_1024x1024_rgb_N360"
OUT = ROOT / "data/angles_sweep_extraction"
TYPES_DICT = {
    "Ultra_Co11": "мелкие зерна",
    "Ultra_Co25": "средние зерна",
    "Ultra_Co8": "средне-мелкие зерна",
    "Ultra_Co6_2": "крупные зерна",
    "Ultra_Co15": "средне-мелкие зерна",
}
STEPS = [1.0, 5.0]
MSL = (5.0, 10.0, 15.0)
TOL = (2, 3, 5)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

for msl, tol in itertools.product(MSL, TOL):
    out = OUT / f"{SRC.name}_msl{int(msl)}_tol{tol}"
    t = time.perf_counter()
    data.sweep_angles(
        SRC,
        out,
        ns=[360],
        step=STEPS,
        class_types=TYPES_DICT,
        tag=f"msl{int(msl)}-tol{tol}",
        force=False,
        run_meta={
            "family": "real",
            "resolution": 1024,
            "tags": ["extraction-sweep"],
            "notes": f"min_segment_len={msl}, angles_tol={tol}",
        },
        workers=20,
        angles_tol=tol,
        min_segment_len=msl,
        keep_contours=False,
        chunksize=64,
    )
    print(f"[done] msl={msl} tol={tol} -> {out} in {time.perf_counter() - t:.0f}s", flush=True)
