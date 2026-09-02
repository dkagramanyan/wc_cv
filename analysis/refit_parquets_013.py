"""Refit stored angle parquets to the combra 0.13.0 schema (mass share, not amplitudes).

Reads each parquet, refits every stored density with the five-parameter
fit, replaces the `angles_gauss_amps` column with `angles_gauss_shares` and
rewrites the file under the current ANGLES_SCHEMA. The original is kept next
to it as `<name>.pre013`. No h5 access or angle re-extraction is involved.

Usage: python refit_parquets_013.py [--dry-run]
"""

import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import combra
from combra.fitting import fit_bimodal_gaussian
from combra.io.schema import ANGLES_SCHEMA

ROOT = Path("/home/david/mnt/ssd_2_sata/phd/wc_cv/data")
PATTERNS = ("angles/*_msl5/angles_n*.parquet", "angles_sweep_extraction/*/angles_n360.parquet")
DRY = "--dry-run" in sys.argv


def refit(path: Path) -> None:
    names = pq.read_schema(path).names
    table = pq.read_table(path)
    rows = table.to_pylist()
    prep_names = table.schema.field("prep_per_step").type.value_type.names
    if "angles_gauss_amps" not in prep_names:
        print(f"skip (already current): {path}")
        return
    n_fits = 0
    for row in rows:
        for st in row["prep_per_step"]:
            x = np.asarray(st["angles_density_x"], float)
            y = np.asarray(st["angles_density_y"], float)
            (x_g, y_g), mus, sigmas, shares, _ = fit_bimodal_gaussian(x, y)
            st.pop("angles_gauss_amps")
            st["angles_gauss_x"] = [float(v) for v in x_g]
            st["angles_gauss_y"] = [float(v) for v in y_g]
            st["angles_gauss_mus"] = [float(v) for v in mus]
            st["angles_gauss_sigmas"] = [float(v) for v in sigmas]
            st["angles_gauss_shares"] = [float(v) for v in shares]
            n_fits += 1
        note = row["run_meta"].get("notes") or ""
        row["run_meta"]["notes"] = (note + "; " if note else "") + f"fits refitted with combra {combra.__version__}"
    out = pa.Table.from_pylist(rows, schema=ANGLES_SCHEMA)
    if DRY:
        print(f"would rewrite {path} ({n_fits} fits)")
        return
    backup = path.with_suffix(path.suffix + ".pre013")
    if not backup.exists():
        shutil.copy2(path, backup)
    pq.write_table(out, path)
    print(f"rewrote {path} ({n_fits} fits; backup {backup.name}); columns were {names}")


for pattern in PATTERNS:
    for f in sorted(glob.glob(str(ROOT / pattern))):
        refit(Path(f))
