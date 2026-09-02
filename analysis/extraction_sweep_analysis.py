"""Tabulate the extraction-parameter sweep written by extraction_sweep.py.

Per (min_segment_len, angles_tol, class): angle count, the stored 5-degree
LSQ fit, the model-free reflex share, the fit's reflex share, the unclaimed
mass, the relative residual, the pixel-grid peaks at 45 and 90 degrees, the
share of the sample in the five most repeated exact values, the observed and
fitted mass in the 180-degree trough and in the edge tails.
"""

import glob
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import norm

from combra import stats

ROOT = Path("/home/david/mnt/ssd_2_sata/phd/wc_cv")
SWEEP = ROOT / "data/angles_sweep_extraction"
OUT = Path(__file__).resolve().parent


def mass_between(a, b, mus, sigmas, amps):
    tot = 0.0
    for mu, sg, amp in zip(mus, sigmas, amps, strict=True):
        z = norm.cdf(360, mu, sg) - norm.cdf(0, mu, sg)
        tot += amp * (norm.cdf(b, mu, sg) - norm.cdf(a, mu, sg)) / z
    return tot / sum(amps)


def peak_ratio(x1, y1, c):
    i = np.flatnonzero(np.isclose(x1, c))
    nb = [np.flatnonzero(np.isclose(x1, c + d)) for d in (-5, -4, -3, -2, 2, 3, 4, 5)]
    nb = [j[0] for j in nb if j.size]
    return float(y1[i[0]] / np.mean(y1[nb])) if i.size and nb else float("nan")


rows = []
hists = {}
for f in sorted(glob.glob(f"{SWEEP}/*_msl*_tol*/angles_n360.parquet")):
    m = re.search(r"_msl(\d+)_tol(\d+)/", f)
    msl, tol = int(m.group(1)), int(m.group(2))
    for r in pq.read_table(f).to_pylist():
        name = r["meta"]["name"].replace("Ultra_", "")
        theta = np.asarray(r["raw"]["angles_series"], float)
        n_img = r["meta"]["n_images"]
        st = next(p for p in r["prep_per_step"] if abs(p["step"] - 5.0) < 1e-9)
        x5, y5 = np.asarray(st["angles_density_x"], float), np.asarray(st["angles_density_y"], float)
        mus, sg, amps = st["angles_gauss_mus"], st["angles_gauss_sigmas"], st["angles_gauss_amps"]
        curve = stats.truncated_bimodal_gaussian(x5, *mus, *sg, *amps)
        rel_res = float(np.sum((curve - y5) ** 2) / np.sum(y5**2))
        x1, y1 = stats.density_histogram(theta, 1.0)
        x1, y1 = np.asarray(x1, float), np.asarray(y1, float)
        vals, cnt = np.unique(np.round(theta, 6), return_counts=True)
        top5 = float(np.sort(cnt)[::-1][:5].sum() / theta.size)
        p_obs = float(np.mean(theta > 180))
        p_fit = mass_between(180, 360, mus, sg, amps)
        trough_obs = float(np.mean((theta >= 170) & (theta < 190)))
        trough_fit = mass_between(170, 190, mus, sg, amps)
        tail_obs = float(np.mean((theta < 20) | (theta >= 340)))
        tail_fit = mass_between(0, 20, mus, sg, amps) + mass_between(340, 360, mus, sg, amps)
        rows.append(
            dict(
                msl=msl, tol=tol, cls=name, n=theta.size, per_img=theta.size / n_img,
                mu1=mus[0], mu2=mus[1], s1=sg[0], s2=sg[1], p=amps[0] / sum(amps),
                p_obs=p_obs, p_fit=p_fit, unclaimed=1 - sum(amps) / 5.0, rel_res=rel_res,
                pk45=peak_ratio(x1, y1, 45), pk90=peak_ratio(x1, y1, 90), top5=top5,
                trough_obs=trough_obs, trough_fit=trough_fit, tail_obs=tail_obs, tail_fit=tail_fit,
            )
        )
        hists[(name, msl, tol)] = (x1, y1, x5, curve, mus, sg, amps)

hdr = "| class | msl | tol | n | per img | mu1 | mu2 | s1 | s2 | p (amp share) | p_obs | p_fit | unclaimed | rel res | pk45 | pk90 | top5 repeated | trough obs/fit | tail obs/fit |"
print(hdr)
print("|" + "---|" * (hdr.count("|") - 1))
for r in sorted(rows, key=lambda r: (r["cls"], r["msl"], r["tol"])):
    print(
        f"| {r['cls']} | {r['msl']} | {r['tol']} | {r['n']} | {r['per_img']:.0f} | {r['mu1']:.1f} | {r['mu2']:.1f} | {r['s1']:.1f} | {r['s2']:.1f} "
        f"| {r['p']:.3f} | {r['p_obs']:.3f} | {r['p_fit']:.3f} | {r['unclaimed']:+.3f} | {r['rel_res']:.4f} | {r['pk45']:.2f} | {r['pk90']:.2f} | {r['top5']:.1%} "
        f"| {r['trough_obs']:.3f}/{r['trough_fit']:.3f} | {r['tail_obs']:.3f}/{r['tail_fit']:.3f} |"
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for cls in sorted({r["cls"] for r in rows}):
    fig, axes = plt.subplots(3, 3, figsize=(17, 11), sharex=True)
    for i, msl in enumerate((5, 10, 15)):
        for j, tol in enumerate((2, 3, 5)):
            ax = axes[i, j]
            x1, y1, x5, curve, mus, sg, amps = hists[(cls, msl, tol)]
            ax.bar(x1, y1, width=1.0, color="0.8")
            xx = np.linspace(0, 360, 721)
            ax.plot(xx, stats.truncated_bimodal_gaussian(xx, *mus, *sg, *amps) / 5.0, "k-", lw=1.2)
            rr = next(r for r in rows if (r["cls"], r["msl"], r["tol"]) == (cls, msl, tol))
            ax.set_title(f"{cls} msl={msl} tol={tol}  n={rr['n']}  mu=({mus[0]:.0f},{mus[1]:.0f}) s=({sg[0]:.0f},{sg[1]:.0f}) p_obs={rr['p_obs']:.3f}", fontsize=9)
            ax.set_xlim(0, 360)
    fig.tight_layout()
    fig.savefig(OUT / f"extraction_sweep_{cls}.png", dpi=100)
    print("saved", OUT / f"extraction_sweep_{cls}.png")
