"""Follow-up to the pilot: two truncated Gaussians + uniform background,
fitted by the grouped likelihood (variant B, h = 1), against LSQ."""
# NOTE: this pilot ran on 2026-09-03 against the combra 0.12.0 parquets, whose
# stored fits carried two amplitudes under `angles_gauss_amps`. The parquets
# were refitted to the 0.13.0 schema afterwards (`angles_gauss_shares`), so the
# stored-fit comparisons here no longer run as-is; the JSON reports and figures
# next to this file are the record of the pilot.


import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize

from pilot_mle import (ANGLES_DIR, HI, M_STARTS, OUT, degenerate, group_counts, lsq_fit, mix_cdf,
                       mix_pdf, pack, rss5, seeds_from_density, sigmoid, unpack, nll_A)
from combra import stats

Q_MAX = 0.5


def unpack_bg(u):
    p, mu, sg = unpack(u[:5])
    return p, mu, sg, Q_MAX * sigmoid(u[5])


def cdf_bg(theta, p, mu, sg, q):
    return (1 - q) * mix_cdf(theta, p, mu, sg) + q * np.clip(theta, 0, HI) / HI


def pdf_bg(theta, p, mu, sg, q):
    return (1 - q) * mix_pdf(theta, p, mu, sg) + q / HI


def nll_bg(u, lo, hi, counts):
    p, mu, sg, q = unpack_bg(u)
    pi = cdf_bg(hi, p, mu, sg, q) - cdf_bg(lo, p, mu, sg, q)
    return -np.sum(counts * np.log(np.maximum(pi, 1e-300)))


DATA = {}


def run(job):
    key, i, u0 = job
    t = time.perf_counter()
    r = minimize(nll_bg, u0, args=DATA[key], method="L-BFGS-B", options={"maxiter": 500})
    return key, i, float(r.fun), r.x.tolist(), time.perf_counter() - t


def main():
    rng = np.random.default_rng(0)
    cases, jobs = {}, []
    for f in sorted(glob.glob(f"{ANGLES_DIR}/o_bc_left_4x_1536_1024x1024_*_msl5/angles_n360.parquet")):
        res = f.split("1536_")[1].split("_rgb")[0].split("_")[1]
        for row in pq.read_table(f).to_pylist():
            key = f"{res}/{row['meta']['name'].replace('Ultra_', '')}"
            theta = np.asarray(row["raw"]["angles_series"], float)
            cases[key] = theta
            counts, lo, hi = group_counts(theta, 1.0)
            DATA[key] = (lo, hi, counts)
            x5, y5 = stats.density_histogram(theta, 5.0)
            for i, s in enumerate(seeds_from_density(x5, y5, M_STARTS, rng)):
                jobs.append((key, i, np.append(s, np.log(0.02 / (Q_MAX - 0.02)))))
    res = {}
    with ProcessPoolExecutor(max_workers=22) as ex:
        for key, i, fun, x, dt in ex.map(run, jobs):
            res.setdefault(key, []).append((fun, x, dt))

    print("| case | n | est | mu1 | mu2 | s1 | s2 | p | q (bg) | p_obs | P>180 | degenerate | nll/n | R ratio | conv | t s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("| case | (c) dmu | (c) sigma ratio-1 | (c) dp | (c) pass | (d) R ratio | LSQ unclaimed mass 1-sum(amp)/5 |")
    crit = []
    out = {}
    for key, theta in cases.items():
        n = theta.size
        rows = res[key]
        funs = np.array([r[0] for r in rows])
        b = int(np.argmin(funs))
        p, mu, sg, q = unpack_bg(np.asarray(rows[b][1]))
        if mu[0] > mu[1]:
            mu, sg, p = mu[::-1], sg[::-1], 1 - p
        conv = float(np.mean((funs - funs[b]) / n < 1e-3))
        L = lsq_fit(theta)
        x5, y5 = L["x5"], L["y5"]
        curve_lsq = stats.truncated_bimodal_gaussian(x5, *L["mu"], *L["sigma"], *L["amps"])
        R_lsq = rss5(x5, y5, curve_lsq)
        R = rss5(x5, y5, 5.0 * pdf_bg(x5, p, mu, sg, q)) / R_lsq
        p_obs = float(np.mean(theta > 180))
        P180 = float(1 - cdf_bg(np.array([180.0]), p, mu, sg, q)[0])
        # raw log-likelihood per observation of the bg model, for comparison with the pilot table
        f_raw = pdf_bg(theta, p, mu, sg, q)
        nll_n = float(-np.mean(np.log(f_raw)))
        dmu = np.abs(mu - L["mu"])
        rs = np.abs(sg / np.asarray(L["sigma"]) - 1)
        dp = abs(p - L["p"])
        c_pass = bool(dmu.max() < 2 and rs.max() < 0.1 and dp < 0.02)
        unclaimed = 1 - sum(L["amps"]) / 5.0
        out[key] = dict(p=float(p), q=float(q), mu=mu.tolist(), sigma=sg.tolist(), nll_per_n=nll_n, R_ratio=R, P180=P180, p_obs=p_obs, conv=conv, lsq=dict(p=L["p"], mu=L["mu"], sigma=L["sigma"], unclaimed=unclaimed))
        print(f"| {key} | {n} | LSQ | {L['mu'][0]:.2f} | {L['mu'][1]:.2f} | {L['sigma'][0]:.2f} | {L['sigma'][1]:.2f} | {L['p']:.4f} | - | {p_obs:.4f} | - | - | - | 1.000 | - | - |")
        print(f"| {key} | {n} | MLE-B h=1 + bg | {mu[0]:.2f} | {mu[1]:.2f} | {sg[0]:.2f} | {sg[1]:.2f} | {p:.4f} | {q:.4f} | {p_obs:.4f} | {P180:.4f} | {','.join(degenerate(p, mu, sg)) or '-'} | {nll_n:.5f} | {R:.3f} | {conv:.1f} | {sum(r[2] for r in rows):.1f} |")
        crit.append(f"| {key} | {np.round(dmu, 2).tolist()} | {np.round(rs, 3).tolist()} | {dp:.4f} | {c_pass} | {R:.3f} | {unclaimed:+.4f} |")
    print()
    print("\n".join(crit))
    json.dump(out, open(f"{OUT}/pilot_bg_report.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    xx = np.linspace(0, 360, 721)
    for ax, (key, theta) in zip(axes.flat, cases.items()):
        x1, y1 = stats.density_histogram(theta, 1.0)
        ax.bar(x1, y1, width=1.0, color="0.8", label="1° histogram")
        o = out[key]
        L = o["lsq"]
        lf = lsq_fit(theta)
        ax.plot(xx, stats.truncated_bimodal_gaussian(xx, *L["mu"], *L["sigma"], *lf["amps"]) / 5.0, "k-", lw=1.2, label="LSQ (5°)/5")
        ax.plot(xx, pdf_bg(xx, o["p"], np.asarray(o["mu"]), np.asarray(o["sigma"]), o["q"]), "r-", lw=1.2, label=f"MLE + uniform bg (q={o['q']:.3f})")
        ax.set_title(f"{key}  n={theta.size}")
        ax.set_xlim(0, 360)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pilot_bg_fits.png", dpi=110)


if __name__ == "__main__":
    main()
