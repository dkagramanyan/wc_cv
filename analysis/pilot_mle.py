"""Pilot (plan task 3): p-parameterized truncated bimodal mixture fitted by
maximum likelihood on raw angles (A) and on grouped counts (B), against
combra's least-squares histogram fit (LSQ), on the three reference parquets.

Usage: python pilot_mle.py synthetic | pilot
"""

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import check_grad, minimize
from scipy.special import ndtr

from combra import fitting, stats
from combra.fitting.fit import _seed_bimodal

LO, HI = 0.0, 360.0
SIG_MIN, SIG_MAX = 1.0, 180.0
SQRT2PI = np.sqrt(2 * np.pi)
M_STARTS = 10
ANGLES_DIR = "/home/david/mnt/ssd_2_sata/phd/wc_cv/data/angles"
OUT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------- model
def _phi(z):
    return np.exp(-0.5 * z * z) / SQRT2PI


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def logit(s):
    return np.log(s / (1.0 - s))


def unpack(u):
    """u = (lam, v1, s1, v2, s2) -> p, mu[2], sg[2]."""
    p = sigmoid(u[0])
    mu = HI * sigmoid(u[1:5:2])
    sg = SIG_MIN + (SIG_MAX - SIG_MIN) * sigmoid(u[2:5:2])
    return p, mu, sg


def pack(p, mu, sg):
    p = float(np.clip(p, 1e-3, 1 - 1e-3))
    mu = np.clip(np.asarray(mu, float), 1.0, HI - 1.0)
    sg = np.clip(np.asarray(sg, float), SIG_MIN + 0.1, SIG_MAX - 1.0)
    r = SIG_MAX - SIG_MIN
    return np.array(
        [logit(p), logit(mu[0] / HI), logit((sg[0] - SIG_MIN) / r), logit(mu[1] / HI), logit((sg[1] - SIG_MIN) / r)]
    )


def comp_pdf(theta, mu, sg):
    a, b = (LO - mu) / sg, (HI - mu) / sg
    Z = ndtr(b) - ndtr(a)
    z = (theta - mu) / sg
    return _phi(z) / (sg * Z), z, Z, a, b


def comp_cdf(theta, mu, sg):
    a, b = (LO - mu) / sg, (HI - mu) / sg
    Z = ndtr(b) - ndtr(a)
    t = np.clip(theta, LO, HI)
    return (ndtr((t - mu) / sg) - ndtr(a)) / Z


def mix_pdf(theta, p, mu, sg):
    return p * comp_pdf(theta, mu[0], sg[0])[0] + (1 - p) * comp_pdf(theta, mu[1], sg[1])[0]


def mix_cdf(theta, p, mu, sg):
    return p * comp_cdf(theta, mu[0], sg[0]) + (1 - p) * comp_cdf(theta, mu[1], sg[1])


# ------------------------------------------------------- likelihoods
def nll_A(u, theta):
    """Negative log-likelihood on raw angles with analytic gradient in u."""
    p, mu, sg = unpack(u)
    g1, z1, Z1, a1, b1 = comp_pdf(theta, mu[0], sg[0])
    g2, z2, Z2, a2, b2 = comp_pdf(theta, mu[1], sg[1])
    f = p * g1 + (1 - p) * g2
    nll = -np.sum(np.log(f))
    w1 = p * g1 / f
    w2 = (1 - p) * g2 / f
    dp = np.sum((g1 - g2) / f)

    def comp_grad(z, Z, a, b, s, w):
        dZdmu = -(_phi(b) - _phi(a)) / s
        dZdsg = -(b * _phi(b) - a * _phi(a)) / s
        return np.sum(w * (z / s - dZdmu / Z)), np.sum(w * ((z * z - 1) / s - dZdsg / Z))

    dmu1, dsg1 = comp_grad(z1, Z1, a1, b1, sg[0], w1)
    dmu2, dsg2 = comp_grad(z2, Z2, a2, b2, sg[1], w2)
    r = SIG_MAX - SIG_MIN
    jac = np.array(
        [
            p * (1 - p),
            mu[0] * (1 - mu[0] / HI),
            (sg[0] - SIG_MIN) * (1 - (sg[0] - SIG_MIN) / r),
            mu[1] * (1 - mu[1] / HI),
            (sg[1] - SIG_MIN) * (1 - (sg[1] - SIG_MIN) / r),
        ]
    )
    return nll, -np.array([dp, dmu1, dsg1, dmu2, dsg2]) * jac


def group_counts(theta, h):
    """Counts per bin centred at k*h (combra.stats.density_histogram convention), all bins."""
    q = np.round(theta / h).astype(np.int64)
    K = int(round(HI / h))
    counts = np.bincount(q, minlength=K + 1).astype(float)
    k = np.arange(K + 1)
    return counts, (k - 0.5) * h, (k + 0.5) * h


def nll_B(u, lo, hi, counts):
    p, mu, sg = unpack(u)
    pi = mix_cdf(hi, p, mu, sg) - mix_cdf(lo, p, mu, sg)
    return -np.sum(counts * np.log(np.maximum(pi, 1e-300)))


# ----------------------------------------------------------- driver
def seeds_from_density(x5, y5, m, rng):
    mu1, mu2, s1, s2, a1, a2 = _seed_bimodal(np.asarray(x5, float), np.asarray(y5, float))
    p0 = a1 / (a1 + a2) if a1 + a2 > 0 else 0.5
    base = (p0, [mu1, mu2], [s1, s2])
    out = [pack(*base)]
    for _ in range(m - 1):
        out.append(
            pack(
                p0 + rng.uniform(-0.1, 0.1),
                np.array([mu1, mu2]) + rng.uniform(-15, 15, 2),
                np.array([s1, s2]) * np.exp(rng.uniform(-0.5, 0.5, 2)),
            )
        )
    return out


DATA = {}  # key -> arrays, filled before forking the pool


def _run_start(job):
    kind, key, i, u0 = job
    t = time.perf_counter()
    if kind == "A":
        r = minimize(nll_A, u0, args=(DATA[key],), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    else:
        r = minimize(nll_B, u0, args=DATA[key], method="L-BFGS-B", options={"maxiter": 500})
    return kind, key, i, float(r.fun), r.x.tolist(), int(r.nit), time.perf_counter() - t, bool(r.success)


def finish(rows, n):
    """rows: list of (fun, x, nit, dt, ok) for one fit; returns summary dict."""
    funs = np.array([r[0] for r in rows])
    best = int(np.argmin(funs))
    p, mu, sg = unpack(np.asarray(rows[best][1]))
    if mu[0] > mu[1]:
        mu, sg, p = mu[::-1], sg[::-1], 1 - p
    conv = float(np.mean((funs - funs[best]) / n < 1e-3))
    return {
        "p": float(p),
        "mu": mu.tolist(),
        "sigma": sg.tolist(),
        "nll": float(funs[best]),
        "conv_frac": conv,
        "time_total": float(sum(r[3] for r in rows)),
        "time_best": float(rows[best][3]),
        "nit_best": rows[best][2],
        "n_ok": int(sum(r[4] for r in rows)),
    }


def degenerate(p, mu, sg):
    reasons = []
    if min(p, 1 - p) < 0.05:
        reasons.append("mass-share")
    if min(mu) < 5 or max(mu) > 355:
        reasons.append("mu-edge")
    if max(sg) > 120:
        reasons.append("pedestal")
    if abs(mu[0] - mu[1]) < max(sg):
        reasons.append("overlap")
    return reasons


def lsq_fit(theta):
    x5, y5 = stats.density_histogram(theta, step=5.0)
    fit = fitting.fit_bimodal_gaussian(x5, y5)
    amps = np.asarray(fit.amps)
    return {
        "p": float(amps[0] / amps.sum()),
        "mu": list(fit.mus),
        "sigma": list(fit.sigmas),
        "amps": amps.tolist(),
        "x5": np.asarray(x5, float),
        "y5": np.asarray(y5, float),
    }


def rss5(x5, y5, curve):
    return float(np.sum((curve - y5) ** 2))


def discretisation_report(theta):
    x1, y1 = stats.density_histogram(theta, step=1.0)
    x1 = np.asarray(x1, float)
    y1 = np.asarray(y1, float)
    peaks = {}
    for c in (90, 135, 180, 225, 270):
        i = np.flatnonzero(np.isclose(x1, c))
        if i.size == 0:
            continue
        nb = [np.flatnonzero(np.isclose(x1, c + d)) for d in (-5, -4, -3, -2, 2, 3, 4, 5)]
        nb = [j[0] for j in nb if j.size]
        peaks[c] = float(y1[i[0]] / np.mean(y1[nb]))
    vals, cnt = np.unique(np.round(theta, 6), return_counts=True)
    top = np.argsort(cnt)[::-1][:5]
    return {
        "peak_ratio_1deg": peaks,
        "top_repeated": [(float(vals[i]), float(cnt[i] / theta.size)) for i in top],
        "share_repeated_ge_1000": float(cnt[cnt >= 1000].sum() / theta.size),
    }


# --------------------------------------------------------- synthetic
def sample_truncated(p, mu, sg, n, rng):
    out = np.empty(n)
    comp = rng.random(n) < p
    for k, mask in ((0, comp), (1, ~comp)):
        m = int(mask.sum())
        draws = np.empty(0)
        while draws.size < m:
            d = rng.normal(mu[k], sg[k], 2 * m)
            draws = np.concatenate([draws, d[(d >= LO) & (d <= HI)]])
        out[mask] = draws[:m]
    return out


def synthetic():
    rng = np.random.default_rng(0)
    truth = (0.78, np.array([106.0, 241.0]), np.array([33.0, 25.0]))
    theta = sample_truncated(*truth, 200_000, rng)
    sub = theta[:5000]
    u0 = pack(0.6, [90, 250], [20, 40])
    err = check_grad(lambda u: nll_A(u, sub)[0], lambda u: nll_A(u, sub)[1], u0)
    print(f"gradient check |num-ana| = {err:.3e}  (scale {np.linalg.norm(nll_A(u0, sub)[1]):.3e})")
    x5, y5 = stats.density_histogram(theta, 5.0)
    seeds = seeds_from_density(x5, y5, M_STARTS, rng)
    rows = []
    for i, s in enumerate(seeds):
        r = minimize(nll_A, s, args=(theta,), jac=True, method="L-BFGS-B")
        rows.append((r.fun, r.x, r.nit, 0.0, r.success))
    a = finish(rows, theta.size)
    counts, lo, hi = group_counts(theta, 1.0)
    rows = []
    for s in seeds:
        r = minimize(nll_B, s, args=(lo, hi, counts), method="L-BFGS-B")
        rows.append((r.fun, r.x, r.nit, 0.0, r.success))
    b = finish(rows, theta.size)
    lsq = lsq_fit(theta)
    print("truth   p=%.4f mu=%s sigma=%s" % (truth[0], truth[1].tolist(), truth[2].tolist()))
    for name, f in (("A raw ", a), ("B h=1 ", b), ("LSQ   ", lsq)):
        print("%s p=%.4f mu=%s sigma=%s" % (name, f["p"], np.round(f["mu"], 2).tolist(), np.round(f["sigma"], 2).tolist()))
    # sigma -> 0 guard: a spike of repeated values must not swallow a component
    spike = np.concatenate([theta, np.full(int(0.03 * theta.size), 90.0)])
    rows = []
    for s in seeds_from_density(*stats.density_histogram(spike, 5.0), M_STARTS, rng):
        r = minimize(nll_A, s, args=(spike,), jac=True, method="L-BFGS-B")
        rows.append((r.fun, r.x, r.nit, 0.0, r.success))
    sp = finish(rows, spike.size)
    print("spike3%% at 90: p=%.4f mu=%s sigma=%s degenerate=%s" % (sp["p"], np.round(sp["mu"], 2).tolist(), np.round(sp["sigma"], 2).tolist(), degenerate(sp["p"], sp["mu"], sp["sigma"])))


# --------------------------------------------------------------- pilot
def pilot():
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(f"{ANGLES_DIR}/o_bc_left_4x_1536_1024x1024_*_msl5/angles_n360.parquet"))
    cases = {}
    jobs = []
    for f in files:
        res = f.split("1024x1024_")[1].split("_rgb")[0]
        for row in pq.read_table(f).to_pylist():
            name = row["meta"]["name"].replace("Ultra_", "")
            key = f"{res}/{name}"
            theta = np.asarray(row["raw"]["angles_series"], float)
            per_image = [np.asarray(a, float) for a in row["raw"]["angles_per_image"]]
            half1 = np.concatenate(per_image[0::2])
            half2 = np.concatenate(per_image[1::2])
            stored5 = next(p for p in row["prep_per_step"] if abs(p["step"] - 5.0) < 1e-9)
            cases[key] = {"theta": theta, "half1": half1, "half2": half2, "stored5": stored5}
            x5, y5 = stats.density_histogram(theta, 5.0)
            seeds = seeds_from_density(x5, y5, M_STARTS, rng)
            DATA[f"A/{key}"] = theta
            jobs += [("A", f"A/{key}", i, s) for i, s in enumerate(seeds)]
            DATA[f"Ajit/{key}"] = theta + rng.uniform(-0.5, 0.5, theta.size)
            jobs += [("A", f"Ajit/{key}", i, s) for i, s in enumerate(seeds)]
            for tag, th in (("Ah1", half1), ("Ah2", half2)):
                DATA[f"{tag}/{key}"] = th
                jobs += [("A", f"{tag}/{key}", i, s) for i, s in enumerate(seeds)]
            for h in (0.1, 1.0, 5.0):
                DATA[f"B{h}/{key}"] = group_counts(theta, h)[::1]
                counts, lo, hi = group_counts(theta, h)
                DATA[f"B{h}/{key}"] = (lo, hi, counts)
                jobs += [("B", f"B{h}/{key}", i, s) for i, s in enumerate(seeds)]
    print(f"{len(cases)} cases, {len(jobs)} optimizer runs", flush=True)

    t0 = time.perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=22) as ex:
        for kind, key, i, fun, x, nit, dt, ok in ex.map(_run_start, jobs, chunksize=1):
            results.setdefault(key, []).append((fun, x, nit, dt, ok))
    print(f"all fits done in {time.perf_counter() - t0:.1f}s wall", flush=True)

    report = {}
    for key, c in cases.items():
        theta = c["theta"]
        n = theta.size
        lsq = lsq_fit(theta)
        st = c["stored5"]
        stored_diff = float(max(np.max(np.abs(np.asarray(st["angles_gauss_mus"]) - lsq["mu"])), np.max(np.abs(np.asarray(st["angles_gauss_sigmas"]) - lsq["sigma"]))))
        A = finish(results[f"A/{key}"], n)
        Aj = finish(results[f"Ajit/{key}"], n)
        Ah1 = finish(results[f"Ah1/{key}"], c["half1"].size)
        Ah2 = finish(results[f"Ah2/{key}"], c["half2"].size)
        Bs = {h: finish(results[f"B{h}/{key}"], n) for h in (0.1, 1.0, 5.0)}
        L1, L2 = lsq_fit(c["half1"]), lsq_fit(c["half2"])
        x5, y5 = lsq["x5"], lsq["y5"]
        curve_lsq = stats.truncated_bimodal_gaussian(x5, *lsq["mu"], *lsq["sigma"], *lsq["amps"])
        R_lsq = rss5(x5, y5, curve_lsq)
        p_obs = float(np.mean(theta > 180))

        def diag(f):
            p, mu, sg = f["p"], np.asarray(f["mu"]), np.asarray(f["sigma"])
            u = pack(p, mu, sg)
            return {
                "degenerate": degenerate(p, mu, sg),
                "R_ratio": rss5(x5, y5, 5.0 * mix_pdf(x5, p, mu, sg)) / R_lsq,
                "nll_per_n": nll_A(u, theta)[0] / n,
                "P_model_gt180": float(1 - mix_cdf(np.array([180.0]), p, mu, sg)[0]),
            }

        rep = {
            "n": int(n),
            "p_obs": p_obs,
            "stored_vs_recomputed_lsq_maxdiff": stored_diff,
            "discretisation": discretisation_report(theta),
            "LSQ": {"p": lsq["p"], "mu": lsq["mu"], "sigma": lsq["sigma"], **diag(lsq)},
            "A": {**A, **diag(A)},
            "A_jitter": {**Aj, **diag(Aj)},
            "B": {str(h): {**b, **diag(b)} for h, b in Bs.items()},
            "split_half": {
                "A_dmu": np.abs(np.asarray(Ah1["mu"]) - Ah2["mu"]).tolist(),
                "A_dp": abs(Ah1["p"] - Ah2["p"]),
                "A_dsigma": np.abs(np.asarray(Ah1["sigma"]) - Ah2["sigma"]).tolist(),
                "LSQ_dmu": np.abs(np.asarray(L1["mu"]) - L2["mu"]).tolist(),
                "LSQ_dp": abs(L1["p"] - L2["p"]),
                "LSQ_dsigma": np.abs(np.asarray(L1["sigma"]) - L2["sigma"]).tolist(),
            },
        }
        report[key] = rep
    with open(f"{OUT}/pilot_report.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print(json.dumps(report, indent=1, default=float))
    plot(cases, report)


def plot(cases, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(cases)
    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    for ax, key in zip(axes.flat, keys):
        theta = cases[key]["theta"]
        x1, y1 = stats.density_histogram(theta, 1.0)
        ax.bar(x1, y1, width=1.0, color="0.8", label="1° histogram")
        xx = np.linspace(0, 360, 721)
        r = report[key]
        L = r["LSQ"]
        amps = np.asarray(cases[key]["stored5"]["angles_gauss_amps"])
        ax.plot(xx, stats.truncated_bimodal_gaussian(xx, *L["mu"], *L["sigma"], *amps) / 5.0, "k-", lw=1.2, label="LSQ (5°)/5")
        A = r["A"]
        ax.plot(xx, mix_pdf(xx, A["p"], np.asarray(A["mu"]), np.asarray(A["sigma"])), "r-", lw=1.2, label="MLE A raw")
        B = r["B"]["1.0"]
        ax.plot(xx, mix_pdf(xx, B["p"], np.asarray(B["mu"]), np.asarray(B["sigma"])), "b--", lw=1.0, label="MLE B h=1")
        ax.set_title(f"{key}  n={r['n']}  p_obs={r['p_obs']:.3f}")
        ax.set_xlim(0, 360)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pilot_fits.png", dpi=110)


if __name__ == "__main__":
    {"synthetic": synthetic, "pilot": pilot}[sys.argv[1]]()
