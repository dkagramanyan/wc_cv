"""Turn pilot_report.json into the criteria tables (a)-(f) of plan task 3."""

import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
rep = json.load(open(f"{OUT}/pilot_report.json"))


def f2(x):
    return f"{x:.2f}"


def crit_c(A, L):
    dmu = np.abs(np.asarray(A["mu"]) - L["mu"])
    rs = np.abs(np.asarray(A["sigma"]) / np.asarray(L["sigma"]) - 1)
    dp = abs(A["p"] - L["p"])
    return dmu, rs, dp, bool(dmu.max() < 2 and rs.max() < 0.1 and dp < 0.02)


print("## Fits (mu1, mu2 | sigma1, sigma2 | p)  and diagnostics\n")
print("| case | n | est | mu1 | mu2 | s1 | s2 | p | p_obs | P>180 | degenerate | nll/n | R ratio | conv | t_10starts s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for key, r in rep.items():
    rows = [("LSQ", r["LSQ"]), ("A", r["A"]), ("A+jit", r["A_jitter"])] + [(f"B h={h}", b) for h, b in r["B"].items()]
    for name, e in rows:
        print(
            f"| {key} | {r['n']} | {name} | {f2(e['mu'][0])} | {f2(e['mu'][1])} | {f2(e['sigma'][0])} | {f2(e['sigma'][1])} "
            f"| {e['p']:.4f} | {r['p_obs']:.4f} | {e['P_model_gt180']:.4f} | {','.join(e['degenerate']) or '-'} "
            f"| {e['nll_per_n']:.5f} | {e['R_ratio']:.3f} | {e.get('conv_frac', float('nan')):.1f} | {e.get('time_total', float('nan')):.1f} |"
        )

print("\n## Criteria for A (raw)\n")
print("| case | (a) non-degenerate | (b) conv frac>=0.5 | (c) |dmu|<2, |s ratio-1|<0.1, |dp|<0.02 | (d) R ratio<=1.1 | (e) split-half A<=LSQ (dmu1,dmu2,dp) | (f) t<=5s | pass |")
print("|---|---|---|---|---|---|---|---|")
for key, r in rep.items():
    A, L, S = r["A"], r["LSQ"], r["split_half"]
    a = not A["degenerate"]
    b = A["conv_frac"] >= 0.5
    dmu, rs, dp, c = crit_c(A, L)
    d = A["R_ratio"] <= 1.1
    e_vals = (S["A_dmu"][0] <= S["LSQ_dmu"][0], S["A_dmu"][1] <= S["LSQ_dmu"][1], S["A_dp"] <= S["LSQ_dp"])
    e = all(e_vals)
    f = A["time_total"] <= 5
    print(
        f"| {key} | {a} | {b} ({A['conv_frac']:.1f}) | {c} (dmu={np.round(dmu,2).tolist()}, rs={np.round(rs,3).tolist()}, dp={dp:.4f}) "
        f"| {d} ({A['R_ratio']:.3f}) | {e} (A {np.round(S['A_dmu'],2).tolist()},{S['A_dp']:.4f} vs LSQ {np.round(S['LSQ_dmu'],2).tolist()},{S['LSQ_dp']:.4f}) "
        f"| {f} ({A['time_total']:.1f}) | {all((a, b, d, e, f))} |"
    )

print("\n## Variant B step sensitivity\n")
print("| case | max |dmu| over h | max |dp| over h | h=0.1 degenerate | h=1 degenerate | h=5 degenerate | R ratio h=0.1/1/5 | t s h=0.1/1/5 |")
print("|---|---|---|---|---|---|---|---|")
for key, r in rep.items():
    B = r["B"]
    mus = np.array([B[h]["mu"] for h in B])
    ps = np.array([B[h]["p"] for h in B])
    dmu = np.max(np.abs(mus[:, None, :] - mus[None, :, :]))
    dp = np.max(np.abs(ps[:, None] - ps[None, :]))
    print(
        f"| {key} | {dmu:.2f} | {dp:.4f} | {','.join(B['0.1']['degenerate']) or '-'} | {','.join(B['1.0']['degenerate']) or '-'} | {','.join(B['5.0']['degenerate']) or '-'} "
        f"| {B['0.1']['R_ratio']:.3f}/{B['1.0']['R_ratio']:.3f}/{B['5.0']['R_ratio']:.3f} | {B['0.1']['time_total']:.1f}/{B['1.0']['time_total']:.1f}/{B['5.0']['time_total']:.1f} |"
    )

print("\n## Discretisation (1-degree peak ratio at grid angles; top repeated values)\n")
print("| case | 90 | 135 | 180 | 225 | 270 | top repeated (value: share) | share in values repeated >=1000x | stored-vs-recomputed LSQ maxdiff |")
print("|---|---|---|---|---|---|---|---|---|")
for key, r in rep.items():
    d = r["discretisation"]
    pk = d["peak_ratio_1deg"]
    top = ", ".join(f"{v:.2f}: {s:.3%}" for v, s in d["top_repeated"][:3])
    print(
        f"| {key} | " + " | ".join(f"{pk.get(str(c), pk.get(c, float('nan'))):.2f}" for c in (90, 135, 180, 225, 270))
        + f" | {top} | {d['share_repeated_ge_1000']:.3%} | {r['stored_vs_recomputed_lsq_maxdiff']:.2e} |"
    )
