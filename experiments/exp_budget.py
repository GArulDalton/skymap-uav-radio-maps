import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
os.makedirs("results", exist_ok=True)
"""Exp 5: length-budgeted risk-aware planning (Algorithm 3) vs fixed-weight planners."""
import numpy as np, json, time, warnings; warnings.filterwarnings('ignore')
from env import *; from methods import fit_pr_kriging; from skygp import SkyGP
from planner import graph, plan, path_stats, budget_plan
TH, NP, MC = 0.0, 40, 256
DELTAS = [0.05, 0.10, 0.20]
out = []
for sd in [1, 2, 3, 4, 5]:
    S = Scenario(sd); node, E, length = graph(S); v = S.idx_valid
    rng = np.random.default_rng(300 + sd); idx = rng.choice(v, int(0.02 * len(v)), replace=False); Y = S.measure(idx)
    true_out = (sinr_db(S.PR[node]) < TH).astype(float)
    g = SkyGP(S).fit(idx, Y); mu, sdv = g.predict(node, return_std=True)
    Mp = fit_pr_kriging(S, idx, Y)[node]
    smp = mu[None] + sdv[None] * np.random.default_rng(sd).standard_normal((MC,) + mu.shape)
    pout = np.stack([(sinr_db(smp[m]) < TH) for m in range(MC)]).mean(0)
    c_prk = (sinr_db(Mp) < TH).astype(float)
    z = S.P[node, 2]; r = np.random.default_rng(700 + sd); pairs = []
    while len(pairs) < NP:
        a, b = r.integers(0, len(node), 2)
        if np.linalg.norm(S.P[node[a], :2] - S.P[node[b], :2]) >= 400 and z[a] >= 60 and z[b] >= 60:
            pairs.append((a, b))
    rec = {}
    for a, b in pairs:
        L0 = path_stats(S, node, plan(node, E, length, pout, a, b, 0.0), true_out)[1]
        p = plan(node, E, length, pout, a, b, 8.0); o, l = path_stats(S, node, p, true_out)
        rec.setdefault('SKYMAP-risk (lambda=8)', []).append((o, l / L0, 1.0, 8.0, 0.0))
        for d in DELTAS:
            for name, c in [('PR-Kriging', c_prk), ('SKYMAP', pout)]:
                t0 = time.time(); p, lam = budget_plan(S, node, E, length, c, a, b, d); dt = time.time() - t0
                o, l = path_stats(S, node, p, true_out)
                rec.setdefault(f'{name} LB-RAP d={d}', []).append((o, l / L0, float(l / L0 <= 1 + d + 1e-9), lam, dt))
    out.append(rec)
    print(sd, {k: (round(100 * np.mean([x[0] for x in v_]), 1), round(np.mean([x[1] for x in v_]), 3), np.mean([x[2] for x in v_]), round(np.mean([x[4] for x in v_]), 2)) for k, v_ in rec.items()}, flush=True)
    json.dump(out, open('results/budget.json', 'w'))
