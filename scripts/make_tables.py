"""Print every numeric table of the paper from results/*.json."""
import os, json, numpy as np
from scipy.stats import wilcoxon
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
J = lambda f: json.load(open("results/" + f))
M = J("map.json"); ms = ["LDPL", "TwoSlope", "IDW", "Kriging", "MLP", "FF-NF", "PINF", "PR-Kriging", "SKYMAP"]
fr = sorted(set(r["frac"] for r in M))
print("\nTable 3  RMSE (dB) vs sampling ratio")
for m in ms:
    print(f"{m:11s}", "  ".join(f"{np.mean([r['rmse'][m] for r in M if r['frac']==f]):.2f}±{np.std([r['rmse'][m] for r in M if r['frac']==f]):.2f}" for f in fr))
a = [r["rmse"]["SKYMAP"] for r in M]; b = [r["rmse"]["PR-Kriging"] for r in M]
print("Wilcoxon SKYMAP vs PR-Kriging:", wilcoxon(a, b))
print("\nTable 4  MAE @2% and time")
for m in ms:
    print(f"{m:11s} MAE {np.mean([r['mae'][m] for r in M if r['frac']==0.02]):.2f}  t2% {np.mean([r['time'][m] for r in M if r['frac']==0.02]):.2f}s  t5% {np.mean([r['time'][m] for r in M if r['frac']==0.05]):.2f}s")
H = [r["hyper"] for r in M if r["frac"] == 0.05]
print("\nTable 5  learned hyper-parameters @5%:", {k: (round(np.mean([h[k] for h in H]), 2), round(np.std([h[k] for h in H]), 2)) for k in H[0]})
A = J("abl.json")
print("\nTable 6  calibration")
for k in ["PR-Kriging", "SKYMAP"]:
    print(k, {m: round(np.mean([c[k][m] for c in A["cal"]]), 3) for m in ["picp68", "picp95", "mpiw95", "nll"]})
print("\nTable 7  ablation / mismatch")
for k in A["abl"][0]: print(f"{k:28s} {np.mean([r[k] for r in A['abl']]):.2f}±{np.std([r[k] for r in A['abl']]):.2f}")
for k in ["SKYMAP", "PR-Kriging", "PINF"]: print("mismatch", k, f"{np.mean([r[k] for r in A['mis']]):.2f}±{np.std([r[k] for r in A['mis']]):.2f}")
T = J("traj.json")
print("\nTable 8  planning (outage %, length m)")
for k in T[0]["paths"]:
    o = [100 * np.mean([x[0] for x in r["paths"][k]]) for r in T]; l = [np.mean([x[1] for x in r["paths"][k]]) for r in T]
    print(f"{k:12s} {np.mean(o):.1f}±{np.std(o):.1f}  {np.mean(l):.0f}±{np.std(l):.0f}")
B = J("budget.json")
print("\nTable 9  LB-RAP (outage %, extra length %, budget satisfaction, mean lambda, time s)")
for k in B[0]:
    o = [100 * np.mean([x[0] for x in r[k]]) for r in B]; l = [100 * (np.mean([x[1] for x in r[k]]) - 1) for r in B]
    print(f"{k:28s} {np.mean(o):.1f}±{np.std(o):.1f}  {np.mean(l):.1f}±{np.std(l):.1f}  {np.mean([x[2] for r in B for x in r[k]]):.2f}  {np.mean([x[3] for r in B for x in r[k]]):.1f}  {np.mean([x[4] for r in B for x in r[k]]):.3f}")
x = np.array([q[1] for r in B for q in r["SKYMAP-risk (lambda=8)"]])
print("fixed lambda=8 queries exceeding +5/+10/+20% length:", (x > 1.05).mean(), (x > 1.10).mean(), (x > 1.20).mean())
Ac = J("active.json"); Iv = J("active_ivr.json")
print("\nAdaptive sampling (RMSE at 3%)")
for s in ["Random", "Space-filling", "SKYMAP-UG"]: print(s, round(np.mean([r[s][-1] for r in Ac]), 2), round(np.std([r[s][-1] for r in Ac]), 2))
print("IVR", round(np.mean([r["SKYMAP-IVR"][-1] for r in Iv]), 2), round(np.std([r["SKYMAP-IVR"][-1] for r in Iv]), 2))
