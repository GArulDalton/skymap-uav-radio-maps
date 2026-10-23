import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
from env import Scenario, L
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
F = "figures/"
import os; os.makedirs(F, exist_ok=True)

# ---- Fig 1: framework (two stages)
fig, ax = plt.subplots(figsize=(7.4, 3.4)); ax.axis("off"); ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.6)
def box(x, y, w, h, t, c, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec="k", lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
ax.add_patch(FancyBboxPatch((0.05, 2.35), 10.3, 2.15, boxstyle="round,pad=0.02", fc="none", ec="0.4", lw=0.8, ls="--"))
ax.add_patch(FancyBboxPatch((0.05, 0.05), 10.3, 2.1, boxstyle="round,pad=0.02", fc="none", ec="0.4", lw=0.8, ls="--"))
ax.text(0.15, 4.32, "Stage I  Radio map construction (Algorithm 1)", fontsize=7.5, weight="bold")
ax.text(0.15, 1.95, "Stage II  Risk aware trajectory planning (Algorithms 2 and 3)", fontsize=7.5, weight="bold")
box(0.25, 2.65, 1.75, 1.4, "Sparse UAV\nRSRP reports\n(x_i, y_i,b)", "#dbe9f6")
box(2.3, 3.3, 1.75, 0.85, "3D city model\nLoS ray marching", "#eeeeee")
box(2.3, 2.5, 1.75, 0.7, "BS tilt and\nantenna pattern", "#eeeeee")
box(4.35, 2.65, 1.85, 1.4, "Physics prior m_b(x)\nridge calibrated\ntwo slope path loss", "#fde2c8")
box(6.5, 2.65, 1.85, 1.4, "Multicell LoS\nheteroscedastic GP\n(type II ML)", "#d7efd4")
box(8.6, 2.65, 1.6, 1.4, "Posterior map\nmean and std\nper cell", "#e8def3")
box(8.6, 0.35, 1.6, 1.35, "Monte Carlo\nSINR outage\nprobability", "#fff3c4")
box(6.5, 0.35, 1.85, 1.35, "Edge cost\nl_e [1 + lambda p_out]\n26 neighbour graph", "#f8d7da")
box(4.35, 0.35, 1.85, 1.35, "LB-RAP bisection\non lambda under\nlength budget", "#f8d7da")
box(2.3, 0.35, 1.75, 1.35, "Dijkstra\nshortest path", "#f8d7da")
box(0.25, 0.35, 1.75, 1.35, "Low outage\ntrajectory", "#dbe9f6")
arr = dict(arrowstyle="->", lw=1)
for p, q in [((2.0, 3.25), (4.35, 3.25)), ((4.05, 3.7), (4.35, 3.6)), ((4.05, 2.85), (4.35, 3.0)), ((6.2, 3.35), (6.5, 3.35)),
             ((8.35, 3.35), (8.6, 3.35)), ((9.4, 2.65), (9.4, 1.7)), ((8.6, 1.02), (8.35, 1.02)), ((6.5, 1.02), (6.2, 1.02)),
             ((4.35, 1.02), (4.05, 1.02)), ((2.3, 1.02), (2.0, 1.02))]:
    ax.annotate("", q, p, arrowprops=arr)
plt.savefig(F + "Fig2.png"); plt.close()

# ---- Fig 2: system model (side view)
from matplotlib.patches import Rectangle, Wedge
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
ax.plot([0, 100], [2, 2], "k-", lw=1)
rng_ = np.random.default_rng(4)
for x0, h in [(14, 12), (24, 20), (40, 9), (52, 7), (66, 14), (82, 18), (91, 8)]:
    ax.add_patch(Rectangle((x0, 2), 6, h, fc="0.85", ec="0.3", hatch="///", lw=0.6))
bs = [(6, 14), (46, 14), (76, 14)]
for k_, (x, h) in enumerate(bs):
    ax.plot([x, x], [2, h], "k-", lw=2); ax.plot(x, h, "k^", ms=8)
    ax.add_patch(Wedge((x, h), 16, -45, -10, fc="0.6", alpha=0.35, ec="none"))
    ax.add_patch(Wedge((x, h), 7, 20, 60, fc="0.6", alpha=0.25, ec="none"))
    ax.text(x, h + 2, "BS %d" % (k_ + 1), ha="center", fontsize=7)
ax.fill_between([0, 100], 30, 44, color="#dbe9f6", alpha=0.6, lw=0)
ax.text(1, 31, "Low altitude corridor (30 to 150 m)", fontsize=7)
ux, uz = 60, 36
ax.plot(ux, uz, marker=(4, 1, 45), ms=14, color="k"); ax.text(ux + 1.5, uz + 2, "UAV", fontsize=7)
ax.annotate("", (ux, uz), (46, 14), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(47, 26, "serving\nLoS link", fontsize=6.5, ha="right")
ax.annotate("", (ux, uz), (76, 14), arrowprops=dict(arrowstyle="->", lw=1.1, ls="--"))
ax.text(72, 26, "LoS\ninterference", fontsize=6.5)
ax.plot([6, 24], [14, 20], ":", lw=1.1, color="k"); ax.plot(24, 20, "kx", ms=6)
ax.text(10, 24, "NLoS (blocked)\nlink", fontsize=6.5)
xs_ = np.linspace(26, 95, 22); zs_ = 32 + 11 * rng_.random(22)
ax.plot(xs_, zs_, "o", ms=2.5, mfc="white", mec="k", mew=0.6)
ax.text(86, 45.5, "o  RSRP samples", fontsize=6.5)
ax.text(8, 5, "main lobe (down tilt)", fontsize=6); ax.text(8.5, 19.5, "side lobe", fontsize=6)
plt.savefig(F + "Fig1.png"); plt.close()

# ---- Fig 2: accuracy vs sampling
M = json.load(open("results/map.json"))
methods = ["LDPL", "TwoSlope", "IDW", "Kriging", "MLP", "FF-NF", "PINF", "PR-Kriging", "SKYMAP"]
lab = {"TwoSlope": "Two-slope (LoS)", "FF-NF": "FF neural field", "PINF": "Physics-inf. neural field"}
fr = sorted(set(r["frac"] for r in M))
tab = {m: {f: [r["rmse"][m] for r in M if r["frac"] == f] for f in fr} for m in methods}
fig, ax = plt.subplots(figsize=(3.5, 2.8))
mk = dict(zip(methods, "vx^sDP<ho"))
for m in methods:
    mu = [np.mean(tab[m][f]) for f in fr]; sd = [np.std(tab[m][f]) for f in fr]
    ax.errorbar([100 * f for f in fr], mu, yerr=sd, marker=mk[m], ms=4, capsize=2, lw=1.8 if m == "SKYMAP" else 1,
                color="k" if m == "SKYMAP" else None, label=lab.get(m, m))
ax.set_xlabel("Sampling ratio (%)"); ax.set_ylabel("RMSE (dB)"); ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 3, 5]); ax.set_xticklabels(["0.5", "1", "2", "3", "5"])
ax.legend(fontsize=6, ncol=2); plt.savefig(F + "Fig3.png"); plt.close()

# ---- Fig 3: map slices (seed 1, z=90 m) — re-fit quickly
from skygp import SkyGP
from methods import fit_pr_kriging
from env import sinr_db
S = Scenario(1); v = S.idx_valid; rng = np.random.default_rng(1020)
idx = rng.choice(v, int(0.02 * len(v)), replace=False); Y = S.measure(idx)
g = SkyGP(S).fit(idx, Y); mu, sd = g.predict(return_std=True); prk = fit_pr_kriging(S, idx, Y)
k = np.where(S.P[:, 2] == 90)[0]; n = S.nx
b = 0
mask = ~S.valid[k].reshape(n, n)
def img(a):
    a = a.reshape(n, n).copy(); a[mask] = np.nan; return a.T
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.1), gridspec_kw=dict(wspace=0.08))
vmin, vmax = np.nanpercentile(S.PR[k, b], [2, 98])
for a_, d, t in zip(axs[:3], [S.PR[k, b], prk[k, b], mu[k, b]], ["(a)", "(b)", "(c)"]):
    im = a_.imshow(img(d), origin="lower", extent=[0, L, 0, L], vmin=vmin, vmax=vmax, cmap="viridis")
    a_.set_title(t, fontsize=8); a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
    a_.plot(*S.BS[b, :2], "r^", ms=5)
cb = plt.colorbar(im, ax=axs[:3], fraction=0.02, pad=0.01); cb.ax.set_title("dBm", fontsize=7)
im2 = axs[3].imshow(img(sd[k, b]), origin="lower", extent=[0, L, 0, L], cmap="magma")
sl = idx[S.P[idx, 2] == 90]; axs[3].plot(S.P[sl, 0], S.P[sl, 1], "c.", ms=2)
axs[3].set_title("(d)", fontsize=8); axs[3].set_xticks([]); axs[3].set_yticks([]); axs[3].grid(False)
axs[3].set_position([0.80, axs[3].get_position().y0, axs[3].get_position().width, axs[3].get_position().height]); plt.colorbar(im2, ax=axs[3], fraction=0.046, pad=0.03)
plt.savefig(F + "Fig4.png"); plt.close()

# ---- Fig 4: calibration + ablation
A = json.load(open("results/abl.json"))
cur = [c["curve"] for c in A["cal"] if "curve" in c][0]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
axs[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="Ideal")
axs[0].plot(cur["q"], cur["SKYMAP"], "o-", color="k", ms=3, label="SKYMAP")
axs[0].plot(cur["q"], cur["PR-Kriging"], "s-", ms=3, label="PR-Kriging")
axs[0].set_xlabel("Nominal coverage"); axs[0].set_ylabel("Empirical coverage"); axs[0].legend(fontsize=7)
axs[0].set_title("(a)", fontsize=8)
names = list(A["abl"][0].keys())
vals = [np.mean([r[n_] for r in A["abl"]]) for n_ in names]; err = [np.std([r[n_] for r in A["abl"]]) for n_ in names]
axs[1].barh(range(len(names)), vals, xerr=err, color=["k"] + ["0.6"] * (len(names) - 1), capsize=2)
axs[1].set_yticks(range(len(names))); axs[1].set_yticklabels(names, fontsize=7); axs[1].invert_yaxis()
axs[1].set_xlabel("RMSE (dB) @ 2%"); axs[1].set_xlim(min(vals) - 0.3, max(vals) + 0.3)
axs[1].set_title("(b)", fontsize=8)
plt.savefig(F + "Fig5.png"); plt.close()

# ---- Fig 5: active sampling
Ac = json.load(open("results/active.json"))
for r, r2 in zip(Ac, json.load(open("results/active_ivr.json"))): r["IVR (SKYMAP)"] = r2["SKYMAP-IVR"]; r["Max-variance (SKYMAP)"] = r.pop("SKYMAP-UG")
xs = [0.5 + 0.5 * i for i in range(len(Ac[0]["Random"]))]
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for s_, m_ in [("Random", "s"), ("Space-filling", "^"), ("Max-variance (SKYMAP)", "o"), ("IVR (SKYMAP)", "D")]:
    arr = np.array([r[s_] for r in Ac])
    ax.errorbar(xs, arr.mean(0), yerr=arr.std(0), marker=m_, ms=4, capsize=2, label=s_,
                color="k" if s_ == "Random" else None)
ax.set_xlabel("Measurement budget (% of volume)"); ax.set_ylabel("RMSE (dB)"); ax.legend(fontsize=7)
plt.savefig(F + "Fig6.png"); plt.close()

# ---- Fig 6: trajectories
T = json.load(open("results/traj.json"))
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw=dict(wspace=0.55))
sl = T[0]["slice"]; xy = np.array(sl["xy"])
sc = axs[0].scatter(xy[:, 0], xy[:, 1], c=np.clip(sl["true"], -10, 20), cmap="RdYlGn", s=9, marker="s")
plt.colorbar(sc, ax=axs[0], label="True SINR, 90 m (dB)", pad=0.02)
cols = {"Shortest": "tab:blue", "PR-Kriging": "tab:orange", "SKYMAP-mean": "tab:purple", "SKYMAP-risk": "k", "Oracle": "tab:cyan"}
for kk, p in T[0]["example"].items():
    p = np.array(p); axs[0].plot(p[:, 0], p[:, 1], "-", color=cols[kk], lw=1.6 if kk == "SKYMAP-risk" else 1.1, label=kk)
axs[0].plot(*np.array(T[0]["example"]["Shortest"])[0, :2], "go", ms=6); axs[0].plot(*np.array(T[0]["example"]["Shortest"])[-1, :2], "r*", ms=9)
axs[0].plot(S.BS[:, 0], S.BS[:, 1], "k^", ms=6)
axs[0].legend(fontsize=6, loc="upper right"); axs[0].set_xlim(0, L); axs[0].set_ylim(0, L); axs[0].set_title("(a)", fontsize=8)
keys = list(cols)
out = [np.mean([np.mean([x[0] for x in r["paths"][kk]]) for r in T]) * 100 for kk in keys]
ln = [np.mean([np.mean([x[1] for x in r["paths"][kk]]) for r in T]) for kk in keys]
ax2 = axs[1]; ax2.bar(range(len(keys)), out, color=[cols[k_] for k_ in keys])
ax2.set_xticks(range(len(keys))); ax2.set_xticklabels([k_.replace("SKYMAP-", "SKYMAP\n").replace("PR-Kriging","PR-\nKriging") for k_ in keys], fontsize=6.5)
ax2.set_ylabel("Outage along path (%)"); ax2.set_title("(b)", fontsize=8)
a3 = ax2.twinx(); a3.plot(range(len(keys)), ln, "ko--", ms=4); a3.set_ylabel("Path length (m)"); a3.grid(False)
plt.savefig(F + "Fig7.png"); plt.close()
print("figs done")

# ---- Fig 8: LB-RAP (Algorithm 3)
Bd = json.load(open("results/budget.json"))
deltas = [0.05, 0.1, 0.2]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.35))
for name, mk_, ls_ in [("SKYMAP", "o", "-"), ("PR-Kriging", "s", "--")]:
    o = [[100 * np.mean([x[0] for x in r[f"{name} LB-RAP d={d}"]]) for r in Bd] for d in deltas]
    l = [[100 * (np.mean([x[1] for x in r[f"{name} LB-RAP d={d}"]]) - 1) for r in Bd] for d in deltas]
    axs[0].errorbar([100 * d for d in deltas], np.mean(o, 1), yerr=np.std(o, 1), marker=mk_, ls=ls_, capsize=2, color="k", mfc="white" if name != "SKYMAP" else "k", label=name + " + LB-RAP")
    axs[1].errorbar([100 * d for d in deltas], np.mean(l, 1), yerr=np.std(l, 1), marker=mk_, ls=ls_, capsize=2, color="k", mfc="white" if name != "SKYMAP" else "k", label=name + " + LB-RAP")
fo = [100 * np.mean([x[0] for x in r["SKYMAP-risk (lambda=8)"]]) for r in Bd]
fl = [100 * (np.mean([x[1] for x in r["SKYMAP-risk (lambda=8)"]]) - 1) for r in Bd]
axs[0].axhline(np.mean(fo), color="0.4", ls=":", label="SKYMAP fixed lambda = 8")
axs[1].axhline(np.mean(fl), color="0.4", ls=":", label="SKYMAP fixed lambda = 8")
axs[1].plot([0, 22], [0, 22], color="0.7", lw=0.8, label="budget limit")
axs[0].set_xlabel("Length budget delta (%)"); axs[0].set_ylabel("Outage along path (%)"); axs[0].set_title("(a)", fontsize=8)
axs[1].set_xlabel("Length budget delta (%)"); axs[1].set_ylabel("Extra length vs shortest (%)"); axs[1].set_title("(b)", fontsize=8)
axs[1].set_xlim(3, 22); axs[1].set_ylim(0, 22); axs[0].set_xticks([5, 10, 20]); axs[1].set_xticks([5, 10, 20])
axs[0].legend(fontsize=6); axs[1].legend(fontsize=6)
plt.savefig(F + "Fig8.png"); plt.close()
print("fig8 done")
