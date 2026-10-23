# Physics Informed Gaussian Process Radio Maps for Risk Aware UAV Trajectory Planning

Reference implementation of **SKYMAP**. The repository reproduces every table and figure of the manuscript submitted to *Wireless Personal Communications* (Springer).

SKYMAP builds a 3D radio map of the low altitude airspace (30 to 150 m) from sparse UAV RSRP measurements. It then plans UAV trajectories that avoid SINR outage.

| Module | File | What it does |
|---|---|---|
| Scenario generator | `src/env.py` | ITU-R P.1410 city, 3GPP TR 36.777 UMa-AV path loss, TR 38.901 BS antenna, 3D correlated shadowing with inter-site correlation |
| Baselines | `src/methods.py` | LDPL, two slope LoS prior, IDW, kriging, physics residual kriging, MLP, Fourier feature neural field, physics informed neural field (NumPy) |
| SKYMAP estimator (Algorithm 1) | `src/skygp.py` | Antenna aware LoS prior + multicell, LoS heteroscedastic Matérn GP; type II ML; posterior mean/std; IVR sampling score |
| Planners (Algorithms 2 and 3) | `src/planner.py` | 26 neighbour graph, Dijkstra with outage weighted edge cost, length budgeted risk aware planner (LB-RAP) |

## Installation

```bash
git clone https://github.com/GArulDalton/skymap-uav-radio-maps.git
cd skymap-uav-radio-maps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs on a CPU. No GPU and no deep learning framework is needed, because the neural baselines are written in NumPy.

## Reproducing the paper

```bash
bash scripts/run_all.sh          # all experiments (about 1.5 h on one CPU core)
python scripts/make_figures.py   # regenerates figures/Fig1.png ... Fig8.png
python scripts/make_tables.py    # prints every table of the paper
```

| Paper item | Script | Output |
|---|---|---|
| Table 3, Table 4, Fig. 3 (accuracy vs sampling, runtime) | `experiments/exp_map.py` | `results/map.json` |
| Table 5 (learned hyper-parameters) | `experiments/exp_map.py` | `results/map.json` |
| Table 6, Table 7, Fig. 5 (calibration, ablation, mismatch) | `experiments/exp_calib_abl.py` | `results/abl.json` |
| Fig. 6 (adaptive sampling) | `experiments/exp_active.py`, `experiments/exp_active_ivr.py` | `results/active*.json` |
| Section 6.6 pilot (anchor-calibrated IVR, city 1 only) | `experiments/exp_active_ivr.py --anchor` | `results/active_ivr_anchor.json` |
| Fig. 1 (system model) and Fig. 2 (framework block diagram) | `scripts/make_figures.py` | `figures/Fig1.png`, `figures/Fig2.png` |
| Table 8, Fig. 7 (risk aware planning) | `experiments/exp_traj.py` | `results/traj.json` |
| Table 9, Fig. 8 (LB-RAP, Algorithm 3) | `experiments/exp_budget.py` | `results/budget.json` |

All random seeds are fixed. Test cities use seeds 1 to 5, and the baseline tuning city uses seed 99. The `results/` folder ships with the exact outputs used in the paper.

## Minimal usage

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from env import Scenario
from skygp import SkyGP

S = Scenario(seed=1)                                   # synthetic city + ground-truth maps
idx = np.random.default_rng(0).choice(S.idx_valid, 664, replace=False)
Y = S.measure(idx)                                      # noisy RSRP of all cells at sampled points
g = SkyGP(S).fit(idx, Y)
mu, sd = g.predict(return_std=True)                     # posterior mean / std (dBm) for every grid point and cell
print(g.hyper())                                        # interpretable hyper-parameters
```

## Citation

If you use this code, please cite the paper (the full reference will be updated on acceptance):

```
G. Arul Dalton, "Physics Informed Gaussian Process Radio Maps for Risk Aware UAV Trajectory Planning,"
Wireless Personal Communications (under review), 2026.
```

## License

MIT. See `LICENSE`.
