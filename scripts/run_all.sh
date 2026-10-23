#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python experiments/exp_map.py        | tee results/map.log
python experiments/exp_calib_abl.py  | tee results/abl.log
python experiments/exp_traj.py       | tee results/traj.log
python experiments/exp_budget.py     | tee results/budget.log
python experiments/exp_active.py     | tee results/active.log
python experiments/exp_active_ivr.py | tee results/active_ivr.log
python experiments/exp_active_ivr.py --anchor | tee results/active_ivr_anchor.log
