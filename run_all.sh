#!/usr/bin/env bash
# Reproduce every result, figure, demo video and browser replay.
# Usage: ./run_all.sh [workers]          (about 6 h on 2 cores; E2 with 30 seeds dominates)
set -euo pipefail
cd "$(dirname "$0")"
W=${1:-2}

python -m pytest -q
python experiments/tune.py --workers "$W"                     # tuning seeds 100-102 -> configs/controllers_tuned.yaml
python experiments/core_experiments.py --workers "$W"         # E0-E8 and ablations
python experiments/e9_resilience.py --seeds 10 --workers "$W"
python experiments/e10_distributed_optimisation.py
python experiments/e11_epidemic.py --seeds 10 --workers "$W"
python experiments/e13_dmpc_stl_epistemic.py --seeds 5 --workers "$W"
python experiments/e14_incident_tidal.py --seeds 10 --dmpc-seeds 5 --workers "$W"
python experiments/e12_marl.py --train-seeds 3 --episodes 120 --workers "$W"
python experiments/e12_ctde_sensitivity.py --workers "$W"
python experiments/make_demo.py
python experiments/sumo_experiments.py --workers "$W"     # needs requirements-sumo.txt
python experiments/make_web_viewer.py
python experiments/make_figures.py
echo "done: see results/"
