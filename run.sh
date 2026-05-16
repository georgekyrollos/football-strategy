#!/usr/bin/env sh
cd /Users/georgekyrollos/Projects/football-strategy
source .venv/bin/activate
python -u run_solver.py 2>&1 | tee q4_full.log
