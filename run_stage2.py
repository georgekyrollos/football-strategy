"""Stage 2 solve: all 16 (to_off, to_def) timeout combinations.

Reads:  q4_full_v.npz          (Stage 1 — already solved)
Writes: q4_to_v_to{a}{b}.npz  for (a,b) in {0..3}²

Run:
    source .venv/bin/activate
    python run_stage2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("Loading chart and punt tables...")
from football_strategy.chart_parser import load_scrimmage_chart, load_punt_chart
from football_strategy.reachability import kickoff_key
from football_strategy.solve_full_game import solve_with_timeouts

sc = load_scrimmage_chart()   # defaults to Ball Control
pc = load_punt_chart()
print("Charts loaded.\n")

solve_with_timeouts(
    sc, pc,
    kickoff_key(q=4, tau=60, delta=0, h=0),   # same start as Stage 1
    stage1_path="q4_full_v.npz",
    out_prefix="q4_to_v",
    use_tqdm=True,
    n_workers=8,
)
