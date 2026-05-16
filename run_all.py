"""Full solve: Stage 1 (no timeouts) then Stage 2 (all 16 TO combinations).

Writes:
  q4_full_v.npz            — Stage 1 solution (Ball Control, Q4, no timeouts)
  q4_to_v_to{a}{b}.npz    — Stage 2 solutions for (a,b) in {0..3}²

Run:
    source .venv/bin/activate
    nohup python run_all.py > solve.log 2>&1 &
    tail -f solve.log
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from football_strategy.chart_parser import load_scrimmage_chart, load_punt_chart, _DEFAULT_CSV
from football_strategy.reachability import kickoff_key
from football_strategy.solve_full_game import solve_full_game, solve_with_timeouts

START_KEY = kickoff_key(q=4, tau=60, delta=0, h=0)

print(f"Chart: {_DEFAULT_CSV}")
sc = load_scrimmage_chart()
pc = load_punt_chart()

# ── Stage 1 ──────────────────────────────────────────────────────────────────
print("\n=== STAGE 1: full backward induction, no timeouts ===")
solve_full_game(
    sc, pc, START_KEY,
    verbose=True,
    n_workers=8,
    save_path="q4_full_v",
)

# ── Stage 2 ──────────────────────────────────────────────────────────────────
print("\n=== STAGE 2: all 16 timeout combinations ===")
solve_with_timeouts(
    sc, pc, START_KEY,
    stage1_path="q4_full_v.npz",
    out_prefix="q4_to_v",
    use_tqdm=True,
    n_workers=8,
)
