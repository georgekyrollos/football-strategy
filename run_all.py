"""Solve Football Strategy Q4 with full timeout support.

Writes:
  q4_full_v.npz            — base solve (no timeouts remaining for either team)
  q4_to_v_to{a}{b}.npz    — all 16 timeout-count combinations (a,b) in {0..3}²

The game-relevant result is q4_to_v_to33.npz (both teams start with 3 TOs).

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

# ── Base solve (θ₁=θ₂=0) ─────────────────────────────────────────────────────
print("\n=== Base solve: backward induction, no timeouts remaining ===")
solve_full_game(
    sc, pc, START_KEY,
    verbose=True,
    n_workers=8,
    save_path="q4_full_v",
)

# ── Full solve: all 16 timeout-count combinations ────────────────────────────
print("\n=== Full solve: all 16 (to_off, to_def) combinations ===")
solve_with_timeouts(
    sc, pc, START_KEY,
    base_path="q4_full_v.npz",
    out_prefix="q4_to_v",
    use_tqdm=True,
    n_workers=8,  # combos at the same level run in parallel, workers split across them
)
