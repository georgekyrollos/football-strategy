"""
Strategy verification script.

Loads the solved Q4 value table and re-solves the LP for a set of canonical
football situations. Prints the equilibrium mixed strategies so we can
sanity-check them against football intuition.

Usage:
    source .venv/bin/activate
    python check_strategies.py
"""

import numpy as np

from football_strategy.chart_parser import load_scrimmage_chart, load_punt_chart
from football_strategy.matrix_game import (
    build_scrimmage_payoff_matrix,
    kickoff_state_value,
    solve_row_strategy,
    solve_col_strategy,
)
from football_strategy.constants import DEFENSE_CARDS, FG_ACTION, PUNT_ACTION
from football_strategy.value_store import ValueStore

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading chart...")
SC = load_scrimmage_chart()   # defaults to Ball Control
PC = load_punt_chart()

print("Loading ValueStore from q4_full_v.npz...")
V = ValueStore.load("q4_full_v.npz")
print(f"  {len(V):,} solved slots loaded\n")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

Q = 4  # all checks are Q4


def action_name(a: int) -> str:
    if a == PUNT_ACTION:
        return "PUNT"
    if a == FG_ACTION:
        return "FG  "
    return f"P{a:02d} "


def print_strategy(label: str, x: int, down: int, to_first: int,
                   delta: int, tau: int, threshold: float = 0.03) -> None:
    """Build payoff matrix, solve both LPs, print strategies."""
    A, actions = build_scrimmage_payoff_matrix(
        SC, PC, Q, tau, x, down, to_first, delta, h=0, V=V
    )
    v_off, p_off = solve_row_strategy(A)
    v_def, p_def = solve_col_strategy(A)

    # sanity: both LPs should agree on value
    value_gap = abs(v_off - v_def)

    print(f"{'='*70}")
    print(f"  {label}")
    pos_str = f"x={x} (own {x})" if x <= 50 else f"x={x} (opp {100-x})"
    print(f"  {pos_str}  |  {down}st/nd/rd/th & {to_first}  |  delta={delta:+d}  |  tau={tau} ({tau*15}s left)")
    print(f"  Game value (off perspective): {v_off:+.4f}   [LP gap {value_gap:.1e}]")
    print()

    # Offense strategy
    print("  OFFENSE strategy (plays >= {:.0f}%):".format(threshold * 100))
    play_groups = []
    for i, a in enumerate(actions):
        if p_off[i] >= threshold:
            play_groups.append((p_off[i], action_name(a)))
    play_groups.sort(reverse=True)
    for prob, name in play_groups:
        bar = "#" * int(prob * 40)
        print(f"    {name}  {prob*100:5.1f}%  {bar}")
    if not play_groups:
        # print top 3 anyway
        top = sorted(enumerate(p_off), key=lambda x: -x[1])[:3]
        for i, prob in top:
            print(f"    {action_name(actions[i])}  {prob*100:5.1f}%  (below threshold)")

    print()

    # Defense strategy
    print("  DEFENSE strategy (cards >= {:.0f}%):".format(threshold * 100))
    def_groups = []
    for j, card in enumerate(DEFENSE_CARDS):
        if p_def[j] >= threshold:
            def_groups.append((p_def[j], card))
    def_groups.sort(reverse=True)
    for prob, card in def_groups:
        bar = "#" * int(prob * 40)
        print(f"    {card}     {prob*100:5.1f}%  {bar}")
    if not def_groups:
        top = sorted(enumerate(p_def), key=lambda x: -x[1])[:3]
        for j, prob in top:
            print(f"    {DEFENSE_CARDS[j]}     {prob*100:5.1f}%  (below threshold)")

    print()

    # Payoff range
    print(f"  Payoff matrix range: [{A.min():+.3f}, {A.max():+.3f}]  "
          f"shape={A.shape}")
    print()


# ---------------------------------------------------------------------------
# Also check kickoff value at various deltas
# ---------------------------------------------------------------------------

def print_kickoff_values(taus=(60, 30, 8)) -> None:
    print(f"{'='*70}")
    print("  KICKOFF VALUES  (kicking team perspective, h=0)")
    print(f"  {'delta':>6}  {'tau=60':>8}  {'tau=30':>8}  {'tau=8':>8}")
    for delta in range(-21, 22, 3):
        vals = []
        for tau in taus:
            try:
                v = kickoff_state_value(Q, tau, delta, h=0, V=V)
                vals.append(f"{v:+.4f}")
            except KeyError:
                vals.append("  N/A  ")
        print(f"  {delta:>+6}  {'  '.join(vals)}")
    print()


# ---------------------------------------------------------------------------
# Run checks
# ---------------------------------------------------------------------------

# 1. Normal midgame situation
print_strategy(
    "MIDFIELD — 1st & 10 — tied — plenty of time",
    x=50, down=1, to_first=10, delta=0, tau=30,
)

# 2. Own territory — 4th & 10 — should punt
print_strategy(
    "OWN 30 — 4th & 10 — tied — should punt",
    x=30, down=4, to_first=10, delta=0, tau=30,
)

# 3. FG range — 4th & 3 — should see FG
print_strategy(
    "OPP 30 (x=70) — 4th & 3 — tied — FG vs go-for-it",
    x=70, down=4, to_first=3, delta=0, tau=20,
)

# 4. Red zone outer (x>=80, plays 17-20 banned)
print_strategy(
    "OPP 15 (x=85) — 1st & 10 — tied — red zone outer",
    x=85, down=1, to_first=10, delta=0, tau=20,
)

# 5. Red zone inner (x>=90, plays 13-20 banned)
print_strategy(
    "OPP 8 (x=92) — 1st & 8 — tied — red zone inner (goal-to-go)",
    x=92, down=1, to_first=8, delta=0, tau=20,
)

# 6. Trailing by 7, late in Q4 — desperation
print_strategy(
    "OWN 20 — 1st & 10 — DOWN 7 — 2 min left (tau=8) — must score",
    x=20, down=1, to_first=10, delta=-7, tau=8,
)

# 7. Leading by 7, late in Q4 — protect lead
print_strategy(
    "OWN 20 — 1st & 10 — UP 7 — 2 min left (tau=8) — kill clock",
    x=20, down=1, to_first=10, delta=7, tau=8,
)

# 8. Trailing badly, late
print_strategy(
    "OWN 20 — 1st & 10 — DOWN 14 — tau=4 (~1 min left) — desperate",
    x=20, down=1, to_first=10, delta=-14, tau=4,
)

# 9. Leading comfortably, late
print_strategy(
    "OWN 20 — 1st & 10 — UP 14 — tau=4 (~1 min left) — safe",
    x=20, down=1, to_first=10, delta=14, tau=4,
)

# 10. Symmetry check: delta=+7 should mirror delta=-7
print_strategy(
    "MIDFIELD — 1st & 10 — UP 7 — tau=30",
    x=50, down=1, to_first=10, delta=7, tau=30,
)
print_strategy(
    "MIDFIELD — 1st & 10 — DOWN 7 — tau=30",
    x=50, down=1, to_first=10, delta=-7, tau=30,
)

# Kickoff table
print_kickoff_values()
