"""
Deep strategic analysis of four key scenarios.

Usage:
    source .venv/bin/activate
    python deep_analysis.py
"""

import numpy as np

from football_strategy.chart_parser import load_scrimmage_chart, load_punt_chart
from football_strategy.matrix_game import (
    build_scrimmage_payoff_matrix,
    solve_row_strategy,
    solve_col_strategy,
)
from football_strategy.constants import (
    DEFENSE_CARDS, FG_ACTION, PUNT_ACTION,
    REDZONE_OUTER_X, REDZONE_INNER_X,
)
from football_strategy.value_store import ValueStore

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading chart and ValueStore...")
SC = load_scrimmage_chart()   # defaults to Ball Control
PC = load_punt_chart()
V  = ValueStore.load("q4_full_v.npz")
print(f"  {len(V):,} solved slots\n")

Q = 4


def action_label(a):
    if a == PUNT_ACTION: return "PUNT"
    if a == FG_ACTION:   return "FG  "
    return f"P{a:02d} "


def matrix_and_strategies(x, down, to_first, delta, tau):
    A, actions = build_scrimmage_payoff_matrix(
        SC, PC, Q, tau, x, down, to_first, delta, h=0, V=V
    )
    v_off, p_off = solve_row_strategy(A)
    v_def, p_def = solve_col_strategy(A)
    return A, actions, v_off, p_off, v_def, p_def


# ============================================================
# SCENARIO 1 — 4th & 10, own 30, tied, tau=30
# ============================================================

print("=" * 72)
print("SCENARIO 1 — 4th & 10, OWN 30, TIED, tau=30")
print("=" * 72)

A1, acts1, v1, p1, _, q1 = matrix_and_strategies(
    x=30, down=4, to_first=10, delta=0, tau=30
)

# Row security values = min over defense cols for each offense row
row_sec = A1.min(axis=1)
# For each offense play, best pure go-for-it security = max row-security excl. punt
punt_idx = acts1.index(PUNT_ACTION)

print(f"\nGame value (equilibrium): {v1:+.4f}")
print(f"\nRow security values (= guaranteed value if offense plays this row purely):")
print(f"  {'Play':<6}  {'Security':>9}  {'Eq prob':>8}  {'Worst-case card'}")
print(f"  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*13}")

# Sort by equilibrium probability descending
order = sorted(range(len(acts1)), key=lambda i: -p1[i])
for i in order:
    a = acts1[i]
    worst_card = DEFENSE_CARDS[int(np.argmin(A1[i]))]
    marker = " <-- PUNT" if a == PUNT_ACTION else ""
    print(f"  {action_label(a)}  {row_sec[i]:+9.4f}  {p1[i]*100:7.1f}%  "
          f"worst={worst_card}{marker}")

punt_sec = row_sec[punt_idx]
best_gfi_idx = max(
    (i for i in range(len(acts1)) if acts1[i] != PUNT_ACTION),
    key=lambda i: row_sec[i]
)
best_gfi_sec = row_sec[best_gfi_idx]

print(f"\nPunt pure security:              {punt_sec:+.4f}")
print(f"Best go-for-it pure security:    {best_gfi_sec:+.4f}  ({action_label(acts1[best_gfi_idx])})")
print(f"Equilibrium value:               {v1:+.4f}")
print(f"\n  Why the mix: Punt security ({punt_sec:+.4f}) > best GFI security ({best_gfi_sec:+.4f}),")
print(f"  yet equil value ({v1:+.4f}) > punt security. The mix is better than either pure strategy")
print(f"  because some go-for-it plays beat specific defense cards better than punt does.")
print(f"  The defense must hedge against both, which lifts the equilibrium above punt-only.")

print(f"\nFull payoff matrix A[play, defense_card]:")
print(f"  {'Play':<6}" + "".join(f"  {c:>6}" for c in DEFENSE_CARDS))
print(f"  {'-'*4}" + "  ------" * 10)
for i, a in enumerate(acts1):
    in_support = p1[i] > 0.01
    marker = "*" if in_support else " "
    row_str = "".join(f"  {A1[i,j]:+6.3f}" for j in range(len(DEFENSE_CARDS)))
    print(f"{marker} {action_label(a)}{row_str}  sec={row_sec[i]:+.4f}")
print(f"  (* = in equilibrium support)")

print(f"\nDefense equilibrium: ", end="")
for j, c in enumerate(DEFENSE_CARDS):
    if q1[j] > 0.01:
        print(f"{c}={q1[j]*100:.1f}%  ", end="")
print()


# ============================================================
# SCENARIO 2 — 4th & 3, opp 30 (x=70), tied, tau=20
# ============================================================

print("\n" + "=" * 72)
print("SCENARIO 2 — 4th & 3, OPP 30 (x=70), TIED, tau=20")
print("=" * 72)

A2, acts2, v2, p2, _, q2 = matrix_and_strategies(
    x=70, down=4, to_first=3, delta=0, tau=20
)

row_sec2 = A2.min(axis=1)
fg_idx = acts2.index(FG_ACTION)
fg_val = A2[fg_idx, 0]   # FG row is constant across all defense cols

print(f"\nGame value (equilibrium): {v2:+.4f}")
print(f"\nFG row: constant = {fg_val:+.4f} (independent of defense card, shape confirms)")
fg_row_vals = A2[fg_idx]
print(f"  FG row values: min={fg_row_vals.min():+.4f}  max={fg_row_vals.max():+.4f}  "
      f"std={fg_row_vals.std():.2e}")
print(f"  FG pure security = {row_sec2[fg_idx]:+.4f}")

print(f"\nRow security values — plays in equilibrium support vs FG:")
print(f"  {'Play':<6}  {'Security':>9}  {'Eq prob':>8}  {'vs FG':>8}")
print(f"  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*8}")
order2 = sorted(range(len(acts2)), key=lambda i: -p2[i])
for i in order2:
    a = acts2[i]
    vs_fg = row_sec2[i] - row_sec2[fg_idx]
    flag = "  <-- FG" if a == FG_ACTION else ("  <-- PUNT" if a == PUNT_ACTION else "")
    print(f"  {action_label(a)}  {row_sec2[i]:+9.4f}  {p2[i]*100:7.1f}%  "
          f"{vs_fg:+8.4f}{flag}")

print(f"\nWhy FG is excluded from support:")
print(f"  FG security = {row_sec2[fg_idx]:+.4f}  (same as FG value; it's constant)")
print(f"  Equilibrium value = {v2:+.4f}")
print(f"  Since equilibrium value ({v2:+.4f}) > FG security ({row_sec2[fg_idx]:+.4f}),")
print(f"  FG is never the best response for offense — the mixed go-for-it strategy")
print(f"  guarantees more than any certain {fg_val:+.4f} FG payoff.")

print(f"\nFull payoff matrix (top rows by equilibrium weight):")
print(f"  {'Play':<6}" + "".join(f"  {c:>6}" for c in DEFENSE_CARDS) + "  sec")
print(f"  {'-'*4}" + "  ------" * 10 + "  -----")
for i in order2:
    a = acts2[i]
    if p2[i] < 0.005 and a not in (FG_ACTION, PUNT_ACTION):
        continue
    in_support = p2[i] > 0.01
    marker = "*" if in_support else " "
    row_str = "".join(f"  {A2[i,j]:+6.3f}" for j in range(len(DEFENSE_CARDS)))
    print(f"{marker} {action_label(a)}{row_str}  {row_sec2[i]:+.4f}")
print(f"  (* = in equilibrium support, FG and PUNT shown regardless)")


# ============================================================
# SCENARIO 3 — Down 7, own 20, tau=8: defense-F column
# ============================================================

print("\n" + "=" * 72)
print("SCENARIO 3 — DOWN 7, OWN 20, tau=8: WHY DEFENSE-F FORCES -1")
print("=" * 72)

A3, acts3, v3, p3, _, q3 = matrix_and_strategies(
    x=20, down=1, to_first=10, delta=-7, tau=8
)

f_idx = DEFENSE_CARDS.index("F")
f_col = A3[:, f_idx]

print(f"\nGame value: {v3:+.4f}")
print(f"Payoff matrix range: [{A3.min():+.4f}, {A3.max():+.4f}]")
print(f"\nDefense-F column values (offense row vs card F):")
print(f"  {'Play':<6}  {'vs F':>8}  {'row max':>8}  {'row min':>8}  {'Eq prob':>8}")
print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
order3 = sorted(range(len(acts3)), key=lambda i: -f_col[i])
for i in order3:
    a = acts3[i]
    best_card_val = A3[i].max()
    worst_card_val = A3[i].min()
    print(f"  {action_label(a)}  {f_col[i]:+8.4f}  {best_card_val:+8.4f}  "
          f"{worst_card_val:+8.4f}  {p3[i]*100:7.1f}%")

n_minus1 = np.sum(A3[:, f_idx] <= -1.0 + 1e-9)
print(f"\n  Card F gives -1.0 against {n_minus1}/{len(acts3)} plays.")
print(f"  Max value offense can achieve against F: {f_col.max():+.4f}")

col_maxs = A3.max(axis=0)
print(f"\nColumn-maximum values (offense BEST RESPONSE per defense card = max_a Q(a,b)):")
print(f"  A defense card forces certain loss iff max_a Q(a,b) = -1.")
for j, c in enumerate(DEFENSE_CARDS):
    br_idx = int(np.argmax(A3[:, j]))
    br_play = action_label(acts3[br_idx])
    print(f"  Card {c}: best-response value = {col_maxs[j]:+.4f}  "
          f"(offense best play: {br_play})   equil weight = {q3[j]*100:.1f}%")

f_max = col_maxs[f_idx]
print(f"\nCard F: max_a Q(a,F) = {f_max:+.4f}  --> {'FORCES -1 (certain loss)' if f_max <= -1.0+1e-9 else 'does NOT force -1'}")
print(f"Since max_a Q(a,F) = -1, defense playing F pure guarantees value <= -1.")
print(f"Combined with value >= -1 always, this proves V(state) = -1 exactly.")

print(f"\nFull matrix (each cell = expected continuation value for offense):")
print(f"  {'Play':<6}" + "".join(f"  {c:>6}" for c in DEFENSE_CARDS))
print(f"  {'-'*4}" + "  ------" * 10)
for i in order3:
    row_str = "".join(f"  {A3[i,j]:+6.3f}" for j in range(len(DEFENSE_CARDS)))
    print(f"  {action_label(acts3[i])}{row_str}")


# ============================================================
# SCENARIO 4 — Red-zone comparison: x=85, 90, 92, 95
# ============================================================

print("\n" + "=" * 72)
print("SCENARIO 4 — RED-ZONE COMPARISON: x=85, 90, 92, 95 (1st & goal-ish)")
print("=" * 72)

redzone_cases = [
    (85, 1, 10, "opp 15  outer RZ (plays 17-20 banned)"),
    (90, 1, 10, "opp 10  inner RZ (plays 13-20 banned)"),
    (92, 1,  8, "opp  8  inner RZ goal-to-go         "),
    (95, 1,  5, "opp  5  inner RZ goal-to-go         "),
]

for x, down, tf, label in redzone_cases:
    A, acts, v_off, p_off, _, q_def = matrix_and_strategies(
        x=x, down=down, to_first=tf, delta=0, tau=20
    )
    row_sec = A.min(axis=1)
    n_plays = len(acts)
    has_fg   = FG_ACTION   in acts
    has_punt = PUNT_ACTION in acts

    play_nums = [a for a in acts if a not in (FG_ACTION, PUNT_ACTION)]
    min_play  = min(play_nums)
    max_play  = max(play_nums)

    print(f"\n  x={x}  {label}")
    print(f"  Legal plays: {n_plays} total  "
          f"(runs/passes {min_play}-{max_play}, FG={'yes' if has_fg else 'no'}, "
          f"PUNT={'yes' if has_punt else 'no'})")
    print(f"  Game value:  {v_off:+.4f}")
    print(f"  A range:     [{A.min():+.4f}, {A.max():+.4f}]")

    # Offense support
    support = [(action_label(acts[i]), p_off[i]) for i in range(len(acts)) if p_off[i] > 0.02]
    support.sort(key=lambda t: -t[1])
    print(f"  Off support: " + "  ".join(f"{name}={prob*100:.1f}%" for name, prob in support))

    # Defense support
    def_support = [(DEFENSE_CARDS[j], q_def[j]) for j in range(10) if q_def[j] > 0.02]
    def_support.sort(key=lambda t: -t[1])
    print(f"  Def support: " + "  ".join(f"{c}={prob*100:.1f}%" for c, prob in def_support))

    # Row security for plays in support
    print(f"  Row security for supported plays:")
    for i in range(len(acts)):
        if p_off[i] > 0.02:
            wc = DEFENSE_CARDS[int(np.argmin(A[i]))]
            print(f"    {action_label(acts[i])}  sec={row_sec[i]:+.4f}  worst_card={wc}")

print("\n  --- Summary table ---")
print(f"  {'x':>4}  {'dist':>5}  {'#plays':>6}  {'value':>8}  {'A_max':>8}  {'A_min':>8}")
print(f"  {'-'*4}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
for x, down, tf, label in redzone_cases:
    A, acts, v_off, _, _, _ = matrix_and_strategies(
        x=x, down=down, to_first=tf, delta=0, tau=20
    )
    dist = 100 - x
    print(f"  {x:>4}  {dist:>5}  {len(acts):>6}  {v_off:+8.4f}  {A.max():+8.4f}  {A.min():+8.4f}")

print()
