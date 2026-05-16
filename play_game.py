#!/usr/bin/env python3
"""
Interactive Football Strategy playtest assistant.

Loads solved Q4 values and for each state:
  - Shows state in readable + technical form
  - Shows both equilibrium mixed strategies with sampling
  - Prompts for played offense action and defense card
  - Resolves the chart outcome concretely (samples LG, PAT, kickoff dice)
  - Advances state; offers state override at any point

Usage:
    source .venv/bin/activate
    python play_game.py
"""

import argparse
import copy
import random
import sys

# Parse --auto flag before heavy imports so --help is fast
_parser = argparse.ArgumentParser(description="Football Strategy playtest assistant")
_parser.add_argument("--auto", action="store_true",
                     help="Auto-roll all dice instead of prompting for results")
_args, _ = _parser.parse_known_args()
AUTO_DICE: bool = _args.auto

import numpy as np

from football_strategy.chart_parser import (
    load_scrimmage_chart, load_pro_style_chart_csv, load_punt_chart,
)
from football_strategy.constants import (
    DEFENSE_CARDS, FG_ACTION, FG_MIN_X, FG_POINTS,
    KICKOFF_NORMAL, KICKOFF_NORMAL_COUNTS, KICKOFF_ONSIDE,
    LG_DIST, PAT_PROB_GOOD, PUNT_ACTION, PUNT_CHART,
    REDZONE_INNER_X, REDZONE_INNER_ILLEGAL,
    REDZONE_OUTER_X, REDZONE_OUTER_ILLEGAL,
    SAFETY_POINTS, TD_POINTS, TICKS_PER_QUARTER, TIMEOUT_REDUCE, TWO_MIN_TICK,
    fg_prob,
)
from football_strategy.matrix_game import (
    build_scrimmage_payoff_matrix,
    kickoff_state_value,
    solve_col_strategy,
    solve_row_strategy,
)
from football_strategy.outcomes import (
    ChoiceOutcome, FirstDownOutcome, FumbleOutcome, IncompleteOutcome,
    InterceptionOutcome, LongGainOutcome, LossOfDownOutcome, PenaltyOutcome,
    YardsOutcome,
)
from football_strategy.states import goal_to_go, legal_offense_actions
from football_strategy.value_store import ValueStore

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading chart...")
SC     = load_scrimmage_chart()
SC_RAW = load_pro_style_chart_csv()   # DataFrame — for displaying raw cell text
PC     = load_punt_chart()

print("Loading ValueStore...")
V = ValueStore.load("q4_full_v.npz")
print(f"  {len(V):,} solved slots\n")

Q = 4

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def roll_lg() -> tuple:
    """Roll the Long Gain table. Returns (yards, dice_description)."""
    d1 = random.randint(1, 6)
    if d1 == 1:
        d2 = random.randint(1, 6)
        yards = 50 + 10 * d2
        return yards, f"die1=1, die2={d2}"
    yards = {2: 50, 3: 45, 4: 40, 5: 35, 6: 30}[d1]
    return yards, f"die1={d1}"


def roll_pat() -> tuple:
    """Roll PAT. Returns (success: bool, roll: int)."""
    roll = random.randint(1, 36)
    return roll <= 34, roll


def roll_2d6() -> tuple:
    d1, d2 = random.randint(1, 6), random.randint(1, 6)
    return d1 + d2, d1, d2


def roll_1d6() -> int:
    return random.randint(1, 6)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class GameState:
    """All game state. `delta` is always from current offense's perspective."""

    def __init__(self):
        self.phase     = "scrimmage"   # "scrimmage" | "kickoff" | "safety_kick" | "over"
        self.x         = 35
        self.down      = 1
        self.to_first  = 10
        self.delta     = 0
        self.tau       = TICKS_PER_QUARTER
        self.q         = Q
        self.h         = 0
        # Absolute score display (independent of symmetric representation)
        self.score_a   = 0
        self.score_b   = 0
        self.a_has_ball = True   # True = Team A is currently the offense
        # Timeouts remaining per team (reset to 3 at halftime)
        self.to_a      = 3
        self.to_b      = 3

    def clone(self):
        return copy.copy(self)

    def recalc_delta(self):
        """Recompute delta from absolute scores + possession."""
        if self.a_has_ball:
            self.delta = self.score_a - self.score_b
        else:
            self.delta = self.score_b - self.score_a


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SEP  = "=" * 70
SEP2 = "-" * 70

def yd(x: int) -> str:
    if x == 50:   return "midfield"
    if x < 50:    return f"own {x}"
    return f"opp {100 - x}"

def down_str(d: int, tf: int) -> str:
    return f"{d}{'st' if d==1 else 'nd' if d==2 else 'rd' if d==3 else 'th'} & {tf}"

def clock_str(tau: int) -> str:
    s = tau * 15
    return f"{s // 60}:{s % 60:02d}"

def display_state(gs: GameState, player_team: str) -> None:
    off_team = "A" if gs.a_has_ball else "B"
    def_team = "B" if gs.a_has_ball else "A"
    player_role = "OFFENSE" if player_team == off_team else "DEFENSE"

    print(SEP)
    print(f"  Q{gs.q}  |  Clock: {clock_str(gs.tau)}  |  "
          f"Score: A={gs.score_a}  B={gs.score_b}  |  TOs: A={gs.to_a}  B={gs.to_b}")
    print(f"  You are Team {player_team} — currently {player_role}")
    print(f"  Possession: Team {off_team}")

    if gs.phase == "scrimmage":
        rz = ""
        if gs.x >= REDZONE_INNER_X:
            rz = "  [INNER RED ZONE — plays 13-20 banned]"
        elif gs.x >= REDZONE_OUTER_X:
            rz = "  [OUTER RED ZONE — plays 17-20 banned]"
        print(f"  {down_str(gs.down, gs.to_first)}  at {yd(gs.x)}{rz}")
        diff = gs.score_a - gs.score_b
        lead = "tied" if diff == 0 else (f"A leads by {diff}" if diff > 0 else f"B leads by {-diff}")
        print(f"  Score: {lead}  (delta from offense view: {gs.delta:+d})")
        print(f"  Technical: x={gs.x}, down={gs.down}, tf={gs.to_first}, "
              f"delta={gs.delta:+d}, tau={gs.tau}")

    elif gs.phase == "kickoff":
        print(f"  KICKOFF — Team {off_team} kicks off")
        print(f"  Technical: delta={gs.delta:+d}, tau={gs.tau}")

    elif gs.phase == "safety_kick":
        print(f"  SAFETY FREE KICK — Team {off_team} kicks")
        print(f"  Technical: delta={gs.delta:+d}, tau={gs.tau}")
    print()


# ---------------------------------------------------------------------------
# Strategy display
# ---------------------------------------------------------------------------

def action_label(a: int) -> str:
    if a == PUNT_ACTION: return "PUNT"
    if a == FG_ACTION:   return "FG  "
    return f"P{a:02d} "


def show_strategies(gs: GameState, player_team: str):
    """Compute, display, and sample both equilibrium strategies.
    Returns (actions, sampled_off_idx, sampled_def_idx) or (None,None,None).
    """
    if gs.phase != "scrimmage":
        print("  (No strategy matrix — not a scrimmage state)")
        return None, None, None

    A, actions = build_scrimmage_payoff_matrix(
        SC, PC, Q, gs.tau, gs.x, gs.down, gs.to_first, gs.delta, h=0, V=V
    )
    v_off, p_off = solve_row_strategy(A)
    _, p_def     = solve_col_strategy(A)

    off_team = "A" if gs.a_has_ball else "B"
    def_team = "B" if gs.a_has_ball else "A"

    print(f"  Game value (Team {off_team} offense perspective): {v_off:+.4f}")
    print()

    # Sample
    s_off = int(np.random.choice(len(actions), p=p_off / p_off.sum()))
    s_def = int(np.random.choice(10,           p=p_def / p_def.sum()))

    you_off = (player_team == off_team)
    you_def = (player_team == def_team)

    # Offense strategy
    print(f"  [OFFENSE — Team {off_team}{'  ← YOU' if you_off else ''}]")
    for i in sorted(range(len(actions)), key=lambda i: -p_off[i]):
        if p_off[i] < 0.01 and i != s_off:
            continue
        arrow = ">>>" if i == s_off else "   "
        bar   = "#" * int(p_off[i] * 28)
        tag   = ""
        if i == s_off:
            tag = "  ← RECOMMENDED" if you_off else "  ← sample"
        print(f"  {arrow} {action_label(actions[i])}  {p_off[i]*100:5.1f}%  {bar}{tag}")
    print()

    # Defense strategy
    print(f"  [DEFENSE — Team {def_team}{'  ← YOU' if you_def else ''}]")
    for j in sorted(range(10), key=lambda j: -p_def[j]):
        if p_def[j] < 0.01 and j != s_def:
            continue
        arrow = ">>>" if j == s_def else "   "
        bar   = "#" * int(p_def[j] * 28)
        tag   = ""
        if j == s_def:
            tag = "  ← RECOMMENDED" if you_def else "  ← sample"
        print(f"  {arrow} {DEFENSE_CARDS[j]}     {p_def[j]*100:5.1f}%  {bar}{tag}")
    print()

    return actions, s_off, s_def


# ---------------------------------------------------------------------------
# Concrete play resolution
# ---------------------------------------------------------------------------

def advance_clock_gs(gs: GameState, ticks: int) -> str:
    """Advance clock in-place; return a note if two-minute warning triggered."""
    note = ""
    if gs.q in (2, 4) and gs.tau > TWO_MIN_TICK and gs.tau - ticks <= TWO_MIN_TICK:
        gs.tau = TWO_MIN_TICK
        note = "  *** TWO-MINUTE WARNING — clock stops at 2:00 ***"
    else:
        gs.tau = max(0, gs.tau - ticks)
    return note


def _auto_timeout(gs: GameState, ticks: int, reduced: int,
                  off_team: str, def_team: str,
                  off_tos: int, def_tos: int) -> int:
    """Auto mode: compare V at normal vs reduced clock to decide TO call."""
    try:
        tau_normal  = max(0, gs.tau - ticks)
        tau_reduced = max(0, gs.tau - reduced)
        v_normal  = V.get_scrimmage(gs.q, tau_normal,  gs.x, gs.down, gs.to_first, gs.delta, gs.h)
        v_reduced = V.get_scrimmage(gs.q, tau_reduced, gs.x, gs.down, gs.to_first, gs.delta, gs.h)
    except (KeyError, AttributeError):
        return ticks   # state not in V (wrong quarter etc.) — skip TO

    EPS = 1e-4
    if v_reduced > v_normal + EPS and off_tos > 0:
        if gs.a_has_ball: gs.to_a -= 1
        else:             gs.to_b -= 1
        print(f"  [AUTO] Team {off_team} (offense) calls timeout  "
              f"(v {v_normal:+.3f} → {v_reduced:+.3f}; clock {ticks}→{reduced} ticks)")
        return reduced
    elif v_reduced < v_normal - EPS and def_tos > 0:
        if gs.a_has_ball: gs.to_b -= 1
        else:             gs.to_a -= 1
        print(f"  [AUTO] Team {def_team} (defense) calls timeout  "
              f"(v {v_normal:+.3f} → {v_reduced:+.3f}; clock {ticks}→{reduced} ticks)")
        return reduced
    return ticks


def _offer_timeout(gs: GameState, ticks: int) -> int:
    """Offer both teams a timeout after a play. Returns final ticks consumed.

    Rules: 3-tick→1, 2-tick→0, 1-tick→0.  Both teams may call independently;
    if either (or both) call, the ticks are reduced.  In auto mode the V-table
    decides which team (if any) calls.
    """
    reduced = TIMEOUT_REDUCE.get(ticks, ticks)
    if reduced == ticks:
        return ticks   # no reduction possible (e.g. PAT = 0 ticks)

    off_team = "A" if gs.a_has_ball else "B"
    def_team = "B" if gs.a_has_ball else "A"
    off_tos  = gs.to_a if gs.a_has_ball else gs.to_b
    def_tos  = gs.to_b if gs.a_has_ball else gs.to_a

    if off_tos == 0 and def_tos == 0:
        return ticks   # no TOs left for either team

    if AUTO_DICE:
        return _auto_timeout(gs, ticks, reduced, off_team, def_team, off_tos, def_tos)

    # Manual mode: prompt each team that still has TOs
    called = False
    if off_tos > 0:
        c = input(f"  Team {off_team} (OFFENSE) call timeout? "
                  f"[{off_tos} left  {ticks}t → {reduced}t] (Y/N): ").strip().upper()
        if c.startswith("Y"):
            if gs.a_has_ball: gs.to_a -= 1
            else:             gs.to_b -= 1
            print(f"  TIMEOUT — Team {off_team}.  Clock reduced: {ticks} → {reduced} ticks")
            called = True

    if def_tos > 0:
        c = input(f"  Team {def_team} (DEFENSE) call timeout? "
                  f"[{def_tos} left  {ticks}t → {reduced}t] (Y/N): ").strip().upper()
        if c.startswith("Y"):
            if gs.a_has_ball: gs.to_b -= 1
            else:             gs.to_a -= 1
            print(f"  TIMEOUT — Team {def_team}.  Clock reduced: {ticks} → {reduced} ticks")
            called = True

    return reduced if called else ticks


def tick_play(gs: GameState, ticks: int) -> str:
    """Offer TO then advance clock. Returns note string (two-minute warning etc.)."""
    final = _offer_timeout(gs, ticks)
    return advance_clock_gs(gs, final)


def half_dist(x: int, pen: int) -> int:
    if pen > 0:   return min(pen, (100 - x) // 2)
    if pen < 0:   return max(pen, -(x // 2))
    return 0


def apply_yards(gs: GameState, yards: int, out_of_bounds: bool,
                auto_first: bool = False) -> tuple:
    """Apply yards to scrimmage state. Returns (events_list, ticks, possession_changed).
    Modifies gs in place. events_list = list of strings describing what happened.
    """
    events = []
    new_x  = gs.x + yards

    # Ticks
    if out_of_bounds:   ticks = 1
    elif yards < 0:     ticks = 2
    elif yards < 20:    ticks = 2
    else:               ticks = 3

    # Touchdown
    if new_x >= 100:
        events.append(f"TOUCHDOWN! (ball in end zone at x={new_x})")
        handle_td(gs, events)
        return events, ticks, True

    # Safety
    if new_x <= 0:
        events.append(f"SAFETY! (ball at x={new_x})")
        handle_safety(gs, events)
        return events, ticks, True

    # Normal
    gs.x = new_x
    new_tf = gs.to_first - yards
    if auto_first or new_tf <= 0:
        gs.down     = 1
        gs.to_first = goal_to_go(new_x)
        events.append(f"First down! Ball at {yd(new_x)}, {down_str(gs.down, gs.to_first)}")
    elif gs.down < 4:
        gs.down    += 1
        gs.to_first = new_tf
        events.append(f"Ball at {yd(new_x)}, {down_str(gs.down, gs.to_first)}")
    else:
        # 4th-down failure
        recv_x  = 100 - new_x
        recv_tf = goal_to_go(recv_x)
        events.append(f"Turnover on downs! Opponent's ball at {yd(recv_x)}")
        swap_possession(gs, recv_x, 1, recv_tf)
        return events, ticks, True

    return events, ticks, False


def handle_td(gs: GameState, events: list) -> None:
    """Score TD + PAT, set up kickoff. Modifies gs."""
    if AUTO_DICE:
        pat_ok, pat_roll = roll_pat()
        events.append(f"  PAT: {'good' if pat_ok else 'NO GOOD'} (roll {pat_roll}/36)")
    else:
        ans = input("  PAT good? (Y/N): ").strip().upper()
        pat_ok = ans.startswith("Y")
        events.append(f"  PAT: {'good' if pat_ok else 'NO GOOD'}")
    pts = TD_POINTS + (1 if pat_ok else 0)
    events.append(f"  +{pts} points total")
    if gs.a_has_ball:
        gs.score_a += pts
    else:
        gs.score_b += pts
    gs.recalc_delta()
    gs.phase = "kickoff"
    events.append(f"  Score now: A={gs.score_a}  B={gs.score_b}")
    events.append(f"  → KICKOFF: Team {'A' if gs.a_has_ball else 'B'} kicks off")


def handle_safety(gs: GameState, events: list) -> None:
    """Defense scores safety, set up safety kick. Modifies gs."""
    # Defense scores 2
    if gs.a_has_ball:
        gs.score_b += SAFETY_POINTS
    else:
        gs.score_a += SAFETY_POINTS
    gs.recalc_delta()
    gs.phase = "safety_kick"
    events.append(f"  +{SAFETY_POINTS} points to defense. Score: A={gs.score_a}  B={gs.score_b}")
    events.append("  → SAFETY KICK: scored-against team kicks from own 20")


def swap_possession(gs: GameState, new_x: int, new_down: int, new_tf: int) -> None:
    gs.x        = new_x
    gs.down     = new_down
    gs.to_first = new_tf
    gs.delta    = -gs.delta
    gs.a_has_ball = not gs.a_has_ball


def resolve_branch(gs: GameState, branch, off_team: str, def_team: str) -> list:
    """Resolve a concrete AND-chain branch. Returns list of event strings.
    Modifies gs in place."""
    primary   = branch[0]
    modifiers = branch[1:]
    auto_first = any(isinstance(m, FirstDownOutcome) for m in modifiers)

    if isinstance(primary, LongGainOutcome):
        yards, dice = roll_lg()
        events = [f"LONG GAIN — rolled {dice} → +{yards} yards"]
        more, ticks, _ = apply_yards(gs, yards, False, auto_first)
        events += more
        note = tick_play(gs, ticks)
        if note: events.append(note)
        return events

    if isinstance(primary, YardsOutcome):
        y  = primary.yards
        ob = primary.out_of_bounds
        events = [f"{'+'if y>=0 else ''}{y} yards{'  (out of bounds)' if ob else ''}"]
        more, ticks, _ = apply_yards(gs, y, ob, auto_first)
        events += more
        note = tick_play(gs, ticks)
        if note: events.append(note)
        return events

    if isinstance(primary, IncompleteOutcome):
        events = ["Incomplete pass"]
        if gs.down < 4:
            gs.down    += 1
            events.append(f"Now {down_str(gs.down, gs.to_first)}")
        else:
            recv_x  = 100 - gs.x
            recv_tf = goal_to_go(recv_x)
            events.append("Turnover on downs (incomplete on 4th)")
            swap_possession(gs, recv_x, 1, recv_tf)
        note = tick_play(gs, 1)
        if note: events.append(note)
        return events

    if isinstance(primary, FumbleOutcome):
        recv_x  = 100 - gs.x
        recv_tf = goal_to_go(recv_x)
        events = ["FUMBLE — defense recovers!"]
        swap_possession(gs, recv_x, 1, recv_tf)
        note = tick_play(gs, 1)
        if note: events.append(note)
        return events

    if isinstance(primary, InterceptionOutcome):
        ret    = primary.return_yards
        x_after = gs.x - ret
        events = [f"INTERCEPTION — returned {ret} yards" if ret else "INTERCEPTION"]
        if x_after <= 0:
            events.append("PICK-SIX!")
            off_saves = gs.a_has_ball
            # swap so defense is now offense, then TD
            swap_possession(gs, 1, 1, 10)
            handle_td(gs, events)
        elif x_after >= 100:
            events.append("Touchback — ball at defense's own 20")
            recv_x  = 20
            recv_tf = goal_to_go(recv_x)
            swap_possession(gs, recv_x, 1, recv_tf)
        else:
            recv_x  = 100 - x_after
            recv_tf = goal_to_go(recv_x)
            events.append(f"Defense ball at {yd(x_after)} (offense frame)")
            swap_possession(gs, recv_x, 1, recv_tf)
        note = tick_play(gs, 2)
        if note: events.append(note)
        return events

    if isinstance(primary, PenaltyOutcome):
        capped = half_dist(gs.x, primary.yards)
        new_x  = max(1, min(99, gs.x + capped))
        events = [f"PENALTY {'+' if primary.yards>=0 else ''}{primary.yards} yds "
                  f"(capped to {capped:+d}) — down repeats"]
        new_tf = gs.to_first - capped
        if auto_first or new_tf <= 0:
            gs.down     = 1
            gs.to_first = goal_to_go(new_x)
        gs.x = new_x
        note = tick_play(gs, 1)
        if note: events.append(note)
        return events

    if isinstance(primary, LossOfDownOutcome):
        capped = half_dist(gs.x, primary.yards)
        new_x  = max(1, min(99, gs.x + capped))
        events = [f"LOSS OF DOWN + {'+' if primary.yards>=0 else ''}{primary.yards} yds"]
        if auto_first or gs.to_first - capped <= 0:
            gs.down     = 1
            gs.to_first = goal_to_go(new_x)
        elif gs.down < 4:
            gs.down    += 1
            gs.to_first = gs.to_first - capped
        else:
            recv_x  = 100 - new_x
            recv_tf = goal_to_go(recv_x)
            events.append("Turnover on downs (LOD on 4th)")
            gs.x = new_x
            swap_possession(gs, recv_x, 1, recv_tf)
            note = tick_play(gs, 1)
            if note: events.append(note)
            return events
        gs.x = new_x
        note = tick_play(gs, 1)
        if note: events.append(note)
        return events

    if isinstance(primary, FirstDownOutcome):
        events = ["Automatic first down"]
        gs.to_first = goal_to_go(gs.x)
        gs.down = 1
        return events

    return [f"(Unknown outcome type: {type(primary).__name__})"]


OUTCOME_HELP = (
    "  Outcome format:  7 | -3 | 7OB | I | F | INT | INT30 | LG45 | TD | S | P7 | P-10 | -15LOD"
)

def apply_manual_outcome(gs: GameState, raw: str) -> list:
    """Parse a human-typed outcome string and apply it to gs.
    Returns list of event strings."""
    s = raw.strip().upper()
    if not s:
        return ["(no outcome entered — state unchanged)"]

    off_team = "A" if gs.a_has_ball else "B"
    def_team = "B" if gs.a_has_ball else "A"

    # Touchdown
    if s == "TD":
        events = ["TOUCHDOWN!"]
        handle_td(gs, events)
        note = tick_play(gs, 2)
        if note: events.append(note)
        return events

    # Safety
    if s in ("S", "SAFETY"):
        events = ["SAFETY!"]
        handle_safety(gs, events)
        note = tick_play(gs, 2)
        if note: events.append(note)
        return events

    # Incomplete
    if s == "I":
        events = ["Incomplete pass"]
        if gs.down < 4:
            gs.down += 1
            events.append(f"Now {down_str(gs.down, gs.to_first)}")
        else:
            recv_x = 100 - gs.x
            swap_possession(gs, recv_x, 1, goal_to_go(recv_x))
            events.append("Turnover on downs (incomplete on 4th)")
        note = tick_play(gs, 1)
        if note: events.append(note)
        return events

    # Fumble
    if s == "F":
        events = ["FUMBLE — defense recovers"]
        recv_x = 100 - gs.x
        swap_possession(gs, recv_x, 1, goal_to_go(recv_x))
        note = tick_play(gs, 1)
        if note: events.append(note)
        return events

    # Interception: INT or INT30
    if s.startswith("INT"):
        ret = int(s[3:]) if len(s) > 3 and s[3:].lstrip("-").isdigit() else 0
        branch = [InterceptionOutcome(return_yards=ret)]
        return resolve_branch(gs, branch, off_team, def_team)

    # Long Gain with explicit yards: LG45
    if s.startswith("LG"):
        remainder = s[2:]
        if remainder.lstrip("-").isdigit():
            yards = int(remainder)
        else:
            yards = int(input("  LG yards gained: ").strip())
        events = [f"Long Gain: +{yards} yards"]
        more, ticks, _ = apply_yards(gs, yards, False, False)
        events += more
        note = tick_play(gs, ticks)
        if note: events.append(note)
        return events

    # Out-of-bounds: 7OB or -3OB
    if s.endswith("OB"):
        yards = int(s[:-2])
        events = [f"{'+'if yards>=0 else ''}{yards} yards (OB)"]
        more, ticks, _ = apply_yards(gs, yards, True, False)
        events += more
        note = tick_play(gs, ticks)
        if note: events.append(note)
        return events

    # Loss of Down: 7LOD / -15LOD
    if s.endswith("LOD"):
        rest = s[:-3]
        if rest.lstrip("-").isdigit():
            yards = int(rest)
            branch = [LossOfDownOutcome(yards=yards)]
            return resolve_branch(gs, branch, off_team, def_team)

    # Penalty: P7 / P-10 / 7P / -10P  (repeat down)
    if s.startswith("P") and s[1:].lstrip("-").isdigit():
        yards = int(s[1:])
        branch = [PenaltyOutcome(yards=yards)]
        return resolve_branch(gs, branch, off_team, def_team)
    if s.endswith("P") and s[:-1].lstrip("-").isdigit():
        yards = int(s[:-1])
        branch = [PenaltyOutcome(yards=yards)]
        return resolve_branch(gs, branch, off_team, def_team)

    # Plain yards: 7 / -3
    if s.lstrip("-").isdigit():
        yards = int(s)
        events = [f"{'+'if yards>=0 else ''}{yards} yards"]
        more, ticks, _ = apply_yards(gs, yards, False, False)
        events += more
        note = tick_play(gs, ticks)
        if note: events.append(note)
        return events

    return [f"(Unrecognised outcome {raw!r} — state unchanged. {OUTCOME_HELP})"]


def resolve_scrimmage_play(gs: GameState, offense_action: int, def_card: str,
                            player_team: str) -> None:
    """Look up chart cell, resolve concretely, update gs."""
    off_team = "A" if gs.a_has_ball else "B"
    def_team = "B" if gs.a_has_ball else "A"

    if offense_action == PUNT_ACTION:
        resolve_punt(gs, def_card, player_team)
        return

    if offense_action == FG_ACTION:
        resolve_fg(gs, player_team)
        return

    # Normal play
    raw_cell = SC_RAW.loc[def_card, str(offense_action)]
    print(f"\n  Chart cell [{def_card} × P{offense_action:02d}] = {raw_cell!r}")

    if not AUTO_DICE:
        # Manual mode: human tells us what happened
        print(OUTCOME_HELP)
        outcome_str = input("  What happened? ").strip()
        events = apply_manual_outcome(gs, outcome_str)
        for e in events:
            print(f"  {e}")
        return

    cell = SC[(def_card, str(offense_action))]

    if isinstance(cell, ChoiceOutcome):
        branch = resolve_or_choice(cell, off_team, def_team)
    else:
        branch = cell

    events = resolve_branch(gs, branch, off_team, def_team)
    for e in events:
        print(f"  {e}")


def resolve_or_choice(cell: ChoiceOutcome, off_team: str, def_team: str):
    """Present OR-choice to the appropriate team; return chosen branch."""
    chooser = off_team if cell.resolver == "offense" else def_team
    print(f"\n  OR-CHOICE (Team {chooser} — {'favored by penalty'} — decides):")
    for i, branch in enumerate(cell.branches):
        desc = " AND ".join(atom_desc(a) for a in branch)
        print(f"    [{i+1}] {desc}")
    while True:
        raw = input(f"  Team {chooser} chooses (1–{len(cell.branches)}): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(cell.branches):
                return cell.branches[idx]
        except ValueError:
            pass
        print("  Invalid choice.")


def atom_desc(atom) -> str:
    if isinstance(atom, YardsOutcome):
        s = f"{'+'if atom.yards>=0 else ''}{atom.yards} yds"
        if atom.out_of_bounds: s += " OB"
        return s
    if isinstance(atom, IncompleteOutcome):    return "Incomplete"
    if isinstance(atom, FumbleOutcome):        return "Fumble"
    if isinstance(atom, LongGainOutcome):      return "Long Gain"
    if isinstance(atom, InterceptionOutcome):  return f"INT ret={atom.return_yards}"
    if isinstance(atom, PenaltyOutcome):       return f"Penalty {atom.yards:+d}P"
    if isinstance(atom, LossOfDownOutcome):    return f"LOD {atom.yards:+d}"
    if isinstance(atom, FirstDownOutcome):     return "1ST"
    return str(atom)


def resolve_punt(gs: GameState, def_card: str, player_team: str) -> None:
    punt_yds, ret_type, ret_yds = PUNT_CHART[def_card]
    land_x = gs.x + punt_yds
    print(f"\n  PUNT — card {def_card}: {punt_yds} yds, return={ret_type}/{ret_yds}")

    if land_x >= 110:
        recv_x = 20
        print(f"  Through end zone → TOUCHBACK at receiver's 20")
    elif land_x >= 100:
        print(f"  Lands in end zone (x={land_x})")
        if ret_type in ("none", "ob"):
            # No return possible — automatic touchback
            recv_x = 20
            print(f"  No return (ret_type={ret_type}) → TOUCHBACK at receiver's 20")
        elif ret_type == "fixed_fumble":
            # Kicking team recovers; stay at kicker's own 20 (touchback-style)
            kicker_x = 20
            print(f"  Fumble return in end zone → kicking team at their own {kicker_x}")
            gs.x = kicker_x; gs.down = 1; gs.to_first = goal_to_go(kicker_x)
            note = tick_play(gs, 1)
            if note: print(f"  {note}")
            return
        else:
            # fixed or lg: receiver may choose touchback or return
            choice = input("  Receiver: (T)ouchback at 20 or (R)eturn? ").strip().upper()
            if choice.startswith("T"):
                recv_x = 20
            elif ret_type == "fixed":
                recv_x = max(1, 100 - (land_x - ret_yds))
                print(f"  Ball at receiver's own {recv_x}")
            else:  # lg
                if AUTO_DICE:
                    yards, dice = roll_lg(); print(f"  LG return: {dice} → {yards} yds")
                else:
                    yards = int(input("  LG return yards: ").strip())
                recv_x = max(1, 100 - (land_x - yards))
                print(f"  Ball at receiver's own {recv_x}")
    else:
        if ret_type in ("ob", "none"):
            recv_x = max(1, 100 - land_x)
        elif ret_type == "fixed":
            recv_x = max(1, 100 - (land_x - ret_yds))
        elif ret_type == "lg":
            if AUTO_DICE:
                yards, dice = roll_lg(); print(f"  LG return: {dice} → {yards} yds")
            else:
                yards = int(input("  LG return yards: ").strip())
            recv_x = max(1, 100 - (land_x - yards))
        elif ret_type == "fixed_fumble":
            # Kicking team recovers; kicker keeps ball (no swap)
            kicker_x = max(1, min(99, land_x - ret_yds))
            kicker_tf = goal_to_go(kicker_x)
            print(f"  Fumble on return — kicking team recovers at {yd(kicker_x)}")
            gs.x = kicker_x; gs.down = 1; gs.to_first = kicker_tf
            note = tick_play(gs, 1)
            if note: print(f"  {note}")
            return
        else:
            recv_x = max(1, 100 - land_x)

    print(f"  Receiver's ball at their own {recv_x}")
    swap_possession(gs, recv_x, 1, goal_to_go(recv_x))
    note = tick_play(gs, 1)
    if note: print(f"  {note}")


def resolve_fg(gs: GameState, player_team: str) -> None:
    p = fg_prob(gs.x)
    dist = 100 - gs.x
    print(f"\n  FIELD GOAL attempt from {dist} yds — P(make)={p*36:.0f}/36")
    if AUTO_DICE:
        roll = random.randint(1, 36)
        made = roll <= round(p * 36)
        print(f"  Roll: {roll}")
    else:
        ans = input("  Result — (M)ade or (X) no good? ").strip().upper()
        made = ans.startswith("M")
    if made:
        print(f"  FIELD GOAL GOOD! +{FG_POINTS} points")
        if gs.a_has_ball: gs.score_a += FG_POINTS
        else:             gs.score_b += FG_POINTS
        gs.recalc_delta()
        gs.phase = "kickoff"
        print(f"  Score: A={gs.score_a}  B={gs.score_b}")
        print(f"  → KICKOFF")
    else:
        print(f"  FIELD GOAL NO GOOD — defense ball at their own 20")
        swap_possession(gs, 20, 1, goal_to_go(20))
    note = tick_play(gs, 1)
    if note: print(f"  {note}")


KICKOFF_HELP = (
    "  Result format:  recv-25 | kicker-40 | TD\n"
    "  (recv-N = receiver's ball at their own N; kicker-N = kicker keeps at their own N)"
)


def _apply_kickoff_result(gs: GameState, raw: str) -> bool:
    """Parse and apply a kickoff result string. Sets gs.phase. Returns True if handled."""
    s = raw.strip().upper()
    if s == "TD":
        print("  KICKOFF RETURN TOUCHDOWN!")
        swap_possession(gs, 1, 1, 10)
        ko_events = []
        handle_td(gs, ko_events)   # sets gs.phase = "kickoff" (scorer kicks next)
        for e in ko_events:
            print(f"  {e}")
        # Leave gs.phase = "kickoff" — handle_td set it correctly
        return True
    if s.startswith("RECV-"):
        try:
            recv_x = int(s[5:])
            print(f"  Receiver's ball at their own {recv_x}")
            swap_possession(gs, recv_x, 1, goal_to_go(recv_x))
            gs.phase = "scrimmage"
            return True
        except ValueError:
            pass
    if s.startswith("KICKER-"):
        try:
            kicker_x = int(s[7:])
            print(f"  Kicking team keeps ball at their own {kicker_x}")
            gs.x = kicker_x; gs.down = 1; gs.to_first = goal_to_go(kicker_x)
            gs.phase = "scrimmage"
            return True
        except ValueError:
            pass
    # Plain number = receiver yard line
    try:
        recv_x = int(s)
        print(f"  Receiver's ball at their own {recv_x}")
        swap_possession(gs, recv_x, 1, goal_to_go(recv_x))
        gs.phase = "scrimmage"
        return True
    except ValueError:
        pass
    print(f"  (Unrecognized kickoff result {raw!r})")
    print(KICKOFF_HELP)
    return False


def resolve_kickoff(gs: GameState) -> None:
    """Handle kickoff interactively, update gs."""
    off_team = "A" if gs.a_has_ball else "B"

    if not AUTO_DICE:
        # Manual mode: human enters what happened on the board (Bug 7 fix: no N/O prompt)
        print(f"\n  Team {off_team} kickoff")
        print(KICKOFF_HELP)
        raw = input("  Kickoff result: ").strip()
        if _apply_kickoff_result(gs, raw):
            note = advance_clock_gs(gs, 1)
            if note: print(f"  {note}")
        return

    # Auto-dice mode: ask Normal or Onside
    choice = input(f"\n  Team {off_team} kickoff — (N)ormal or (O)nside? ").strip().upper()

    # Safety-kick offset: add 25 yards to all yard-line results (Bug 2 fix)
    safety_offset = 25 if gs.phase == "safety_kick" else 0

    if choice.startswith("O"):
        # Onside
        roll = roll_1d6()
        adj  = 0 if gs.delta < 0 else 1   # +1 if not trailing
        adj_roll = min(roll + adj, 6)
        print(f"  Onside die: {roll} + {adj} adj = {adj_roll}")
        who, x_val = KICKOFF_ONSIDE[adj_roll]
        if who == "kicker":
            print(f"  KICKING TEAM recovers at their own {x_val}")
            gs.x = x_val; gs.down = 1; gs.to_first = goal_to_go(x_val)
        else:
            print(f"  Receiving team ball at their own {x_val}")
            swap_possession(gs, x_val, 1, goal_to_go(x_val))
        gs.phase = "scrimmage"

    else:
        # Normal kickoff
        total, d1, d2 = roll_2d6()
        print(f"  Kickoff dice: {d1}+{d2}={total}")
        res_type, param = KICKOFF_NORMAL[total]

        if res_type == "receiver_ball":
            recv_x = min(99, param + safety_offset)
            print(f"  Receiver's ball at their own {recv_x}")
            swap_possession(gs, recv_x, 1, goal_to_go(recv_x))
            gs.phase = "scrimmage"

        elif res_type == "long_gain":
            extra = param or 0
            yards, dice = roll_lg()
            actual = yards + extra + safety_offset
            if actual >= 100:
                offset_str = f"+{safety_offset}" if safety_offset else ""
                print(f"  LG+{extra}{offset_str} = {actual} — KICKOFF RETURN TOUCHDOWN!")
                swap_possession(gs, 1, 1, 10)
                ko_events = []
                handle_td(gs, ko_events)
                for e in ko_events: print(f"  {e}")
                # handle_td sets gs.phase = "kickoff" — leave it (Bug 3 fix)
            else:
                recv_x = max(1, min(99, actual))
                offset_str = f"+{safety_offset}" if safety_offset else ""
                print(f"  LG {dice}+{extra}{offset_str}={actual} → receiver's own {recv_x}")
                swap_possession(gs, recv_x, 1, goal_to_go(recv_x))
                gs.phase = "scrimmage"

        elif res_type == "fumble":
            # Reroll — kicking team recovers unless second fumble
            print(f"  FUMBLE on kickoff! Reroll...")
            total2, d1, d2 = roll_2d6()
            print(f"  Reroll: {d1}+{d2}={total2}")
            res2, p2 = KICKOFF_NORMAL[total2]
            if res2 == "fumble":
                recv_x = min(99, 20 + safety_offset)
                print(f"  Second fumble — receiving team ball at their own {recv_x}")
                swap_possession(gs, recv_x, 1, goal_to_go(recv_x))
            else:
                # Kicking team recovers
                if res2 == "receiver_ball":
                    recv_x_frame = p2
                elif res2 == "long_gain":
                    yards, dice = roll_lg()
                    recv_x_frame = yards + (p2 or 0)
                else:
                    recv_x_frame = 25  # penalty/other proxy
                kicker_x = max(1, min(99, 100 - recv_x_frame + safety_offset))
                print(f"  Kicking team recovers at their own {yd(kicker_x)}")
                gs.x = kicker_x; gs.down = 1; gs.to_first = goal_to_go(kicker_x)
            gs.phase = "scrimmage"

        elif res_type == "penalty":
            print(f"  PENALTY on kickoff (−10 yds to yard line) — reroll...")
            total2, d1, d2 = roll_2d6()
            print(f"  Reroll: {d1}+{d2}={total2}")
            res2, p2 = KICKOFF_NORMAL[total2]
            base = p2 if res2 == "receiver_ball" else 25
            recv_x = max(1, min(99, base + param + safety_offset))  # param = -10
            print(f"  Receiver's ball at their own {recv_x}")
            swap_possession(gs, recv_x, 1, goal_to_go(recv_x))
            gs.phase = "scrimmage"

    note = advance_clock_gs(gs, 1)
    if note: print(f"  {note}")


# ---------------------------------------------------------------------------
# State override
# ---------------------------------------------------------------------------

def state_override(gs: GameState) -> None:
    print(f"\n  {SEP2}")
    print("  STATE OVERRIDE — enter new state values (blank = keep current)")
    print(f"  Current: phase={gs.phase}, x={gs.x}, down={gs.down}, tf={gs.to_first}, "
          f"delta={gs.delta}, tau={gs.tau}")
    print(f"  Scores: A={gs.score_a}  B={gs.score_b}  a_has_ball={gs.a_has_ball}")

    def prompt(label, current, cast=int):
        raw = input(f"  {label} [{current}]: ").strip()
        return cast(raw) if raw else current

    new_phase = input(f"  phase [scrimmage/kickoff/safety_kick] [{gs.phase}]: ").strip()
    if new_phase in ("scrimmage", "kickoff", "safety_kick"):
        gs.phase = new_phase

    if gs.phase == "scrimmage":
        gs.x        = prompt("x (yards from own goal, 1-99)", gs.x)
        gs.down     = prompt("down (1-4)", gs.down)
        gs.to_first = prompt("to_first (yards to 1st down)", gs.to_first)

    gs.score_a   = prompt("score_a", gs.score_a)
    gs.score_b   = prompt("score_b", gs.score_b)
    a_ball_str   = input(f"  a_has_ball [{'Y' if gs.a_has_ball else 'N'}]: ").strip().upper()
    if a_ball_str in ("Y", "N"):
        gs.a_has_ball = (a_ball_str == "Y")
    gs.tau       = prompt("tau (ticks remaining)", gs.tau)
    gs.to_a      = prompt("to_a (Team A timeouts, 0-3)", gs.to_a)
    gs.to_b      = prompt("to_b (Team B timeouts, 0-3)", gs.to_b)
    gs.recalc_delta()
    print(f"  State updated: x={gs.x}, down={gs.down}, tf={gs.to_first}, "
          f"delta={gs.delta:+d}, tau={gs.tau}, to_a={gs.to_a}, to_b={gs.to_b}")
    print(f"  {SEP2}\n")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def prompt_offense_action(gs: GameState, actions: list) -> int:
    labels = {action_label(a).strip(): a for a in actions}
    labels.update({str(a): a for a in actions})
    while True:
        raw = input("  Offense played (play number or PUNT/FG, or O=override): ").strip().upper()
        if raw == "O":
            return -1
        if raw in labels:
            return labels[raw]
        try:
            n = int(raw)
            if n in actions:
                return n
        except ValueError:
            pass
        print(f"  Legal actions: {[action_label(a).strip() for a in actions]}")


def prompt_def_card() -> str:
    while True:
        raw = input("  Defense card (A-J, or O=override): ").strip().upper()
        if raw == "O":
            return "O"
        if raw in DEFENSE_CARDS:
            return raw
        print(f"  Valid cards: {DEFENSE_CARDS}")


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def main():
    gs = GameState()

    print(SEP)
    print("  FOOTBALL STRATEGY — Q4 Playtest")
    print(SEP)
    player_team = input("  You are Team (A or B): ").strip().upper()
    if player_team not in ("A", "B"):
        player_team = "A"

    kicker = input("  Which team kicks off? (A or B): ").strip().upper()
    if kicker not in ("A", "B"):
        kicker = "A"

    gs.a_has_ball = (kicker == "A")
    gs.phase = "kickoff"
    gs.delta = 0

    print(f"\n  Team {player_team} vs opponent. Team {kicker} kicks off. Q4 begins.\n")

    while gs.phase != "over":
        # Check game over
        if gs.tau == 0:
            diff = gs.score_a - gs.score_b
            print(SEP)
            print(f"  GAME OVER — Final: A={gs.score_a}  B={gs.score_b}")
            if diff > 0:   print("  Team A wins!")
            elif diff < 0: print("  Team B wins!")
            else:           print("  Tie game.")
            break

        display_state(gs, player_team)

        # Override option before strategy
        ov = input("  (O)verride state, or press Enter to continue: ").strip().upper()
        if ov == "O":
            state_override(gs)
            continue

        # --- KICKOFF ---
        if gs.phase in ("kickoff", "safety_kick"):
            print(f"  Kickoff value (kicker perspective): ", end="")
            try:
                kv = kickoff_state_value(Q, gs.tau, gs.delta, gs.h, V)
                print(f"{kv:+.4f}")
            except Exception:
                print("N/A")

            if gs.phase == "safety_kick":
                print("  Safety kick: kicker's choice — normal kickoff (+25 yds) or punt from own 20")

            input("  Press Enter to resolve kickoff... ")
            resolve_kickoff(gs)
            print()
            continue

        # --- SCRIMMAGE ---
        actions, s_off, s_def = show_strategies(gs, player_team)
        if actions is None:
            state_override(gs)
            continue

        # Prompt for actual plays
        print(SEP2)
        off_action = prompt_offense_action(gs, actions)
        if off_action == -1:
            state_override(gs)
            continue

        def_card = prompt_def_card()
        if def_card == "O":
            state_override(gs)
            continue

        print()
        resolve_scrimmage_play(gs, off_action, def_card, player_team)
        gs.recalc_delta()
        print()

    print(SEP)
    print("  Thanks for playing!")


if __name__ == "__main__":
    main()
