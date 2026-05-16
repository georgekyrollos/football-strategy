"""
State dataclasses for the Football Strategy full-game solver.

Symmetric possession convention (enforced everywhere):
  - x is always measured from the CURRENT OFFENSE's own goal line.
    The offense drives toward x = 100 (opponent's goal line).
  - delta = offense_score - defense_score  (from current offense's perspective)
  - h = +1 if the current offense's team receives the second-half kickoff,
         -1 if the current defense's team receives it,
          0  in Q3/Q4 (second half kickoff already resolved)
  - On every possession change apply:
        new_x     = 100 - old_x
        new_delta = -old_delta
        new_h     = -old_h
    and the caller negates the continuation value (swapped=True).

No artificial clipping of delta or to_first — store exact reachable values.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from football_strategy.constants import (
    PLAYS_BASE, PUNT_ACTION, FG_ACTION,
    REDZONE_OUTER_X, REDZONE_OUTER_ILLEGAL,
    REDZONE_INNER_X, REDZONE_INNER_ILLEGAL,
    FG_MIN_X,
)


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScrimmageState:
    q:        int   # quarter: 1–4
    tau:      int   # ticks remaining in quarter: 0–60
    x:        int   # yards from current offense's own goal line: 0–100
    down:     int   # current down: 1–4
    to_first: int   # yards to first down (exact, no clipping)
    delta:    int   # offense_score - defense_score (exact, no clipping)
    h:        int   # second-half kickoff: +1/-1 for Q1/Q2; 0 for Q3/Q4


@dataclass(frozen=True)
class KickoffState:
    q:     int   # quarter: 1–4
    tau:   int   # ticks remaining: 0–60
    delta: int   # kicker_score - receiver_score (kicker is "current offense")
    h:     int   # same h convention; 0 for Q3/Q4


@dataclass(frozen=True)
class SafetyKickState:
    """State after a safety: scored-against team free-kicks from own 20."""
    q:     int
    tau:   int
    delta: int   # free_kicker_score - receiver_score
    h:     int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def goal_to_go(x: int) -> int:
    """Yards to first down when inside (or at) the red zone.
    Normal: 10 yards.  Near the goal: max(1, 100 - x).
    """
    return max(1, min(10, 100 - x))


def terminal_utility(delta: int) -> int:
    """Win/tie/loss utility at game end (from current offense's perspective)."""
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def legal_offense_actions(x: int, down: int) -> List[int]:
    """Return legal offense action codes for a given field position and down.

    Base plays 1–20 subject to red-zone restrictions:
      plays 17–20 illegal at x ≥ REDZONE_OUTER_X (within opponent's 20)
      plays 13–16 illegal at x ≥ REDZONE_INNER_X (within opponent's 10)
    PUNT_ACTION added on 4th down only.
    FG_ACTION added whenever x ≥ FG_MIN_X.
    """
    actions: List[int] = list(PLAYS_BASE)

    # Apply red-zone play restrictions
    if x >= REDZONE_INNER_X:
        actions = [a for a in actions if a not in REDZONE_INNER_ILLEGAL]
    if x >= REDZONE_OUTER_X:
        actions = [a for a in actions if a not in REDZONE_OUTER_ILLEGAL]

    if down == 4:
        actions.append(PUNT_ACTION)

    if x >= FG_MIN_X:
        actions.append(FG_ACTION)

    return actions
