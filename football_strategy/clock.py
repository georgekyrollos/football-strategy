"""
Clock arithmetic for the Football Strategy solver.

This module ONLY computes tick arithmetic — it does NOT create states or
decide what happens at quarter boundaries.  Those decisions belong to
solve_full_game.py.
"""

from __future__ import annotations

from typing import Tuple

from football_strategy.constants import TWO_MIN_TICK
from football_strategy.outcomes import (
    FirstDownOutcome,
    FumbleOutcome,
    IncompleteOutcome,
    InterceptionOutcome,
    LongGainOutcome,
    LossOfDownOutcome,
    PenaltyOutcome,
    SingleAtom,
    YardsOutcome,
)


def ticks_for_atom(atom: SingleAtom, yards: int = 0) -> int:
    """Return the number of clock ticks consumed by a primary atom.

    ``yards`` is the resolved yardage for a YardsOutcome (already known at
    call time).  For all other types the ``yards`` parameter is ignored.

    Rules:
      YardsOutcome:
        out_of_bounds → 1  (clock stops)
        yards < 0     → 2  (loss, clock runs)
        yards < 20    → 2  (short inbounds gain)   [see TODO in constants.py]
        yards >= 20   → 3  (long inbounds gain)    [see TODO in constants.py]
      LongGainOutcome:   3  (always a long gain)
      IncompleteOutcome: 1  (clock stops)
      FumbleOutcome:     1
      InterceptionOutcome: 2
      PenaltyOutcome:    1
      LossOfDownOutcome: 1  (penalty variant)
      FirstDownOutcome:  0  (modifier only, never a primary atom)
    """
    if isinstance(atom, YardsOutcome):
        if atom.out_of_bounds:
            return 1
        if yards < 0:
            return 2
        if yards < 20:
            return 2
        return 3

    if isinstance(atom, LongGainOutcome):
        return 3

    if isinstance(atom, IncompleteOutcome):
        return 1

    if isinstance(atom, FumbleOutcome):
        return 1

    if isinstance(atom, InterceptionOutcome):
        return 2

    if isinstance(atom, (PenaltyOutcome, LossOfDownOutcome)):
        return 1

    if isinstance(atom, FirstDownOutcome):
        return 0

    raise TypeError(f"Unknown atom type: {type(atom)!r}")


def advance_clock(tau: int, q: int, ticks: int) -> Tuple[int, bool]:
    """Advance the clock by ``ticks`` ticks in quarter ``q``.

    Returns ``(new_tau, hit_two_min_warning)``.

    The two-minute warning applies only in Q2 and Q4:
      - If ``tau > TWO_MIN_TICK`` and ``tau - ticks <= TWO_MIN_TICK``,
        the clock stops at ``TWO_MIN_TICK`` (returns ``True``).
    Otherwise the clock simply decrements: ``max(0, tau - ticks)``.

    Quarter-boundary logic (``new_tau == 0`` → next quarter) is handled
    by solve_full_game.py, NOT here.
    """
    if q in (2, 4) and tau > TWO_MIN_TICK and tau - ticks <= TWO_MIN_TICK:
        return (TWO_MIN_TICK, True)
    return (max(0, tau - ticks), False)
