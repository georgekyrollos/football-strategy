"""
Typed outcome objects produced by the chart parser.

A chart cell can be:
  - A single Branch (AND-chain of atoms applied in order), or
  - A ChoiceOutcome (OR-branches; the team indicated by `resolver` picks the best one).

OR-branch resolver rules (from the Football Strategy rulebook):
  "When a penalty creates a choice, the team favored by the penalty chooses."
  - Penalty with yards > 0 (advances offense)  → offense is favored → resolver="offense"
  - Penalty with yards < 0 (backs offense up)   → defense is favored → resolver="defense"
  - "I OR -10P": incomplete OR penalty against offense → defense favored → resolver="defense"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Union


# ---------------------------------------------------------------------------
# Atom types (elements of an AND-chain)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YardsOutcome:
    yards: int
    out_of_bounds: bool = False
    is_safety_flag: bool = False   # "-10S" suffix in chart; clock treated as inbounds


@dataclass(frozen=True)
class IncompleteOutcome:
    pass


@dataclass(frozen=True)
class FumbleOutcome:
    pass


@dataclass(frozen=True)
class LongGainOutcome:
    """Resolved probabilistically using LG_DIST from constants.py."""
    pass


@dataclass(frozen=True)
class InterceptionOutcome:
    return_yards: int = 0
    """Positive = defender runs toward original offense's own end zone (yards gained by defender)."""


@dataclass(frozen=True)
class PenaltyOutcome:
    yards: int
    """Positive = offense gains yardage; negative = offense loses yardage.
    Down repeats (no down increment) unless combined with FirstDownOutcome.
    """


@dataclass(frozen=True)
class LossOfDownOutcome:
    yards: int
    """Like PenaltyOutcome but the down is also burned (advances by 1)."""


@dataclass(frozen=True)
class FirstDownOutcome:
    """Automatic first down modifier.  Consumes 0 ticks (applied after primary atom)."""
    pass


# ---------------------------------------------------------------------------
# Composite types
# ---------------------------------------------------------------------------

SingleAtom = Union[
    YardsOutcome,
    IncompleteOutcome,
    FumbleOutcome,
    LongGainOutcome,
    InterceptionOutcome,
    PenaltyOutcome,
    LossOfDownOutcome,
    FirstDownOutcome,
]

Branch = List[SingleAtom]   # AND-chain; first atom consumes ticks, rest are modifiers


@dataclass(frozen=True)
class ChoiceOutcome:
    """An OR-choice between two or more branches.

    resolver = "offense" : offense picks the branch yielding the highest value
    resolver = "defense" : defense picks the branch yielding the lowest value
                          (i.e., minimises the offense's continuation value)
    """
    branches: Tuple[Branch, ...]
    resolver: str   # "offense" | "defense"


# Top-level cell outcome type
CellOutcome = Union[Branch, ChoiceOutcome]


# ---------------------------------------------------------------------------
# Punt result (built from PUNT_CHART constant, not parsed from CSV)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PuntResult:
    punt_yards:   int
    return_type:  str   # "ob" | "none" | "fixed" | "lg" | "fixed_fumble"
    return_yards: int   # relevant for "fixed" and "fixed_fumble"; 0 otherwise
