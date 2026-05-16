"""
Parse the Pro Style offense-defense chart CSV and the punt chart into typed
CellOutcome / PuntResult objects consumed by transitions.py.

parse_atom / parse_expr adapted from old/transition_quarter_score.py.
Key differences from the old code:
  - "LG"        → LongGainOutcome()         (not a sentinel yards=10**9)
  - "<y>OB"     → YardsOutcome(out_of_bounds=True)   (OB flag preserved for clock)
  - "<y>S"      → YardsOutcome(is_safety_flag=True)
  - OR-branches → ChoiceOutcome with resolver="offense"|"defense"
  - AND-chains  → Branch (List[SingleAtom])
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd

from football_strategy.constants import DEFENSE_CARDS, PUNT_CHART
from football_strategy.outcomes import (
    Branch,
    CellOutcome,
    ChoiceOutcome,
    FirstDownOutcome,
    FumbleOutcome,
    IncompleteOutcome,
    InterceptionOutcome,
    LongGainOutcome,
    LossOfDownOutcome,
    PenaltyOutcome,
    PuntResult,
    SingleAtom,
    YardsOutcome,
)

# ---------------------------------------------------------------------------
# Compiled regexes (covers every token present in the Pro Style CSV)
# ---------------------------------------------------------------------------

_OR_SPLIT  = re.compile(r"\s+OR\s+",  flags=re.IGNORECASE)
_AND_SPLIT = re.compile(r"\s+AND\s+", flags=re.IGNORECASE)

_LG_RE    = re.compile(r"^LG$",                      flags=re.IGNORECASE)
_FIRST_RE = re.compile(r"^(1ST|FIRST)$",             flags=re.IGNORECASE)
_INT_RE   = re.compile(r"^INT(?P<ret>-?\d+)?$",       flags=re.IGNORECASE)
_LOD_RE   = re.compile(r"^(?P<y>-?\d+)LOD$",         flags=re.IGNORECASE)
_PEN_RE   = re.compile(r"^(?P<y>-?\d+)P$",           flags=re.IGNORECASE)
_OB_RE    = re.compile(r"^(?P<y>-?\d+)OB$",          flags=re.IGNORECASE)
_S_RE     = re.compile(r"^(?P<y>-?\d+)S$",           flags=re.IGNORECASE)
_YARDS_RE = re.compile(r"^-?\d+$")


# ---------------------------------------------------------------------------
# Atom parser
# ---------------------------------------------------------------------------

def parse_atom(tok: str) -> SingleAtom:
    """Convert a single chart token string into a typed SingleAtom.

    Raises ValueError for unrecognised tokens.
    """
    tok = (tok or "").strip()
    if not tok:
        return YardsOutcome(yards=0)

    up = tok.upper()

    if up == "I":
        return IncompleteOutcome()

    if up == "F":
        return FumbleOutcome()

    if _LG_RE.match(tok):
        return LongGainOutcome()

    if _FIRST_RE.match(tok):
        return FirstDownOutcome()

    m = _INT_RE.match(tok)
    if m:
        ret_str = m.group("ret")
        ret = int(ret_str) if ret_str is not None else 0
        return InterceptionOutcome(return_yards=ret)

    m = _LOD_RE.match(tok)
    if m:
        return LossOfDownOutcome(yards=int(m.group("y")))

    m = _PEN_RE.match(tok)
    if m:
        return PenaltyOutcome(yards=int(m.group("y")))

    m = _OB_RE.match(tok)
    if m:
        return YardsOutcome(yards=int(m.group("y")), out_of_bounds=True)

    m = _S_RE.match(tok)
    if m:
        return YardsOutcome(yards=int(m.group("y")), is_safety_flag=True)

    if _YARDS_RE.match(tok):
        return YardsOutcome(yards=int(tok))

    raise ValueError(f"Unrecognised chart token: {tok!r}")


# ---------------------------------------------------------------------------
# OR-branch resolver determination
# ---------------------------------------------------------------------------

def _resolver_for_or_branches(branches: List[Branch]) -> str:
    """Determine which team picks among the OR-branches.

    Rule (from Football Strategy rulebook):
      "The team favored by the penalty chooses."
    A branch that contains a PenaltyOutcome or LossOfDownOutcome with positive
    yards (offense gains) favors the OFFENSE.
    A branch that contains a PenaltyOutcome or LossOfDownOutcome with negative
    yards (offense loses) favors the DEFENSE.
    An IncompleteOutcome branch counts as neutral (offense prefers higher yards,
    so the *other* branch with positive penalty favors offense, or the negative
    penalty branch makes defense the resolver).

    Practical heuristic used here:
      - Scan all atoms in all branches for PenaltyOutcome / LossOfDownOutcome.
      - If any such atom has yards < 0 → defense resolves (penalty hurts offense).
      - If any such atom has yards > 0 → offense resolves (penalty helps offense).
      - If no penalty found in any branch (shouldn't happen in OR cells), default to offense.
    """
    for branch in branches:
        for atom in branch:
            if isinstance(atom, (PenaltyOutcome, LossOfDownOutcome)):
                if atom.yards < 0:
                    return "defense"
    # Default: offense resolves (penalty favors offense, or yards gain)
    return "offense"


# ---------------------------------------------------------------------------
# Expression parser
# ---------------------------------------------------------------------------

def parse_expr(token: str) -> CellOutcome:
    """Parse a full chart cell string into a CellOutcome.

    Single branch  → Branch (List[SingleAtom])
    Multiple branches (OR-separated) → ChoiceOutcome
    Each branch is an AND-chain of atoms.
    """
    token = (token or "").strip()
    if not token:
        return [YardsOutcome(yards=0)]

    raw_branches: List[Branch] = []
    for part in _OR_SPLIT.split(token):
        part = part.strip()
        if not part:
            continue
        chain: Branch = [
            parse_atom(p.strip())
            for p in _AND_SPLIT.split(part)
            if p.strip()
        ]
        if not chain:
            chain = [YardsOutcome(yards=0)]
        raw_branches.append(chain)

    if not raw_branches:
        return [YardsOutcome(yards=0)]

    if len(raw_branches) == 1:
        return raw_branches[0]

    # Multiple branches → OR-choice
    resolver = _resolver_for_or_branches(raw_branches)
    return ChoiceOutcome(
        branches=tuple(raw_branches),
        resolver=resolver,
    )


# ---------------------------------------------------------------------------
# Chart loaders
# ---------------------------------------------------------------------------

BALL_CONTROL_CSV = "Football Strategy Ball Control.csv"
PRO_STYLE_CSV    = "Football Strategy Pro Style.csv"

_DEFAULT_CSV = BALL_CONTROL_CSV   # active chart used by all solvers and scripts


def load_pro_style_chart_csv(csv_path: str = _DEFAULT_CSV) -> pd.DataFrame:
    """Load an offense-defense chart CSV into a DataFrame.

    Works for any properly-formatted chart (Ball Control, Pro Style, etc.).
    Returns a (10 × 20) DataFrame indexed by defense card ("A"–"J") with
    string columns "1"–"20".
    """
    df = pd.read_csv(csv_path, header=0)
    df = df.rename(columns={df.columns[0]: "DEF"}).set_index("DEF")
    df.columns = [str(c).strip() for c in df.columns]

    expected_cols = [str(i) for i in range(1, 21)]
    df = df[expected_cols]

    def clean(v):
        if pd.isna(v):
            return ""
        return str(v).strip()

    chart = df.apply(lambda col: col.map(clean))

    expected_rows = list("ABCDEFGHIJ")
    missing = [r for r in expected_rows if r not in chart.index]
    if missing:
        raise ValueError(f"Missing defense rows in CSV: {missing}")

    return chart


def load_scrimmage_chart(
    csv_path: str = _DEFAULT_CSV,
) -> Dict[Tuple[str, str], CellOutcome]:
    """Parse all 200 scrimmage chart cells into typed CellOutcome objects.

    Returns a dict keyed by (defense_card, play_number_str), e.g. ("F", "4").
    """
    df = load_pro_style_chart_csv(csv_path)
    chart: Dict[Tuple[str, str], CellOutcome] = {}

    for def_card in df.index:
        for col in df.columns:
            token = df.loc[def_card, col]
            try:
                chart[(def_card, col)] = parse_expr(token)
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse cell ({def_card!r}, play {col}): {token!r}"
                ) from exc

    return chart


def load_punt_chart() -> Dict[str, PuntResult]:
    """Build a punt-chart dict from the PUNT_CHART constant.

    Returns a dict keyed by defense card ("A"–"J"), each value a PuntResult.
    """
    result: Dict[str, PuntResult] = {}
    for card in DEFENSE_CARDS:
        punt_yards, return_type, return_yards = PUNT_CHART[card]
        result[card] = PuntResult(
            punt_yards=punt_yards,
            return_type=return_type,
            return_yards=return_yards if return_yards is not None else 0,
        )
    return result
