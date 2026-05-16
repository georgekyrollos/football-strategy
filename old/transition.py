# transition.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
import re
import pandas as pd

LONG_GAIN_YARDS = 50  # TODO: confirm exact "LG" meaning later

@dataclass(frozen=True)
class State:
    x: int        # yards from own goal line, 1..99
    down: int     # 1..4
    to_first: int # 1..10

@dataclass(frozen=True)
class Transition:
    next_state: Optional[State]  # None if terminal
    reward: int                  # points for offense on this drive
    terminal: Optional[str]      # "TD", "TURNOVER", "TOD", "SAFETY", or None

# ----- parsed outcomes -----
@dataclass(frozen=True)
class OutcomeYards: yards: int

@dataclass(frozen=True)
class OutcomeIncomplete: pass

@dataclass(frozen=True)
class OutcomeTurnover:
    kind: str     # "F" or "INT"
    detail: str   # original token

@dataclass(frozen=True)
class OutcomePenalty:
    yards: int
    auto_first: bool
    loss_of_down: bool

@dataclass(frozen=True)
class OutcomeSafety:
    yards: int

@dataclass(frozen=True)
class OutcomeChoice:
    options: Tuple["Outcome", ...]

Outcome = Union[
    OutcomeYards,
    OutcomeIncomplete,
    OutcomeTurnover,
    OutcomePenalty,
    OutcomeSafety,
    OutcomeChoice,
]

_int_re = re.compile(r"^INT.*$", re.IGNORECASE)
_safety_re = re.compile(r"^(?P<yd>-?\d+)\s*S$", re.IGNORECASE)
_lod_re = re.compile(r"^(?P<yd>-?\d+)\s*LOD$", re.IGNORECASE)
_pen_and1st_re = re.compile(r"^(?P<yd>-?\d+)\s*P\s*AND\s*1ST$", re.IGNORECASE)
_pen_re = re.compile(r"^(?P<yd>-?\d+)\s*P$", re.IGNORECASE)
_num_re = re.compile(r"^-?\d+$")

def parse_cell(token: str) -> Outcome:
    tok = str(token).strip()
    if not tok:
        raise ValueError("Empty chart token")

    if " OR " in tok:
        parts = [p.strip() for p in tok.split(" OR ")]
        return OutcomeChoice(tuple(parse_cell(p) for p in parts))

    # out-of-bounds marker doesn't affect drive-only transition (no clock model yet)
    if tok.upper().endswith("OB"):
        tok = tok[:-2].strip()

    if tok.upper() == "I":
        return OutcomeIncomplete()

    if tok.upper() == "F":
        return OutcomeTurnover(kind="F", detail=tok.upper())

    if _int_re.match(tok):
        return OutcomeTurnover(kind="INT", detail=tok.upper())

    if tok.upper() == "LG":
        return OutcomeYards(LONG_GAIN_YARDS)

    m = _safety_re.match(tok)
    if m:
        return OutcomeSafety(yards=int(m.group("yd")))

    m = _lod_re.match(tok)
    if m:
        return OutcomePenalty(yards=int(m.group("yd")), auto_first=False, loss_of_down=True)

    m = _pen_and1st_re.match(tok)
    if m:
        return OutcomePenalty(yards=int(m.group("yd")), auto_first=True, loss_of_down=False)

    m = _pen_re.match(tok)
    if m:
        return OutcomePenalty(yards=int(m.group("yd")), auto_first=False, loss_of_down=False)

    if _num_re.match(tok):
        return OutcomeYards(int(tok))

    raise NotImplementedError(f"Unrecognized token: {token!r}")

def _first_down_to_first(new_x: int) -> int:
    # after first down: 10 yards unless inside the 10
    return min(10, max(1, 100 - new_x))

def _td() -> Transition:
    return Transition(next_state=None, reward=7, terminal="TD")

def _safety() -> Transition:
    return Transition(next_state=None, reward=-2, terminal="SAFETY")

def _turnover() -> Transition:
    return Transition(next_state=None, reward=0, terminal="TURNOVER")

def _tod() -> Transition:
    return Transition(next_state=None, reward=0, terminal="TOD")

def _apply_gain(s: State, gain: int) -> Transition:
    new_x = s.x + gain

    # touchdown / safety bounds
    if new_x >= 100:
        return _td()
    if new_x <= 0:
        return _safety()

    # achieved first down?
    new_to_first = s.to_first - gain
    if new_to_first <= 0:
        return Transition(
            next_state=State(x=new_x, down=1, to_first=_first_down_to_first(new_x)),
            reward=0,
            terminal=None,
        )

    # otherwise, advance down
    new_down = s.down + 1
    if new_down >= 5:
        return _tod()

    return Transition(
        next_state=State(x=new_x, down=new_down, to_first=new_to_first),
        reward=0,
        terminal=None,
    )

def _apply_incomplete(s: State) -> Transition:
    new_down = s.down + 1
    if new_down >= 5:
        return _tod()
    return Transition(next_state=State(x=s.x, down=new_down, to_first=s.to_first), reward=0, terminal=None)

def _apply_penalty(s: State, p: OutcomePenalty) -> Transition:
    # V1 simplification:
    # - Move ball by p.yards
    # - If auto_first: 1st & 10 (or goal-to-go)
    # - If loss_of_down: advance down (and still move ball)
    # - Else: repeat down (distance updates with the moved ball)

    new_x = s.x + p.yards
    if new_x >= 100:
        return _td()
    if new_x <= 0:
        return _safety()

    if p.auto_first:
        return Transition(
            next_state=State(x=new_x, down=1, to_first=_first_down_to_first(new_x)),
            reward=0,
            terminal=None,
        )

    new_to_first = s.to_first - p.yards
    if new_to_first <= 0:
        return Transition(
            next_state=State(x=new_x, down=1, to_first=_first_down_to_first(new_x)),
            reward=0,
            terminal=None,
        )

    new_down = s.down + (1 if p.loss_of_down else 0)
    if new_down >= 5:
        return _tod()

    return Transition(next_state=State(x=new_x, down=new_down, to_first=new_to_first), reward=0, terminal=None)

def transitions_from_outcome(s: State, out: Outcome) -> List[Transition]:
    if isinstance(out, OutcomeChoice):
        res: List[Transition] = []
        for opt in out.options:
            res.extend(transitions_from_outcome(s, opt))
        return res

    if isinstance(out, OutcomeYards):
        return [_apply_gain(s, out.yards)]

    if isinstance(out, OutcomeIncomplete):
        return [_apply_incomplete(s)]

    if isinstance(out, OutcomeTurnover):
        return [_turnover()]

    if isinstance(out, OutcomePenalty):
        return [_apply_penalty(s, out)]

    if isinstance(out, OutcomeSafety):
        # V1: treat as safety terminal, regardless of yard value shown
        return [_safety()]

    raise TypeError(f"Unhandled outcome type: {type(out)}")

def successors(chart: pd.DataFrame, s: State, off_play: int, def_play: str) -> List[Transition]:
    token = chart.loc[def_play, str(off_play)]
    out = parse_cell(token)
    return transitions_from_outcome(s, out)
