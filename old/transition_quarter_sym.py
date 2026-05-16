# Symmetric transition model (no explicit possession in state)
# Current offense always drives toward 100; delta = offense - defense.
# When possession changes we mirror the field and flip delta sign, and
# expose a `swapped` flag so callers can flip continuation values.
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from transition_quarter_score import (
    State as FullState,
    Transition as FullTransition,
    OFFENSE_ACTIONS,
    DEFENSE_PLAYS,
    PUNT,
    US,
    successors as full_successors,
)


@dataclass(frozen=True)
class State:
    x: int         # yards from current offense goal line, 0..100
    down: int      # 1..4
    to_first: int  # yards to first down (relative to offense direction)
    t: int         # plays remaining in quarter
    delta: int     # score diff: current offense - defense


@dataclass(frozen=True)
class Transition:
    next_state: Optional[State]
    reward: float
    swapped: bool  # True if possession changed


def _to_full_state(s: State) -> FullState:
    # Encode symmetric state as "US" possessing in the full model.
    return FullState(
        poss=US,
        x=s.x,
        down=s.down,
        to_first=s.to_first,
        t=s.t,
        delta=s.delta,
    )


def _from_full_state(fs: FullState) -> tuple[State, bool]:
    """
    Convert a full-state (with poss) to symmetric offense-facing state.
    If poss != US, mirror the field and flip delta sign; mark swapped.
    """
    if fs.poss == US:
        return (
            State(x=fs.x, down=fs.down, to_first=fs.to_first, t=fs.t, delta=fs.delta),
            False,
        )

    # Defense took over: mirror field and flip delta so it remains
    # offense-minus-defense from the new offense's perspective.
    mirrored_x = 100 - fs.x
    mirrored_delta = -fs.delta
    return (
        State(
            x=mirrored_x,
            down=fs.down,
            to_first=fs.to_first,
            t=fs.t,
            delta=mirrored_delta,
        ),
        True,
    )


def successors(chart, s: State, offense_action: int, def_play: str) -> List[Transition]:
    """
    Mirror-aware wrapper around the quarter-score transition logic.
    """
    full_state = _to_full_state(s)
    outs: List[Transition] = []

    for tr in full_successors(chart, full_state, offense_action, def_play):
        if tr.next_state is None:
            outs.append(Transition(None, tr.reward, False))
            continue

        sym_next, swapped = _from_full_state(tr.next_state)
        outs.append(Transition(sym_next, tr.reward, swapped))

    return outs
