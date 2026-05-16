# transition_quarter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re

US, THEM = 0, 1

# Offense actions: 1..20 plus a punt action
PUNT = 21
OFFENSE_ACTIONS = list(range(1, 21)) + [PUNT]
DEFENSE_PLAYS = list("ABCDEFGHIJ")

# -------------------------
# Score-diff compression
# -------------------------
DELTA_MIN, DELTA_MAX = -14, 14

def clip_delta(d: int) -> int:
    return DELTA_MIN if d < DELTA_MIN else (DELTA_MAX if d > DELTA_MAX else d)

# -------------------------
# Token regexes (covers ALL tokens in Pro Style)
# -------------------------
_OR_SPLIT = re.compile(r"\s+OR\s+", flags=re.IGNORECASE)
_AND_SPLIT = re.compile(r"\s+AND\s+", flags=re.IGNORECASE)

_LG_RE = re.compile(r"^LG$", flags=re.IGNORECASE)
_FIRST_RE = re.compile(r"^(1ST|FIRST)$", flags=re.IGNORECASE)

_INT_RE = re.compile(r"^INT(?P<ret>-?\d+)?$", flags=re.IGNORECASE)

_PEN_RE = re.compile(r"^(?P<y>-?\d+)P$", flags=re.IGNORECASE)
_LOD_RE = re.compile(r"^(?P<y>-?\d+)LOD$", flags=re.IGNORECASE)

_YARDS_RE = re.compile(r"^(?P<y>-?\d+)$")
_YARDS_OB_RE = re.compile(r"^(?P<y>-?\d+)OB$", flags=re.IGNORECASE)
_YARDS_S_RE = re.compile(r"^(?P<y>-?\d+)S$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class State:
    poss: int      # 0=US offense, 1=THEM offense
    x: int         # physical yardline from OUR goal line, 0..100
    down: int      # 1..4
    to_first: int  # yards to first
    t: int         # plays remaining in quarter
    delta: int     # score differential (US - THEM), clipped to [DELTA_MIN, DELTA_MAX]


@dataclass(frozen=True)
class Transition:
    next_state: Optional[State]
    reward: float
    terminal: Optional[str]  # "TD", "SAFETY", "END", None


@dataclass(frozen=True)
class Leaf:
    kind: str   # "yards", "incomplete", "penalty", "lod", "fumble", "int", "first"
    yards: int = 0
    ret: int = 0


# -------------------------
# Parsing: expression -> branches (OR) of chains (AND)
# -------------------------
def parse_atom(tok: str) -> Leaf:
    tok = (tok or "").strip()
    if not tok:
        return Leaf("yards", yards=0)

    up = tok.upper()
    if up == "I":
        return Leaf("incomplete")
    if up == "F":
        return Leaf("fumble")
    if _LG_RE.match(tok):
        # sentinel for LG (interpreted using state in apply_leaf)
        return Leaf("yards", yards=10**9)
    if _FIRST_RE.match(tok):
        return Leaf("first")

    m = _INT_RE.match(tok)
    if m:
        ret = m.group("ret")
        return Leaf("int", ret=int(ret) if ret is not None else 0)

    m = _LOD_RE.match(tok)
    if m:
        return Leaf("lod", yards=int(m.group("y")))

    m = _PEN_RE.match(tok)
    if m:
        return Leaf("penalty", yards=int(m.group("y")))

    m = _YARDS_OB_RE.match(tok)
    if m:
        return Leaf("yards", yards=int(m.group("y")))

    m = _YARDS_S_RE.match(tok)
    if m:
        return Leaf("yards", yards=int(m.group("y")))

    m = _YARDS_RE.match(tok)
    if m:
        return Leaf("yards", yards=int(m.group("y")))

    raise ValueError(f"Unrecognized chart atom: {tok!r}")


def parse_expr(token: str) -> List[List[Leaf]]:
    """
    Returns: list of chains, one per OR-branch.
      - OR splits into branches
      - AND splits into a chain inside each branch
    """
    token = (token or "").strip()
    if not token:
        return [[Leaf("yards", yards=0)]]

    branches: List[List[Leaf]] = []
    for part in _OR_SPLIT.split(token):
        part = part.strip()
        if not part:
            continue
        chain = [parse_atom(p.strip()) for p in _AND_SPLIT.split(part) if p.strip()]
        if not chain:
            chain = [Leaf("yards", yards=0)]
        branches.append(chain)

    if not branches:
        branches = [[Leaf("yards", yards=0)]]
    return branches


# -------------------------
# Football mechanics helpers
# -------------------------
def clamp_x(x: int) -> int:
    return 0 if x < 0 else (100 if x > 100 else x)


def offense_dir(poss: int) -> int:
    # US drives toward 100, THEM drives toward 0
    return +1 if poss == US else -1


def dist_to_goal(poss: int, x_phys: int) -> int:
    return (100 - x_phys) if poss == US else x_phys


def first_down_to_first(poss: int, x_phys: int) -> int:
    d = dist_to_goal(poss, x_phys)
    return max(1, min(10, d))


def kickoff_state(receiving_poss: int, t: int, delta: int) -> State:
    x = 20 if receiving_poss == US else 80
    return State(poss=receiving_poss, x=x, down=1, to_first=10, t=t, delta=delta)


def new_possession_state(new_poss: int, x_phys: int, t: int, delta: int) -> State:
    x_phys = clamp_x(x_phys)
    return State(
        poss=new_poss,
        x=x_phys,
        down=1,
        to_first=first_down_to_first(new_poss, x_phys),
        t=t,
        delta=delta,
    )


# -------------------------
# Punt model (Pass 1)
# -------------------------
NET_PUNT_YARDS = 41

def punt_transition(s: State) -> Transition:
    if s.t <= 0:
        return Transition(None, 0.0, "END")
    t2 = s.t - 1
    sign = offense_dir(s.poss)

    y2 = s.x + sign * NET_PUNT_YARDS

    # Touchback
    if s.poss == US and y2 >= 100:
        return Transition(kickoff_state(receiving_poss=THEM, t=t2, delta=s.delta), 0.0, None)
    if s.poss == THEM and y2 <= 0:
        return Transition(kickoff_state(receiving_poss=US, t=t2, delta=s.delta), 0.0, None)

    y2 = clamp_x(y2)
    return Transition(new_possession_state(1 - s.poss, y2, t2, s.delta), 0.0, None)


# -------------------------
# Apply leaves (single + chain)
# -------------------------
LG_EXPECTED = 50  # updated from your note

def apply_leaf(s: State, leaf: Leaf, *, consume_time: bool) -> Transition:
    if consume_time:
        if s.t <= 0:
            return Transition(None, 0.0, "END")
        t2 = s.t - 1
    else:
        t2 = s.t

    sign = offense_dir(s.poss)

    def score_for(off_poss: int, points: int, tag: str) -> Transition:
        # delta is US - THEM
        d2 = s.delta + (points if off_poss == US else -points)
        d2 = clip_delta(d2)
        return Transition(kickoff_state(receiving_poss=1 - off_poss, t=t2, delta=d2), 0.0, tag)

    def score_check(off_poss: int, x2: int) -> Optional[Transition]:
        # TD
        if off_poss == US and x2 >= 100:
            return score_for(off_poss, 7, "TD")
        if off_poss == THEM and x2 <= 0:
            return score_for(off_poss, 7, "TD")

            # Safety (DEFENSE scores)
        if off_poss == US and x2 <= 0:
            return score_for(1 - off_poss, 2, "SAFETY")
        if off_poss == THEM and x2 >= 100:
            return score_for(1 - off_poss, 2, "SAFETY")


    if leaf.kind == "first":
        return Transition(
            State(s.poss, s.x, 1, first_down_to_first(s.poss, s.x), t2, s.delta),
            0.0,
            None
        )

    if leaf.kind == "incomplete":
        x2 = s.x
        down2 = s.down + 1
        if down2 >= 5:
            return Transition(new_possession_state(1 - s.poss, x2, t2, s.delta), 0.0, None)
        return Transition(State(s.poss, x2, down2, s.to_first, t2, s.delta), 0.0, None)

    if leaf.kind == "penalty":
        y = leaf.yards
        x2 = s.x + sign * y
        sc = score_check(s.poss, x2)
        if sc is not None:
            return sc
        x2 = clamp_x(x2)

        # penalty repeats down in this model
        to2 = s.to_first - y
        if to2 <= 0:
            return Transition(State(s.poss, x2, 1, first_down_to_first(s.poss, x2), t2, s.delta), 0.0, None)
        return Transition(State(s.poss, x2, s.down, to2, t2, s.delta), 0.0, None)

    if leaf.kind == "lod":
        # Loss Of Down: apply yardage, AND you burn a down.
        y = leaf.yards
        x2 = s.x + sign * y
        sc = score_check(s.poss, x2)
        if sc is not None:
            return sc
        x2 = clamp_x(x2)

        to2 = s.to_first - y
        if to2 <= 0:
            return Transition(State(s.poss, x2, 1, first_down_to_first(s.poss, x2), t2, s.delta), 0.0, None)

        down2 = s.down + 1
        if down2 >= 5:
            return Transition(new_possession_state(1 - s.poss, x2, t2, s.delta), 0.0, None)
        return Transition(State(s.poss, x2, down2, to2, t2, s.delta), 0.0, None)

    if leaf.kind == "yards":
        y = leaf.yards

        # LG sentinel: expected big gain, capped by goal line
        if y >= 10**8:
            y = min(LG_EXPECTED, dist_to_goal(s.poss, s.x))

        x2 = s.x + sign * y
        sc = score_check(s.poss, x2)
        if sc is not None:
            return sc
        x2 = clamp_x(x2)

        to2 = s.to_first - y
        if to2 <= 0:
            return Transition(State(s.poss, x2, 1, first_down_to_first(s.poss, x2), t2, s.delta), 0.0, None)

        down2 = s.down + 1
        if down2 >= 5:
            return Transition(new_possession_state(1 - s.poss, x2, t2, s.delta), 0.0, None)
        return Transition(State(s.poss, x2, down2, to2, t2, s.delta), 0.0, None)

    if leaf.kind == "fumble":
        return Transition(new_possession_state(1 - s.poss, s.x, t2, s.delta), 0.0, None)

    if leaf.kind == "int":
        r = leaf.ret
        new_poss = 1 - s.poss
        x2 = s.x - sign * r  # defense returns opposite direction

        # pick-6 check
        if new_poss == US and x2 >= 100:
            return Transition(kickoff_state(receiving_poss=THEM, t=t2, delta=clip_delta(s.delta + 7)), 0.0, "TD")
        if new_poss == THEM and x2 <= 0:
            return Transition(kickoff_state(receiving_poss=US, t=t2, delta=clip_delta(s.delta - 7)), 0.0, "TD")

        x2 = clamp_x(x2)
        return Transition(new_possession_state(new_poss, x2, t2, s.delta), 0.0, None)

    raise RuntimeError(f"Unknown leaf kind: {leaf.kind}")


def apply_chain(s: State, chain: List[Leaf]) -> Transition:
    """
    Apply an AND-chain within a single play:
      - first leaf consumes time
      - remaining leaves are modifiers (no extra time)
    """
    if not chain:
        return Transition(None, 0.0, "END")

    tr = apply_leaf(s, chain[0], consume_time=True)
    if tr.next_state is None or tr.terminal is not None:
        return tr

    s2 = tr.next_state
    for lf in chain[1:]:
        tr2 = apply_leaf(s2, lf, consume_time=False)
        if tr2.next_state is None or tr2.terminal is not None:
            return Transition(tr2.next_state, 0.0, tr2.terminal)
        s2 = tr2.next_state

    return Transition(s2, 0.0, None)


def successors(chart, s: State, offense_action: int, def_play: str) -> List[Transition]:
    """
    Returns 1+ transitions.
    If chart cell has OR, we return multiple transitions (solver can treat as offense choice).
    """
    if offense_action == PUNT:
        return [punt_transition(s)]

    token = str(chart.loc[def_play, str(offense_action)]).strip()
    branches = parse_expr(token)
    return [apply_chain(s, chain) for chain in branches]
