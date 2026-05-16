"""
Forward BFS reachability over (q, tau) layers.

State key encoding:
  ScrimmageKey  = ("s",  q, tau, x, down, to_first, delta, h)
  KickoffKey    = ("k",  q, tau, delta, h)
  SafetyKickKey = ("sk", q, tau, delta, h)

Key optimisation — core-level transition cache:
  The successor CORES (x, down, to_first, delta, h) for a given
  (core, action, def_card) are independent of (q, tau).  We compute them once
  and reuse across all (q, tau) layers.  This avoids re-running apply_branch
  for each time a core is encountered at a new clock position.

Progress reporting:
  BFS is FIFO (deque) so states are processed approximately in increasing-tau
  order.  A summary line is printed whenever a new (q, tau) pair is seen for
  the first time and the previous pair is fully drained, giving a per-layer
  progress view.  Set verbose=False to suppress all output.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from football_strategy.clock import advance_clock
from football_strategy.constants import (
    DEFENSE_CARDS,
    FG_ACTION,
    PUNT_ACTION,
    TICKS_PER_QUARTER,
)
from football_strategy.states import legal_offense_actions
from football_strategy.transitions import (
    ChoiceTemplates,
    SuccessorTemplate,
    TransitionResult,
    field_goal_templates,
    normal_kickoff_templates,
    onside_kickoff_templates,
    punt_templates,
    safety_kick_option_normal,
    safety_kick_option_punt,
    scrimmage_play_templates,
)


# ---------------------------------------------------------------------------
# Key construction helpers
# ---------------------------------------------------------------------------

def scrimmage_key(q: int, tau: int, x: int, down: int, to_first: int, delta: int, h: int) -> tuple:
    return ("s", q, tau, x, down, to_first, delta, h)


def kickoff_key(q: int, tau: int, delta: int, h: int) -> tuple:
    return ("k", q, tau, delta, h)


def safety_kick_key(q: int, tau: int, delta: int, h: int) -> tuple:
    return ("sk", q, tau, delta, h)


def key_phase(key: tuple) -> str:
    return key[0]


def key_qt(key: tuple) -> Tuple[int, int]:
    return (key[1], key[2])


# ---------------------------------------------------------------------------
# Core-level transition cache
# ---------------------------------------------------------------------------
# Maps (phase, core, action_or_None, def_card_or_None)
#   → list of (successor_phase, successor_core, ticks, swapped)
# This is phase-coded so kickoff/safety_kick can share the same cache dict.

_CORE_CACHE: Dict[tuple, List[Tuple[str, tuple, int, bool]]] = {}


def _cached_scrimmage_core_succs(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    x: int, down: int, to_first: int, delta: int, h: int,
) -> List[Tuple[str, tuple, int, bool]]:
    """Return all (phase, core, ticks, swapped) successor entries for a scrimmage core.

    Results are cached by (x, down, to_first, delta, h) since transitions
    are independent of (q, tau).
    """
    cache_key = ("s", x, down, to_first, delta, h)
    if cache_key in _CORE_CACHE:
        return _CORE_CACHE[cache_key]

    actions = legal_offense_actions(x, down)
    succs: List[Tuple[str, tuple, int, bool]] = []

    for action in actions:
        if action == FG_ACTION:
            result = field_goal_templates(x, down, to_first, delta, h)
            _extract_succs(result, succs)
        elif action == PUNT_ACTION:
            for def_card in DEFENSE_CARDS:
                result = punt_templates(x, down, to_first, delta, h, def_card)
                _extract_succs(result, succs)
        else:
            for def_card in DEFENSE_CARDS:
                result = scrimmage_play_templates(
                    scrimmage_chart, x, down, to_first, delta, h, action, def_card,
                )
                _extract_succs(result, succs)

    # Deduplicate (same (phase, core, ticks, swapped) combo can arise from many action/card pairs)
    seen: Set[tuple] = set()
    deduped = []
    for entry in succs:
        if entry not in seen:
            seen.add(entry)
            deduped.append(entry)

    _CORE_CACHE[cache_key] = deduped
    return deduped


def _cached_kickoff_core_succs(delta: int, h: int) -> List[Tuple[str, tuple, int, bool]]:
    cache_key = ("k", delta, h)
    if cache_key in _CORE_CACHE:
        return _CORE_CACHE[cache_key]

    succs: List[Tuple[str, tuple, int, bool]] = []
    for tmpls in (normal_kickoff_templates(delta, h), onside_kickoff_templates(delta, h)):
        _extract_succs(tmpls, succs)

    seen: Set[tuple] = set()
    deduped = []
    for entry in succs:
        if entry not in seen:
            seen.add(entry)
            deduped.append(entry)

    _CORE_CACHE[cache_key] = deduped
    return deduped


def _cached_safety_kick_core_succs(delta: int, h: int) -> List[Tuple[str, tuple, int, bool]]:
    cache_key = ("sk", delta, h)
    if cache_key in _CORE_CACHE:
        return _CORE_CACHE[cache_key]

    succs: List[Tuple[str, tuple, int, bool]] = []
    for tmpls in (safety_kick_option_normal(delta, h), safety_kick_option_punt(delta, h)):
        _extract_succs(tmpls, succs)

    seen: Set[tuple] = set()
    deduped = []
    for entry in succs:
        if entry not in seen:
            seen.add(entry)
            deduped.append(entry)

    _CORE_CACHE[cache_key] = deduped
    return deduped


def _extract_succs(
    result: TransitionResult,
    out: List[Tuple[str, tuple, int, bool]],
) -> None:
    """Flatten a TransitionResult into (phase, core, ticks, swapped) entries."""
    if isinstance(result, list):
        for t in result:
            if t.phase != "terminal":
                out.append((t.phase, t.core, t.ticks, t.swapped))
    else:  # ChoiceTemplates
        for branch in result.branches:
            for t in branch:
                if t.phase != "terminal":
                    out.append((t.phase, t.core, t.ticks, t.swapped))


# ---------------------------------------------------------------------------
# Template → successor state key (after clock advancement)
# ---------------------------------------------------------------------------

def _succ_phase_core_to_key(
    phase: str, core: tuple, q: int, tau: int, ticks: int,
) -> Optional[tuple]:
    """Compute the full state key for a successor given its core and clock cost."""
    new_tau, _warn = advance_clock(tau, q, ticks)

    if new_tau == 0:
        if q < 4:
            q_next   = q + 1
            tau_next = TICKS_PER_QUARTER
        else:
            return None   # Q4 end → terminal, not a reachable state key
    else:
        q_next   = q
        tau_next = new_tau

    if phase == "scrimmage":
        x, down, tf, delta, h = core
        return scrimmage_key(q_next, tau_next, x, down, tf, delta, h)
    if phase == "kickoff":
        delta, h = core
        return kickoff_key(q_next, tau_next, delta, h)
    if phase == "safety_kick":
        delta, h = core
        return safety_kick_key(q_next, tau_next, delta, h)
    return None


# ---------------------------------------------------------------------------
# BFS reachability
# ---------------------------------------------------------------------------

def build_reachable_states(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    start_key: tuple,
    *,
    verbose: bool = True,
) -> Dict[Tuple[int, int], Set[tuple]]:
    """Forward BFS from start_key.

    Returns ``layers``: dict mapping (q, tau) → set of full state keys reachable
    at that (q, tau) pair.  Terminal (q=4, tau=0) states are excluded.

    Progress is printed per (q, tau) layer as each is fully explored, showing:
      BFS (q=4, tau=58): 1234 new states [total 45678, cache 890 cores, 12.3s]
    """
    layers: Dict[Tuple[int, int], Set[tuple]] = {}
    visited: Set[tuple] = set()
    frontier: deque = deque([start_key])
    visited.add(start_key)

    qt0 = key_qt(start_key)
    layers.setdefault(qt0, set()).add(start_key)

    total   = 1
    t_start = time.time()

    # Track per-(q,tau) stats for progress reporting
    qt_counts: Dict[Tuple[int, int], int] = {qt0: 1}

    prev_qt  = qt0
    last_log = time.time()

    while frontier:
        key = frontier.popleft()
        q, tau = key_qt(key)

        # Progress: log whenever we finish a layer
        if verbose:
            now = time.time()
            if now - last_log >= 5.0:
                print(
                    f"  BFS (q={q}, tau={tau:>2}): {qt_counts.get((q,tau),0)} states in layer"
                    f"  [total {total}, cache {len(_CORE_CACHE)} cores, {now-t_start:.0f}s]"
                )
                last_log = now

        for succ_key in _successors_of_key(key, scrimmage_chart, punt_chart, q, tau):
            if succ_key not in visited:
                visited.add(succ_key)
                sq, stau = key_qt(succ_key)
                layers.setdefault((sq, stau), set()).add(succ_key)
                qt_counts[(sq, stau)] = qt_counts.get((sq, stau), 0) + 1
                frontier.append(succ_key)
                total += 1

    if verbose:
        elapsed = time.time() - t_start
        print(
            f"  BFS done: {total} reachable states, "
            f"{len(_CORE_CACHE)} cached cores, {elapsed:.1f}s"
        )

    return layers


def _successors_of_key(
    key: tuple,
    scrimmage_chart: Dict,
    punt_chart: Dict,
    q: int,
    tau: int,
) -> List[tuple]:
    """Return all successor state keys from a given state key using the core cache."""
    phase = key[0]

    if phase == "s":
        _, _q, _tau, x, down, to_first, delta, h = key
        core_succs = _cached_scrimmage_core_succs(scrimmage_chart, punt_chart, x, down, to_first, delta, h)
    elif phase == "k":
        _, _q, _tau, delta, h = key
        core_succs = _cached_kickoff_core_succs(delta, h)
    elif phase == "sk":
        _, _q, _tau, delta, h = key
        core_succs = _cached_safety_kick_core_succs(delta, h)
    else:
        return []

    result = []
    for (succ_phase, succ_core, ticks, _swapped) in core_succs:
        succ_key = _succ_phase_core_to_key(succ_phase, succ_core, q, tau, ticks)
        if succ_key is not None:
            result.append(succ_key)
    return result
