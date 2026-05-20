"""
Zero-sum matrix game LP solver and payoff matrix builder.

LP solver functions (_is_feasible_row_lp, solve_row_strategy, solve_col_strategy)
are copied verbatim from old/solve_quarter_sym.py and adapted for scipy linprog.

The payoff matrix builder constructs the m × 10 matrix for a scrimmage state:
  rows = legal offense actions (plays 1-20 + optional punt/FG)
  cols = 10 defense cards A-J

Clock arithmetic (advance_clock) happens here, in _lookup_val, NOT in transitions.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from football_strategy.clock import advance_clock
from football_strategy.constants import (
    DEFENSE_CARDS,
    TICKS_PER_QUARTER,
)
from football_strategy.states import legal_offense_actions, terminal_utility
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
from football_strategy.constants import FG_ACTION, PUNT_ACTION


# ---------------------------------------------------------------------------
# LP solvers (copied from old/solve_quarter_sym.py; no changes except imports)
# ---------------------------------------------------------------------------

def _fictitious_play(A: np.ndarray, n_iter: int = 30_000) -> Tuple[float, np.ndarray, np.ndarray]:
    """HiGHS-independent fallback: fictitious play for zero-sum matrix games.

    Converges to the Nash equilibrium; used when HiGHS returns no solution.
    Returns (v, p, q) where p is the row player's mixed strategy, q the column
    player's mixed strategy, and v the game value from the row player's perspective.
    """
    import warnings
    m, n = A.shape
    counts_r = np.ones(m, dtype=float)
    counts_c = np.ones(n, dtype=float)
    for _ in range(n_iter):
        defense_mix = counts_c / counts_c.sum()
        counts_r[int(np.argmax(A @ defense_mix))] += 1.0
        offense_mix = counts_r / counts_r.sum()
        counts_c[int(np.argmin(A.T @ offense_mix))] += 1.0
    p = counts_r / counts_r.sum()
    q = counts_c / counts_c.sum()
    v = float(np.min(A.T @ p))
    warnings.warn(f"HiGHS LP failed; fictitious-play fallback used (v≈{v:.4f})")
    return v, p, q


def _is_feasible_row_lp(A: np.ndarray, p: np.ndarray, v: float, tol: float = 1e-8) -> bool:
    if np.any(p < -tol):
        return False
    if abs(float(np.sum(p)) - 1.0) > 1e-6:
        return False
    viol = (-A.T @ p) + v  # <= 0
    return float(np.max(viol)) <= tol


def solve_row_strategy(A: np.ndarray, *, tol: float = 1e-8) -> Tuple[float, np.ndarray]:
    """Solve for the row player's (offense) optimal mixed strategy.

    Returns (v, p) where v is the game value and p is the mixed strategy.
    """
    import warnings
    A = np.asarray(A, dtype=float)
    if not np.all(np.isfinite(A)):
        warnings.warn(f"Payoff matrix has non-finite values ({(~np.isfinite(A)).sum()} entries); replacing with 0.0")
        A = np.where(np.isfinite(A), A, 0.0)
    m, n = A.shape
    v_lo = float(np.min(A)) - 1e-9
    v_hi = float(np.max(A)) + 1e-9

    c = np.zeros(m + 1, dtype=float)
    c[-1] = -1.0  # minimize -v

    A_ub = np.zeros((n, m + 1), dtype=float)
    b_ub = np.zeros(n, dtype=float)
    A_ub[:, :m] = -A.T
    A_ub[:, m] = 1.0

    A_eq = np.zeros((1, m + 1), dtype=float)
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0], dtype=float)

    bounds = [(0.0, 1.0)] * m + [(v_lo, v_hi)]

    candidates: List[Tuple[float, np.ndarray]] = []
    last_msg = None
    for method in ("highs", "highs-ipm", "highs-ds"):
        res = linprog(
            c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method=method,
            options={"presolve": True},
        )
        last_msg = res.message
        if res.x is None:
            continue

        p = res.x[:m].astype(float)
        v = float(res.x[m])

        p[p < 0] = 0.0
        s = float(np.sum(p))
        p = (np.ones(m, dtype=float) / m) if s <= 0 else (p / s)

        if res.success or _is_feasible_row_lp(A, p, v, tol=tol):
            return v, p

        candidates.append((v, p))

    # Strict tolerance failed — accept the best solution with looser tolerances.
    # Constraint violation of 1e-3 is negligible for game values in [-1, 1].
    for loose_tol in (1e-5, 1e-4, 1e-3, 1e-2):
        for v, p in candidates:
            if _is_feasible_row_lp(A, p, v, tol=loose_tol):
                return v, p

    # Absolute last resort: project p to uniform, bound v by minimax.
    if candidates:
        _, p = candidates[0]
        v = float(np.min(A.T @ p))  # guaranteed lower bound on game value
        return v, p

    # HiGHS returned no solution at all — use fictitious play (always terminates).
    v, p, _q = _fictitious_play(A)
    return v, p


def _is_feasible_col_lp(A: np.ndarray, q: np.ndarray, w: float, tol: float = 1e-8) -> bool:
    if np.any(q < -tol):
        return False
    if abs(float(np.sum(q)) - 1.0) > 1e-6:
        return False
    viol = (A @ q) - w  # <= 0
    return float(np.max(viol)) <= tol


def solve_col_strategy(A: np.ndarray, *, tol: float = 1e-8) -> Tuple[float, np.ndarray]:
    """Solve for the column player's (defense) optimal mixed strategy.

    Returns (w, q) where w is the game value and q is the mixed strategy.
    """
    import warnings
    A = np.asarray(A, dtype=float)
    if not np.all(np.isfinite(A)):
        warnings.warn(f"Payoff matrix has non-finite values ({(~np.isfinite(A)).sum()} entries); replacing with 0.0")
        A = np.where(np.isfinite(A), A, 0.0)
    m, n = A.shape
    w_lo = float(np.min(A)) - 1e-9
    w_hi = float(np.max(A)) + 1e-9

    c = np.zeros(n + 1, dtype=float)
    c[-1] = 1.0  # minimize w

    A_ub = np.zeros((m, n + 1), dtype=float)
    b_ub = np.zeros(m, dtype=float)
    A_ub[:, :n] = A
    A_ub[:, n] = -1.0

    A_eq = np.zeros((1, n + 1), dtype=float)
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0], dtype=float)

    bounds = [(0.0, 1.0)] * n + [(w_lo, w_hi)]

    candidates: List[Tuple[float, np.ndarray]] = []
    last_msg = None
    for method in ("highs", "highs-ipm", "highs-ds"):
        res = linprog(
            c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method=method,
            options={"presolve": True},
        )
        last_msg = res.message
        if res.x is None:
            continue

        q = res.x[:n].astype(float)
        w = float(res.x[n])

        q[q < 0] = 0.0
        s = float(np.sum(q))
        q = (np.ones(n, dtype=float) / n) if s <= 0 else (q / s)

        if res.success or _is_feasible_col_lp(A, q, w, tol=tol):
            return w, q

        candidates.append((w, q))

    for loose_tol in (1e-5, 1e-4, 1e-3, 1e-2):
        for w, q in candidates:
            if _is_feasible_col_lp(A, q, w, tol=loose_tol):
                return w, q

    if candidates:
        _, q = candidates[0]
        w = float(np.max(A @ q))  # guaranteed upper bound on game value
        return w, q

    # HiGHS returned no solution at all — use fictitious play (always terminates).
    _v, _p, q = _fictitious_play(A)
    w = float(np.max(A @ q))
    return w, q


# ---------------------------------------------------------------------------
# Value table lookup with clock advancement
# ---------------------------------------------------------------------------

# Key types (matching solve_full_game.py conventions):
#   ScrimmageKey  = ("s", q, tau, x, down, to_first, delta, h)
#   KickoffKey    = ("k", q, tau, delta, h)
#   SafetyKickKey = ("sk", q, tau, delta, h)
#   TerminalKey   = ("terminal",)

def _make_state_key(
    q: int, tau: int, tmpl: SuccessorTemplate,
) -> tuple:
    """Build a V-dict lookup key from a SuccessorTemplate and the resulting (q, tau)."""
    if tmpl.phase == "terminal":
        return ("terminal",)
    if tmpl.phase == "scrimmage":
        x, down, tf, delta, h = tmpl.core
        return ("s", q, tau, x, down, tf, delta, h)
    if tmpl.phase == "kickoff":
        delta, h = tmpl.core
        return ("k", q, tau, delta, h)
    if tmpl.phase == "safety_kick":
        delta, h = tmpl.core
        return ("sk", q, tau, delta, h)
    raise ValueError(f"Unknown phase: {tmpl.phase!r}")


def lookup_val(
    V: Dict,
    tmpl: SuccessorTemplate,
    q: int,
    tau: int,
) -> float:
    """Apply clock arithmetic then look up V; negate if swapped.

    Quarter-boundary: if new_tau == 0 and q < 4 → use (q+1, TICKS_PER_QUARTER).
                      if new_tau == 0 and q == 4 → terminal_utility.
    """
    new_tau, _hit_warn = advance_clock(tau, q, tmpl.ticks)

    # Determine successor quarter
    if new_tau == 0:
        if q < 4:
            q_next   = q + 1
            tau_next = TICKS_PER_QUARTER
        else:
            # Q4 ended → terminal utility
            if tmpl.phase == "scrimmage":
                delta_in_core = tmpl.core[3]   # (x, down, to_first, delta, h)
            elif tmpl.phase in ("kickoff", "safety_kick"):
                delta_in_core = tmpl.core[0]   # (delta, h)
            else:
                delta_in_core = 0

            raw_val = float(terminal_utility(delta_in_core))
            if tmpl.swapped:
                raw_val = -raw_val
            return raw_val
    else:
        q_next   = q
        tau_next = new_tau

    # Hot path: call typed accessor directly — avoids __getitem__ dispatch,
    # key-tuple allocation, and phase-string comparison on every lookup.
    # AttributeError fallback keeps plain-dict V working for tests.
    phase = tmpl.phase
    try:
        if phase == "scrimmage":
            x, down, tf, delta, h = tmpl.core
            v = V.get_scrimmage(q_next, tau_next, x, down, tf, delta, h)
        elif phase == "kickoff":
            delta, h = tmpl.core
            v = V.get_kickoff(q_next, tau_next, delta, h)
        else:  # safety_kick
            delta, h = tmpl.core
            v = V.get_safety(q_next, tau_next, delta, h)
    except AttributeError:
        v = float(V[_make_state_key(q_next, tau_next, tmpl)])
    if tmpl.swapped:
        v = -v
    return v


def _resolve_templates(
    result: TransitionResult,
    V: Dict,
    q: int,
    tau: int,
) -> float:
    """Resolve a TransitionResult to a scalar expected value.

    ChoiceTemplates: offense takes max, defense takes min (per resolver field).
    """
    if isinstance(result, list):
        return sum(t.prob * lookup_val(V, t, q, tau) for t in result)

    # ChoiceTemplates
    branch_vals = [
        sum(t.prob * lookup_val(V, t, q, tau) for t in branch)
        for branch in result.branches
    ]
    if result.resolver == "offense":
        return max(branch_vals)
    else:
        return min(branch_vals)


# ---------------------------------------------------------------------------
# Payoff matrix builder for a scrimmage state
# ---------------------------------------------------------------------------

def build_scrimmage_payoff_matrix(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    q: int,
    tau: int,
    x: int,
    down: int,
    to_first: int,
    delta: int,
    h: int,
    V: Dict,
) -> Tuple[np.ndarray, List[int]]:
    """Build the payoff matrix A[offense_action_idx, defense_card_idx].

    Returns (A, offense_actions) where offense_actions is the list of action codes
    corresponding to each row of A.
    """
    offense_actions = legal_offense_actions(x, down)
    n_off  = len(offense_actions)
    n_def  = len(DEFENSE_CARDS)   # always 10

    A = np.zeros((n_off, n_def), dtype=float)

    # FG row (if present): constant across all defense columns (def_card irrelevant)
    fg_result_cache: Optional[float] = None
    fg_row_idx: Optional[int] = None
    if FG_ACTION in offense_actions:
        fg_row_idx = offense_actions.index(FG_ACTION)
        tmpls = field_goal_templates(x, down, to_first, delta, h)
        fg_result_cache = _resolve_templates(tmpls, V, q, tau)

    for ii, action in enumerate(offense_actions):
        if action == FG_ACTION:
            A[ii, :] = fg_result_cache
            continue

        for jj, def_card in enumerate(DEFENSE_CARDS):
            if action == PUNT_ACTION:
                result = punt_templates(x, down, to_first, delta, h, def_card)
            else:
                result = scrimmage_play_templates(
                    scrimmage_chart, x, down, to_first, delta, h, action, def_card,
                )
            A[ii, jj] = _resolve_templates(result, V, q, tau)

    return A, offense_actions


# ---------------------------------------------------------------------------
# Kickoff state value
# ---------------------------------------------------------------------------

def kickoff_state_value(
    q: int,
    tau: int,
    delta: int,
    h: int,
    V: Dict,
) -> float:
    """Value of a KickoffState: kicker maximises over normal vs onside."""
    normal_tmpls  = normal_kickoff_templates(delta, h)
    onside_tmpls  = onside_kickoff_templates(delta, h)

    v_normal = sum(t.prob * lookup_val(V, t, q, tau) for t in normal_tmpls)
    v_onside = sum(t.prob * lookup_val(V, t, q, tau) for t in onside_tmpls)
    return max(v_normal, v_onside)


# ---------------------------------------------------------------------------
# Safety kick state value
# ---------------------------------------------------------------------------

def safety_kick_state_value(
    q: int,
    tau: int,
    delta: int,
    h: int,
    V: Dict,
) -> float:
    """Value of a SafetyKickState: free kicker maximises over two options."""
    normal_tmpls = safety_kick_option_normal(delta, h)
    punt_tmpls   = safety_kick_option_punt(delta, h)

    v_normal = sum(t.prob * lookup_val(V, t, q, tau) for t in normal_tmpls)
    v_punt   = sum(t.prob * lookup_val(V, t, q, tau) for t in punt_tmpls)
    return max(v_normal, v_punt)


# ---------------------------------------------------------------------------
# Timeout-aware variants
# ---------------------------------------------------------------------------
# These functions integrate the post-play timeout sub-game into the cell
# value computation.
#
# Design:
#   After a play produces outcome (ticks=k, next_core=C), both teams
#   simultaneously decide: call TO or not.  Calling reduces ticks k → k'.
#   The resulting 2×2 zero-sum sub-game is solved analytically.
#
# Reference: football_strategy/to_value_store.py (ToValueStore class).
# ---------------------------------------------------------------------------

from football_strategy.constants import TIMEOUT_REDUCE


def _solve_to_subgame(
    a: float, b: float, c: float,
    off_has_to: bool, def_has_to: bool,
) -> float:
    """Solve the post-play timeout decision (pure strategy).

        a = V(k', to_off-1, to_def  )  offense calls
        b = V(k', to_off,   to_def-1)  defense calls
        c = V(k,  to_off,   to_def  )  neither calls

    Zero-sum: reduced clock benefits exactly one side, so at most one team
    calls — never both.  Because at most one call occurs, the order of
    decisions is irrelevant: whoever benefits calls, the other does not.

        offense calls iff a > c  →  value = a
        defense calls iff b < c  →  value = b
        otherwise neither calls  →  value = c
    """
    if not off_has_to and not def_has_to:
        return c

    if not off_has_to:
        return min(b, c)            # defense calls iff beneficial

    if not def_has_to:
        return max(a, c)            # offense calls iff beneficial

    # Both have TOs — at most one will call.
    return max(a, min(b, c))


def _lookup_with_to(
    to_store: "ToValueStore",           # type: ignore[name-defined]
    tmpl: SuccessorTemplate,
    q: int,
    tau: int,
    to_off: int,
    to_def: int,
) -> float:
    """lookup_val extended with the timeout decision sub-game.

    For a single SuccessorTemplate with ticks=k:
    - If k cannot be reduced (PAT=0, or TIMEOUT_REDUCE[k]==k): plain lookup.
    - Otherwise: build a reduced template (ticks=k') and solve the timeout
      sub-game.  Three slices are consulted:

        c  : (to_off,   to_def  )  — neither calls, ticks=k
        a  : (to_off-1, to_def  )  — offense calls,  ticks=k'
        b  : (to_off,   to_def-1)  — defense calls,  ticks=k'

    The "both call" slice is never needed (zero-sum: at most one team calls).
    """
    k = tmpl.ticks
    k_prime = TIMEOUT_REDUCE.get(k, k)

    if k_prime == k:
        # No TO benefit possible for this outcome
        return lookup_val(to_store.get(to_off, to_def), tmpl, q, tau)

    # Reduced-clock template (same core / phase / swapped)
    tmpl_r = SuccessorTemplate(
        prob=tmpl.prob,
        phase=tmpl.phase,
        core=tmpl.core,
        ticks=k_prime,
        swapped=tmpl.swapped,
        delta_change=tmpl.delta_change,
    )

    # c: neither calls — normal ticks, same TOs
    c = lookup_val(to_store.get(to_off, to_def), tmpl, q, tau)

    # a: offense calls — reduced ticks, offense loses 1 TO
    # b: defense calls — reduced ticks, defense loses 1 TO
    # "both call" case eliminated (zero-sum: at most one side benefits).
    #
    # k'=0: advance_clock(tau, q, 0) = tau, so the lookup targets the SAME tau
    # in the predecessor store.  States may be absent if genuinely unreachable
    # via 0-tick transitions (e.g. safety_kick at tau=7).  Fall back to c.
    effective_off = to_off > 0
    a = 0.0
    if effective_off:
        try:
            a = lookup_val(to_store.get(to_off - 1, to_def), tmpl_r, q, tau)
        except KeyError:
            effective_off = False

    effective_def = to_def > 0
    b = 0.0
    if effective_def:
        try:
            b = lookup_val(to_store.get(to_off, to_def - 1), tmpl_r, q, tau)
        except KeyError:
            effective_def = False

    return _solve_to_subgame(a, b, c, effective_off, effective_def)


def _resolve_templates_to(
    result: TransitionResult,
    to_store: "ToValueStore",           # type: ignore[name-defined]
    q: int,
    tau: int,
    to_off: int,
    to_def: int,
) -> float:
    """_resolve_templates extended with the 2×2 timeout sub-game.

    Mirrors _resolve_templates but calls _lookup_with_to instead of lookup_val.
    """
    if isinstance(result, list):
        return sum(
            t.prob * _lookup_with_to(to_store, t, q, tau, to_off, to_def)
            for t in result
        )

    # ChoiceTemplates — resolver is "offense" or "defense"
    branch_vals = [
        sum(
            t.prob * _lookup_with_to(to_store, t, q, tau, to_off, to_def)
            for t in branch
        )
        for branch in result.branches
    ]
    if result.resolver == "offense":
        return max(branch_vals)
    else:
        return min(branch_vals)


def build_scrimmage_payoff_matrix_to(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    q: int,
    tau: int,
    x: int,
    down: int,
    to_first: int,
    delta: int,
    h: int,
    to_store: "ToValueStore",           # type: ignore[name-defined]
    to_off: int,
    to_def: int,
) -> Tuple[np.ndarray, List[int]]:
    """Payoff matrix with timeout-adjusted cell values.

    Drop-in replacement for build_scrimmage_payoff_matrix when to_off > 0 or
    to_def > 0.  For (to_off=0, to_def=0) the results are identical to Stage 1
    because _lookup_with_to degenerates to a plain lookup_val call.

    Returns (A, offense_actions) — same contract as build_scrimmage_payoff_matrix.
    """
    offense_actions = legal_offense_actions(x, down)
    n_off = len(offense_actions)
    n_def = len(DEFENSE_CARDS)

    A = np.zeros((n_off, n_def), dtype=float)

    # FG row: constant across all defense columns
    fg_result_cache: Optional[float] = None
    fg_row_idx: Optional[int] = None
    if FG_ACTION in offense_actions:
        fg_row_idx = offense_actions.index(FG_ACTION)
        tmpls = field_goal_templates(x, down, to_first, delta, h)
        fg_result_cache = _resolve_templates_to(tmpls, to_store, q, tau, to_off, to_def)

    for ii, action in enumerate(offense_actions):
        if ii == fg_row_idx:
            A[ii, :] = fg_result_cache
            continue

        for jj, def_card in enumerate(DEFENSE_CARDS):
            if action == PUNT_ACTION:
                result = punt_templates(x, down, to_first, delta, h, def_card)
            else:
                result = scrimmage_play_templates(
                    scrimmage_chart, x, down, to_first, delta, h, action, def_card,
                )
            A[ii, jj] = _resolve_templates_to(result, to_store, q, tau, to_off, to_def)

    return A, offense_actions


def kickoff_state_value_to(
    q: int,
    tau: int,
    delta: int,
    h: int,
    to_store: "ToValueStore",           # type: ignore[name-defined]
    to_off: int,
    to_def: int,
) -> float:
    """Kickoff value with timeout sub-game (kicker maximises normal vs onside)."""
    normal_tmpls = normal_kickoff_templates(delta, h)
    onside_tmpls = onside_kickoff_templates(delta, h)

    v_normal = sum(
        t.prob * _lookup_with_to(to_store, t, q, tau, to_off, to_def)
        for t in normal_tmpls
    )
    v_onside = sum(
        t.prob * _lookup_with_to(to_store, t, q, tau, to_off, to_def)
        for t in onside_tmpls
    )
    return max(v_normal, v_onside)


def safety_kick_state_value_to(
    q: int,
    tau: int,
    delta: int,
    h: int,
    to_store: "ToValueStore",           # type: ignore[name-defined]
    to_off: int,
    to_def: int,
) -> float:
    """Safety kick value with timeout sub-game (free kicker maximises two options)."""
    normal_tmpls = safety_kick_option_normal(delta, h)
    punt_tmpls   = safety_kick_option_punt(delta, h)

    v_normal = sum(
        t.prob * _lookup_with_to(to_store, t, q, tau, to_off, to_def)
        for t in normal_tmpls
    )
    v_punt = sum(
        t.prob * _lookup_with_to(to_store, t, q, tau, to_off, to_def)
        for t in punt_tmpls
    )
    return max(v_normal, v_punt)
