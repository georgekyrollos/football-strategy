# solve_quarter.py
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import linprog
from functools import lru_cache
from typing import Dict, Tuple, List, Optional, Set, Callable

from read_matrix import load_pro_style_chart_csv
from transition_quarter_score import (
    State, successors,
    OFFENSE_ACTIONS, DEFENSE_PLAYS,
    US, THEM, PUNT,
)

# ============================================================
# NO BUCKETING VERSION
#   - Exact x (0..100) and exact to_first (clipped only)
#   - Exact delta (score differential) clipped to [-14, 14]
#   - Terminal utility: tanh(delta / TANH_SCALE)
#
# IMPORTANT (objective):
#   This file assumes "win-utility" is handled ONLY by delta in the state
#   + terminal_utility at t=0.
#   Therefore we IGNORE transition.reward in DP by default to avoid
#   double-counting if your transition already updates delta.
#   (If your transition encodes score only in reward and does NOT update delta,
#    set USE_TRANSITION_REWARD=True and make sure delta is updated consistently.)
# ============================================================

# -----------------------------
# Config: bounds + terminal utility
# -----------------------------
DELTA_MIN = -14
DELTA_MAX = 14

# Allowed upper bound (saturating) on to_first to prevent pathological blowups
# NOTE: this is NOT "bucketing by distance"; it only clips very large values.
TO_FIRST_MAX = 30

# Smooth win utility
TANH_SCALE = 7.0  # U(delta)=tanh(delta/TANH_SCALE)

# Whether to add transition.reward inside DP (default: False for win-utility)
USE_TRANSITION_REWARD = False

# -----------------------------
# Abstract “core” state (time is separate)
# -----------------------------
Core = Tuple[int, int, int, int, int]  # (poss, x, down, to_first, delta)

# -----------------------------
# Utility + clipping helpers
# -----------------------------
def clip_delta(d: int) -> int:
    return DELTA_MIN if d < DELTA_MIN else (DELTA_MAX if d > DELTA_MAX else d)

def clip_to_first(to_first: int) -> int:
    if to_first < 1:
        return 1
    if to_first > TO_FIRST_MAX:
        return TO_FIRST_MAX
    return to_first

def clip_x(x: int) -> int:
    if x < 0:
        return 0
    if x > 100:
        return 100
    return x

def terminal_utility(delta: int) -> float:
    return float(math.tanh(delta / TANH_SCALE))

def poss_str(poss: int) -> str:
    return "US" if poss == US else "THEM"

# -----------------------------
# Reporting plan (sample lots of states)
# -----------------------------
def state_from_core(c: Core, t: int) -> State:
    poss, x, down, to_first, delta = c
    return State(
        poss=int(poss),
        x=int(x),
        down=int(down),
        to_first=int(to_first),
        t=int(t),
        delta=int(delta),
    )

def _core_to_state_for_print(core: Core, t: int) -> State:
    return state_from_core(core, t=t)

def make_report_plan(
    layers: List[Set[Core]],
    T: int,
    *,
    seed: int = 0,
    t_step: int = 5,
    per_t: int = 50,
    max_total: int = 2000,
) -> List[Tuple[str, int, Core]]:
    """
    Pick many reachable (t, core) pairs to inspect.

    - Samples evenly across time (every t_step)
    - Adds targeted "football situation" filters
    - No bucketing used for solving; this is JUST for reporting.
    """
    rng = np.random.default_rng(seed)
    picked: List[Tuple[str, int, Core]] = []
    used: Set[Tuple[int, Core]] = set()

    def add(label: str, t: int, core: Core):
        key = (t, core)
        if key in used:
            return
        used.add(key)
        picked.append((label, t, core))

    # 1) Broad random coverage across time
    for t in range(T, 0, -t_step):
        cores = list(layers[t])
        if not cores:
            continue
        k = min(per_t, len(cores))
        idxs = rng.choice(len(cores), size=k, replace=False)
        for i in idxs:
            add("RANDOM", t, cores[int(i)])
        if len(picked) >= max_total:
            return picked[:max_total]

    # 2) Targeted “interesting” situations
    def is_redzone(c: Core) -> bool:
        _poss, x, _down, _to_first, _delta = c
        return x >= 80

    def is_goal_to_go(c: Core) -> bool:
        _poss, x, _down, to_first, _delta = c
        return x >= 90 and to_first <= 10

    def is_backed_up(c: Core) -> bool:
        _poss, x, _down, _to_first, _delta = c
        return x <= 10

    def is_4th_and_short(c: Core) -> bool:
        _poss, _x, down, to_first, _delta = c
        return down == 4 and to_first <= 2

    def is_4th_and_long(c: Core) -> bool:
        _poss, _x, down, to_first, _delta = c
        return down == 4 and to_first >= 8

    def is_3rd_and_long(c: Core) -> bool:
        _poss, _x, down, to_first, _delta = c
        return down == 3 and to_first >= 8

    def is_midfield(c: Core) -> bool:
        _poss, x, down, to_first, _delta = c
        return 45 <= x <= 55 and down == 1 and to_first == 10

    def is_close_game(c: Core) -> bool:
        return abs(c[-1]) <= 3

    def is_down_big(c: Core) -> bool:
        # US losing big => delta <= -10
        return c[-1] <= -10

    def is_up_big(c: Core) -> bool:
        return c[-1] >= 10

    situations: List[Tuple[str, Callable[[Core], bool], int]] = [
        ("MIDFIELD_1ST10", is_midfield, 120),
        ("REDZONE", is_redzone, 120),
        ("GOAL_TO_GO", is_goal_to_go, 120),
        ("BACKED_UP", is_backed_up, 120),
        ("4TH_SHORT", is_4th_and_short, 160),
        ("4TH_LONG", is_4th_and_long, 160),
        ("3RD_LONG", is_3rd_and_long, 160),
        ("CLOSE_GAME", is_close_game, 160),
        ("DOWN_BIG", is_down_big, 120),
        ("UP_BIG", is_up_big, 120),
    ]

    for label, pred, want_k in situations:
        got = 0
        for t in range(T, 0, -1):
            if got >= want_k:
                break
            for core in layers[t]:
                if pred(core):
                    add(label, t, core)
                    got += 1
                    if got >= want_k:
                        break
        if len(picked) >= max_total:
            return picked[:max_total]

    return picked[:max_total]

# -----------------------------
# Robust LP solvers (HiGHS "Unknown" handling)
# -----------------------------
def _is_feasible_row_lp(A: np.ndarray, p: np.ndarray, v: float, tol: float = 1e-8) -> bool:
    if np.any(p < -tol):
        return False
    if abs(float(np.sum(p)) - 1.0) > 1e-6:
        return False
    viol = (-A.T @ p) + v  # <= 0
    return float(np.max(viol)) <= tol

def solve_row_strategy(A: np.ndarray, *, tol: float = 1e-8) -> Tuple[float, np.ndarray]:
    """
    Zero-sum matrix game: ROW maximizes, COL minimizes payoff A.
    Returns (v, p_row). Accepts HiGHS 'Unknown' if feasible.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape={A.shape}")
    if not np.all(np.isfinite(A)):
        bad = np.argwhere(~np.isfinite(A))
        i, j = map(int, bad[0])
        raise ValueError(f"A contains non-finite at ({i},{j}): {A[i,j]}")

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

    raise RuntimeError(f"Row LP failed after fallbacks. Last message: {last_msg}")

def _is_feasible_col_lp(A: np.ndarray, q: np.ndarray, w: float, tol: float = 1e-8) -> bool:
    if np.any(q < -tol):
        return False
    if abs(float(np.sum(q)) - 1.0) > 1e-6:
        return False
    viol = (A @ q) - w  # <= 0
    return float(np.max(viol)) <= tol

def solve_col_strategy(A: np.ndarray, *, tol: float = 1e-8) -> Tuple[float, np.ndarray]:
    """
    Minimizing column player:
      minimize w
      s.t. A q <= w, sum(q)=1, q>=0
    Returns (w, q).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape={A.shape}")
    if not np.all(np.isfinite(A)):
        bad = np.argwhere(~np.isfinite(A))
        i, j = map(int, bad[0])
        raise ValueError(f"A contains non-finite at ({i},{j}): {A[i,j]}")

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

    raise RuntimeError(f"Col LP failed after fallbacks. Last message: {last_msg}")

# -----------------------------
# Core conversions (NO BUCKETING)
# -----------------------------
def core_of_state(s: State) -> Core:
    return (
        int(s.poss),
        clip_x(int(s.x)),
        int(s.down),
        clip_to_first(int(s.to_first)),
        clip_delta(int(s.delta)),
    )

# -----------------------------
# Transition caching (time-blind core cache)
# -----------------------------
def make_core_successor_cache(chart):
    """
    Cache (next_core, reward) for each (core, action, def_play).
    Uses representative State(core, t=1). Successors mapped back to (clipped) core.
    """
    @lru_cache(maxsize=None)
    def succ_core(core: Core, off_action: int, d: str) -> Tuple[Tuple[Optional[Core], float], ...]:
        s = state_from_core(core, t=1)
        outs: List[Tuple[Optional[Core], float]] = []
        for tr in successors(chart, s, off_action, d):
            if tr.next_state is None:
                outs.append((None, float(tr.reward)))
            else:
                outs.append((core_of_state(tr.next_state), float(tr.reward)))
        return tuple(outs)

    return succ_core

# -----------------------------
# Cell resolution + payoff matrix (US-value convention)
# -----------------------------
def resolve_cell_us_value(
    succ_core,
    core: Core,
    off_action: int,
    def_play: str,
    V_next: Dict[Core, float],
) -> float:
    """
    US payoff for one cell (off_action, def_play) at a CORE state.

    OR-branches are resolved by current offense:
      - poss==US => max branch
      - poss==THEM => min branch (THEM minimizes US)

    Win-utility mode:
      - Usually ignore transition.reward (score should live in next_state.delta).
    """
    poss = core[0]
    delta_now = core[-1]
    vals: List[float] = []

    for nxt_core, r in succ_core(core, off_action, def_play):
        add = float(r) if USE_TRANSITION_REWARD else 0.0

        if nxt_core is None:
            vals.append(add + terminal_utility(delta_now))
        else:
            if nxt_core not in V_next:
                raise KeyError(f"Missing V_next for successor core={nxt_core} from core={core}")
            vals.append(add + float(V_next[nxt_core]))

    if not vals:
        return terminal_utility(delta_now)

    return max(vals) if poss == US else min(vals)

def build_payoff_matrix_us(succ_core, core: Core, V_next: Dict[Core, float]) -> np.ndarray:
    m, n = len(OFFENSE_ACTIONS), len(DEFENSE_PLAYS)
    A = np.zeros((m, n), dtype=float)
    for ii, off_action in enumerate(OFFENSE_ACTIONS):
        for jj, d in enumerate(DEFENSE_PLAYS):
            A[ii, jj] = resolve_cell_us_value(succ_core, core, off_action, d, V_next)
    return A

# -----------------------------
# Reachability layers on CORE
# -----------------------------
def build_layers_core(chart, start_core: Core, T: int, *, verbose=True) -> List[Set[Core]]:
    """
    layers[t] = set of cores reachable from start_core with exactly t plays remaining.
    Built backward:
      layers[T] = {start_core}
      layers[t-1] = one-step successors of layers[t]
    """
    succ_core = make_core_successor_cache(chart)
    layers: List[Set[Core]] = [set() for _ in range(T + 1)]
    layers[T] = {start_core}

    for t in range(T, 0, -1):
        nxt: Set[Core] = set()
        for core in layers[t]:
            for off_action in OFFENSE_ACTIONS:
                for d in DEFENSE_PLAYS:
                    for nxt_core, _r in succ_core(core, off_action, d):
                        if nxt_core is not None:
                            nxt.add(nxt_core)
        layers[t - 1] = nxt
        if verbose:
            print(f"States with {t-1} plays remaining: {len(layers[t-1])}")

    return layers

# -----------------------------
# Pretty printing + diagnostics
# -----------------------------
def print_offense_mix(p: np.ndarray, *, header: Optional[str] = None):
    if header:
        print(header)
    for idx, act in enumerate(OFFENSE_ACTIONS):
        label = "PUNT" if act == PUNT else f"{act:>2}"
        print(f"{label:>4}: {float(p[idx]):.6f}")
    print("Total probability:", float(np.sum(p)))

def print_defense_mix(q: np.ndarray, *, header: Optional[str] = None):
    if header:
        print(header)
    for j, d in enumerate(DEFENSE_PLAYS):
        print(f"{d}: {float(q[j]):.6f}")
    print("Total probability:", float(np.sum(q)))

def diagnostics(A: np.ndarray, p: np.ndarray, q: np.ndarray, v: float) -> Dict[str, float]:
    ev = float(p @ A @ q)
    row_br_value = float(np.max(A @ q))
    col_br_value = float(np.min(p @ A))
    gap = float(row_br_value - col_br_value)
    return {
        "ev": ev,
        "v_us": float(v),
        "best_response_gap": gap,
        "row_br_value": row_br_value,
        "col_br_value": col_br_value,
    }

def print_state_block(
    label: str,
    s: State,
    v: float,
    p_off: np.ndarray,
    q_def: np.ndarray,
    *,
    who_off: str,
    who_def: str,
    diag: Optional[Dict[str, float]] = None,
):
    print("\n==============================")
    print(label)
    print(f"State: poss={poss_str(s.poss)}, x={s.x}, down={s.down}, to_first={s.to_first}, t={s.t}, delta={s.delta}")
    print("V_us =", float(v))
    print(f"Policy for {who_off} offense (includes PUNT):")
    print_offense_mix(p_off)
    print(f"Policy for {who_def} defense (A..J):")
    print_defense_mix(q_def)
    if diag is not None:
        print("Diagnostics:")
        for k, val in diag.items():
            print(f"{k:>20s}: {val:.9f}")

def print_report_block(label: str, t: int, core: Core, rec: dict):
    s = _core_to_state_for_print(core, t=t)
    who_off = rec["who_off"]
    who_def = rec["who_def"]
    v = rec["v"]
    p_off = rec["p_off"]
    q_def = rec["q_def"]
    diag = rec["diag"]
    print_state_block(f"{label}  (t={t})", s, v, p_off, q_def, who_off=who_off, who_def=who_def, diag=diag)

# -----------------------------
# Main solver
# -----------------------------
def solve_quarter(
    T: int = 60,
    csv_path: str = "Football Strategy Pro Style.csv",
    start_state: Optional[State] = None,
    *,
    verbose: bool = True,
):
    chart = load_pro_style_chart_csv(csv_path)
    succ_core = make_core_successor_cache(chart)

    if start_state is None:
        start_state = State(poss=US, x=20, down=1, to_first=10, t=T, delta=0)
    start_core = core_of_state(start_state)

    if verbose:
        print(f"Building reachable layers up to T={T} ...")
    layers = build_layers_core(chart, start_core, T, verbose=verbose)

    report_plan = make_report_plan(
        layers, T,
        seed=0,
        t_step=5,
        per_t=60,
        max_total=2000,
    )
    report_keys: Set[Tuple[int, Core]] = {(t, core) for _lab, t, core in report_plan}
    reports: Dict[Tuple[int, Core], dict] = {}

    # Base layer: t=0 -> terminal utility based on delta only
    V_prev: Dict[Core, float] = {c: terminal_utility(c[-1]) for c in layers[0]}

    stored: Dict[str, object] = {}

    for t in range(1, T + 1):
        if verbose:
            print(f"\nSolving layer t={t}/{T} (|states|={len(layers[t])}) ...")

        V_curr: Dict[Core, float] = {}

        for core in layers[t]:
            A_us = build_payoff_matrix_us(succ_core, core, V_prev)
            poss = core[0]

            if poss == US:
                v, p_us = solve_row_strategy(A_us)
                _w, q_them = solve_col_strategy(A_us)
                v_us = float(v)
                V_curr[core] = v_us

                if (t, core) in report_keys:
                    reports[(t, core)] = {
                        "who_off": "US",
                        "who_def": "THEM",
                        "p_off": p_us,
                        "q_def": q_them,
                        "v": v_us,
                        "diag": diagnostics(A_us, p_us, q_them, v),
                    }

                if (t == T) and (core == start_core):
                    stored["who_off"] = "US"
                    stored["who_def"] = "THEM"
                    stored["p_off"] = p_us
                    stored["q_def"] = q_them
                    stored["v"] = v_us
                    stored["diag"] = diagnostics(A_us, p_us, q_them, v)

            else:
                # THEM offense minimizes US; transform by B=-A_us
                B = -A_us
                vB, p_them = solve_row_strategy(B)
                _w, q_us = solve_col_strategy(B)
                v_us = float(-vB)
                V_curr[core] = v_us

                if (t, core) in report_keys:
                    # gap check in B-space; still useful
                    reports[(t, core)] = {
                        "who_off": "THEM",
                        "who_def": "US",
                        "p_off": p_them,
                        "q_def": q_us,
                        "v": v_us,
                        "diag": diagnostics(B, p_them, q_us, vB),
                    }

        V_prev = V_curr

    return start_state, stored, report_plan, reports


if __name__ == "__main__":
    T = 60
    s0, stored, report_plan, reports = solve_quarter(
        T=T,
        csv_path="Football Strategy Pro Style.csv",
        verbose=True,
    )

    print("\n" + "=" * 30)
    print("START (s0)")
    print_state_block(
        "START (s0)",
        s0,
        float(stored["v"]),
        stored["p_off"],
        stored["q_def"],
        who_off=str(stored["who_off"]),
        who_def=str(stored["who_def"]),
        diag=stored["diag"],
    )

    print("\n" + "=" * 30)
    print(f"REPORTING {len(report_plan)} REACHABLE STATES (stored {len(reports)})")

    report_plan_sorted = sorted(report_plan, key=lambda x: (x[0], -x[1]))
    for label, t, core in report_plan_sorted:
        key = (t, core)
        rec = reports.get(key)
        if rec is None:
            continue
        print_report_block(label, t, core, rec)
