# solve_quarter_sym.py
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import linprog
from functools import lru_cache
from typing import Dict, Tuple, List, Optional, Set, Callable

from read_matrix import load_pro_style_chart_csv
from transition_quarter_sym import (
    State, successors,
    OFFENSE_ACTIONS, DEFENSE_PLAYS,
    PUNT,
)

# ============================================================
# SYMMETRIC SOLVER (NO poss IN STATE)
#
# - State is always "current offense drives toward 100"
# - delta = offense - defense
# - If possession changes, continuation value flips sign
#
# Terminal utility: tanh(delta / TANH_SCALE)
# (tanh is odd, which is crucial for symmetry correctness)
# ============================================================

# -----------------------------
# Config
# -----------------------------
DELTA_MIN = -14
DELTA_MAX = 14

TO_FIRST_MAX = 30
TANH_SCALE = 7.0
USE_TRANSITION_REWARD = False  # you currently encode score in delta updates

# -----------------------------
# Core state (time separate)
# -----------------------------
Core = Tuple[int, int, int, int]  # (x, down, to_first, delta)

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

def state_from_core(c: Core, t: int) -> State:
    x, down, to_first, delta = c
    return State(x=int(x), down=int(down), to_first=int(to_first), t=int(t), delta=int(delta))

def core_of_state(s: State) -> Core:
    return (
        clip_x(int(s.x)),
        int(s.down),
        clip_to_first(int(s.to_first)),
        clip_delta(int(s.delta)),
    )

# -----------------------------
# Robust LP solvers (your same fallbacks)
# -----------------------------
def _is_feasible_row_lp(A: np.ndarray, p: np.ndarray, v: float, tol: float = 1e-8) -> bool:
    if np.any(p < -tol):
        return False
    if abs(float(np.sum(p)) - 1.0) > 1e-6:
        return False
    viol = (-A.T @ p) + v  # <= 0
    return float(np.max(viol)) <= tol

def solve_row_strategy(A: np.ndarray, *, tol: float = 1e-8) -> Tuple[float, np.ndarray]:
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    v_lo = float(np.min(A)) - 1e-9
    v_hi = float(np.max(A)) + 1e-9

    # variables: p[0..m-1], v
    c = np.zeros(m + 1, dtype=float)
    c[-1] = -1.0  # minimize -v

    # -A^T p + v <= 0
    A_ub = np.zeros((n, m + 1), dtype=float)
    b_ub = np.zeros(n, dtype=float)
    A_ub[:, :m] = -A.T
    A_ub[:, m] = 1.0

    # sum(p)=1
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
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    w_lo = float(np.min(A)) - 1e-9
    w_hi = float(np.max(A)) + 1e-9

    # variables: q[0..n-1], w
    c = np.zeros(n + 1, dtype=float)
    c[-1] = 1.0  # minimize w

    # A q - w <= 0
    A_ub = np.zeros((m, n + 1), dtype=float)
    b_ub = np.zeros(m, dtype=float)
    A_ub[:, :n] = A
    A_ub[:, n] = -1.0

    # sum(q)=1
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
# Transition caching (time-blind core cache)
# -----------------------------
def make_core_successor_cache(chart):
    """
    Cache successors for (core, action, def_play).
    Each successor includes whether possession swapped (role swap).
    """
    @lru_cache(maxsize=None)
    def succ_core(core: Core, off_action: int, d: str) -> Tuple[Tuple[Optional[Core], float, bool], ...]:
        s = state_from_core(core, t=1)
        outs: List[Tuple[Optional[Core], float, bool]] = []
        for tr in successors(chart, s, off_action, d):
            if tr.next_state is None:
                outs.append((None, float(tr.reward), bool(tr.swapped)))
            else:
                outs.append((core_of_state(tr.next_state), float(tr.reward), bool(tr.swapped)))
        return tuple(outs)
    return succ_core

# -----------------------------
# Cell resolution + payoff matrix (OFFENSE-VALUE convention)
# -----------------------------
def resolve_cell_off_value(
    succ_core,
    core: Core,
    off_action: int,
    def_play: str,
    V_next: Dict[Core, float],
) -> float:
    """
    Payoff for one cell (off_action, def_play) at core, from CURRENT OFFENSE POV.

    - OR-branches are resolved by offense: choose max branch
    - If possession changes: continuation value flips sign (zero-sum role swap)
    - In win-utility mode, transition.reward usually ignored (score lives in delta)
    """
    delta_now = core[-1]
    vals: List[float] = []

    for nxt_core, r, swapped in succ_core(core, off_action, def_play):
        add = float(r) if USE_TRANSITION_REWARD else 0.0

        if nxt_core is None:
            vals.append(add + terminal_utility(delta_now))
            continue

        if nxt_core not in V_next:
            raise KeyError(f"Missing V_next for successor core={nxt_core} from core={core}")

        v = float(V_next[nxt_core])
        if swapped:
            v = -v
        vals.append(add + v)

    if not vals:
        return terminal_utility(delta_now)

    return max(vals)

def build_payoff_matrix_off(succ_core, core: Core, V_next: Dict[Core, float]) -> np.ndarray:
    m, n = len(OFFENSE_ACTIONS), len(DEFENSE_PLAYS)
    A = np.zeros((m, n), dtype=float)
    for ii, off_action in enumerate(OFFENSE_ACTIONS):
        for jj, d in enumerate(DEFENSE_PLAYS):
            A[ii, jj] = resolve_cell_off_value(succ_core, core, off_action, d, V_next)
    return A

# -----------------------------
# Reachability layers on CORE
# -----------------------------
def build_layers_core(chart, start_core: Core, T: int, *, verbose=True) -> List[Set[Core]]:
    succ_core = make_core_successor_cache(chart)
    layers: List[Set[Core]] = [set() for _ in range(T + 1)]
    layers[T] = {start_core}

    for t in range(T, 0, -1):
        nxt: Set[Core] = set()
        for core in layers[t]:
            for off_action in OFFENSE_ACTIONS:
                for d in DEFENSE_PLAYS:
                    for nxt_core, _r, _sw in succ_core(core, off_action, d):
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
        "v_off": float(v),
        "best_response_gap": gap,
        "row_br_value": row_br_value,
        "col_br_value": col_br_value,
    }

def print_state_block(label: str, s: State, v: float, p_off: np.ndarray, q_def: np.ndarray, diag: Optional[Dict[str, float]] = None):
    print("\n==============================")
    print(label)
    print(f"State: x={s.x}, down={s.down}, to_first={s.to_first}, t={s.t}, delta(off-def)={s.delta}")
    print("V_offense =", float(v))
    print("Policy for OFFENSE (includes PUNT):")
    print_offense_mix(p_off)
    print("Policy for DEFENSE (A..J):")
    print_defense_mix(q_def)
    if diag is not None:
        print("Diagnostics:")
        for k, val in diag.items():
            print(f"{k:>20s}: {val:.9f}")

# -----------------------------
# Reporting plan (adapted; labels still useful)
# -----------------------------
def make_report_plan(
    layers: List[Set[Core]],
    T: int,
    *,
    seed: int = 0,
    t_step: int = 5,
    per_t: int = 50,
    max_total: int = 2000,
) -> List[Tuple[str, int, Core]]:
    rng = np.random.default_rng(seed)
    picked: List[Tuple[str, int, Core]] = []
    used: Set[Tuple[int, Core]] = set()

    def add(label: str, t: int, core: Core):
        key = (t, core)
        if key in used:
            return
        used.add(key)
        picked.append((label, t, core))

    # broad random coverage across time
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

    # targeted situations
    def is_redzone(c: Core) -> bool:
        x, _down, _to_first, _delta = c
        return x >= 80

    def is_goal_to_go(c: Core) -> bool:
        x, _down, to_first, _delta = c
        return x >= 90 and to_first <= 10

    def is_backed_up(c: Core) -> bool:
        x, _down, _to_first, _delta = c
        return x <= 10

    def is_4th_and_short(c: Core) -> bool:
        _x, down, to_first, _delta = c
        return down == 4 and to_first <= 2

    def is_4th_and_long(c: Core) -> bool:
        _x, down, to_first, _delta = c
        return down == 4 and to_first >= 8

    def is_3rd_and_long(c: Core) -> bool:
        _x, down, to_first, _delta = c
        return down == 3 and to_first >= 8

    def is_midfield(c: Core) -> bool:
        x, down, to_first, _delta = c
        return 45 <= x <= 55 and down == 1 and to_first == 10

    def is_close_game(c: Core) -> bool:
        return abs(c[-1]) <= 3

    def is_down_big(c: Core) -> bool:
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
        start_state = State(x=20, down=1, to_first=10, t=T, delta=0)

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
            A = build_payoff_matrix_off(succ_core, core, V_prev)
            v, p_off = solve_row_strategy(A)
            _w, q_def = solve_col_strategy(A)

            v_off = float(v)
            V_curr[core] = v_off

            if (t, core) in report_keys:
                reports[(t, core)] = {
                    "p_off": p_off,
                    "q_def": q_def,
                    "v": v_off,
                    "diag": diagnostics(A, p_off, q_def, v),
                }

            if (t == T) and (core == start_core):
                stored["p_off"] = p_off
                stored["q_def"] = q_def
                stored["v"] = v_off
                stored["diag"] = diagnostics(A, p_off, q_def, v)

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
        diag=stored["diag"],
    )

    print("\n" + "=" * 30)
    print(f"REPORTING {len(report_plan)} REACHABLE STATES (stored {len(reports)})")

    report_plan_sorted = sorted(report_plan, key=lambda x: (x[0], -x[1]))
    for label, t, core in report_plan_sorted:
        rec = reports.get((t, core))
        if rec is None:
            continue
        s = state_from_core(core, t=t)
        print_state_block(
            f"{label}  (t={t})",
            s,
            rec["v"],
            rec["p_off"],
            rec["q_def"],
            diag=rec["diag"],
        )
