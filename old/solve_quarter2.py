# solve_quarter.py
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from functools import lru_cache
from typing import Dict, Tuple, List, Optional, Set, Callable, Any

from read_matrix import load_pro_style_chart_csv
from transition_quarter import (
    State, successors,
    OFFENSE_ACTIONS, DEFENSE_PLAYS,
    US, THEM, PUNT
)

Core = Tuple[int, int, int, int]  # (poss, x, down, to_first)

# -----------------------------
# Numerical stability knobs
# -----------------------------
# HiGHS "Status 15: Unknown; primal_status Basic" is commonly triggered by bad scaling / huge free bounds.
# Bound the value variable to something realistic for this game (way looser than needed, but not insane).
V_BOUND = 1000.0


# -----------------------------
# LP solvers (ROW-max, COL-min)
# -----------------------------
def solve_row_strategy(A: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Zero-sum matrix game: ROW maximizes, COL minimizes payoff A.
    Returns (v, p_row).
    """
    m, n = A.shape

    # vars: p_0..p_{m-1}, v
    # maximize v <=> minimize -v
    c = np.zeros(m + 1, dtype=float)
    c[-1] = -1.0

    # A^T p >= v  <=>  -A^T p + v <= 0
    A_ub = np.zeros((n, m + 1), dtype=float)
    b_ub = np.zeros(n, dtype=float)
    for j in range(n):
        A_ub[j, :m] = -A[:, j]
        A_ub[j, m] = 1.0

    # sum p = 1
    A_eq = np.zeros((1, m + 1), dtype=float)
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0], dtype=float)

    # Tighten bounds for numerical stability:
    # p_i in [0,1], v in [-V_BOUND, V_BOUND]
    bounds = [(0.0, 1.0)] * m + [(-V_BOUND, V_BOUND)]

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs"
    )
    if not res.success:
        raise RuntimeError(f"Row LP failed: {res.message}")

    p = res.x[:m].astype(float)
    v = float(res.x[m])
    return v, p


def solve_col_strategy(A: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Zero-sum matrix game: ROW maximizes, COL minimizes payoff A.
    Compute the COL minimizer strategy q and value w:
      minimize w
      s.t. A q <= w, sum q=1, q>=0
    Returns (w, q_col).
    """
    m, n = A.shape

    # vars: q_0..q_{n-1}, w
    c = np.zeros(n + 1, dtype=float)
    c[-1] = 1.0  # minimize w

    # A q <= w  <=>  A q - w <= 0
    A_ub = np.zeros((m, n + 1), dtype=float)
    b_ub = np.zeros(m, dtype=float)
    for i in range(m):
        A_ub[i, :n] = A[i, :]
        A_ub[i, n] = -1.0

    # sum q = 1
    A_eq = np.zeros((1, n + 1), dtype=float)
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0], dtype=float)

    bounds = [(0.0, 1.0)] * n + [(-V_BOUND, V_BOUND)]

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs"
    )
    if not res.success:
        raise RuntimeError(f"Col LP failed: {res.message}")

    q = res.x[:n].astype(float)
    w = float(res.x[n])
    return w, q


# -----------------------------
# Core utilities
# -----------------------------
def core_of_state(s: State) -> Core:
    return (s.poss, s.x, s.down, s.to_first)


def state_from_core(c: Core, t: int) -> State:
    poss, x, down, to_first = c
    return State(poss=poss, x=x, down=down, to_first=to_first, t=t)


def poss_str(poss: int) -> str:
    return "US" if poss == US else "THEM"


# -----------------------------
# Transition caching (time-blind core cache)
# -----------------------------
def make_core_successor_cache(chart):
    """
    Cache (next_core, reward) branches for each (core, off_action, def_play).

    We call successors() with t=1 as a sentinel to generate a valid one-play transition,
    then drop time and keep only (core -> next_core) plus reward.
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

    OR-branches resolved by the CURRENT OFFENSE:
      - if poss==US: offense chooses max branch
      - if poss==THEM: offense chooses min branch (THEM chooses to minimize US)
    """
    poss, _, _, _ = core
    vals: List[float] = []

    for nxt_core, r in succ_core(core, off_action, def_play):
        if nxt_core is None:
            vals.append(r)
        else:
            if nxt_core not in V_next:
                raise KeyError(f"Missing V_next for successor core={nxt_core} from core={core}")
            vals.append(r + float(V_next[nxt_core]))

    if not vals:
        return 0.0

    return max(vals) if poss == US else min(vals)


def build_payoff_matrix_us(succ_core, core: Core, V_next: Dict[Core, float]) -> np.ndarray:
    """
    Build A_us where rows = offense actions (OFFENSE_ACTIONS), cols = defense plays (DEFENSE_PLAYS),
    entries are US payoffs.
    """
    m, n = len(OFFENSE_ACTIONS), len(DEFENSE_PLAYS)
    A = np.zeros((m, n), dtype=float)
    for ii, off_action in enumerate(OFFENSE_ACTIONS):
        for jj, d in enumerate(DEFENSE_PLAYS):
            A[ii, jj] = resolve_cell_us_value(succ_core, core, off_action, d, V_next)
    return A


# -----------------------------
# Reachability layers on CORE
# -----------------------------
def build_layers_core(
    succ_core,
    start_core: Core,
    T: int,
    *,
    verbose: bool = True,
) -> List[Set[Core]]:
    """
    layers[t] = set of cores reachable from start_core with exactly t plays remaining.
    Built backward:
      layers[T] = {start_core}
      layers[t-1] = one-step successors of layers[t]
    """
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
# Sampling helper: pick REACHABLE states with PREFERRED LARGE t
# -----------------------------
def pick_samples(
    layers: List[Set[Core]],
    T: int,
    label_and_predicates: List[Tuple[str, Callable[[Core], bool], int]],
    *,
    prefer_t_min: int = 25,
) -> List[Tuple[str, int, Core]]:
    """
    For each (label, predicate, k), pick up to k cores that satisfy predicate.
    Prefer larger t (earlier quarter).
    """
    picked: List[Tuple[str, int, Core]] = []
    used: Set[Tuple[int, Core]] = set()

    for label, pred, k in label_and_predicates:
        found = 0

        # Pass 1: prefer larger t
        for t in range(T, max(prefer_t_min, 1) - 1, -1):
            for core in layers[t]:
                if not pred(core):
                    continue
                key = (t, core)
                if key in used:
                    continue
                picked.append((label, t, core))
                used.add(key)
                found += 1
                if found >= k:
                    break
            if found >= k:
                break

        # Pass 2: allow smaller t if needed
        if found < k:
            for t in range(min(prefer_t_min - 1, T), 0, -1):
                for core in layers[t]:
                    if not pred(core):
                        continue
                    key = (t, core)
                    if key in used:
                        continue
                    picked.append((label, t, core))
                    used.add(key)
                    found += 1
                    if found >= k:
                        break
                if found >= k:
                    break

    return picked


# -----------------------------
# Pretty printing
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


def print_diagnostics(diag: Dict[str, float]):
    print("Diagnostics:")
    for k in ("ev", "v_us", "best_response_gap", "row_br_value", "col_br_value"):
        if k in diag:
            print(f"  {k:>18s}: {diag[k]:.9f}")


def print_state_header(label: str, s: State):
    print("\n==============================")
    print(label)
    print(f"State: poss={poss_str(s.poss)}, x={s.x}, down={s.down}, to_first={s.to_first}, t={s.t}")


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
        start_state = State(poss=US, x=20, down=1, to_first=10, t=T)
    start_core = core_of_state(start_state)

    if verbose:
        print(f"Building reachable layers up to T={T} ...")
    layers = build_layers_core(succ_core, start_core, T, verbose=verbose)

    # More variety in samples:
    preds: List[Tuple[str, Callable[[Core], bool], int]] = [
        ("4th & 1 (US) - red zone-ish",         lambda c: c[0] == US and c[2] == 4 and c[3] == 1 and c[1] >= 70, 2),
        ("4th & long (US) - punt territory",    lambda c: c[0] == US and c[2] == 4 and 8 <= c[3] <= 20 and c[1] <= 40, 2),
        ("4th & short (US) - borderline",       lambda c: c[0] == US and c[2] == 4 and 1 <= c[3] <= 3 and 40 <= c[1] <= 65, 2),
        ("3rd & long (US)",                     lambda c: c[0] == US and c[2] == 3 and 8 <= c[3] <= 20, 2),
        ("3rd & medium (US)",                   lambda c: c[0] == US and c[2] == 3 and 3 <= c[3] <= 7, 2),
        ("2nd & short (US)",                    lambda c: c[0] == US and c[2] == 2 and 1 <= c[3] <= 3, 2),
        ("Goal-to-go (US) x>=90",               lambda c: c[0] == US and c[1] >= 90 and c[3] <= 5, 2),
        ("Backed up (US) x<=10",                lambda c: c[0] == US and c[1] <= 10, 2),
        ("THEM offense (sample) x<=10",         lambda c: c[0] == THEM and c[1] <= 10, 2),
        ("THEM offense (sample) midfield-ish",  lambda c: c[0] == THEM and 40 <= c[1] <= 60, 2),
    ]
    samples = pick_samples(layers, T, preds, prefer_t_min=25)

    if verbose:
        print("\n\n=== SAMPLE PLAN (reachable, prefers larger t) ===")
        for lab, t, core in samples:
            poss, x, down, to_first = core
            print(f"- {lab:32s} t={t:2d}  (poss={poss_str(poss)}, x={x}, down={down}, to_first={to_first})")

    # Store value/policy only for start + samples
    want: Set[Tuple[int, Core]] = {(T, start_core)}
    for _lab, t, core in samples:
        want.add((t, core))

    # DP base: at 0 plays remaining, V = 0 for all cores in layer[0]
    V_prev: Dict[Core, float] = {c: 0.0 for c in layers[0]}

    # Stored outputs:
    # stored[(t,core)] = {
    #   "v_us": float,
    #   "off_who": "US"/"THEM", "p_off": np.ndarray,
    #   "def_who": "US"/"THEM", "q_def": np.ndarray,
    #   "diag": { ... }
    # }
    stored: Dict[Tuple[int, Core], Dict[str, Any]] = {}

    for t in range(1, T + 1):
        if verbose:
            print(f"\nSolving layer t={t}/{T} (|states|={len(layers[t])}) ...")

        V_curr: Dict[Core, float] = {}

        for core in layers[t]:
            A_us = build_payoff_matrix_us(succ_core, core, V_prev)
            poss, _, _, _ = core

            key = (t, core)
            need_store = (key in want)

            if poss == US:
                # US offense = ROW maximizer on A_us; THEM defense = COL minimizer on A_us
                v, p_us = solve_row_strategy(A_us)
                V_curr[core] = float(v)

                if need_store:
                    _w, q_them_def = solve_col_strategy(A_us)

                    # Nash / saddle diagnostics (row=max, col=min)
                    Aq = A_us @ q_them_def
                    pA = p_us @ A_us
                    row_br = float(np.max(Aq))
                    col_br = float(np.min(pA))
                    ev = float(p_us @ (A_us @ q_them_def))
                    gap = float(row_br - col_br)

                    stored[key] = {
                        "v_us": float(v),
                        "off_who": "US",
                        "p_off": p_us,
                        "def_who": "THEM",
                        "q_def": q_them_def,
                        "diag": {
                            "v_us": float(v),
                            "ev": ev,
                            "row_br_value": row_br,
                            "col_br_value": col_br,
                            "best_response_gap": gap,
                        }
                    }

            else:
                # THEM offense wants to MINIMIZE US; US defense wants to MAXIMIZE US.
                # Encode as ROW-max game on B = -A_us:
                #   THEM maximizing B == minimizing A_us
                #   COL minimizing B == maximizing A_us  (US defense)
                B = -A_us
                vB, p_them = solve_row_strategy(B)
                v_us = float(-vB)
                V_curr[core] = v_us

                if need_store:
                    _wB, q_us_def = solve_col_strategy(B)

                    # Diagnostics in US terms for the (row=min, col=max) game on A_us:
                    # THEM best response to q minimizes US: min_i (A q)_i
                    # US best response to p maximizes US: max_j (p^T A)_j
                    Aq = A_us @ q_us_def
                    pA = p_them @ A_us
                    row_br_min = float(np.min(Aq))
                    col_br_max = float(np.max(pA))
                    ev = float(p_them @ (A_us @ q_us_def))
                    gap = float(col_br_max - row_br_min)

                    stored[key] = {
                        "v_us": v_us,
                        "off_who": "THEM",
                        "p_off": p_them,
                        "def_who": "US",
                        "q_def": q_us_def,
                        "diag": {
                            "v_us": v_us,
                            "ev": ev,
                            "row_br_value": row_br_min,
                            "col_br_value": col_br_max,
                            "best_response_gap": gap,
                        }
                    }

        V_prev = V_curr

    return start_state, samples, stored


# -----------------------------
# Script entry
# -----------------------------
if __name__ == "__main__":
    T = 60
    s0, samples, stored = solve_quarter(
        T=T,
        csv_path="Football Strategy Pro Style.csv",
        verbose=True
    )

    # Print START state (t=T)
    start_core = core_of_state(s0)
    key0 = (T, start_core)
    rec0 = stored[key0]

    print("\n" + "=" * 30)
    print("START (s0)")
    print_state_header("START (s0)", s0)
    print("V_us =", rec0["v_us"])
    print_offense_mix(rec0["p_off"], header=f"Policy for {rec0['off_who']} offense (includes PUNT):")
    print_defense_mix(rec0["q_def"], header=f"Policy for {rec0['def_who']} defense (A..J):")
    print_diagnostics(rec0["diag"])

    print("\n\n=== OTHER REACHABLE SITUATIONS (with meaningful t) ===")
    for label, t, core in samples:
        s = state_from_core(core, t=t)
        key = (t, core)
        rec = stored[key]

        print_state_header(label, s)
        print("V_us =", rec["v_us"])
        print_offense_mix(rec["p_off"], header=f"Policy for {rec['off_who']} offense (includes PUNT):")
        print_defense_mix(rec["q_def"], header=f"Policy for {rec['def_who']} defense (A..J):")
        print_diagnostics(rec["diag"])
