# solve_quarter.py
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from functools import lru_cache
from typing import Dict, Tuple, List, Optional, Set, Callable

from read_matrix import load_pro_style_chart_csv
from transition_quarter import (
    State, successors,
    OFFENSE_ACTIONS, DEFENSE_PLAYS,
    US, THEM, PUNT
)

Core = Tuple[int, int, int, int]  # (poss, x, down, to_first)


# -----------------------------
# LP: ONE solve per state (row player)
# -----------------------------
def solve_row_strategy(A: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Zero-sum matrix game: ROW maximizes, COL minimizes payoff A.
    Returns (v, p_row).
    """
    m, n = A.shape

    # variables: p_0..p_{m-1}, v
    # maximize v  <=> minimize -v
    c = np.zeros(m + 1, dtype=float)
    c[-1] = -1.0

    # constraints: A^T p >= v  <=>  -A^T p + v <= 0
    A_ub = np.zeros((n, m + 1), dtype=float)
    b_ub = np.zeros(n, dtype=float)
    for j in range(n):
        A_ub[j, :m] = -A[:, j]
        A_ub[j, m] = 1.0

    # sum p = 1
    A_eq = np.zeros((1, m + 1), dtype=float)
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0], dtype=float)

    bounds = [(0.0, None)] * m + [(-1e6, 1e6)]

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
    Cache the (next_core, reward) branches for each (core, action, def_play).
    We call successors with t=1 just to get a valid 'one play happens' transition,
    then drop the time component and keep only core transitions.
    """
    @lru_cache(maxsize=None)
    def succ_core(core: Core, off_action: int, d: str) -> Tuple[Tuple[Optional[Core], float], ...]:
        s = state_from_core(core, t=1)  # IMPORTANT: just "one play available" sentinel
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

    OR-branches are resolved by the CURRENT OFFENSE:
      - if poss==US: offense chooses max branch
      - if poss==THEM: offense chooses min branch (THEM chooses to minimize US)
    """
    poss, _, _, _ = core
    vals: List[float] = []

    for nxt_core, r in succ_core(core, off_action, def_play):
        if nxt_core is None:
            vals.append(r)
        else:
            # This *should* always exist if layers are built correctly.
            # If it doesn't, it's a bug in layer construction; fail loudly.
            if nxt_core not in V_next:
                raise KeyError(f"Missing V_next for successor core={nxt_core} from core={core}")
            vals.append(r + float(V_next[nxt_core]))

    if not vals:
        return 0.0

    return max(vals) if poss == US else min(vals)


def build_payoff_matrix_us(succ_core, core: Core, V_next: Dict[Core, float]) -> np.ndarray:
    """
    Build A_us where rows = offense actions (1..20,PUNT), cols = defense plays (A..J),
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
# Sampling helper: pick REACHABLE states with PREFERRED LARGE t
# -----------------------------
def pick_samples(
    layers: List[Set[Core]],
    T: int,
    label_and_predicates: List[Tuple[str, Callable[[Core], bool], int]],
    *,
    prefer_t_min: int = 20,
) -> List[Tuple[str, int, Core]]:
    """
    For each (label, predicate, k), pick up to k cores that satisfy predicate.
    We prefer samples with *large t* (early quarter) and try to avoid duplicates.
    """
    picked: List[Tuple[str, int, Core]] = []
    used: Set[Tuple[int, int, int, int, int]] = set()  # (t, poss, x, down, to_first)

    for label, pred, k in label_and_predicates:
        found = 0

        # Pass 1: try t from T down to prefer_t_min
        for t in range(T, max(prefer_t_min, 1) - 1, -1):
            for core in layers[t]:
                if not pred(core):
                    continue
                key = (t, *core)
                if key in used:
                    continue
                picked.append((label, t, core))
                used.add(key)
                found += 1
                if found >= k:
                    break
            if found >= k:
                break

        # Pass 2 (fallback): if still not enough, allow smaller t
        if found < k:
            for t in range(min(prefer_t_min - 1, T), 0, -1):
                for core in layers[t]:
                    if not pred(core):
                        continue
                    key = (t, *core)
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


def print_state_block(label: str, s: State, v: float, p: np.ndarray, *, who: str):
    print("\n==============================")
    print(label)
    print(f"State: poss={poss_str(s.poss)}, x={s.x}, down={s.down}, to_first={s.to_first}, t={s.t}")
    print("V_us =", float(v))
    print(f"Policy for {who} offense (includes PUNT):")
    print_offense_mix(p)


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
    layers = build_layers_core(chart, start_core, T, verbose=verbose)

    # Choose “situational” samples guaranteed reachable (and prefer large t)
    preds: List[Tuple[str, Callable[[Core], bool], int]] = [
        ("4th & 1 (US) - red zone-ish", lambda c: c[0] == US and c[2] == 4 and c[3] == 1 and c[1] >= 70, 2),
        ("4th & long (US) - punt territory", lambda c: c[0] == US and c[2] == 4 and 8 <= c[3] <= 20 and c[1] <= 40, 2),
        ("3rd & long (US)", lambda c: c[0] == US and c[2] == 3 and 8 <= c[3] <= 20, 2),
        ("Backed up (US) x<=10", lambda c: c[0] == US and c[1] <= 10, 2),
        ("Midfield 1st & 10 (US)", lambda c: c[0] == US and c[2] == 1 and c[3] == 10 and 45 <= c[1] <= 55, 2),
        ("THEM offense (sample)", lambda c: c[0] == THEM and c[2] in (1, 2, 3, 4), 2),
    ]
    samples = pick_samples(layers, T, preds, prefer_t_min=25)

    if verbose:
        print("\n\n=== SAMPLE PLAN (reachable, prefers larger t) ===")
        for lab, t, core in samples:
            poss, x, down, to_first = core
            print(f"- {lab:30s}  t={t:2d}  (poss={poss_str(poss)}, x={x}, down={down}, to_first={to_first})")

    # We will store value+policy ONLY for:
    # - start state at (T, start_core)
    # - sampled states at their specific (t, core)
    want: Set[Tuple[int, Core]] = {(T, start_core)}
    for _lab, t, core in samples:
        want.add((t, core))

    # Base layer: t=0 values are 0 for all cores in layer[0]
    V_prev: Dict[Core, float] = {c: 0.0 for c in layers[0]}

    # Stored outputs for requested (t, core)
    stored_value: Dict[Tuple[int, Core], float] = {}
    stored_policy: Dict[Tuple[int, Core], Tuple[str, np.ndarray]] = {}  # (who, p)

    # DP forward in "remaining plays" t
    for t in range(1, T + 1):
        if verbose:
            print(f"\nSolving layer t={t}/{T} (|states|={len(layers[t])}) ...")

        V_curr: Dict[Core, float] = {}

        for core in layers[t]:
            A_us = build_payoff_matrix_us(succ_core, core, V_prev)
            poss, _, _, _ = core

            if poss == US:
                v, p_us = solve_row_strategy(A_us)
                V_curr[core] = float(v)

                if (t, core) in want:
                    stored_value[(t, core)] = float(v)
                    stored_policy[(t, core)] = ("US", p_us)

            else:
                # THEM is offense and wants to minimize US value:
                # Solve max game on B=-A_us for THEM, then flip sign back.
                B = -A_us
                vB, p_them = solve_row_strategy(B)
                V_curr[core] = float(-vB)

                if (t, core) in want:
                    stored_value[(t, core)] = float(-vB)
                    stored_policy[(t, core)] = ("THEM", p_them)

        V_prev = V_curr

    return start_state, samples, stored_value, stored_policy


# -----------------------------
# Script entry
# -----------------------------
if __name__ == "__main__":
    T = 60
    s0, samples, stored_value, stored_policy = solve_quarter(
        T=T,
        csv_path="Football Strategy Pro Style.csv",
        verbose=True
    )

    # Print START state (t=T)
    start_core = (s0.poss, s0.x, s0.down, s0.to_first)
    key0 = (T, start_core)
    who0, p0 = stored_policy[key0]
    v0 = stored_value[key0]
    print("\n" + "=" * 30)
    print("START (s0)")
    print_state_block("START (s0)", s0, v0, p0, who=who0)

    print("\n\n=== OTHER REACHABLE SITUATIONS (with meaningful t) ===")
    for label, t, core in samples:
        s = state_from_core(core, t=t)
        key = (t, core)
        who, p = stored_policy[key]
        v = stored_value[key]
        print_state_block(label, s, v, p, who=who)
