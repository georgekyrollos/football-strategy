"""
Full 4-quarter backward induction solver for Football Strategy.

Solve order (high tau → low tau within each quarter, Q4 → Q1):

  1. Build all reachable state keys via forward BFS (reachability.py).
  2. Seed terminal values: (q=4, tau=0) states → terminal_utility(delta).
  3. Q4: tau = 1..60 in increasing order.
  4. Q3/tau=0 boundary: same core as Q4/tau=60.
  5. Q3: tau = 1..60.
  6. Halftime boundary (Q2/tau=0 → Q3 kickoff).
  7. Q2: tau = 1..60.
  8. Q1/tau=0 boundary: same core as Q2/tau=60.
  9. Q1: tau = 1..60.

Returns V: Dict[tuple, float] — all reachable state keys → equilibrium value.

Parallel layer solve
--------------------
Each tau layer is embarrassingly parallel: states at the same (q, tau) look
up V only at already-solved lower-tau entries and never conflict with each
other.  Set n_workers > 1 to enable fork-based multiprocessing.

Implementation uses `multiprocessing.get_context('fork')` so that the large
V dict is inherited copy-on-write by workers (zero pickling cost for V).
Workers only write back a (key, float) pair — the parent collects and updates
V after each layer.  A new Pool is created per tau layer so workers always see
the latest V snapshot.  Pool-creation overhead is ~30–80 ms per layer on
macOS; this is dominated by LP solve time once layers are large (>500 states).
"""

from __future__ import annotations

import multiprocessing
import os
import time
from typing import Dict, List, Optional, Set, Tuple

from football_strategy.constants import TICKS_PER_QUARTER
from football_strategy.value_store import ValueStore
from football_strategy.matrix_game import (
    build_scrimmage_payoff_matrix,
    kickoff_state_value,
    safety_kick_state_value,
    solve_row_strategy,
)
from football_strategy.reachability import (
    build_reachable_states,
    kickoff_key,
    safety_kick_key,
    scrimmage_key,
)
from football_strategy.states import terminal_utility


# ---------------------------------------------------------------------------
# Fork-worker globals — set in parent immediately before Pool() is called.
# Workers inherit these via copy-on-write; no pickling of large objects.
# ---------------------------------------------------------------------------

_WORKER_SC: Optional[Dict] = None       # scrimmage chart
_WORKER_PC: Optional[Dict] = None       # punt chart
_WORKER_V:  Optional[ValueStore] = None # value table (read-only in workers)


def _solve_key_worker(key: tuple) -> Tuple[tuple, float]:
    """Worker entry point: solve one state key using fork-inherited globals."""
    q, tau = key[1], key[2]
    return key, _solve_key(_WORKER_SC, _WORKER_PC, key, q, tau, _WORKER_V)


def _store(V, key: tuple, value: float) -> None:
    """Write-back helper: typed accessor for ValueStore, plain setitem for dict."""
    if isinstance(V, ValueStore):
        phase = key[0]
        if phase == "s":
            _, q, tau, x, down, tf, delta, h = key
            V.set_scrimmage(q, tau, x, down, tf, delta, h, value)
        elif phase == "k":
            _, q, tau, delta, h = key
            V.set_kickoff(q, tau, delta, h, value)
        else:  # "sk"
            _, q, tau, delta, h = key
            V.set_safety(q, tau, delta, h, value)
    else:
        V[key] = value


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_full_game(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    start_key: tuple,
    *,
    verbose: bool = False,
    n_workers: int = 1,
    parallel_min_layer: int = 200,
    use_value_store: bool = True,
    save_path: Optional[str] = None,
) -> object:
    """Run 4-quarter backward induction.

    Parameters
    ----------
    scrimmage_chart     : output of chart_parser.load_scrimmage_chart()
    punt_chart          : output of chart_parser.load_punt_chart()
    start_key           : a KickoffKey for the opening kickoff, e.g.
                          kickoff_key(q=1, tau=60, delta=0, h=1)
    verbose             : if True, print per-layer progress
    n_workers           : number of parallel workers (1 = serial).
                          Uses fork-based multiprocessing; requires a
                          POSIX platform.  On macOS with scipy/numpy this
                          is safe for pure-compute workers.
    parallel_min_layer  : only use the pool when a layer has at least this
                          many states (avoids fork overhead on tiny layers).
    """
    t_total = time.time()

    if verbose:
        print(f"Building reachable states (forward BFS, workers={n_workers})...")

    layers = build_reachable_states(scrimmage_chart, punt_chart, start_key, verbose=verbose)

    if verbose:
        total_states = sum(len(s) for s in layers.values())
        print(f"Reachable states: {total_states:,} across {len(layers)} (q,tau) layers")

    V = ValueStore() if use_value_store else {}

    # --- Seed Q4 terminal boundary ---
    for key in layers.get((4, 0), set()):
        _store(V, key, _terminal_value_for_key(key))

    # --- Q4 ---
    _solve_quarter(scrimmage_chart, punt_chart, q=4, layers=layers, V=V,
                   verbose=verbose, n_workers=n_workers, parallel_min_layer=parallel_min_layer)

    # --- Q3/tau=0 = Q4/tau=60 ---
    _copy_quarter_boundary(from_q=4, to_q=3, layers=layers, V=V)

    # --- Q3 ---
    _solve_quarter(scrimmage_chart, punt_chart, q=3, layers=layers, V=V,
                   verbose=verbose, n_workers=n_workers, parallel_min_layer=parallel_min_layer)

    # --- Halftime boundary ---
    _resolve_halftime_boundary(layers=layers, V=V, verbose=verbose)

    # --- Q2 ---
    _solve_quarter(scrimmage_chart, punt_chart, q=2, layers=layers, V=V,
                   verbose=verbose, n_workers=n_workers, parallel_min_layer=parallel_min_layer)

    # --- Q1/tau=0 = Q2/tau=60 ---
    _copy_quarter_boundary(from_q=2, to_q=1, layers=layers, V=V)

    # --- Q1 ---
    _solve_quarter(scrimmage_chart, punt_chart, q=1, layers=layers, V=V,
                   verbose=verbose, n_workers=n_workers, parallel_min_layer=parallel_min_layer)

    elapsed = time.time() - t_total
    if verbose:
        try:
            v_start = V[start_key]
            v_str = f"  V(start)={v_start:.8f}"
        except Exception:
            v_str = ""
        print(f"Solve complete: {len(V):,} states, total wall time {elapsed:.1f}s{v_str}")

    if save_path is not None and isinstance(V, ValueStore):
        V.save(save_path)
        if verbose:
            actual = save_path if save_path.endswith(".npz") else save_path + ".npz"
            print(f"  on-disk size: {os.path.getsize(actual) / 1e6:.0f} MB")

    return V


# ---------------------------------------------------------------------------
# Quarter solver (serial + parallel)
# ---------------------------------------------------------------------------

def _solve_quarter(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    q: int,
    layers: Dict,
    V: ValueStore,
    verbose: bool,
    n_workers: int,
    parallel_min_layer: int,
) -> None:
    """Solve all states in quarter q from tau=1 to tau=60.

    For each tau layer the inner loop is:

        V[key] = _solve_key(sc, pc, key, q, tau, V)

    When n_workers > 1 and the layer is large enough, this is parallelised by
    forking n_workers processes that each inherit a snapshot of V via
    copy-on-write.  A fresh pool is created for every tau layer so workers
    always see the fully updated V from all previous layers.
    """
    global _WORKER_SC, _WORKER_PC, _WORKER_V

    for tau in range(1, TICKS_PER_QUARTER + 1):
        keys = list(layers.get((q, tau), set()))
        if not keys:
            continue

        t0 = time.time()
        use_parallel = (n_workers > 1) and (len(keys) >= parallel_min_layer)

        if use_parallel:
            # Snapshot globals for forked workers
            _WORKER_SC = scrimmage_chart
            _WORKER_PC = punt_chart
            _WORKER_V  = V

            ctx = multiprocessing.get_context("fork")
            chunk = max(1, len(keys) // (n_workers * 8))
            with ctx.Pool(n_workers) as pool:
                results = pool.map(_solve_key_worker, keys, chunksize=chunk)

            for key, val in results:
                _store(V, key, val)
        else:
            for key in keys:
                _store(V, key, _solve_key(scrimmage_chart, punt_chart, key, q, tau, V))

        if verbose:
            n_s  = sum(1 for k in keys if k[0] == "s")
            n_ko = sum(1 for k in keys if k[0] == "k")
            n_sk = sum(1 for k in keys if k[0] == "sk")
            mode = f"{n_workers}×par" if use_parallel else "serial"
            elapsed = time.time() - t0
            print(
                f"  Q{q} tau={tau:>2d}: {len(keys):>8,} states"
                f"  (scrim={n_s:,}, ko={n_ko}, sk={n_sk})"
                f"  [{mode}, {elapsed:.1f}s]"
            )


# ---------------------------------------------------------------------------
# Per-state dispatcher
# ---------------------------------------------------------------------------

def _solve_key(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    key: tuple,
    q: int,
    tau: int,
    V: ValueStore,
) -> float:
    phase = key[0]

    if phase == "s":
        _, _q, _tau, x, down, to_first, delta, h = key
        A, _actions = build_scrimmage_payoff_matrix(
            scrimmage_chart, punt_chart,
            q, tau, x, down, to_first, delta, h, V,
        )
        v, _p = solve_row_strategy(A)
        return float(v)

    if phase == "k":
        _, _q, _tau, delta, h = key
        return kickoff_state_value(q, tau, delta, h, V)

    if phase == "sk":
        _, _q, _tau, delta, h = key
        return safety_kick_state_value(q, tau, delta, h, V)

    raise ValueError(f"Unknown state key phase: {phase!r}")


# ---------------------------------------------------------------------------
# Quarter boundary helpers
# ---------------------------------------------------------------------------

def _copy_quarter_boundary(
    from_q: int,
    to_q: int,
    layers: Dict,
    V: ValueStore,
) -> None:
    """Map (to_q, tau=0) states → value of (from_q, tau=60) state with same core."""
    tau_src = TICKS_PER_QUARTER
    for key in layers.get((to_q, 0), set()):
        src_key = _replace_qt(key, from_q, tau_src)
        if src_key in V:
            _store(V, key, V[src_key])
        else:
            _store(V, key, _terminal_value_for_key(key))


def _resolve_halftime_boundary(
    layers: Dict,
    V: ValueStore,
    verbose: bool,
) -> None:
    """Resolve Q2/tau=0 → value of the appropriate Q3 kickoff state.

    h encodes who receives the Q3 kickoff (from current offense's perspective):
      h = +1: current offense receives Q3 kickoff → defense kicks off in Q3
              Q3 KickoffKey has kicker = defense → delta flips: ko_delta = -delta
      h = -1: current defense receives Q3 kickoff → offense kicks off in Q3
              Q3 KickoffKey has kicker = offense → ko_delta = delta

    After halftime h resets to 0 for Q3/Q4.
    """
    tau_q3 = TICKS_PER_QUARTER

    for key in layers.get((2, 0), set()):
        phase = key[0]

        if phase == "s":
            _, _q, _tau, x, down, to_first, delta, h = key
        elif phase in ("k", "sk"):
            _, _q, _tau, delta, h = key
        else:
            continue

        if h == 0:
            continue

        if h == +1:
            ko_key = kickoff_key(3, tau_q3, -delta, 0)
            ko_val = V.get(ko_key)
            _store(V, key, -float(ko_val) if ko_val is not None else float(terminal_utility(delta)))
        else:  # h == -1
            ko_key = kickoff_key(3, tau_q3, delta, 0)
            ko_val = V.get(ko_key)
            _store(V, key, float(ko_val) if ko_val is not None else float(terminal_utility(delta)))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _terminal_value_for_key(key: tuple) -> float:
    phase = key[0]
    if phase == "s":
        delta = key[6]
    elif phase in ("k", "sk"):
        delta = key[3]
    else:
        delta = 0
    return float(terminal_utility(delta))


def _replace_qt(key: tuple, q: int, tau: int) -> tuple:
    lst = list(key)
    lst[1] = q
    lst[2] = tau
    return tuple(lst)


# ---------------------------------------------------------------------------
# Timeout-aware solve
# ---------------------------------------------------------------------------
# TOs are NOT added to the state tuple.  Instead we maintain 16 separate
# ValueStore instances (one per (to_off, to_def) ∈ {0..3}²) and solve them
# in order of increasing to_off + to_def.  The (0,0) base case is seeded
# from the no-timeout solve.  For each new (to_off, to_def), the solve is a
# full backward induction pass using the TO-aware cell computation: after
# each play outcome (ticks=k, next_core), the post-play sub-game determines
# which team (if any) calls a timeout (see matrix_game._solve_to_subgame).
# ---------------------------------------------------------------------------

# Fork-worker globals for the TO solve pass
_TO_WORKER_SC: Optional[Dict] = None
_TO_WORKER_PC: Optional[Dict] = None
_TO_WORKER_V:  Optional[ValueStore] = None   # V currently being built
_TO_WORKER_STORE = None                       # ToValueStore with predecessors
_TO_WORKER_OFF: int = 0
_TO_WORKER_DEF: int = 0


def _run_combo_worker(args: tuple) -> str:
    """Top-level worker for combo-level parallelism (fork context).

    Runs one full (to_off, to_def) backward induction pass, saves the result
    to disk, and returns the save path.  All heavy state (to_store, layers,
    charts) is inherited copy-on-write from the parent fork.
    """
    sc, pc, to_off, to_def, to_store, layers, save_path, n_workers_inner = args
    V_new = _solve_one_to_combo(
        sc, pc, to_off, to_def, to_store, layers,
        verbose=False, n_workers=n_workers_inner,
    )
    V_new.save(save_path)
    return save_path


def _solve_key_to_worker(key: tuple) -> Tuple[tuple, float]:
    """Worker entry point for the TO solve pass (fork-inherited globals)."""
    q, tau = key[1], key[2]
    return key, _solve_key_to(
        _TO_WORKER_SC, _TO_WORKER_PC, key, q, tau,
        _TO_WORKER_V, _TO_WORKER_STORE, _TO_WORKER_OFF, _TO_WORKER_DEF,
    )


def _solve_key_to(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    key: tuple,
    q: int,
    tau: int,
    V_new: ValueStore,
    to_store: object,   # ToValueStore
    to_off: int,
    to_def: int,
) -> float:
    """Per-state dispatcher for the TO-aware solve pass.

    Mirrors _solve_key but uses build_scrimmage_payoff_matrix_to,
    kickoff_state_value_to, and safety_kick_state_value_to from matrix_game.
    """
    from football_strategy.matrix_game import (
        build_scrimmage_payoff_matrix_to,
        kickoff_state_value_to,
        safety_kick_state_value_to,
    )
    phase = key[0]

    if phase == "s":
        _, _q, _tau, x, down, to_first, delta, h = key
        A, _actions = build_scrimmage_payoff_matrix_to(
            scrimmage_chart, punt_chart,
            q, tau, x, down, to_first, delta, h,
            to_store, to_off, to_def,
        )
        v, _p = solve_row_strategy(A)
        return float(v)

    if phase == "k":
        _, _q, _tau, delta, h = key
        return kickoff_state_value_to(q, tau, delta, h, to_store, to_off, to_def)

    if phase == "sk":
        _, _q, _tau, delta, h = key
        return safety_kick_state_value_to(q, tau, delta, h, to_store, to_off, to_def)

    raise ValueError(f"Unknown state key phase: {phase!r}")


def _solve_quarter_to(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    q: int,
    layers: Dict,
    V_new: ValueStore,
    to_store: object,   # ToValueStore
    to_off: int,
    to_def: int,
    verbose: bool,
    n_workers: int,
    parallel_min_layer: int,
    pbar=None,          # optional tqdm bar; updated once per tau layer
) -> None:
    """Quarter solve for one (to_off, to_def) combo — writes into V_new using to_store for lookups."""
    global _TO_WORKER_SC, _TO_WORKER_PC, _TO_WORKER_V
    global _TO_WORKER_STORE, _TO_WORKER_OFF, _TO_WORKER_DEF

    q_states = 0
    for tau in range(1, TICKS_PER_QUARTER + 1):
        keys = list(layers.get((q, tau), set()))
        if pbar is not None:
            pbar.update(1)
        if not keys:
            continue

        t0 = time.time()
        use_parallel = (n_workers > 1) and (len(keys) >= parallel_min_layer)

        if use_parallel:
            _TO_WORKER_SC    = scrimmage_chart
            _TO_WORKER_PC    = punt_chart
            _TO_WORKER_V     = V_new
            _TO_WORKER_STORE = to_store
            _TO_WORKER_OFF   = to_off
            _TO_WORKER_DEF   = to_def

            ctx   = multiprocessing.get_context("fork")
            chunk = max(1, len(keys) // (n_workers * 8))
            with ctx.Pool(n_workers) as pool:
                results = pool.map(_solve_key_to_worker, keys, chunksize=chunk)

            for key, val in results:
                _store(V_new, key, val)
        else:
            for key in keys:
                _store(
                    V_new, key,
                    _solve_key_to(
                        scrimmage_chart, punt_chart, key, q, tau,
                        V_new, to_store, to_off, to_def,
                    ),
                )

        q_states += len(keys)
        if verbose:
            n_s  = sum(1 for k in keys if k[0] == "s")
            n_ko = sum(1 for k in keys if k[0] == "k")
            n_sk = sum(1 for k in keys if k[0] == "sk")
            mode = f"{n_workers}×par" if use_parallel else "serial"
            elapsed = time.time() - t0
            print(
                f"  Q{q} tau={tau:>2d}: {len(keys):>8,} states"
                f"  (scrim={n_s:,}, ko={n_ko}, sk={n_sk})"
                f"  [{mode}, {elapsed:.1f}s]"
            )


def _solve_one_to_combo(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    to_off: int,
    to_def: int,
    to_store: object,   # ToValueStore — contains all predecessors
    layers: Dict,       # reachable-states layers (shared across all TO combos)
    *,
    verbose: bool = False,
    n_workers: int = 1,
    parallel_min_layer: int = 200,
    pbar=None,          # optional tqdm bar for tau-layer progress
) -> ValueStore:
    """Run one full backward induction pass for (to_off, to_def).

    All lookups use to_store (which must already have (to_off-1, to_def),
    (to_off, to_def-1), (to_off-1, to_def-1), and (to_off, to_def) for
    lower-tau entries — the last is V_new itself, written as we go).

    Returns a new ValueStore for this (to_off, to_def).
    """
    V_new = ValueStore()

    # For the 2×2 sub-game lookups at lower tau, we need to_store.get(to_off, to_def)
    # to return V_new (the table being built at this TO combo) for the "neither calls"
    # branch.  Register V_new into to_store under its key so lookup_val can find it.
    # This is safe because lookups always go to LOWER tau than the state being solved.
    to_store.add(to_off, to_def, V_new)

    # --- Seed Q4 terminal boundary ---
    for key in layers.get((4, 0), set()):
        _store(V_new, key, _terminal_value_for_key(key))

    # --- Q4 ---
    _solve_quarter_to(
        scrimmage_chart, punt_chart, q=4, layers=layers, V_new=V_new,
        to_store=to_store, to_off=to_off, to_def=to_def,
        verbose=verbose, n_workers=n_workers, parallel_min_layer=parallel_min_layer,
        pbar=pbar,
    )

    # --- Q3/tau=0 = Q4/tau=60 ---
    _copy_quarter_boundary(from_q=4, to_q=3, layers=layers, V=V_new)

    # --- Q3 ---
    _solve_quarter_to(
        scrimmage_chart, punt_chart, q=3, layers=layers, V_new=V_new,
        to_store=to_store, to_off=to_off, to_def=to_def,
        verbose=verbose, n_workers=n_workers, parallel_min_layer=parallel_min_layer,
        pbar=pbar,
    )

    # --- Halftime boundary ---
    _resolve_halftime_boundary(layers=layers, V=V_new, verbose=False)

    # --- Q2 ---
    _solve_quarter_to(
        scrimmage_chart, punt_chart, q=2, layers=layers, V_new=V_new,
        to_store=to_store, to_off=to_off, to_def=to_def,
        verbose=verbose, n_workers=n_workers, parallel_min_layer=parallel_min_layer,
        pbar=pbar,
    )

    # --- Q1/tau=0 = Q2/tau=60 ---
    _copy_quarter_boundary(from_q=2, to_q=1, layers=layers, V=V_new)

    # --- Q1 ---
    _solve_quarter_to(
        scrimmage_chart, punt_chart, q=1, layers=layers, V_new=V_new,
        to_store=to_store, to_off=to_off, to_def=to_def,
        verbose=verbose, n_workers=n_workers, parallel_min_layer=parallel_min_layer,
        pbar=pbar,
    )

    return V_new


def solve_with_timeouts(
    scrimmage_chart: Dict,
    punt_chart: Dict,
    start_key: tuple,
    *,
    base_path: str = "q4_full_v.npz",
    out_prefix: str = "q4_to_v",
    verbose: bool = False,
    use_tqdm: bool = True,
    n_workers: int = 1,
    parallel_min_layer: int = 200,
) -> object:
    """Solve all 16 (to_off, to_def) timeout-count combinations.

    Parameters
    ----------
    scrimmage_chart : output of chart_parser.load_scrimmage_chart()
    punt_chart      : output of chart_parser.load_punt_chart()
    start_key       : opening kickoff key (same as solve_full_game)
    base_path       : path to the (0,0) base solve (no timeouts remaining)
    out_prefix      : prefix for output files; combo (a,b) → '<prefix>_to{a}{b}.npz'
    verbose         : if True, print per-layer progress
    n_workers       : parallel workers for each combo (fork-based)

    Returns
    -------
    ToValueStore with all 16 solved instances loaded.
    The game-relevant result is store.get(3, 3) — both teams start with 3 TOs.
    """
    from football_strategy.to_value_store import ToValueStore
    from football_strategy.constants import MAX_TOS

    t_wall = time.time()

    print("Building reachable-state layers (shared across all TO combos)...")
    print("  (this may take a few minutes for the full game BFS)")
    layers = build_reachable_states(scrimmage_chart, punt_chart, start_key, verbose=verbose)
    total_states = sum(len(s) for s in layers.values())
    print(f"  {total_states:,} reachable states across {len(layers)} (q,tau) layers\n")

    to_store = ToValueStore()

    # Load (0,0) base case (no timeouts remaining)
    print(f"Loading base solve from {base_path} ...")
    v00 = ValueStore.load(base_path)
    to_store.add(0, 0, v00)
    print(f"  {len(v00):,} solved slots loaded\n")

    # Group combos by total so we evict only after all combos at each total are done.
    # evict_stale(T) removes (a,b) with a+b+2 ≤ T — safe only once every combo at
    # total=T has been solved (earlier eviction would prematurely remove predecessors
    # still needed by sibling combos at the same total level).
    from itertools import groupby
    combos_by_total = {
        total: list(grp)
        for total, grp in groupby(
            ToValueStore.solve_order(MAX_TOS), key=lambda pair: pair[0] + pair[1]
        )
    }

    try:
        from tqdm import tqdm as _tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    n_combos = sum(len(v) for v in combos_by_total.values())
    # 4 quarters × 60 tau layers per combo
    tau_steps_per_combo = 4 * TICKS_PER_QUARTER

    outer_bar = (
        _tqdm(total=n_combos, desc="TO combos", unit="combo", position=0, leave=True)
        if use_tqdm and _have_tqdm else None
    )

    try:
        for total in sorted(combos_by_total):
            combos = combos_by_total[total]
            t0 = time.time()

            # Combos at the same total level are independent: they only read
            # from lower-total tables (already fully solved).  Run them in
            # parallel forked processes, splitting available workers evenly.
            n_parallel = min(len(combos), n_workers)
            workers_each = max(1, n_workers // n_parallel)

            if n_parallel > 1:
                if verbose:
                    print(f"\n--- total={total}: {len(combos)} combos in parallel "
                          f"({n_parallel} processes × {workers_each} workers) ---")

                tasks = [
                    (scrimmage_chart, punt_chart, to_off, to_def, to_store, layers,
                     f"{out_prefix}_to{to_off}{to_def}", workers_each)
                    for to_off, to_def in combos
                ]
                ctx = multiprocessing.get_context("fork")
                with ctx.Pool(n_parallel) as pool:
                    pool.map(_run_combo_worker, tasks)

                # Reload all results from disk into to_store
                for to_off, to_def in combos:
                    path = f"{out_prefix}_to{to_off}{to_def}.npz"
                    to_store.add(to_off, to_def, ValueStore.load(path))

                if outer_bar is not None:
                    outer_bar.update(len(combos))

            else:
                # Single combo (or n_workers=1): run sequentially with full workers
                for to_off, to_def in combos:
                    if outer_bar is not None:
                        outer_bar.set_description(f"combo ({to_off},{to_def}) [total={total}]")

                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"Solving (to_off={to_off}, to_def={to_def})  [total={total}]")
                        print(f"{'='*60}")

                    inner_bar = (
                        _tqdm(
                            total=tau_steps_per_combo,
                            desc=f"  ({to_off},{to_def}) τ-layers",
                            unit="τ", position=1, leave=False,
                        )
                        if use_tqdm and _have_tqdm else None
                    )
                    try:
                        V_new = _solve_one_to_combo(
                            scrimmage_chart, punt_chart,
                            to_off, to_def, to_store, layers,
                            verbose=verbose, n_workers=n_workers,
                            parallel_min_layer=parallel_min_layer,
                            pbar=inner_bar,
                        )
                    finally:
                        if inner_bar is not None:
                            inner_bar.close()

                    save_path_s = f"{out_prefix}_to{to_off}{to_def}"
                    V_new.save(save_path_s)

                    elapsed = time.time() - t0
                    if verbose:
                        print(f"  Saved {save_path_s}.npz  ({len(V_new):,} slots, {elapsed:.1f}s)")

                    if outer_bar is not None:
                        outer_bar.update(1)
                        outer_bar.set_postfix(saved=f"to{to_off}{to_def}", elapsed=f"{elapsed:.0f}s")

            if verbose:
                print(f"  total={total} done in {time.time()-t0:.1f}s")

            # Evict tables no longer needed as predecessors
            to_store.evict_stale(total)
    finally:
        if outer_bar is not None:
            outer_bar.close()

    elapsed_total = time.time() - t_wall
    print(f"\nTO solve complete — {elapsed_total/3600:.2f}h total")
    print(f"Output files: {out_prefix}_to{{a}}{{b}}.npz  (16 combos)")
    print(f"Primary result (3 TOs each): {out_prefix}_to33.npz")

    return to_store
