"""
ToValueStore — manages 16 ValueStore instances, one per (to_off, to_def) pair.

Timeout counts are not added to the state tuple; instead we keep one ValueStore
per (to_off, to_def) ∈ {0..3}² and solve them in order of increasing to_off+to_def.

  total=0: (0,0)   ← base case (no timeouts remaining)
  total=1: (1,0), (0,1)
  ...
  total=6: (3,3)   ← game-relevant result (both teams start with 3 TOs)

Peak RAM during solve: at most 3 ValueStore instances simultaneously
  (the current combo + its 2 immediate predecessors for the sub-game lookups).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from football_strategy.value_store import ValueStore
from football_strategy.constants import MAX_TOS


class ToValueStore:
    """Container for all solved (to_off, to_def) value tables.

    Each (to_off, to_def) pair maps to a separate ValueStore that holds
    V(q, tau, state_core) under the assumption that the current offense has
    to_off timeouts and the current defense has to_def timeouts remaining.

    Usage (solve loop):
        store = ToValueStore()
        store.add(0, 0, ValueStore.load("q4_full_v.npz"))   # base case (no TOs remaining)
        for to_off, to_def in solve_order(MAX_TOS):
            V_new = solve_one_combo(to_off, to_def, store)  # builds new ValueStore
            store.add(to_off, to_def, V_new)
            V_new.save(f"q4_to_v_to{to_off}{to_def}")
            store.evict_stale(to_off + to_def)              # free RAM
    """

    __slots__ = ("_stores",)

    def __init__(self) -> None:
        self._stores: Dict[Tuple[int, int], ValueStore] = {}

    # ------------------------------------------------------------------
    # Store management
    # ------------------------------------------------------------------

    def add(self, to_off: int, to_def: int, vs: ValueStore) -> None:
        """Register a solved ValueStore for this (to_off, to_def) combination."""
        self._stores[(to_off, to_def)] = vs

    def get(self, to_off: int, to_def: int) -> ValueStore:
        """Return the ValueStore for (to_off, to_def). Raises KeyError if missing."""
        key = (to_off, to_def)
        if key not in self._stores:
            raise KeyError(
                f"ToValueStore: ({to_off}, {to_def}) not loaded. "
                f"Available: {sorted(self._stores)}"
            )
        return self._stores[key]

    def has(self, to_off: int, to_def: int) -> bool:
        return (to_off, to_def) in self._stores

    def evict(self, to_off: int, to_def: int) -> None:
        """Release a ValueStore from RAM (it has already been saved to disk)."""
        self._stores.pop((to_off, to_def), None)

    def evict_stale(self, just_finished_total: int) -> None:
        """Evict (to_off, to_def) pairs that cannot be predecessors of any
        not-yet-solved combination.

        A pair (a, b) is needed as a predecessor only by combos (a+1, b),
        (a, b+1), (a+1, b+1) — all at total a+b+1 or a+b+2.  Once
        just_finished_total >= a+b+2, those combos are done and (a,b) is stale.
        Conservative rule: evict (a,b) when a+b+2 <= just_finished_total.
        """
        to_evict = [
            (a, b) for (a, b) in list(self._stores)
            if a + b + 2 <= just_finished_total
        ]
        for key in to_evict:
            self._stores.pop(key)
        if to_evict:
            freed = len(to_evict)
            print(f"[ToValueStore] Evicted {freed} store(s) after total={just_finished_total}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_all(self, prefix: str) -> None:
        """Save every loaded store to '<prefix>_to{a}{b}.npz'."""
        for (a, b), vs in self._stores.items():
            path = f"{prefix}_to{a}{b}"
            vs.save(path)

    @classmethod
    def load(cls, prefix: str, pairs: Optional[List[Tuple[int, int]]] = None) -> "ToValueStore":
        """Load stores from disk.

        Args:
            prefix: Common filename prefix (e.g. 'q4_to_v').
            pairs:  List of (to_off, to_def) pairs to load.
                    Defaults to all 16 combinations {0..3}².
        """
        if pairs is None:
            pairs = [(a, b) for a in range(MAX_TOS + 1) for b in range(MAX_TOS + 1)]
        tvs = cls()
        for a, b in pairs:
            path = f"{prefix}_to{a}{b}.npz"
            if os.path.exists(path):
                tvs.add(a, b, ValueStore.load(path))
            else:
                print(f"[ToValueStore] Warning: {path} not found, skipping ({a},{b})")
        return tvs

    # ------------------------------------------------------------------
    # Solve-order helper
    # ------------------------------------------------------------------

    @staticmethod
    def solve_order(max_tos: int = MAX_TOS) -> List[Tuple[int, int]]:
        """Return (to_off, to_def) pairs in increasing to_off + to_def order.

        (0, 0) is omitted — it is seeded externally from Stage 1.
        Within each total, offense-heavy combinations come first.
        """
        result = []
        for total in range(1, 2 * max_tos + 1):
            for to_off in range(min(total, max_tos), -1, -1):
                to_def = total - to_off
                if 0 <= to_def <= max_tos:
                    result.append((to_off, to_def))
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        keys = sorted(self._stores)
        return f"ToValueStore({len(keys)} stores loaded: {keys})"

    def loaded_pairs(self) -> List[Tuple[int, int]]:
        return sorted(self._stores)
