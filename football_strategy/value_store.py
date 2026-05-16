"""
Dense NumPy-backed value store replacing the Dict[tuple, float] V table.

Quarter-aware layout
---------------------
Q3/Q4 scrimmage   V_s34[q-3, tau, x-1, down-1, to_first-1, delta+OFFSET]
  shape: (2, 61, 99, 4, TO_FIRST_MAX, 2*DELTA_RANGE+1)   — no h dimension (h=0 always)

Q1/Q2 scrimmage   V_s12[q-1, tau, x-1, down-1, to_first-1, delta+OFFSET, h_idx]
  shape: (2, 61, 99, 4, TO_FIRST_MAX, 2*DELTA_RANGE+1, 2)  — h∈{-1,+1}, h_idx=(h+1)>>1

Q3/Q4 kickoff     V_k34[q-3, tau, delta+OFFSET]         shape: (2, 61, N_DELTA)
Q1/Q2 kickoff     V_k12[q-1, tau, delta+OFFSET, h_idx]  shape: (2, 61, N_DELTA, 2)
(same shapes for safety_kick)

Memory at float64:
  s34  ≈ 823 MB   s12 ≈ 1.646 GB   ko/sk arrays ≈ negligible
  Total ≈ 2.47 GB  (vs 4.94 GB for a monolithic q×h array)

Sentinel value for "not yet solved": NaN.
The lookup raises KeyError if a NaN is read (bug detection).
"""

from __future__ import annotations

import numpy as np

from football_strategy.constants import (
    DELTA_MAX,
    DELTA_MIN,
    TICKS_PER_QUARTER,
    TO_FIRST_MAX,
)

# ---------------------------------------------------------------------------
# Dimension sizes
# ---------------------------------------------------------------------------

_N_TAU        = TICKS_PER_QUARTER + 1   # 0..60
_N_X          = 99                       # x ∈ [1, 99]
_N_DOWN       = 4                        # down ∈ [1, 4]
_N_TF         = TO_FIRST_MAX             # to_first ∈ [1, 30]
_DELTA_OFFSET = -DELTA_MIN               # 35 → index = delta + 35
_N_DELTA      = DELTA_MAX - DELTA_MIN + 1  # 71
_N_H          = 2                        # Q1/Q2 only: h_idx ∈ {0,1} for h ∈ {-1,+1}

_NAN = np.float64("nan")


def _h_idx(h: int) -> int:
    """Map h ∈ {-1, +1} → array index {0, 1}."""
    return (h + 1) >> 1  # -1 → 0, +1 → 1


# ---------------------------------------------------------------------------
# ValueStore
# ---------------------------------------------------------------------------

class ValueStore:
    """Dense NumPy-backed value table with quarter-aware h-dimension handling.

    Q3/Q4 states carry h=0 and use arrays without an h axis.
    Q1/Q2 states carry h∈{-1,+1} and use arrays with a 2-element h axis.

    Public API mirrors Dict[tuple, float]:
        vs[key]     → float
        vs[key] = v
        key in vs
        len(vs)     → number of set (non-NaN) slots
    """

    __slots__ = (
        "_s12", "_s34",
        "_k12", "_k34",
        "_sk12", "_sk34",
        "_count",
    )

    def __init__(self) -> None:
        # Q1/Q2 scrimmage: h ∈ {-1, +1}
        self._s12 = np.full(
            (2, _N_TAU, _N_X, _N_DOWN, _N_TF, _N_DELTA, _N_H),
            _NAN, dtype=np.float64,
        )
        # Q3/Q4 scrimmage: no h dimension
        self._s34 = np.full(
            (2, _N_TAU, _N_X, _N_DOWN, _N_TF, _N_DELTA),
            _NAN, dtype=np.float64,
        )
        # Kickoff/safety kick Q1/Q2 (h ∈ {-1,+1})
        self._k12  = np.full((2, _N_TAU, _N_DELTA, _N_H), _NAN, dtype=np.float64)
        self._sk12 = np.full((2, _N_TAU, _N_DELTA, _N_H), _NAN, dtype=np.float64)
        # Kickoff/safety kick Q3/Q4 (no h)
        self._k34  = np.full((2, _N_TAU, _N_DELTA), _NAN, dtype=np.float64)
        self._sk34 = np.full((2, _N_TAU, _N_DELTA), _NAN, dtype=np.float64)

        # Fast O(1) count of solved slots
        self._count = 0

        self._print_sizes()

    # ------------------------------------------------------------------
    # Startup memory report
    # ------------------------------------------------------------------

    def _print_sizes(self) -> None:
        parts = [
            ("s12  (Q1/Q2 scrim)", self._s12),
            ("s34  (Q3/Q4 scrim)", self._s34),
            ("k12  (Q1/Q2 ko)   ", self._k12),
            ("k34  (Q3/Q4 ko)   ", self._k34),
            ("sk12 (Q1/Q2 sk)   ", self._sk12),
            ("sk34 (Q3/Q4 sk)   ", self._sk34),
        ]
        total = sum(a.nbytes for _, a in parts)
        print("[ValueStore] Allocated arrays:")
        for name, arr in parts:
            mb = arr.nbytes / 1e6
            print(f"  {name} shape={arr.shape}  {mb:,.0f} MB")
        print(f"  Total: {total/1e9:.3f} GB")

    # ------------------------------------------------------------------
    # Typed accessors (hot path — called directly from lookup_val)
    # ------------------------------------------------------------------

    def get_scrimmage(self, q: int, tau: int, x: int, down: int,
                      to_first: int, delta: int, h: int) -> float:
        di = delta + _DELTA_OFFSET
        if q <= 2:
            v = self._s12[q-1, tau, x-1, down-1, to_first-1, di, _h_idx(h)]
        else:
            v = self._s34[q-3, tau, x-1, down-1, to_first-1, di]
        if np.isnan(v):
            raise KeyError(
                f"scrimmage state not solved: q={q} tau={tau} x={x} "
                f"d={down} tf={to_first} delta={delta} h={h}"
            )
        return float(v)

    def set_scrimmage(self, q: int, tau: int, x: int, down: int,
                      to_first: int, delta: int, h: int, value: float) -> None:
        di = delta + _DELTA_OFFSET
        if q <= 2:
            idx = (q-1, tau, x-1, down-1, to_first-1, di, _h_idx(h))
            if np.isnan(self._s12[idx]):
                self._count += 1
            self._s12[idx] = value
        else:
            idx = (q-3, tau, x-1, down-1, to_first-1, di)
            if np.isnan(self._s34[idx]):
                self._count += 1
            self._s34[idx] = value

    def get_kickoff(self, q: int, tau: int, delta: int, h: int) -> float:
        di = delta + _DELTA_OFFSET
        if q <= 2:
            v = self._k12[q-1, tau, di, _h_idx(h)]
        else:
            v = self._k34[q-3, tau, di]
        if np.isnan(v):
            raise KeyError(
                f"kickoff state not solved: q={q} tau={tau} delta={delta} h={h}"
            )
        return float(v)

    def set_kickoff(self, q: int, tau: int, delta: int, h: int, value: float) -> None:
        di = delta + _DELTA_OFFSET
        if q <= 2:
            idx = (q-1, tau, di, _h_idx(h))
            if np.isnan(self._k12[idx]):
                self._count += 1
            self._k12[idx] = value
        else:
            idx = (q-3, tau, di)
            if np.isnan(self._k34[idx]):
                self._count += 1
            self._k34[idx] = value

    def get_safety(self, q: int, tau: int, delta: int, h: int) -> float:
        di = delta + _DELTA_OFFSET
        if q <= 2:
            v = self._sk12[q-1, tau, di, _h_idx(h)]
        else:
            v = self._sk34[q-3, tau, di]
        if np.isnan(v):
            raise KeyError(
                f"safety_kick state not solved: q={q} tau={tau} delta={delta} h={h}"
            )
        return float(v)

    def set_safety(self, q: int, tau: int, delta: int, h: int, value: float) -> None:
        di = delta + _DELTA_OFFSET
        if q <= 2:
            idx = (q-1, tau, di, _h_idx(h))
            if np.isnan(self._sk12[idx]):
                self._count += 1
            self._sk12[idx] = value
        else:
            idx = (q-3, tau, di)
            if np.isnan(self._sk34[idx]):
                self._count += 1
            self._sk34[idx] = value

    # ------------------------------------------------------------------
    # Dict-like interface keyed by state tuples
    # ------------------------------------------------------------------

    def __getitem__(self, key: tuple) -> float:
        phase = key[0]
        if phase == "s":
            _, q, tau, x, down, tf, delta, h = key
            return self.get_scrimmage(q, tau, x, down, tf, delta, h)
        if phase == "k":
            _, q, tau, delta, h = key
            return self.get_kickoff(q, tau, delta, h)
        if phase == "sk":
            _, q, tau, delta, h = key
            return self.get_safety(q, tau, delta, h)
        raise KeyError(f"unknown phase {phase!r}")

    def __setitem__(self, key: tuple, value: float) -> None:
        phase = key[0]
        if phase == "s":
            _, q, tau, x, down, tf, delta, h = key
            self.set_scrimmage(q, tau, x, down, tf, delta, h, value)
        elif phase == "k":
            _, q, tau, delta, h = key
            self.set_kickoff(q, tau, delta, h, value)
        elif phase == "sk":
            _, q, tau, delta, h = key
            self.set_safety(q, tau, delta, h, value)
        else:
            raise KeyError(f"unknown phase {phase!r}")

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple):
            return False
        try:
            self[key]
            return True
        except KeyError:
            return False

    def get(self, key: tuple, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Memory info
    # ------------------------------------------------------------------

    def nbytes(self) -> int:
        """Total bytes occupied by all arrays."""
        return (self._s12.nbytes + self._s34.nbytes +
                self._k12.nbytes + self._k34.nbytes +
                self._sk12.nbytes + self._sk34.nbytes)

    def nbytes_str(self) -> str:
        return f"{self.nbytes() / 1e9:.3f} GB"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all arrays to a compressed .npz file.

        Compression is aggressive because most slots are NaN (unsolved).
        Typical Q4-only full solve: ~200-400 MB on disk vs 2.47 GB in RAM.
        """
        np.savez_compressed(
            path,
            s12=self._s12, s34=self._s34,
            k12=self._k12, k34=self._k34,
            sk12=self._sk12, sk34=self._sk34,
            count=np.array([self._count], dtype=np.int64),
        )
        actual = path if path.endswith(".npz") else path + ".npz"
        print(f"[ValueStore] Saved to {actual}  ({len(self):,} solved slots)")

    @classmethod
    def load(cls, path: str) -> "ValueStore":
        """Load a previously saved ValueStore from a .npz file."""
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path)
        vs = cls.__new__(cls)
        vs._s12  = data["s12"]
        vs._s34  = data["s34"]
        vs._k12  = data["k12"]
        vs._k34  = data["k34"]
        vs._sk12 = data["sk12"]
        vs._sk34 = data["sk34"]
        vs._count = int(data["count"][0])
        print(f"[ValueStore] Loaded {path}  ({vs._count:,} solved slots)")
        return vs

