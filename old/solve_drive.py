# solve_drive.py
import numpy as np
from scipy.optimize import linprog

from read_matrix import load_pro_style_chart_csv
from transition import State, successors

OFFENSE_PLAYS = list(range(1, 21))
DEFENSE_PLAYS = list("ABCDEFGHIJ")


def print_offense_mix(p, *, header=None):
    """Print offense mixed strategy p over plays 1..20."""
    if header:
        print(header)
    for play in range(1, 21):
        print(f"{play:>2}: {float(p[play-1]):.6f}")
    print("Total probability:", float(np.sum(p)))


def solve_zero_sum_value(A: np.ndarray):
    """
    Zero-sum matrix game value for offense payoff A (m x n).
    Returns (v, p) where p is offense mixed strategy over rows.
    """
    m, n = A.shape

    # vars: p_0..p_{m-1}, v
    c = np.zeros(m + 1)
    c[-1] = -1.0  # maximize v <-> minimize -v

    # for each column j: sum_i p_i A[i,j] >= v  <->  -A^T p + v <= 0
    A_ub = np.zeros((n, m + 1))
    b_ub = np.zeros(n)
    for j in range(n):
        A_ub[j, :m] = -A[:, j]
        A_ub[j, m] = 1.0

    # sum p = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * m + [(-10, 10)]  # safe for drive rewards

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs"
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    p = res.x[:m].astype(float)  # avoid np.float64 printing later
    v = float(res.x[m])
    return v, p


def build_layers(chart, start_states, H: int):
    """
    layers[h] = set of states reachable from start_states with exactly h plays remaining.
      layers[H] = start_states
      layers[h-1] = one-step successors of layers[h]
    """
    layers = [set() for _ in range(H + 1)]
    layers[H] = set(start_states)

    for h in range(H, 0, -1):
        nxt = set()
        for s in layers[h]:
            for off in OFFENSE_PLAYS:
                for d in DEFENSE_PLAYS:
                    for tr in successors(chart, s, off_play=off, def_play=d):
                        if tr.next_state is not None:
                            nxt.add(tr.next_state)
        layers[h - 1] = nxt

    return layers


def payoff_cell_value(chart, s: State, off_play: int, def_play: str, V_next: dict):
    """
    Value for one cell (off,def) at state s:
      - compute successor transitions
      - if cell is OR-choice, offense picks best successor -> max
    """
    best = -1e9
    for tr in successors(chart, s, off_play=off_play, def_play=def_play):
        if tr.next_state is None or tr.terminal is not None:
            val = float(tr.reward)
        else:
            val = float(tr.reward) + float(V_next[tr.next_state])
        if val > best:
            best = val
    return best


def build_payoff_matrix(chart, s: State, V_next: dict) -> np.ndarray:
    m = len(OFFENSE_PLAYS)
    n = len(DEFENSE_PLAYS)
    A = np.zeros((m, n), dtype=float)
    for ii, off in enumerate(OFFENSE_PLAYS):
        for jj, d in enumerate(DEFENSE_PLAYS):
            A[ii, jj] = payoff_cell_value(chart, s, off, d, V_next)
    return A


def solve_drive(H: int = 10, csv_path: str = "Football Strategy Pro Style.csv", start_states=None, *, verbose=True):
    """
    Computes finite-horizon equilibrium policies for ALL layers.
    Returns:
      Vh[(h, s)] = value at state s with h plays remaining
      Pih[(h, s)] = offense mixed strategy at state s with h plays remaining
      layers = list of sets of states
    """
    chart = load_pro_style_chart_csv(csv_path)

    if start_states is None:
        start_states = [State(x=20, down=1, to_first=10)]

    layers = build_layers(chart, start_states, H)

    if verbose:
        for h in range(H + 1):
            print(f"States with {h} plays remaining: {len(layers[h])}")

    # Base: V0(s)=0 for all s in layer[0]
    V_prev = {s: 0.0 for s in layers[0]}

    Vh = {(0, s): 0.0 for s in layers[0]}
    Pih = {}  # (h, s) -> p

    for h in range(1, H + 1):
        if verbose:
            print(f"Solving horizon h={h}/{H} ...")

        V_curr = {}
        for s in layers[h]:
            A = build_payoff_matrix(chart, s, V_prev)
            v, p = solve_zero_sum_value(A)
            V_curr[s] = v
            Vh[(h, s)] = v
            Pih[(h, s)] = p

        V_prev = V_curr

    return Vh, Pih, layers


def best_available_h(layers, s: State, h_pref: int):
    """Pick an h to display: prefer h_pref; otherwise nearest h where s exists."""
    if 0 <= h_pref < len(layers) and s in layers[h_pref]:
        return h_pref
    candidates = [h for h in range(len(layers)) if s in layers[h]]
    if not candidates:
        return None
    # nearest to preference
    return min(candidates, key=lambda h: abs(h - h_pref))


def show_state_at_h(Vh, Pih, layers, s: State, h_pref: int, *, print_full=True):
    h = best_available_h(layers, s, h_pref)
    if h is None:
        print("\nState never appears in any layer:", s)
        return

    v = float(Vh[(h, s)])
    print(f"\nState: {s}   (showing policy at h={h} plays remaining)   V={v:.6f}")

    p = Pih[(h, s)]
    if print_full:
        print_offense_mix(p)
    else:
        top = sorted([(i + 1, float(p[i])) for i in range(20)], key=lambda x: -x[1])[:8]
        print("Top plays:", [(pl, round(pr, 4)) for pl, pr in top])


if __name__ == "__main__":
    H = 45
    Vh, Pih, layers = solve_drive(H=H, csv_path="Football Strategy Pro Style.csv", verbose=True)

    # States you want to inspect
    test_states = [
        State(x=20, down=1, to_first=10),  # start-ish
        State(x=20, down=3, to_first=10),  # 3rd & 10
        State(x=20, down=4, to_first=1),   # 4th & 1
        State(x=95, down=1, to_first=5),   # near goal line
        State(x=5,  down=1, to_first=10),  # backed up
        State(x=60, down=2, to_first=3),   # 2nd & 3
    ]

    # Which "plays remaining" do you want to view policies at?
    # - For start state, you'd typically view h=H
    # - For sanity checks across generic drive contexts, pick something like h_view=10 or 20
    h_view = 45

    for s in test_states:
        show_state_at_h(Vh, Pih, layers, s, h_pref=h_view, print_full=True)
