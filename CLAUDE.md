# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a full-game equilibrium solver for Avalon Hill's board game **Football Strategy**, intended as a publishable extension of Sean McCulloch's paper *A Game-Theoretic Intelligent Agent for the Board Game Football Strategy* (PDF included). The goal is to replace McCulloch's hand-tuned heuristic utilities with exact backward-induction continuation values, solving the game as a finite-horizon two-player zero-sum stochastic game.

The design document is `plan.txt` — it is the authoritative specification for what to build.

## Environment

Python 3.9 virtual environment in `.venv/`. Activate before running anything:

```
source .venv/bin/activate
```

Key dependencies: `numpy`, `scipy` (linprog with HiGHS), `pandas`.

Run a script:
```
python old/solve_quarter_sym.py
```

There is no test runner or build system yet. The `old/` directory contains earlier solver iterations; new code should be placed at the project root under the module structure described in `plan.txt` (section 4.11).

## Data files

- `Football Strategy Pro Style.csv` — the offense-defense chart: 10 defense rows (A–J) × 20 offense play columns (1–20). This is the primary input to all solvers.
- `Football Strategy Pro Style.xlsx` — same chart in Excel format.

## Architecture

### Core abstractions (as evolved in `old/`)

**State** — scrimmage state tuple. Two conventions exist:

- *Explicit-possession* (`transition_quarter_score.py`): `State(poss, x, down, to_first, t, delta)` where `poss` is `US=0` or `THEM=1`, `x` is yardline from Team 1's goal, `t` is plays remaining, `delta = US - THEM` clipped to `[-14, 14]`.
- *Symmetric/possession-less* (`transition_quarter_sym.py`, preferred for new work): `State(x, down, to_first, t, delta)` where `x` is always measured from the **current offense's** goal line and `delta = offense - defense`. When possession changes, `x → 100 - x` and `delta → -delta`, and the continuation value is negated. This roughly halves state-space size.

**Chart loading** (`read_matrix.py`): `load_pro_style_chart_csv()` returns a `(10 × 20)` pandas DataFrame indexed by defense row (`"A"`–`"J"`) and offense column (`"1"`–`"20"`). Access a cell: `chart.loc["F", "4"]`.

**Transition engine** (`transition_quarter_score.py`): parses chart cell tokens (e.g., `"LG"`, `"INT-20"`, `"-3OB"`, `"7P OR 7"`) into typed leaves and applies game rules to produce `Transition(next_state, reward, terminal)` lists. Handles: yards, incomplete, fumble, interception, penalty, LOD, out-of-bounds, safety, long gain, punt. `successors(chart, state, offense_action, def_play)` is the main entry point.

**Solver pattern** (established in `solve_quarter_sym.py` / `solve_quarter_score.py`):
1. **Forward reachability** (`build_layers_core`): BFS from start states to enumerate all reachable `(core, t)` states, grouped by time layer.
2. **Backward induction**: iterate `t = 0 → T`, for each reachable state at each `t`, build the `|O(s)| × 10` payoff matrix and solve the zero-sum matrix game.
3. **Matrix game LP**: offense maximizes, defense minimizes via `scipy.optimize.linprog` (HiGHS method). LP variables: offense mixed strategy `p` over rows + value `v`. Returns `(v, p)`.
4. **Terminal utility**: `tanh(delta / TANH_SCALE)` for quarter-only solvers. The full-game solver (`plan.txt` §3.1) uses `+1/0/-1` (win/tie/loss).

### Key constants

| Name | Value | Meaning |
|---|---|---|
| `DELTA_MIN/MAX` | -14 / 14 | Score-diff clipping bounds |
| `TO_FIRST_MAX` | 30 | Max tracked yards-to-first |
| `TANH_SCALE` | 7.0 | Quarter terminal utility scale |
| `PUNT` | 21 | Offense action index for punt |
| Time tick | 15 sec | All clock values in 15-sec units |
| Quarter ticks | 60 | Ticks per quarter |

### Red-zone play restrictions

Plays 17–20 illegal when `x >= 80` (offense's 80+, i.e., defender's 1–20).
Plays 13–16 illegal when `x >= 90` (defender's 1–10).

### Long gain table

`LG` resolves probabilistically: `P(+30)=P(+35)=P(+40)=P(+45)=P(+50)=1/6`; for `+60` through `+110` each has probability `1/36`. Always computed as exact expectation, never sampled.

## Full-game solver plan

`plan.txt` contains the complete specification. The target module structure (§4.11):

```
football_strategy/
├── constants.py       # dice/kickoff/LG/kicking tables, clock constants
├── states.py          # ScrimmageState, KickoffState, SafetyKickState
├── outcomes.py        # typed chart outcome classes
├── chart_parser.py    # parse Pro Style chart into structured outcomes
├── clock.py           # quarter transitions, 2-min stoppage, timeout effects
├── transitions.py     # all transition logic
├── reachability.py    # forward reachability traversal
├── matrix_game.py     # LP solver for matrix-game value and strategies
├── solve_full_game.py # backward induction solver
└── diagnostics.py     # reporting and paper figures
```

Development stages (§4.12): Stage 1 = full rules without timeouts; Stage 2 = add timeouts; Stage 3 = experiments and paper outputs.
