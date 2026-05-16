"""
Full rules engine for the Football Strategy solver.

Transitions return SuccessorTemplate lists (structural templates + tick cost).
Clock advancement happens in the solve layer (matrix_game.py / solve_full_game.py),
NOT here. This makes templates fully cacheable keyed on (core, action, def_card).

Possession convention: x is always from the CURRENT OFFENSE's own goal line.
On every swap: new_x = 100 - old_x, new_delta = -old_delta, new_h = -old_h.
The solve layer negates the continuation value whenever swapped=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from football_strategy.constants import (
    DEFENSE_CARDS,
    DELTA_MAX,
    DELTA_MIN,
    FG_MIN_X,
    FG_POINTS,
    KICKOFF_NORMAL,
    KICKOFF_NORMAL_COUNTS,
    KICKOFF_ONSIDE,
    LG_DIST,
    PAT_PROB_GOOD,
    PAT_PROB_MISS,
    PUNT_CHART,
    SAFETY_POINTS,
    TD_POINTS,
    TO_FIRST_MAX,
    fg_prob,
)
from football_strategy.outcomes import (
    Branch,
    CellOutcome,
    ChoiceOutcome,
    FirstDownOutcome,
    FumbleOutcome,
    IncompleteOutcome,
    InterceptionOutcome,
    LongGainOutcome,
    LossOfDownOutcome,
    PenaltyOutcome,
    PuntResult,
    SingleAtom,
    YardsOutcome,
)
from football_strategy.states import goal_to_go


# ---------------------------------------------------------------------------
# SuccessorTemplate
# ---------------------------------------------------------------------------

@dataclass
class SuccessorTemplate:
    """Structural successor descriptor returned by all transition functions.

    The solve layer uses these as:
        new_tau, hit_warn = advance_clock(tau, q, tmpl.ticks)
        succ_val = V[make_key(q_next, tau_next, tmpl)]
        if tmpl.swapped:
            succ_val = -succ_val
        cell_value += tmpl.prob * succ_val
    """
    prob:         float
    phase:        str           # "scrimmage" | "kickoff" | "safety_kick" | "terminal"
    core:         Optional[tuple]  # ScrimmageCore / KickoffCore / None for terminal
    ticks:        int
    swapped:      bool          # True → possession changed; caller negates value
    delta_change: int           # score points added (positive = offense gains)


@dataclass
class ChoiceTemplates:
    """OR-choice between two lists of SuccessorTemplate.

    The payoff matrix builder resolves this:
      resolver="offense" → take the branch with higher expected value (max for offense)
      resolver="defense" → take the branch with lower expected value (min for offense)
    """
    branches:  Tuple[List[SuccessorTemplate], ...]
    resolver:  str   # "offense" | "defense"


# Transition result type
TransitionResult = Union[List[SuccessorTemplate], ChoiceTemplates]


# ---------------------------------------------------------------------------
# Core tuple helpers
# ---------------------------------------------------------------------------
# ScrimmageCore = (x, down, to_first, delta, h)
# KickoffCore   = (delta, h)
# SafetyKickCore = (delta, h)  — same shape as KickoffCore, different phase

def _sc(x: int, down: int, to_first: int, delta: int, h: int) -> tuple:
    """Build a ScrimmageCore tuple (with delta and to_first clipping)."""
    delta    = max(DELTA_MIN,    min(DELTA_MAX,    delta))
    to_first = max(1,            min(TO_FIRST_MAX, to_first))
    return (x, down, to_first, delta, h)


def _kc(delta: int, h: int) -> tuple:
    """Build a KickoffCore / SafetyKickCore tuple (with delta clipping)."""
    delta = max(DELTA_MIN, min(DELTA_MAX, delta))
    return (delta, h)


# ---------------------------------------------------------------------------
# Penalty / half-distance cap
# ---------------------------------------------------------------------------

def half_distance_cap(x: int, pen_yards: int) -> int:
    """Cap penalty at half distance to the relevant goal.

    pen_yards > 0: offense gains yards toward opponent's goal.
                   Cap: min(pen_yards, (100 - x) // 2).
    pen_yards < 0: offense loses yards toward own goal.
                   Cap: max(pen_yards, -(x // 2)).
    """
    if pen_yards > 0:
        return min(pen_yards, (100 - x) // 2)
    elif pen_yards < 0:
        return max(pen_yards, -(x // 2))
    return 0


# ---------------------------------------------------------------------------
# PAT inline expansion
# ---------------------------------------------------------------------------

def _pat_kickoff_templates(
    scorer_delta_before: int,
    scorer_h: int,
    ticks: int,
) -> List[SuccessorTemplate]:
    """Expand a TD into two weighted templates (PAT good + PAT miss).

    Both lead to a kickoff where the scoring team kicks off.
    delta_change includes the TD points (passed in scorer_delta_before already
    reflects the state BEFORE the TD; we add TD_POINTS here plus PAT).
    """
    # After TD + PAT good: scorer has TD_POINTS + 1 more than before TD
    # After TD + PAT miss: scorer has TD_POINTS more than before TD
    # In kickoff state, scorer is the kicker; perspective is kicker's.
    # delta_change in SuccessorTemplate is from CURRENT offense's view.
    td_and_pat_good = TD_POINTS + 1
    td_only         = TD_POINTS

    return [
        SuccessorTemplate(
            prob=PAT_PROB_GOOD,
            phase="kickoff",
            core=_kc(scorer_delta_before + td_and_pat_good, scorer_h),
            ticks=ticks,
            swapped=False,   # scorer stays as the "kicker" in kickoff state
            delta_change=td_and_pat_good,
        ),
        SuccessorTemplate(
            prob=PAT_PROB_MISS,
            phase="kickoff",
            core=_kc(scorer_delta_before + td_only, scorer_h),
            ticks=ticks,
            swapped=False,
            delta_change=td_only,
        ),
    ]


# ---------------------------------------------------------------------------
# Single atom application (branch head)
# ---------------------------------------------------------------------------

def _apply_yards(
    x: int, down: int, to_first: int, delta: int, h: int, yards: int,
) -> List[SuccessorTemplate]:
    """Apply a plain yards gain/loss. Returns list of SuccessorTemplate."""
    new_x = x + yards

    # Touchdown
    if new_x >= 100:
        return _pat_kickoff_templates(
            scorer_delta_before=delta,
            scorer_h=h,
            ticks=_yards_ticks(yards, out_of_bounds=False),
        )

    # Safety (offense pushed back into own end zone)
    if new_x <= 0:
        # Defense scores SAFETY_POINTS; from current offense's perspective: delta drops by 2
        # Scored-against team (current offense) becomes the free kicker.
        new_delta = delta - SAFETY_POINTS
        return [SuccessorTemplate(
            prob=1.0,
            phase="safety_kick",
            core=_kc(new_delta, h),
            ticks=_yards_ticks(yards, out_of_bounds=False),
            swapped=False,   # current offense becomes safety kicker
            delta_change=-SAFETY_POINTS,
        )]

    # Normal gain/loss
    return [_normal_scrimmage_template(
        x=x, new_x=new_x, down=down, to_first=to_first, delta=delta, h=h,
        yards=yards, out_of_bounds=False,
    )]


def _yards_ticks(yards: int, out_of_bounds: bool) -> int:
    if out_of_bounds:
        return 1
    if yards < 0:
        return 2
    if yards < 20:
        return 2
    return 3


def _normal_scrimmage_template(
    x: int, new_x: int, down: int, to_first: int, delta: int, h: int,
    yards: int, out_of_bounds: bool,
) -> SuccessorTemplate:
    """Build a SuccessorTemplate for a normal scrimmage continuation (no score)."""
    ticks = _yards_ticks(yards, out_of_bounds)

    new_to_first = to_first - yards

    if new_to_first <= 0:
        # First down
        new_down     = 1
        new_to_first = goal_to_go(new_x)
    elif down < 4:
        # Next down, same series
        new_down = down + 1
    else:
        # Turnover on downs (4th down failure)
        recv_x     = 100 - new_x
        recv_delta = -delta
        recv_h     = -h
        recv_tf    = goal_to_go(recv_x)
        return SuccessorTemplate(
            prob=1.0, phase="scrimmage",
            core=_sc(recv_x, 1, recv_tf, recv_delta, recv_h),
            ticks=ticks, swapped=True, delta_change=0,
        )

    return SuccessorTemplate(
        prob=1.0, phase="scrimmage",
        core=_sc(new_x, new_down, new_to_first, delta, h),
        ticks=ticks, swapped=False, delta_change=0,
    )


# ---------------------------------------------------------------------------
# Branch application (AND-chain)
# ---------------------------------------------------------------------------

def apply_branch(
    x: int, down: int, to_first: int, delta: int, h: int,
    branch: Branch,
) -> List[SuccessorTemplate]:
    """Apply an AND-chain branch. First atom drives clock; modifiers after."""
    if not branch:
        return [SuccessorTemplate(1.0, "scrimmage", _sc(x, down, to_first, delta, h), 0, False, 0)]

    primary = branch[0]
    modifiers = branch[1:]

    # Expand LG into weighted list first, then apply modifiers per outcome
    if isinstance(primary, LongGainOutcome):
        templates: List[SuccessorTemplate] = []
        for (lg_yards, prob) in LG_DIST:
            sub = _apply_yards_with_modifiers(x, down, to_first, delta, h, lg_yards, False, modifiers)
            for t in sub:
                templates.append(SuccessorTemplate(
                    prob=t.prob * prob,
                    phase=t.phase, core=t.core,
                    ticks=3,  # LG always costs 3 ticks
                    swapped=t.swapped, delta_change=t.delta_change,
                ))
        return templates

    if isinstance(primary, YardsOutcome):
        return _apply_yards_with_modifiers(
            x, down, to_first, delta, h, primary.yards, primary.out_of_bounds, modifiers,
        )

    if isinstance(primary, IncompleteOutcome):
        # Clock stops (1 tick). Down advances, no yardage change. No modifiers expected.
        if down < 4:
            return [SuccessorTemplate(
                1.0, "scrimmage",
                _sc(x, down + 1, to_first, delta, h),
                ticks=1, swapped=False, delta_change=0,
            )]
        else:
            # Incomplete on 4th down → turnover on downs
            recv_x  = 100 - x
            recv_tf = goal_to_go(recv_x)
            return [SuccessorTemplate(
                1.0, "scrimmage",
                _sc(recv_x, 1, recv_tf, -delta, -h),
                ticks=1, swapped=True, delta_change=0,
            )]

    if isinstance(primary, FumbleOutcome):
        # Defense recovers; immediate possession swap at same spot.
        recv_x  = 100 - x
        recv_tf = goal_to_go(recv_x)
        return [SuccessorTemplate(
            1.0, "scrimmage",
            _sc(recv_x, 1, recv_tf, -delta, -h),
            ticks=1, swapped=True, delta_change=0,
        )]

    if isinstance(primary, InterceptionOutcome):
        return _apply_interception(x, down, to_first, delta, h, primary.return_yards)

    if isinstance(primary, PenaltyOutcome):
        return _apply_penalty_with_modifiers(
            x, down, to_first, delta, h, primary.yards, advance_down=False, modifiers=modifiers,
        )

    if isinstance(primary, LossOfDownOutcome):
        return _apply_penalty_with_modifiers(
            x, down, to_first, delta, h, primary.yards, advance_down=True, modifiers=modifiers,
        )

    if isinstance(primary, FirstDownOutcome):
        # Pure modifier — treat as a first-down reset at current spot
        new_tf = goal_to_go(x)
        return [SuccessorTemplate(
            1.0, "scrimmage", _sc(x, 1, new_tf, delta, h),
            ticks=0, swapped=False, delta_change=0,
        )]

    raise TypeError(f"Unknown primary atom: {type(primary)!r}")


def _apply_yards_with_modifiers(
    x: int, down: int, to_first: int, delta: int, h: int,
    yards: int, out_of_bounds: bool,
    modifiers: Branch,
) -> List[SuccessorTemplate]:
    """Apply yardage, then apply any FirstDownOutcome modifiers."""
    new_x = x + yards

    # Touchdown
    if new_x >= 100:
        ticks = _yards_ticks(yards, out_of_bounds)
        return _pat_kickoff_templates(delta, h, ticks)

    # Safety
    if new_x <= 0:
        ticks = _yards_ticks(yards, out_of_bounds)
        return [SuccessorTemplate(
            1.0, "safety_kick", _kc(delta - SAFETY_POINTS, h),
            ticks=ticks, swapped=False, delta_change=-SAFETY_POINTS,
        )]

    ticks = _yards_ticks(yards, out_of_bounds)
    new_to_first = to_first - yards

    # Check for FirstDownOutcome modifier
    has_first_down = any(isinstance(m, FirstDownOutcome) for m in modifiers)

    if has_first_down or new_to_first <= 0:
        new_down     = 1
        new_to_first = goal_to_go(new_x)
    elif down < 4:
        new_down = down + 1
    else:
        # 4th-down failure
        recv_x  = 100 - new_x
        recv_tf = goal_to_go(recv_x)
        return [SuccessorTemplate(
            1.0, "scrimmage", _sc(recv_x, 1, recv_tf, -delta, -h),
            ticks=ticks, swapped=True, delta_change=0,
        )]

    return [SuccessorTemplate(
        1.0, "scrimmage", _sc(new_x, new_down, new_to_first, delta, h),
        ticks=ticks, swapped=False, delta_change=0,
    )]


def _apply_penalty_with_modifiers(
    x: int, down: int, to_first: int, delta: int, h: int,
    pen_yards: int, advance_down: bool, modifiers: Branch,
) -> List[SuccessorTemplate]:
    """Apply a penalty (PenaltyOutcome or LossOfDownOutcome)."""
    capped = half_distance_cap(x, pen_yards)
    new_x  = x + capped

    # Clamp field position (shouldn't reach end zones via penalty, but be safe)
    new_x = max(1, min(99, new_x))

    new_to_first = to_first - capped

    has_first_down = any(isinstance(m, FirstDownOutcome) for m in modifiers)

    if has_first_down or new_to_first <= 0:
        new_down     = 1
        new_to_first = goal_to_go(new_x)
    elif advance_down:
        if down < 4:
            new_down = down + 1
        else:
            # 4th-down LOD failure
            recv_x  = 100 - new_x
            recv_tf = goal_to_go(recv_x)
            return [SuccessorTemplate(
                1.0, "scrimmage", _sc(recv_x, 1, recv_tf, -delta, -h),
                ticks=1, swapped=True, delta_change=0,
            )]
    else:
        # PenaltyOutcome: down repeats
        new_down = down

    return [SuccessorTemplate(
        1.0, "scrimmage", _sc(new_x, new_down, new_to_first, delta, h),
        ticks=1, swapped=False, delta_change=0,
    )]


def _apply_interception(
    x: int, down: int, to_first: int, delta: int, h: int,
    return_yards: int,
) -> List[SuccessorTemplate]:
    """Apply an interception with ``return_yards`` toward offense's own end zone."""
    # Defender runs toward current offense's own end zone
    x_after = x - return_yards

    # Pick-6: defender reaches or passes offense's goal line
    if x_after <= 0:
        # Defense scores TD; from current offense's perspective: delta -= TD_POINTS
        # The DEFENSE (now the scorer) becomes the kickoff team.
        # After possession swap: defense is "offense" in kickoff state.
        # We represent this as: swapped=True, phase="kickoff",
        # core = KickoffCore from the new-offense (former defense) perspective.
        # Their delta = -(delta - TD_POINTS) = -delta + TD_POINTS
        # Their h = -h
        # PAT belongs to the intercepting team.
        scorer_delta = -delta  # intercepting team's perspective before PAT
        templates = []
        pat_good = TD_POINTS + 1
        pat_miss = TD_POINTS
        templates.append(SuccessorTemplate(
            prob=PAT_PROB_GOOD,
            phase="kickoff",
            core=_kc(scorer_delta + pat_good, -h),
            ticks=2, swapped=True,
            delta_change=-(pat_good),  # from current offense's perspective
        ))
        templates.append(SuccessorTemplate(
            prob=PAT_PROB_MISS,
            phase="kickoff",
            core=_kc(scorer_delta + pat_miss, -h),
            ticks=2, swapped=True,
            delta_change=-(pat_miss),
        ))
        return templates

    # Touchback: defender catches in/runs into own end zone (offense perspective: x_after >= 100)
    if x_after >= 100:
        # Ball placed at 20 from receiver's (intercepting team's) own goal
        recv_x  = 20
        recv_tf = goal_to_go(recv_x)
        return [SuccessorTemplate(
            1.0, "scrimmage", _sc(recv_x, 1, recv_tf, -delta, -h),
            ticks=2, swapped=True, delta_change=0,
        )]

    # Normal interception return
    recv_x  = 100 - x_after   # symmetric frame for new offense (the intercepting team)
    recv_tf = goal_to_go(recv_x)
    return [SuccessorTemplate(
        1.0, "scrimmage", _sc(recv_x, 1, recv_tf, -delta, -h),
        ticks=2, swapped=True, delta_change=0,
    )]


# ---------------------------------------------------------------------------
# Scrimmage play (plays 1–20)
# ---------------------------------------------------------------------------

def scrimmage_play_templates(
    scrimmage_chart: Dict[Tuple[str, str], CellOutcome],
    x: int, down: int, to_first: int, delta: int, h: int,
    offense_action: int,
    def_card: str,
) -> TransitionResult:
    """Return transition templates for a normal scrimmage play.

    If the chart cell is a ChoiceOutcome, returns a ChoiceTemplates.
    Otherwise returns a flat List[SuccessorTemplate].
    """
    cell: CellOutcome = scrimmage_chart[(def_card, str(offense_action))]

    if isinstance(cell, ChoiceOutcome):
        branch_results = []
        for branch in cell.branches:
            branch_results.append(
                apply_branch(x, down, to_first, delta, h, branch)
            )
        return ChoiceTemplates(
            branches=tuple(branch_results),
            resolver=cell.resolver,
        )
    else:
        # Single branch (list of atoms)
        return apply_branch(x, down, to_first, delta, h, cell)


# ---------------------------------------------------------------------------
# Punt (PUNT_ACTION)
# ---------------------------------------------------------------------------

def punt_templates(
    x: int, down: int, to_first: int, delta: int, h: int,
    def_card: str,
) -> TransitionResult:
    """Return transition templates for a punt.

    Land position is from current offense's own goal (offense drives toward 100).
    """
    punt_yds, ret_type, ret_yds = PUNT_CHART[def_card]
    land_x = x + punt_yds   # in offense's frame

    # Through the end zone → forced touchback
    if land_x >= 110:
        recv_x  = 20
        recv_tf = goal_to_go(recv_x)
        return [SuccessorTemplate(
            1.0, "scrimmage", _sc(recv_x, 1, recv_tf, -delta, -h),
            ticks=1, swapped=True, delta_change=0,
        )]

    # Lands IN the end zone (10-yard deep end zone beyond the goal line)
    if land_x >= 100:
        # Receiver strategic choice: take the return OR touchback at 20
        touchback_tmpl = [SuccessorTemplate(
            1.0, "scrimmage", _sc(20, 1, goal_to_go(20), -delta, -h),
            ticks=1, swapped=True, delta_change=0,
        )]
        return_tmpls = _punt_return_templates(x, land_x, ret_type, ret_yds, delta, h)
        # Receiving team (current defense) picks the better option for them.
        # From current offense's perspective that's a min (defense maximizes their own gain).
        return ChoiceTemplates(
            branches=(touchback_tmpl, return_tmpls),
            resolver="defense",
        )

    # Normal punt (land_x < 100)
    return _punt_return_templates(x, land_x, ret_type, ret_yds, delta, h)


def _punt_return_templates(
    x: int, land_x: int, ret_type: str, ret_yds: int, delta: int, h: int,
) -> List[SuccessorTemplate]:
    """Resolve a single punt return type into SuccessorTemplate list."""
    if ret_type in ("ob", "none"):
        recv_x  = 100 - land_x
        recv_tf = goal_to_go(max(1, recv_x))
        recv_x  = max(1, recv_x)
        return [SuccessorTemplate(
            1.0, "scrimmage", _sc(recv_x, 1, recv_tf, -delta, -h),
            ticks=1, swapped=True, delta_change=0,
        )]

    if ret_type == "fixed":
        # Ball returned ret_yds back toward the punter (toward offense's own end)
        final_x = land_x - ret_yds    # still in offense's frame
        return _punt_final_position(final_x, delta, h)

    if ret_type == "lg":
        templates: List[SuccessorTemplate] = []
        for (lg_y, prob) in LG_DIST:
            final_x = land_x - lg_y
            sub = _punt_final_position(final_x, delta, h)
            for t in sub:
                templates.append(SuccessorTemplate(
                    prob=t.prob * prob, phase=t.phase, core=t.core,
                    ticks=t.ticks, swapped=t.swapped, delta_change=t.delta_change,
                ))
        return templates

    if ret_type == "fixed_fumble":
        # Returner gains ret_yds but kicking team recovers the fumble at that spot.
        # Kicking team (current offense) gets ball at final_x from their own goal.
        # No possession swap.
        final_x = land_x - ret_yds
        punter_x  = max(1, min(99, final_x))
        punter_tf = goal_to_go(punter_x)
        return [SuccessorTemplate(
            1.0, "scrimmage", _sc(punter_x, 1, punter_tf, delta, h),
            ticks=1, swapped=False, delta_change=0,
        )]

    raise ValueError(f"Unknown punt return type: {ret_type!r}")


def _punt_final_position(final_x: int, delta: int, h: int) -> List[SuccessorTemplate]:
    """Build a template for punt return landing at final_x (offense's frame).

    final_x is the ball position after the return, in current offense's coordinate frame.
    The receiving team gets the ball; we swap to their perspective.

    If final_x <= 0 the returner has crossed the punting team's goal line — punt
    return touchdown for the receiving team.  (This arises from card-A LG returns
    that travel past the punter's end zone.)
    """
    if final_x <= 0:
        # Punt return TD: receiving team (current defense) scores.
        # From current offense (punter) perspective: delta drops by TD + PAT.
        scorer_delta = -delta   # receiver's own delta before the score
        return [
            SuccessorTemplate(
                PAT_PROB_GOOD, "kickoff",
                _kc(scorer_delta + TD_POINTS + 1, -h),
                ticks=1, swapped=True,
                delta_change=-(TD_POINTS + 1),
            ),
            SuccessorTemplate(
                PAT_PROB_MISS, "kickoff",
                _kc(scorer_delta + TD_POINTS, -h),
                ticks=1, swapped=True,
                delta_change=-TD_POINTS,
            ),
        ]

    recv_x  = max(1, min(99, final_x))
    recv_tf = goal_to_go(recv_x)
    # Receiving team gets ball: flip to their perspective
    new_x  = max(1, 100 - recv_x)
    new_tf = goal_to_go(new_x)
    return [SuccessorTemplate(
        1.0, "scrimmage", _sc(new_x, 1, new_tf, -delta, -h),
        ticks=1, swapped=True, delta_change=0,
    )]


# ---------------------------------------------------------------------------
# Field goal (FG_ACTION)
# ---------------------------------------------------------------------------

def field_goal_templates(
    x: int, down: int, to_first: int, delta: int, h: int,
) -> List[SuccessorTemplate]:
    """Field goal attempt from position x.

    Made  → offense +3 points, then offense kicks off.
    Missed → defense gets ball at their own 20 (from their perspective).
    """
    p = fg_prob(x)

    made_templates = _pat_kickoff_templates_fg(delta, h, p)
    miss_x  = 20   # defense ball at their own 20
    miss_tf = goal_to_go(miss_x)

    return [
        *made_templates,
        SuccessorTemplate(
            prob=1.0 - p,
            phase="scrimmage",
            core=_sc(miss_x, 1, miss_tf, -delta, -h),
            ticks=1, swapped=True, delta_change=0,
        ),
    ]


def _pat_kickoff_templates_fg(delta: int, h: int, prob: float) -> List[SuccessorTemplate]:
    """FG made (no PAT) — just add FG_POINTS and go to kickoff."""
    return [SuccessorTemplate(
        prob=prob,
        phase="kickoff",
        core=_kc(delta + FG_POINTS, h),
        ticks=1, swapped=False, delta_change=FG_POINTS,
    )]


# ---------------------------------------------------------------------------
# Kickoff: normal
# ---------------------------------------------------------------------------

def _recv_scrimmage(recv_x: int, ko_delta: int, ko_h: int) -> SuccessorTemplate:
    """Receiving team gets ball at recv_x from their own goal. Possession swaps."""
    recv_x  = max(1, min(99, recv_x))
    recv_tf = goal_to_go(recv_x)
    # After swap: new offense is the receiver. Their delta = -ko_delta, h = -ko_h.
    return SuccessorTemplate(
        1.0, "scrimmage",
        _sc(recv_x, 1, recv_tf, -ko_delta, -ko_h),
        ticks=1, swapped=True, delta_change=0,
    )


def normal_kickoff_templates(
    delta: int, h: int,
) -> List[SuccessorTemplate]:
    """Build SuccessorTemplate list for a normal kickoff.

    Fumble and penalty rows recursively reroll the 2d6 table.
    """
    templates: List[SuccessorTemplate] = []

    for roll, count in KICKOFF_NORMAL_COUNTS.items():
        base_prob = count / 36
        result_type, param = KICKOFF_NORMAL[roll]

        if result_type == "receiver_ball":
            t = _recv_scrimmage(param, delta, h)
            templates.append(SuccessorTemplate(
                base_prob, t.phase, t.core, t.ticks, t.swapped, t.delta_change,
            ))

        elif result_type == "fumble":
            # Reroll to determine recovery yard line.
            # KICKING team takes possession at that yard line.
            # Second fumble (reroll also = fumble) reverts to RECEIVING team.
            for sub_roll, sub_count in KICKOFF_NORMAL_COUNTS.items():
                sub_prob = sub_count / 36
                sub_type, sub_param = KICKOFF_NORMAL[sub_roll]
                if sub_type == "fumble":
                    # Second fumble → receiver gets ball at their own 20
                    t = _recv_scrimmage(20, delta, h)
                    templates.append(SuccessorTemplate(
                        base_prob * sub_prob, t.phase, t.core, t.ticks, t.swapped, t.delta_change,
                    ))
                else:
                    # Kicker recovers; recv_x is yard line in receiver's frame.
                    # From kicker's own goal: kicker_x = 100 - recv_x.
                    recv_x   = _kickoff_reroll_yard_line(sub_type, sub_param, penalty_offset=0)
                    kicker_x = max(1, min(99, 100 - recv_x))
                    kicker_tf = goal_to_go(kicker_x)
                    templates.append(SuccessorTemplate(
                        base_prob * sub_prob,
                        "scrimmage",
                        _sc(kicker_x, 1, kicker_tf, delta, h),
                        ticks=1, swapped=False, delta_change=0,
                    ))

        elif result_type == "penalty":
            # param = -10; reroll for yard line then subtract 10
            penalty_offset = param  # e.g. -10
            for sub_roll, sub_count in KICKOFF_NORMAL_COUNTS.items():
                sub_prob = sub_count / 36
                sub_type, sub_param = KICKOFF_NORMAL[sub_roll]
                recv_x = _kickoff_reroll_yard_line(sub_type, sub_param, penalty_offset)
                t = _recv_scrimmage(recv_x, delta, h)
                templates.append(SuccessorTemplate(
                    base_prob * sub_prob, t.phase, t.core, t.ticks, t.swapped, t.delta_change,
                ))

        elif result_type == "long_gain":
            extra = param  # 0 or 5
            for (lg_y, lg_prob) in LG_DIST:
                actual = lg_y + extra
                if actual >= 100:
                    # Kickoff return TD (receiving team scores)
                    # From kicker's perspective: delta drops by TD + PAT
                    scorer_delta = -delta  # receiver's perspective
                    templates.append(SuccessorTemplate(
                        base_prob * lg_prob * PAT_PROB_GOOD,
                        "kickoff",
                        _kc(scorer_delta + TD_POINTS + 1, -h),
                        ticks=1, swapped=True,
                        delta_change=-(TD_POINTS + 1),
                    ))
                    templates.append(SuccessorTemplate(
                        base_prob * lg_prob * PAT_PROB_MISS,
                        "kickoff",
                        _kc(scorer_delta + TD_POINTS, -h),
                        ticks=1, swapped=True,
                        delta_change=-TD_POINTS,
                    ))
                else:
                    recv_x = max(1, actual)
                    t = _recv_scrimmage(recv_x, delta, h)
                    templates.append(SuccessorTemplate(
                        base_prob * lg_prob, t.phase, t.core, t.ticks, t.swapped, t.delta_change,
                    ))

    return templates


def _kickoff_reroll_yard_line(
    sub_type: str, sub_param: Optional[int], penalty_offset: int,
) -> int:
    """Determine the yard line from a kickoff reroll result.

    Used for fumble (offset=0) and penalty (offset e.g. -10) rows.
    """
    if sub_type == "receiver_ball":
        return max(1, sub_param + penalty_offset)
    elif sub_type == "fumble":
        # Second fumble → receiving team still gets ball; no explicit yard line in rule.
        # Use 20 as a reasonable default (receiving team's own 20).
        return max(1, 20 + penalty_offset)
    elif sub_type == "long_gain":
        extra = sub_param or 0
        # Use the mean LG yard value (50 yards) as a proxy for the reroll case.
        # TODO: For exact rule fidelity, expand LG here too. Low-frequency event.
        return max(1, 50 + extra + penalty_offset)
    elif sub_type == "penalty":
        # Nested penalty on reroll; use 10-yard line as proxy.
        return max(1, 10 + penalty_offset)
    return max(1, 20 + penalty_offset)


# ---------------------------------------------------------------------------
# Kickoff: onside
# ---------------------------------------------------------------------------

def onside_kickoff_templates(
    delta: int, h: int,
) -> List[SuccessorTemplate]:
    """Build SuccessorTemplate list for an onside kickoff."""
    # +1 to roll if kicker is NOT trailing (tied or ahead); cap at 6
    adjustment = 0 if delta < 0 else 1
    templates: List[SuccessorTemplate] = []

    for base_roll in range(1, 7):
        prob = 1 / 6
        adjusted_roll = min(base_roll + adjustment, 6)
        who, x_val = KICKOFF_ONSIDE[adjusted_roll]

        if who == "kicker":
            # Kicker retains ball at x_val from their own goal; no swap
            tf = goal_to_go(x_val)
            templates.append(SuccessorTemplate(
                prob, "scrimmage",
                _sc(x_val, 1, tf, delta, h),
                ticks=1, swapped=False, delta_change=0,
            ))
        else:
            # Receiver gets ball at x_val from their own goal; swap
            tf = goal_to_go(x_val)
            templates.append(SuccessorTemplate(
                prob, "scrimmage",
                _sc(x_val, 1, tf, -delta, -h),
                ticks=1, swapped=True, delta_change=0,
            ))

    return templates


# ---------------------------------------------------------------------------
# Safety kick resolution
# ---------------------------------------------------------------------------

def safety_kick_option_normal(
    delta: int, h: int,
) -> List[SuccessorTemplate]:
    """Normal kickoff table + 25 yards added to each result's yard line."""
    base = normal_kickoff_templates(delta, h)
    # Add 25 to the receiver's starting position for each template
    adjusted: List[SuccessorTemplate] = []
    for t in base:
        if t.phase == "scrimmage" and not t.swapped:
            # Kicker retained ball (onside-style result) — shift forward too
            x, down, tf_val, d, hh = t.core
            new_x  = min(99, x + 25)
            new_tf = goal_to_go(new_x)
            adjusted.append(SuccessorTemplate(
                t.prob, t.phase, _sc(new_x, down, new_tf, d, hh),
                t.ticks, t.swapped, t.delta_change,
            ))
        elif t.phase == "scrimmage" and t.swapped:
            # Receiver got ball; their position (in their own frame) shifts by 25
            x, down, tf_val, d, hh = t.core
            new_x  = min(99, x + 25)
            new_tf = goal_to_go(new_x)
            adjusted.append(SuccessorTemplate(
                t.prob, t.phase, _sc(new_x, down, new_tf, d, hh),
                t.ticks, t.swapped, t.delta_change,
            ))
        else:
            # TD / kickoff phase outcomes — carry through unchanged
            adjusted.append(t)
    return adjusted


def safety_kick_option_punt(
    delta: int, h: int,
) -> List[SuccessorTemplate]:
    """Safety free kick via punt-style: uniform over all 10 PUNT_CHART cards.

    Kicker punts from their own 20 (x=20 in kicker's frame).
    """
    templates: List[SuccessorTemplate] = []
    prob_each = 1.0 / len(DEFENSE_CARDS)
    for card in DEFENSE_CARDS:
        sub = punt_templates(
            x=20, down=4, to_first=10,   # down/to_first don't matter for punt logic
            delta=delta, h=h, def_card=card,
        )
        # sub is a List (no ChoiceTemplates for normal punt at x=20; land_x=20+max70=90<100)
        if isinstance(sub, list):
            for t in sub:
                templates.append(SuccessorTemplate(
                    t.prob * prob_each, t.phase, t.core,
                    t.ticks, t.swapped, t.delta_change,
                ))
        else:
            # ChoiceTemplates — shouldn't happen from x=20 but handle defensively
            for branch in sub.branches:
                for t in branch:
                    templates.append(SuccessorTemplate(
                        t.prob * prob_each / len(sub.branches), t.phase, t.core,
                        t.ticks, t.swapped, t.delta_change,
                    ))
    return templates
