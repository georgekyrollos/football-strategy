"""
All numeric constants, probability tables, and game parameters.
No imports from this project.
"""

# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

TICKS_PER_QUARTER = 60     # 1 tick = 15 sec; 60 ticks = 15-minute quarter
TWO_MIN_TICK = 8           # automatic clock stop at tau=8 in Q2 and Q4

# Ticks consumed per outcome category.
# TODO: Confirm whether a 20-yard in-bounds gain costs 30 sec (2 ticks) or
# 45 sec (3 ticks).  The printed Time Keeping Table wording uses "0-20" and
# "20+" as the two buckets, leaving it ambiguous whether exactly 20 yards is
# "short" or "long".  Current assumption: gains of 0-19 yards → 2 ticks;
# gains of ≥20 yards → 3 ticks (i.e., 20 is in the "long" bucket).
# Clarify with the physical rulebook before finalising.
TICKS = {
    "gain_short":    2,   # 0–19 yd inbounds gain  (30 sec) — SEE TODO ABOVE
    "gain_long":     3,   # ≥20 yd inbounds gain   (45 sec) — SEE TODO ABOVE
    "loss":          2,   # any loss               (30 sec)
    "out_of_bounds": 1,   # any out-of-bounds play (15 sec)
    "incomplete":    1,   # incomplete pass        (15 sec)
    "interception":  2,   # interception           (30 sec)
    "penalty":       1,   # penalty                (15 sec)
    "fumble":        1,   # fumble                 (15 sec)
    "kickoff":       1,   # kickoff                (15 sec)
    "field_goal":    1,   # field goal attempt     (15 sec)
    "punt":          1,   # punt                   (15 sec)
    "pat":           0,   # extra point            ( 0 sec)
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

TD_POINTS     = 6
SAFETY_POINTS = 2
FG_POINTS     = 3
PAT_PROB_GOOD = 34 / 36
PAT_PROB_MISS =  2 / 36

# ---------------------------------------------------------------------------
# Field goal success probabilities
# Each entry: (min_dist, max_dist, probability)
# distance = 100 - x  (yards from offense's current position to opponent's goal)
# ---------------------------------------------------------------------------

FG_PROB_TABLE = [
    ( 1, 12, 32 / 36),
    (13, 22, 28 / 36),
    (23, 32, 18 / 36),
    (33, 38, 12 / 36),
    (39, 45,  2 / 36),
]
FG_MIN_X = 55   # offense must be at x ≥ 55 (at most 45 yd from opponent's goal)


def fg_prob(x: int) -> float:
    """Return FG success probability for a ball at position x (offense frame)."""
    dist = 100 - x
    for lo, hi, p in FG_PROB_TABLE:
        if lo <= dist <= hi:
            return p
    return 0.0   # out of range — should not happen for legal FG attempt


# ---------------------------------------------------------------------------
# Long Gain (LG) distribution
# Each entry: (gain_yards, probability)
# Die 1 roll: 2-6 give fixed gains 50, 45, 40, 35, 30.
# Die 1 roll: 1 gives 50 + 10 * (die 2), i.e. 60–110 each with prob 1/36.
# Total: 5*(1/6) + 6*(1/36) = 30/36 + 6/36 = 1  ✓
# ---------------------------------------------------------------------------

LG_DIST = [
    ( 30, 1/6),
    ( 35, 1/6),
    ( 40, 1/6),
    ( 45, 1/6),
    ( 50, 1/6),
    ( 60, 1/36),
    ( 70, 1/36),
    ( 80, 1/36),
    ( 90, 1/36),
    (100, 1/36),
    (110, 1/36),
]

# ---------------------------------------------------------------------------
# Normal kickoff (2d6)
# Each entry: roll → (result_type, param)
# result_type values: "receiver_ball", "fumble", "penalty", "long_gain"
# ---------------------------------------------------------------------------

KICKOFF_NORMAL = {
     2: ("fumble",         None),
     3: ("penalty",         -10),  # -10 yards applied after rerolling for yard line
     4: ("receiver_ball",    10),
     5: ("receiver_ball",    15),
     6: ("receiver_ball",    20),
     7: ("receiver_ball",    25),
     8: ("receiver_ball",    30),
     9: ("receiver_ball",    35),
    10: ("receiver_ball",    40),
    11: ("long_gain",          0),  # LG table result directly (not "from yard 40")
    12: ("long_gain",          5),  # LG table result + 5 yards
}
# Number of 2d6 ways to make each roll (sums to 36)
KICKOFF_NORMAL_COUNTS = {
     2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
     8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
}
# Fumble note: roll again (same 2d6) to determine yard line.
#   Second fumble on the reroll → receiving team still gets the ball.
# Penalty note: the -10 is applied relative to the yard-line produced by a reroll.

# ---------------------------------------------------------------------------
# Onside kickoff (1d6)
# Add +1 to roll if kicking team is NOT trailing (tied or ahead); cap at 6.
# Values: (who_gets_ball, x_from_that_teams_own_goal)
#   "kicker"   → kicking team keeps the ball; x measured from kicker's own goal
#   "receiver" → receiving team gets ball;    x measured from receiver's own goal
# ---------------------------------------------------------------------------

KICKOFF_ONSIDE = {
    1: ("kicker",    40),   # kicker's ball at kicker's 40  → x=40 in kicker's frame
    2: ("kicker",    40),
    3: ("receiver",  65),   # receiver's ball at kicker's 35 → x=65 from receiver's goal
    4: ("receiver",  65),
    5: ("receiver",  65),
    6: ("receiver",  70),   # receiver's ball at kicker's 30 → x=70 from receiver's goal
}

# ---------------------------------------------------------------------------
# Punt chart (indexed by defense card A–J)
# format: (punt_yards, return_type, return_yards)
#   punt_yards  : how far the ball travels from the line of scrimmage
#   return_type : "ob"           = out of bounds, no return
#                 "none"         = no return (fair catch / downed)
#                 "fixed"        = fixed return yardage toward punter
#                 "lg"           = return uses Long Gain table
#                 "fixed_fumble" = fixed return then kicking team recovers fumble
#   return_yards: yards returned toward punter's end zone (ignored for "ob"/"none"/"lg")
# Also used for the safety free-kick option (from kicker's own 20).
# ---------------------------------------------------------------------------

PUNT_CHART = {
    "A": (70, "lg",           0),
    "B": (60, "fixed",       10),
    "C": (50, "ob",           0),
    "D": (50, "fixed",       10),
    "E": (40, "fixed",       10),
    "F": (30, "none",         0),
    "G": (40, "none",         0),
    "H": (60, "fixed",       20),
    "I": (50, "none",         0),
    "J": (50, "fixed_fumble", 20),
}

# ---------------------------------------------------------------------------
# State space bounds (truncation for tractability)
# ---------------------------------------------------------------------------

DELTA_MIN    = -35   # clip offense_score - defense_score at ±35
DELTA_MAX    =  35
TO_FIRST_MAX =  30   # cap yards-to-first-down at 30

# ---------------------------------------------------------------------------
# Timeout clock reduction
# ---------------------------------------------------------------------------

# After a timeout is called, the just-finished play's ticks are reduced:
#   3-tick play → 1 tick  (45 s → 15 s)
#   2-tick play → 0 ticks (30 s → 0 s)
#   1-tick play → 0 ticks (15 s → 0 s)
TIMEOUT_REDUCE = {1: 0, 2: 0, 3: 1}
MAX_TOS = 3   # timeouts per team per half

# ---------------------------------------------------------------------------
# Legal play identifiers
# ---------------------------------------------------------------------------

PLAYS_BASE    = list(range(1, 21))   # standard offense plays 1–20
PUNT_ACTION   = 21                   # 4th-down only
FG_ACTION     = 22                   # when x ≥ FG_MIN_X
DEFENSE_CARDS = list("ABCDEFGHIJ")   # 10 defense cards

# ---------------------------------------------------------------------------
# Red-zone restrictions (x = yards from current offense's own goal line)
# ---------------------------------------------------------------------------

REDZONE_OUTER_X       = 80                    # plays 17–20 illegal at x ≥ 80
REDZONE_OUTER_ILLEGAL = frozenset(range(17, 21))
REDZONE_INNER_X       = 90                    # plays 13–16 illegal at x ≥ 90
REDZONE_INNER_ILLEGAL = frozenset(range(13, 17))
