"""Overnight Q4 full-game solve. Run via: sh run.sh"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from football_strategy.chart_parser import load_scrimmage_chart, load_punt_chart
from football_strategy.reachability import kickoff_key
from football_strategy.solve_full_game import solve_full_game

sc = load_scrimmage_chart()   # defaults to Ball Control
pc = load_punt_chart()

V = solve_full_game(
    sc, pc,
    kickoff_key(q=4, tau=60, delta=0, h=0),
    verbose=True,
    n_workers=8,
    save_path="q4_full_v",
)
