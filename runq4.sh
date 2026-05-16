#!/usr/bin/env bash
# Full Q4 solve: Stage 1 (no timeouts) then Stage 2 (all 16 TO combinations).
# Logs to solve.log. Safe to close terminal after starting.
set -e

source .venv/bin/activate
nohup python run_all.py > solve.log 2>&1 &

echo "Solve running in background (PID $!)"
echo "Watch progress: tail -f solve.log"
