#!/usr/bin/env bash
# Creates .venv and installs all dependencies.
# Run once on a new machine before running runq4.sh.
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy==2.0.2 scipy==1.13.1 pandas==2.3.3 tqdm==4.67.3

echo ""
echo "Setup complete. Run: bash runq4.sh"
