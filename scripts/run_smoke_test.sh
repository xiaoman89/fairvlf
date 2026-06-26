#!/usr/bin/env bash
# ==========================================================================
# FairVLF smoke test runner (for AutoDL or any Linux GPU box).
# Verifies the pipeline end-to-end on a tiny subset before a full run.
# ==========================================================================
set -e

# Move to repo root (this script lives in scripts/).
cd "$(dirname "$0")/.."

# Make the package importable.
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "=== Step 1: loss sanity checks (CPU, no model) ==="
python scripts/test_losses.py

echo ""
echo "=== Step 2: generate a tiny dummy dataset ==="
python scripts/make_dummy_manifest.py --n 56

echo ""
echo "=== Step 3: run the smoke-test training config ==="
echo "(this downloads the backbone the first time; be patient)"
python -m fairvlf.train --config configs/smoke_test.yaml

echo ""
echo "=== Smoke test finished. Check the newest folder under runs/ ==="
ls -dt runs/*/ | head -1
