#!/usr/bin/env bash
# Local smoke test for the GTX 1050 (4 GB) — fast sanity, not a real experiment.
# Usage: bash scripts/smoke_test.sh
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

echo "==> uv sync"
uv sync --quiet

echo "==> environment check"
uv run python scripts/check_env.py

echo "==> unit tests (gradcheck + R^2 equivalence)"
if compgen -G "tests/test_*.py" > /dev/null; then
    uv run pytest -q
else
    echo "    (no tests yet — added in M2)"
fi

# Tiny end-to-end training run (added in M3): a ~<=1M-param complex-NoPE model
# on the 'copy' task for a few hundred steps; asserts the loss decreases.
if [ -f experiments/train.py ]; then
    echo "==> tiny training run (smoke config)"
    uv run python -m experiments.train --smoke --task copy
else
    echo "==> tiny training run skipped (experiments/train.py lands in M3)"
fi

echo "==> smoke test complete"
