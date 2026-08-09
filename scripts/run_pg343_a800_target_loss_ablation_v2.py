"""PG-343 v2 launcher for the pre-registered target-loss ablation.

The implementation stays identical to the audited PG-343 runner.  This thin
launcher only supplies a new seed triplet so the v1 artifact remains
reproducible and its code hash remains meaningful.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_pg343_a800_target_conditioned_smoke as _base


SEEDS = (34321, 34322, 34323)


def main() -> int:
    _base.SEEDS = SEEDS
    return _base.main()


if __name__ == "__main__":
    raise SystemExit(main())
