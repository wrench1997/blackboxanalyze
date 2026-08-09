"""PG-344 A800 launcher with a distinct seed block.

The implementation lives in the audited PG-343 runner; this thin launcher
only changes the pre-registered seed identities and keeps the same gates,
context firewall, candidate-only semantics, and promotion closures.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_pg343_a800_target_conditioned_smoke as base  # noqa: E402

SEEDS = (34401, 34402, 34403)


def main() -> int:
    previous = base.SEEDS
    base.SEEDS = SEEDS
    try:
        return base.main()
    finally:
        base.SEEDS = previous


if __name__ == "__main__":
    raise SystemExit(main())
