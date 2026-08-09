"""PG-347 wrapper: reuse PG-346 structured slot runner with new seeds."""

from __future__ import annotations

import sys
from pathlib import Path

# When invoked as a file, Python puts ``scripts/`` ahead of the workspace and
# an unrelated site-packages ``scripts`` package can win the import.  Pin the
# repository root before importing the PG-346 base runner so the remote smoke
# uses the locked local implementation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_pg346_a800_structured_target_slot_smoke as base


SEEDS = (34701, 34702, 34703)


def main() -> int:
    previous = base.SEEDS
    base.SEEDS = SEEDS
    try:
        return base.main()
    finally:
        base.SEEDS = previous


if __name__ == "__main__":
    raise SystemExit(main())
