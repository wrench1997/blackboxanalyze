"""PG-343 v4 launcher: target-only SFT with a higher bounded learning rate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_pg343_a800_target_conditioned_smoke as _base


SEEDS = (34341, 34342, 34343)


def main() -> int:
    _base.SEEDS = SEEDS
    return _base.main()


if __name__ == "__main__":
    raise SystemExit(main())
