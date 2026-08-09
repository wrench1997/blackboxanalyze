"""Deterministically build the PG-379 implementation-A route manifest.

This script only hashes local source and writes a manifest.  It does not build
or run Docker, open a socket, or contact a target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # module invocation: ``python -m fixtures.pg379.impl_a.build_manifest``
    from .app import manifest, validate_manifest
except ImportError:  # direct deterministic script invocation from repo root
    from app import manifest, validate_manifest


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "manifest_v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    document = manifest()
    validation = validate_manifest(document)
    if validation.get("status") != "passed":
        raise SystemExit(f"manifest validation failed: {validation}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
