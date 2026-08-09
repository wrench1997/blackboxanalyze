"""Build a PG-340 append-only vocabulary from the frozen PG-339 vocabulary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "research" / "pg339_multi_shape_vocabulary_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg340_balanced_axis_vocabulary_v1.json"
SCHEMA = "pg340-balanced-axis-vocabulary-v1"
FORBIDDEN = ("family=", "implementation=", "route=", "url=", "path=", "payload=", "payload_", "raw_", "response_body=", "oracle=", "evaluator=")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def build(base_path: Path = DEFAULT_BASE) -> dict[str, Any]:
    base = json.loads(base_path.read_text(encoding="utf-8-sig"))
    if not isinstance(base, Mapping):
        raise ValueError("base vocabulary must be an object")
    tokens: list[str] = []
    forbidden: list[str] = []
    for token in list(base.get("context_tokens") or []):
        token = str(token)
        if any(marker in token.casefold() for marker in FORBIDDEN):
            forbidden.append(token)
        elif token not in tokens:
            tokens.append(token)
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "diagnostic_only" if tokens and not forbidden else "blocked",
        "base_vocabulary_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "context_tokens": tokens,
        "target_tokens": [],
        "context_vocabulary_size": len(tokens),
        "target_vocabulary_size": 0,
        "forbidden_tokens": sorted(set(forbidden)),
        "append_only": True,
        "holdout_rows_used_for_vocabulary": False,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["vocabulary_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-340 append-only vocabulary")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build(args.base)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "context_vocabulary_size": result["context_vocabulary_size"], "vocabulary_sha256": result["vocabulary_sha256"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if result["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
