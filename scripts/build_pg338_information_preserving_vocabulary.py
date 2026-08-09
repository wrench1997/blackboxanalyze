"""Build an append-only context/target vocabulary for PG-338."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg338_information_preserving_process_token_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg338_information_preserving_vocabulary_v1.json"
SCHEMA = "pg338-information-preserving-vocabulary-v1"
FORBIDDEN = ("family=", "implementation=", "route=", "route_literal=", "source=", "image=", "path=", "url=", "payload=", "payload_", "raw_", "response_body=", "response_body_text=", "oracle=", "evaluator=", "canary=")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def build(dataset: dict[str, Any]) -> dict[str, Any]:
    context: list[str] = []
    target: list[str] = []
    forbidden: set[str] = set()
    for row in list(dataset.get("records") or []):
        for token in [str(t) for t in row.get("context_tokens") or []]:
            if any(marker in token.casefold() for marker in FORBIDDEN):
                forbidden.add(token)
            elif token not in context:
                context.append(token)
        for token in [str(t) for t in row.get("target_tokens") or []]:
            if token not in target:
                target.append(token)
    status = "diagnostic_only" if not forbidden and context and target and dataset.get("status") == "diagnostic_only_full_axis_cross_implementation" else "blocked"
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": status,
        "dataset_sha256": str(dataset.get("dataset_sha256", "")),
        "context_tokens": context,
        "target_tokens": target,
        "context_vocabulary_size": len(context),
        "target_vocabulary_size": len(target),
        "forbidden_tokens": sorted(forbidden),
        "append_only": True,
        "full_axis_required": True,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["vocabulary_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-338 full-axis vocabulary")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    result = build(dataset)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if result["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
