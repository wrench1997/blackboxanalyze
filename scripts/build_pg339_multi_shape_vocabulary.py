"""Build the append-only context vocabulary used by the PG-339 smoke.

The vocabulary is assembled from already-frozen ontology manifests, never from
PG-339 holdout rows.  This keeps the shape holdout out of vocabulary fitting
while still allowing the decoder to evaluate an existing abstract token
coordinate system.  No records, wire values, payloads, responses, or oracle
answers are copied into the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASES = (
    ROOT / "research" / "pg333_three_impl_get_post_diagnostic_vocabulary_v1.json",
    ROOT / "research" / "pg338_information_preserving_vocabulary_v1.json",
)
DEFAULT_OUTPUT = ROOT / "research" / "pg339_multi_shape_vocabulary_v1.json"
SCHEMA = "pg339-multi-shape-vocabulary-v1"
FORBIDDEN = (
    "family=", "implementation=", "route=", "route_literal=", "source=",
    "image=", "path=", "url=", "payload=", "payload_", "raw_",
    "response_body=", "response_body_text=", "oracle=", "evaluator=", "canary=",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"vocabulary manifest must be an object: {path}")
    return value


def build(base_paths: Sequence[Path] = DEFAULT_BASES) -> dict[str, Any]:
    context: list[str] = []
    forbidden: set[str] = set()
    source_hashes: list[dict[str, str]] = []
    for path in base_paths:
        manifest = _load(path)
        source_hashes.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for token in list(manifest.get("context_tokens") or []):
            token = str(token)
            if any(marker in token.casefold() for marker in FORBIDDEN):
                forbidden.add(token)
            elif token not in context:
                context.append(token)
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "diagnostic_only" if context and not forbidden else "blocked",
        "source_manifests": source_hashes,
        "context_tokens": context,
        "target_tokens": [],
        "context_vocabulary_size": len(context),
        "target_vocabulary_size": 0,
        "forbidden_tokens": sorted(forbidden),
        "append_only": True,
        "holdout_rows_used_for_vocabulary": False,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    result["vocabulary_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-339 append-only vocabulary from frozen manifests")
    parser.add_argument("--base", type=Path, action="append", dest="bases")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build(tuple(args.bases) if args.bases else DEFAULT_BASES)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if args.json else {"status": result["status"], "context_vocabulary_size": result["context_vocabulary_size"], "vocabulary_sha256": result["vocabulary_sha256"]}, ensure_ascii=False))
    return 0 if result["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
