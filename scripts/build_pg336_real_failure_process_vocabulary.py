"""Build the append-only context/target vocabulary for PG-336 diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg336_real_failure_process_token_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg336_real_failure_process_vocabulary_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build(data: dict[str, Any], *, base: dict[str, Any] | None = None) -> dict[str, Any]:
    current_context = {str(token) for token in list(data.get("context_tokens") or [])}
    current_target = {str(token) for token in list(data.get("target_tokens") or [])}
    base = base if isinstance(base, dict) else {}
    context = sorted({str(token) for token in list(base.get("context_tokens") or [])} | current_context)
    target = sorted({str(token) for token in list(base.get("target_tokens") or [])} | current_target)
    result = {
        "schema_version": "pg336-real-failure-process-vocabulary-v1",
        "status": "diagnostic_only",
        "append_only": True,
        "dataset_sha256": data.get("dataset_sha256"),
        "base_vocabulary_sha256": base.get("vocabulary_sha256"),
        "context_tokens": context,
        "target_tokens": target,
        "reserved": ["[BOS]", "[EOS]", "[TARGET_BOS]", "[TARGET_EOS]"],
        "coverage": {"real_failure_trace": True, "ask_preflight": True, "negative_review": True, "seed_holdout": True, "independent_implementation_holdout": False},
        "training_eligibility": {"allowed": False, "reason": "single_implementation_process_diagnostic"},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["vocabulary_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-336 process-token vocabulary")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base", type=Path)
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    base = json.loads(args.base.read_text(encoding="utf-8-sig")) if args.base else None
    result = build(data, base=base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "context": len(result["context_tokens"]), "target": len(result["target_tokens"]), "vocabulary_sha256": result["vocabulary_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
