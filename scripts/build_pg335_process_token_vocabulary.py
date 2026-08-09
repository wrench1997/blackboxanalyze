"""Build PG-335 vocabulary from the sanitized process-token dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg335_real_process_token_diagnostic_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg335_real_process_token_vocabulary_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build(data: dict[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": "pg335-real-process-token-vocabulary-v1",
        "status": "diagnostic_only",
        "dataset_sha256": data.get("dataset_sha256"),
        "append_only": True,
        "context_tokens": sorted({str(token) for token in list(data.get("context_tokens") or [])}),
        "target_tokens": sorted({str(token) for token in list(data.get("target_tokens") or [])}),
        "reserved": ["[BOS]", "[CTX_END]", "[TARGET_BOS]", "[TARGET_EOS]"],
        "training_eligibility": {"allowed": False, "reason": "source_grounded_masks_and_diagnostic_failure_rows"},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["vocabulary_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-335 process-token vocabulary")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(json.loads(args.dataset.read_text(encoding="utf-8-sig")))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "context": len(result["context_tokens"]), "target": len(result["target_tokens"]), "vocabulary_sha256": result["vocabulary_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
