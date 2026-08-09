"""Append PG-350 abstract oracle/negative-control target tokens to PG-349 v8."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "research" / "pg349_dynamic_typed_vocabulary_v8.json"
DEFAULT_DATASET = ROOT / "research" / "pg350_oracle_slot_source_rows_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg350_oracle_slot_vocabulary_v1.json"
ORACLE_VALUES = ("none", "unknown", "reflection", "response_shape", "parser_shape", "dom_shape", "typed_state_delta", "typed_effect", "negative_no_effect")
NEGATIVE_VALUES = ("unknown", "not_observed", "not_required", "matched_triplet")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extend(base: dict[str, Any], *, base_sha256: str, dataset_path: Path) -> dict[str, Any]:
    if (base.get("vocabulary_policy") or {}).get("append_only") is not True:
        raise ValueError("PG-350 base vocabulary must be append-only")
    additions = [*(f"oracle_ref={value}" for value in ORACLE_VALUES), *(f"negative_control_presence_ref={value}" for value in NEGATIVE_VALUES)]
    context = list(dict.fromkeys(str(token) for token in base.get("context_tokens") or []))
    target = list(dict.fromkeys([*(str(token) for token in base.get("target_tokens") or []), *additions]))
    shared = sorted((set(context) | set(target) | set(str(token) for token in base.get("shared_tokens") or [])) - {"[PAD]", "[UNK]"})
    result = dict(base)
    result["schema_version"] = "pg350-oracle-slot-vocabulary-v1"
    result["base_vocabulary"] = str(DEFAULT_BASE.relative_to(ROOT))
    result["base_vocabulary_sha256"] = base_sha256
    result["oracle_slot_dataset"] = str(dataset_path.resolve().relative_to(ROOT.resolve()))
    result["oracle_slot_dataset_sha256"] = _file_sha(dataset_path)
    result["oracle_slot_values"] = {"oracle_ref": list(ORACLE_VALUES), "negative_control_presence_ref": list(NEGATIVE_VALUES)}
    result["context_tokens"] = sorted(context)
    result["target_tokens"] = sorted(target)
    result["shared_tokens"] = shared
    counts = dict(result.get("counts") or {})
    counts.update({"context_total": len(context), "target_total": len(target), "shared_total": len(shared), "oracle_slot_reserved": len(additions)})
    result["counts"] = counts
    policy = dict(result.get("vocabulary_policy") or {})
    policy.update({"append_only": True, "raw_literal_tokens_allowed": False, "oracle_slot_abstract_only": True, "negative_control_answer_in_context": False, "runtime_binding": "evaluator_only"})
    result["vocabulary_policy"] = policy
    result["promotion"] = {"training": False, "memory": False, "payload": False, "vulnerability": False}
    result["vocabulary_sha256"] = ""
    result["vocabulary_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-vocabulary", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = json.loads(args.base_vocabulary.read_text(encoding="utf-8-sig"))
    result = extend(base, base_sha256=_file_sha(args.base_vocabulary), dataset_path=args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.get("status"), "counts": result.get("counts"), "vocabulary_sha256": result["vocabulary_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
