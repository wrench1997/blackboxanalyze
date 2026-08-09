"""Build the PG-386 fixture-bound payload-generation training view.

The persisted dataset contains only abstract contexts and a bounded output
class (``fixture_double_layer_value`` or ``ask``).  It deliberately does not
store a URL, wire body, raw canary, response body, or evaluator answer.  The
runtime decoder maps the class to an ephemeral string only after a reviewed
local fixture has been selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "research/pg385_filter_repair_adversarial_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research/pg386_fixture_payload_generation_dataset_v1.json"
SCHEMA_VERSION = "pg386-fixture-payload-generation-dataset-v1"
FORBIDDEN = ("http://", "https://", "payload=", "wire=", "response_body", "<script", "javascript:")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError("PG-385 source dataset is malformed")
    return value


def _label(row: Mapping[str, Any]) -> str:
    targets = {str(token).split("=", 1)[0]: str(token).split("=", 1)[1] for token in row.get("target_tokens", []) if "=" in str(token)}
    if targets.get("safe_to_send") == "1" and targets.get("encoding_ref") == "double_layer_order_sensitive":
        return "fixture_double_layer_value"
    return "ask"


def build_dataset(input_path: Path = DEFAULT_INPUT) -> dict[str, Any]:
    source = _load(input_path)
    records: list[dict[str, Any]] = []
    for source_row in source["records"]:
        if not isinstance(source_row, Mapping):
            raise ValueError("source record is not an object")
        context = [str(token) for token in source_row.get("context_tokens", [])]
        if any(any(fragment in token.casefold() for fragment in FORBIDDEN) for token in context):
            raise ValueError("raw/evaluator marker reached PG-386 context")
        if source_row.get("raw_payload_stored") is not False or source_row.get("raw_response_body_stored") is not False or source_row.get("oracle_answer_in_context") is not False:
            raise ValueError("source row does not satisfy PG-386 firewall")
        output_class = _label(source_row)
        record = {
            "record_id": str(source_row.get("record_id", "")),
            "split": str(source_row.get("split", "")),
            "implementation_id": str(source_row.get("implementation_id", "")),
            "source_hash": str(source_row.get("source_hash", "")),
            "seed": int(source_row.get("seed", 0)),
            "method": str(source_row.get("method", "")),
            "scenario_id": str(source_row.get("scenario_id", "")),
            "role": str(source_row.get("role", "")),
            "context_tokens": context,
            "payload_output_class": output_class,
            "payload_grammar_id": "pg386-local-filter-canary-v1",
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "oracle_answer_in_context": False,
            "training_eligible": False,
            "promotion": {
                "training_allowed": False,
                "memory_promotion_allowed": False,
                "payload_catalog_promotion_allowed": False,
                "vulnerability_claim_allowed": False,
            },
        }
        record["record_sha256"] = _sha(record)
        records.append(record)
    if not records or not any(row["split"] == "train" for row in records) or not any(row["split"] == "implementation_holdout" for row in records):
        raise ValueError("PG-386 dataset needs train and implementation_holdout rows")
    counts = {
        "records": len(records),
        "train": sum(row["split"] == "train" for row in records),
        "implementation_holdout": sum(row["split"] == "implementation_holdout" for row in records),
        "methods_get": sum(row["method"] == "GET" for row in records),
        "methods_post": sum(row["method"] == "POST" for row in records),
        "output_classes": {name: sum(row["payload_output_class"] == name for row in records) for name in sorted({row["payload_output_class"] for row in records})},
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "abstract_fixture_payload_candidate_only",
        "objective": "filtered local canary -> model judges feedback -> bounded fixture output class -> ephemeral local string",
        "source_dataset": str(input_path),
        "source_dataset_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "counts": counts,
        "output_contract": {
            "model_may_emit_fixture_bound_string": True,
            "model_may_emit_arbitrary_string": False,
            "grammar_id": "pg386-local-filter-canary-v1",
            "raw_string_persisted": False,
            "raw_string_in_context": False,
            "evaluator_last_hop_only": True,
            "loopback_only": True,
        },
        "records": records,
        "training_eligible": 0,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    result["dataset_sha256"] = _sha(result)
    return result


def write_dataset(output_path: Path = DEFAULT_OUTPUT, input_path: Path = DEFAULT_INPUT) -> dict[str, Any]:
    result = build_dataset(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_dataset(args.output, args.input)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_dataset", "write_dataset"]
