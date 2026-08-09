"""Build a PG-361 abstract payload-shape target view.

The input is the already collected PG-350 evaluator-side source-row view and
the local PG-348 registry.  This script does not contact a target and does
not recover raw requests/responses.  It adds only a bounded
``syntax_category_ref`` target slot, with a source-metadata attestation hash,
and leaves the dataset diagnostic-only until an independent runtime and
neural capability audit pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import _target_tokens, sha256_json, validate_pg331_source_row
from app.pg361_payload_shape_slots import ALLOWED_SYNTAX_CATEGORIES, syntax_attestation


DEFAULT_SOURCE = ROOT / "research" / "pg350_oracle_slot_source_rows_v1.json"
DEFAULT_REGISTRY = ROOT / "fixtures" / "pg348" / "registry_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg361_payload_shape_slot_source_rows_v1.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _syntax_category(record: Mapping[str, Any]) -> str:
    """Map an attested surface description to a bounded grammar class.

    This is intentionally a surface rule, not an exploit classifier.  The
    evaluator stores the mapping hash; the model sees only the resulting
    category token.  Unknown source shapes fail closed instead of being
    silently bucketed into ``marker``.
    """

    explicit = str(record.get("syntax_category_ref", "")).casefold().replace("-", "_")
    if explicit:
        if explicit not in ALLOWED_SYNTAX_CATEGORIES or explicit in {"none", "unknown"}:
            raise ValueError("registry syntax_category_ref is not allow-listed")
        return explicit
    role = str(record.get("parameter_role", "")).casefold().replace("-", "_")
    response = str(record.get("response_shape", "")).casefold().replace("-", "_")
    script = str(record.get("script_surface", "")).casefold().replace("-", "_")
    transport = str(record.get("transport_method", "")).casefold()
    if not role or not response or not transport:
        raise ValueError("registry surface is incomplete for syntax category")
    if role in {"attribute_value", "path_segment", "fragment_identifier"} or "attribute" in response or "fragment" in response:
        return "delimiter_boundary"
    if role == "json_value" or "json" in response or "json" in script:
        return "structured_value"
    if "error_shape" in response or "parser" in script:
        return "parser_node"
    if "302" in response or "redirect" in script:
        return "redirect_control"
    if script not in {"", "none"}:
        return "expression_node" if any(word in script for word in ("event", "dialog", "inline", "module")) else "state_transition"
    if transport == "post" and role in {"form_field", "note_text", "notice_state", "step_index"}:
        return "boolean_branch"
    return "marker"


def build(source: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    registry_rows = list(registry.get("records") or [])
    by_source_hash = {str(row.get("source_hash")): row for row in registry_rows if row.get("source_hash")}
    source_rows = list(source.get("records") or [])
    if not source_rows:
        raise ValueError("source dataset contains no records")
    output_rows: list[dict[str, Any]] = []
    attestations: dict[str, dict[str, str]] = {}
    syntax_counts: Counter[str] = Counter()
    failures: list[str] = []
    for source_row in source_rows:
        row = copy.deepcopy(dict(source_row))
        source_meta = dict(row.get("source_meta") or {})
        registry_row = by_source_hash.get(str(source_meta.get("source_digest")))
        if registry_row is None:
            failures.append(f"missing_registry_source:{row.get('record_id', '')}")
            continue
        category = _syntax_category(registry_row)
        attestation = syntax_attestation(registry_row, category)
        target = dict(row.get("target_projection") or {})
        target["syntax_category_ref"] = category
        row["target_projection"] = target
        row["target_tokens"] = _target_tokens(target)
        # This is a diagnostic target-slot view.  It never upgrades old
        # operator/replay status merely because a slot was appended.
        row["operator_reviewed"] = False
        row["training_eligible"] = False
        promotion = dict(row.get("promotion") or {})
        promotion.update(
            {
                "training_eligible": False,
                "memory_promotion_allowed": False,
                "payload_catalog_promotion_allowed": False,
                "vulnerability_claim_allowed": False,
            }
        )
        row["promotion"] = promotion
        row.pop("record_sha256", None)
        row["record_sha256"] = sha256_json(row)
        check = validate_pg331_source_row(row)
        if not check["valid"]:
            failures.extend(f"{row.get('record_id', '')}:{item}" for item in check["failures"])
            continue
        output_rows.append(row)
        attestations[str(row["record_id"])] = attestation
        syntax_counts[category] += 1
    if failures:
        raise ValueError("PG-361 source-row validation failed: " + ", ".join(sorted(failures)[:10]))
    source_hash = sha256_json(source)
    registry_hash = sha256_json(registry)
    return {
        "schema_version": "pg361-payload-shape-slot-source-rows-v1",
        "status": "completed_syntax_slot_diagnostic_only",
        "source_dataset_sha256": source_hash,
        "source_registry_sha256": registry_hash,
        "records": output_rows,
        "syntax_category_attestations": attestations,
        "counts": {
            "records": len(output_rows),
            "source_records": len(source_rows),
            "syntax_category_unique": len(syntax_counts),
            "syntax_category_counts": dict(sorted(syntax_counts.items())),
            "training_eligible_rows": 0,
            "implementation_holdout_rows": sum(row.get("split") == "implementation_holdout" for row in output_rows),
            "train_rows": sum(row.get("split") == "train" for row in output_rows),
        },
        "information_gate": {
            "source_metadata_only": True,
            "raw_payload_in_context": False,
            "raw_response_in_context": False,
            "syntax_category_is_evaluator_attested": True,
            "predictive_entropy": "not_run",
            "status": "diagnostic_only",
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-361 abstract syntax-category target slots")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(_load(args.source), _load(args.registry))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
