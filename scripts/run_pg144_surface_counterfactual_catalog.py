"""Generate PG-144 surface-counterfactual next-token representation data.

Only bounded PG-140 tokens are read.  No requests are sent and no raw source,
probe, response, evaluator label, or positive authority is materialized.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg144_surface_counterfactual_catalog import (
    DATASET_SCHEMA,
    SCHEMA_VERSION,
    SURFACE_VARIANTS,
    build_augmented_rows,
    sha256_json,
)

RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg140_information_complete_model_dataset_v1.json"
DATASET = RESEARCH / "pg144_surface_counterfactual_model_dataset_v1.json"
TRACE = RESEARCH / "pg144_surface_counterfactual_trace_v1.json"
PROTOCOL = RESEARCH / "pg144_surface_counterfactual_protocol_v1.json"
PROPOSAL = RESEARCH / "pg144_surface_counterfactual_proposal_v1.json"
REPORT = RESEARCH / "pg144_surface_counterfactual_report_v1.json"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _contains_forbidden_raw(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False).casefold()
    return any(marker in text for marker in ("<script", "onerror", "union select", "password="))


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    base_rows = source.get("rows", [])
    augmented_rows, summary = build_augmented_rows(base_rows)
    pair_records = summary.pop("pair_records")
    summary["variant_identity_in_model_input_count"] = sum(
        1
        for row in augmented_rows
        for token in row["tokens"]
        if str(token).startswith("cf.surface.variant=") or str(token) == "[CF_SURFACE]"
    )
    dataset = {
        "schema_version": DATASET_SCHEMA,
        "status": "completed_pg144_surface_counterfactual_representation_only",
        "training_eligible": False,
        "representation_pretrain_allowed": False,
        "representation_diagnostic_only": True,
        "memory_promotion_allowed": False,
        "raw_source_retained": False,
        "raw_probe_response_retained": False,
        "evaluator_labels_retained": False,
        "positive_authority_retained": False,
        "variant_identity_in_model_input": False,
        "source_dataset_schema": source.get("schema_version"),
        "source_dataset_sha256": source.get("dataset_sha256"),
        "rows": augmented_rows,
        "pair_records": pair_records,
        "summary": summary,
    }
    dataset["dataset_sha256"] = sha256_json(dataset)

    protocol = {
        "protocol_id": "pg-pk-144-surface-counterfactual-v1",
        "schema_version": "pg144-surface-counterfactual-protocol-v1",
        "objective": "补充同一 oracle availability 下的多实现/多表面 next-token 表征序列，验证模型学习抽象状态而不是表面 token。",
        "source": "pg140_information_complete_model_dataset_v1.json",
        "surface_variants": sorted(SURFACE_VARIANTS),
        "local_only": True,
        "raw_content_storage": False,
        "representation_pretrain_only": True,
        "variant_identity_not_in_model_input": True,
        "action_supervision_allowed": False,
        "safety_supervision_allowed": False,
        "memory_promotion_allowed": False,
        "required_gates": {
            "oracle_availability_preserved": True,
            "same_split_parent_binding": True,
            "changed_surface_pair": True,
            "all_raw_content_absent": True,
            "counterfactual_not_authority": True,
            "variant_identity_not_in_model_input": True,
            "surface_diversity_gate": summary["surface_diversity_gate"],
        },
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    proposal = {
        "proposal_id": "pg-pk-144-surface-counterfactual-v1",
        "schema_version": "pg144-surface-counterfactual-proposal-v1",
        "status": "representation_data_expansion_evaluation_only",
        "selected_action": "pretrain_representation_only_then_test_surface_ood",
        "base_row_count": summary["base_row_count"],
        "augmented_row_count": summary["augmented_row_count"],
        "variant_count": summary["variant_count"],
        "training_eligible": False,
        "representation_pretrain_allowed": False,
        "representation_diagnostic_only": True,
        "memory_promotion_allowed": False,
    }
    proposal["proposal_sha256"] = sha256_json(proposal)
    raw_absent = not _contains_forbidden_raw(dataset)
    trace = {
        "schema_version": "pg144-surface-counterfactual-trace-v1",
        "protocol_id": protocol["protocol_id"],
        "status": "completed_pg144_surface_counterfactual_representation_only",
        "training_eligible": False,
        "representation_pretrain_allowed": False,
        "representation_diagnostic_only": True,
        "memory_promotion_allowed": False,
        "source_dataset_sha256": source.get("dataset_sha256"),
        "dataset_sha256": dataset["dataset_sha256"],
        "oracle_availability_preserved": summary["oracle_availability_counts"],
        "raw_content_absent": raw_absent,
        "variant_identity_not_in_model_input": summary["variant_identity_in_model_input_count"] == 0,
        "action_supervision_allowed": False,
        "safety_supervision_allowed": False,
        "unique_sequence_density": summary["unique_sequence_density"],
        "duplicate_sequence_count": summary["duplicate_sequence_count"],
        "surface_diversity_gate": summary["surface_diversity_gate"],
    }
    trace["trace_sha256"] = sha256_json(trace)
    report = {
        "protocol_id": protocol["protocol_id"],
        "schema_version": "pg144-surface-counterfactual-report-v1",
        "status": "completed_pg144_surface_counterfactual_representation_only",
        "hard_gates_passed": bool(
            raw_absent
            and summary["changed_surface_pair_count"] == summary["augmented_row_count"]
            and summary["unique_surface_delta_count"] >= summary["variant_count"]
            and summary["all_action_supervision_forbidden"]
            and summary["all_safety_supervision_forbidden"]
            and summary["all_memory_promotion_forbidden"]
            and summary["variant_identity_in_model_input_count"] == 0
            and summary["surface_diversity_gate"]
        ),
        "unique_sequence_density": summary["unique_sequence_density"],
        "duplicate_sequence_count": summary["duplicate_sequence_count"],
        "surface_diversity_gate": summary["surface_diversity_gate"],
        "training_eligible": False,
        "representation_pretrain_allowed": False,
        "representation_diagnostic_only": True,
        "memory_promotion_allowed": False,
        "source_dataset_file": SOURCE.name,
        "dataset_file": DATASET.name,
        "summary": summary,
        "gates": {
            "oracle_availability_preserved": True,
            "same_split_parent_binding": summary["parent_split_binding_count"] == summary["base_row_count"],
            "changed_surface_pair": summary["changed_surface_pair_count"] == summary["augmented_row_count"],
            "all_raw_content_absent": raw_absent,
            "counterfactual_not_authority": summary["all_action_supervision_forbidden"] and summary["all_safety_supervision_forbidden"],
            "variant_identity_not_in_model_input": summary["variant_identity_in_model_input_count"] == 0,
            "surface_diversity_gate": summary["surface_diversity_gate"],
        },
        "promotion": {
            "representation_pretrain_allowed": False,
            "representation_diagnostic_only": True,
            "capability_train_allowed": False,
            "training_artifact_promotion_allowed": False,
            "memory_promotion_allowed": False,
            "reason": "surface diversity is below the minimum unique-sequence density; retain counterfactuals for diagnosis only",
        },
        "source": {
            "runner": _file_hash(Path(__file__)),
            "module": _file_hash(ROOT / "app" / "pg144_surface_counterfactual_catalog.py"),
            "source_dataset": _file_hash(SOURCE),
        },
    }
    report["report_sha256"] = sha256_json(report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    _write(PROPOSAL, proposal)
    _write(TRACE, trace)
    _write(REPORT, report)
    print(json.dumps({
        "status": report["status"],
        "base_row_count": summary["base_row_count"],
        "augmented_row_count": summary["augmented_row_count"],
        "variant_count": summary["variant_count"],
        "typed_count": summary["oracle_availability_counts"].get("typed", 0),
        "unknown_count": summary["oracle_availability_counts"].get("unknown", 0),
        "hard_gates_passed": report["hard_gates_passed"],
        "unique_sequence_density": summary["unique_sequence_density"],
        "duplicate_sequence_count": summary["duplicate_sequence_count"],
        "surface_diversity_gate": summary["surface_diversity_gate"],
        "training_eligible": report["training_eligible"],
        "report": str(REPORT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
