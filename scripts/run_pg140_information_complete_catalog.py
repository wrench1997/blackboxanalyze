"""Build PG-140's evaluator-side information-complete catalog.

This is a schema-repair/representation-preparation run, not a vulnerability
scanner and not a capability-training promotion.  It replays only the local
bounded fixtures already used by PG-139 and emits no raw probe or response
content.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg140_information_complete_catalog import (
    MODEL_DATASET_SCHEMA,
    SCHEMA_VERSION,
    build_catalog_row,
    build_manifest,
    make_context_index,
    quality_summary,
    sha256_json,
)


RESEARCH = ROOT / "research"
CATALOG = RESEARCH / "pg140_information_complete_catalog_v1.json"
MODEL_DATASET = RESEARCH / "pg140_information_complete_model_dataset_v1.json"
REPORT = RESEARCH / "pg140_information_complete_catalog_report_v1.json"
TRACE = RESEARCH / "pg140_information_complete_catalog_trace_v1.json"
PROTOCOL = RESEARCH / "pg140_information_complete_catalog_protocol_v1.json"
PROPOSAL = RESEARCH / "pg140_information_complete_catalog_proposal_v1.json"


def _load_pg139() -> Any:
    path = ROOT / "scripts" / "run_pg139_value_head_loio.py"
    spec = importlib.util.spec_from_file_location("pg139_runner_for_pg140_catalog", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-139 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_hash(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rows_by_role(fold: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any]]]:
    result: list[tuple[str, str, Mapping[str, Any]]] = []
    for role in ("train", "dev"):
        result.extend((role, role, row) for row in fold[role])
    for holdout_name, rows in fold["holdout"].items():
        result.extend(("holdout", holdout_name, row) for row in rows)
    return result


def main() -> None:
    runner = _load_pg139()
    targets = asyncio.run(runner._collect())
    folds = runner.PG138._build_folds(targets)
    context_index = make_context_index(targets)
    catalog_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    replayable: dict[str, bool] = {}
    binding_missing = 0

    for fold_name, fold in folds.items():
        for role, role_name, row in _rows_by_role(fold):
            examples = runner._examples([row])
            tokens = examples[0]["tokens"]
            context = next(iter(context_index.get(str(row.get("step_id")), [])), None)
            if context is None:
                binding_missing += 1
            model_row, catalog_row, can_replay = build_catalog_row(
                row,
                fold=fold_name,
                role=role_name if role == "holdout" else role,
                tokens=tokens,
                context=context,
            )
            catalog_rows.append(catalog_row)
            model_rows.append(model_row)
            replayable[catalog_row["catalog_row_id"]] = can_replay

    source_hashes = {
        "runner_pg139": _file_hash("scripts/run_pg139_value_head_loio.py"),
        "catalog_module": _file_hash("app/pg140_information_complete_catalog.py"),
        "parser_variant": _file_hash("app/pg139_parser_variant.py"),
        "causal_model": _file_hash("app/pg136_causal_token_lm.py"),
    }
    manifest = build_manifest(catalog_rows, source_hashes=source_hashes)
    quality = quality_summary(catalog_rows, replayable)
    quality["context_binding_missing_count"] = binding_missing
    quality["all_catalog_row_hashes_valid"] = all(
        row.get("catalog_row_sha256") == sha256_json({key: value for key, value in row.items() if key != "catalog_row_sha256"})
        for row in catalog_rows
    )
    quality["all_required_information_explicit"] = binding_missing == 0 and quality["all_catalog_row_hashes_valid"]

    catalog = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_pg140_information_complete_catalog",
        "training_eligible": False,
        "hard_gates_passed": False,
        "memory_promotion_allowed": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "model_labels_stored": False,
        "manifest": manifest,
        "rows": catalog_rows,
    }
    catalog["catalog_sha256"] = sha256_json(catalog)

    model_dataset = {
        "schema_version": MODEL_DATASET_SCHEMA,
        "status": "representation_pretrain_and_capability_candidate_only",
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "labels_in_model_rows": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "rows": model_rows,
    }
    model_dataset["dataset_sha256"] = sha256_json(model_dataset)

    protocol = {
        "protocol_id": "pg-pk-140-information-complete-catalog-v1",
        "schema_version": "pg140-information-complete-catalog-protocol-v1",
        "objective": "将模型最小投影与 evaluator-side provenance/oracle/evidence 目录解耦，并把缺失值显式化。",
        "local_only": True,
        "raw_content_storage": False,
        "required_manifest_fields": [
            "catalog_schema_version",
            "row_provenance_manifest_sha256",
            "tokenizer_schema_version",
            "tokenizer_config_sha256",
            "source_implementation_manifest_sha256",
            "oracle_contract_manifest_sha256",
            "split_manifest_sha256",
            "omission_policy",
        ],
        "learning_stages": ["schema_repair", "representation_pretrain", "capability_train", "memory_promotion"],
        "capability_train_requires_original_missing_zero": True,
        "memory_promotion_requires_cross_seed_implementation_ood_review": True,
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    proposal = {
        "proposal_id": "pg-pk-140-information-complete-catalog-v1",
        "schema_version": "pg140-information-complete-catalog-proposal-v1",
        "status": "evaluation_only_catalog_repair",
        "selected_action": "repair_catalog_before_training_action_head",
        "observed_gap": quality["original_missing_field_counts"],
        "capability_train_candidate_count": quality["capability_train_candidate_count"],
        "training_eligible": False,
        "memory_promotion_allowed": False,
    }
    proposal["proposal_sha256"] = sha256_json(proposal)
    trace = {
        "schema_version": "pg140-information-complete-catalog-trace-v1",
        "protocol_id": protocol["protocol_id"],
        "status": "completed_pg140_information_complete_catalog",
        "training_eligible": False,
        "hard_gates_passed": False,
        "memory_promotion_allowed": False,
        "fresh_reset_per_episode": True,
        "get_post_replayed": True,
        "matched_negative_control": True,
        "raw_source_saved": False,
        "raw_probe_response_saved": False,
        "row_provenance_manifest_sha256": manifest["row_provenance_manifest_sha256"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "catalog_sha256": catalog["catalog_sha256"],
        "dataset_sha256": model_dataset["dataset_sha256"],
    }
    trace["trace_sha256"] = sha256_json(trace)
    report = {
        "protocol_id": protocol["protocol_id"],
        "schema_version": "pg140-information-complete-catalog-report-v1",
        "status": "completed_pg140_information_complete_catalog",
        "hard_gates_passed": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "catalog_file": CATALOG.name,
        "model_dataset_file": MODEL_DATASET.name,
        "quality": quality,
        "manifest": {
            "row_provenance_manifest_sha256": manifest["row_provenance_manifest_sha256"],
            "tokenizer_config_sha256": manifest["tokenizer_config_sha256"],
            "source_implementation_manifest_sha256": manifest["source_implementation_manifest_sha256"],
            "oracle_contract_manifest_sha256": manifest["oracle_contract_manifest_sha256"],
            "split_manifest_sha256": manifest["split_manifest_sha256"],
        },
        "learning_policy": {
            "incomplete_rows": "schema_repair_or_representation_pretrain_only",
            "replayable_complete_rows": "capability_train_candidate_only",
            "memory_promotion": "forbidden_until_cross_seed_implementation_ood_and_manual_gates",
        },
        "promotion": {
            "training_artifact_promotion_allowed": False,
            "memory_promotion_allowed": False,
            "reason": "catalog repair is not evidence of model capability",
        },
        "source_hashes": source_hashes,
    }
    report["report_sha256"] = sha256_json(report)

    _write(CATALOG, catalog)
    _write(MODEL_DATASET, model_dataset)
    _write(PROTOCOL, protocol)
    _write(PROPOSAL, proposal)
    _write(TRACE, trace)
    _write(REPORT, report)
    print(json.dumps({
        "status": report["status"],
        "catalog_rows": quality["catalog_row_count"],
        "original_missing": quality["original_missing_field_counts"],
        "explicit_unknown": quality["explicit_unknown_field_counts"],
        "capability_train_candidates": quality["capability_train_candidate_count"],
        "training_eligible": report["training_eligible"],
        "memory_promotion_allowed": report["memory_promotion_allowed"],
        "report": str(REPORT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

