"""Fresh PG-361 dynamic collection wrapper.

The existing PG-348 collector performs the loopback GET/POST replay and
typed candidate/reference/negative/replay sidecar.  This wrapper keeps that
runtime unchanged, appends the source-attested ``syntax_category_ref`` slot,
and writes a new diagnostic artifact.  It never contacts a public URL and
never upgrades rows to training eligibility by itself.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import _target_tokens, sha256_json, validate_pg331_source_row
from scripts.build_pg361_payload_shape_slot_dataset import _syntax_category
from scripts.collect_pg348_dynamic_typed_rows import DEFAULT_REGISTRY, ROLES, collect as collect_pg348, load_registry


DEFAULT_DATASET = ROOT / "research" / "pg361_dynamic_syntax_typed_source_rows_v1.json"
DEFAULT_SIDECARS = ROOT / "research" / "pg361_dynamic_syntax_typed_sidecars_v1.json"
DEFAULT_REPORT = ROOT / "research" / "pg361_dynamic_syntax_typed_collection_report_v1.json"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(registry: Mapping[str, Any], *, max_records: int | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_dataset, sidecars, base_report = collect_pg348(registry, operator_reviewed=False, max_records=max_records)
    registry_rows = {str(row.get("source_hash")): row for row in list(registry.get("records") or []) if row.get("source_hash")}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    syntax_counts: dict[str, int] = {}
    for source_row in list(base_dataset.get("records") or []):
        row = copy.deepcopy(dict(source_row))
        record = registry_rows.get(str((row.get("source_meta") or {}).get("source_digest")))
        if record is None:
            failures.append(f"missing_registry_record:{row.get('record_id', '')}")
            continue
        category = _syntax_category(record)
        target = dict(row.get("target_projection") or {})
        target["syntax_category_ref"] = category
        evaluator = dict(row.get("evaluator_sidecar") or {})
        variant = str(target.get("probe_variant_ref", "none"))
        failure = str(target.get("question", "none")) == "ask_failure"
        typed = evaluator.get("typed_available") is True and evaluator.get("fresh_reset") is True
        target["oracle_ref"] = "unknown" if failure or not typed else "negative_no_effect" if variant == "negative_control" else "typed_effect"
        matched = (
            evaluator.get("negative_control") is True
            and evaluator.get("reference_present") is True
            and evaluator.get("candidate_present") is True
            and evaluator.get("fresh_reset") is True
        )
        target["negative_control_presence_ref"] = "matched_triplet" if matched else "unknown"
        row["target_projection"] = target
        row["target_tokens"] = _target_tokens(target)
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
        rows.append(row)
        syntax_counts[category] = syntax_counts.get(category, 0) + 1
    if failures:
        raise ValueError("PG-361 live collection validation failed: " + ", ".join(sorted(failures)[:20]))
    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    dataset = {
        "schema_version": "pg361-dynamic-syntax-typed-source-rows-v1",
        "status": "completed_dynamic_syntax_diagnostic_only",
        "records": rows,
        "source_registry_sha256": sha256_json(registry),
        "runtime_image_digest": base_dataset.get("runtime_image_digest"),
        "base_collector_report_sha256": sha256_json(base_report),
        "counts": {
            "records": len(rows),
            "routes": int(base_dataset.get("counts", {}).get("routes", 0)),
            "train_rows": sum(row.get("split") == "train" for row in rows),
            "implementation_holdout_rows": sum(row.get("split") == "implementation_holdout" for row in rows),
            "get_rows": sum(str(row.get("target_projection", {}).get("transport_ref")) in {"get_query", "get_path", "get_fragment"} for row in rows),
            "post_rows": sum(str(row.get("target_projection", {}).get("transport_ref")) in {"post_form", "post_json"} for row in rows),
            "typed_positive_rows": sum(row.get("evaluator_sidecar", {}).get("typed_available") is True for row in rows),
            "failure_rows": sum(row.get("target_projection", {}).get("question") == "ask_failure" for row in rows),
            "syntax_category_unique": len(syntax_counts),
            "syntax_category_counts": dict(sorted(syntax_counts.items())),
            "training_eligible_rows": 0,
        },
        "collection_contract": {
            "fresh_reset_per_role": True,
            "candidate_reference_negative_replay": True,
            "typed_evidence_sha256": True,
            "failure_action_change": True,
            "loopback_only": True,
            "external_network": False,
            "raw_payload_stored": False,
            "raw_response_stored": False,
        },
        "promotion": promotion,
    }
    sidecar_doc = dict(sidecars)
    sidecar_doc["schema_version"] = "pg361-dynamic-syntax-typed-sidecars-v1"
    sidecar_doc["promotion"] = promotion
    report = {
        "schema_version": "pg361-dynamic-syntax-typed-collection-report-v1",
        "status": dataset["status"],
        "counts": dataset["counts"],
        "base_report": base_report,
        "syntax_category_unique": len(syntax_counts),
        "source_registry_sha256": dataset["source_registry_sha256"],
        "runtime_image_digest": dataset["runtime_image_digest"],
        "training_eligible": False,
        "promotion": promotion,
    }
    return dataset, sidecar_doc, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect PG-361 fresh dynamic syntax-slot rows")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sidecars", type=Path, default=DEFAULT_SIDECARS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    dataset, sidecars, report = collect(registry, max_records=args.max_records)
    for path, value in ((args.dataset, dataset), (args.sidecars, sidecars), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "dataset_sha256": _file_sha(args.dataset), "report_sha256": _file_sha(args.report)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
