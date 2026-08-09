"""Build a non-training, privacy-safe PG-331 source-row selection manifest.

Rows are validated independently of a supplied source audit.  The resulting
manifest contains only one-way row/record references, aggregate reasons and
unchanged split/implementation group counts; it never copies the source rows
or their context/target tokens.  A later reviewed tool must re-open the
original dataset by hash before any training workflow, which remains disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import sha256_json, validate_pg331_source_row  # noqa: E402


SCHEMA_VERSION = "pg331-audited-row-selection-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _safe_ref(value: Any) -> str:
    return _sha256_bytes(str(value or "").encode("utf-8"))


def select_audited_rows(dataset: Mapping[str, Any], source_audit: Mapping[str, Any], *, dataset_sha256: str = "", source_audit_sha256: str = "") -> dict[str, Any]:
    """Validate each row and return safe references for valid/incomplete lanes."""
    document = dict(dataset) if isinstance(dataset, Mapping) else {}
    audit = dict(source_audit) if isinstance(source_audit, Mapping) else {}
    values = document.get("records")
    if not isinstance(values, list):
        raise ValueError("PG-331 selection dataset records must be a list")
    valid_refs: list[dict[str, str]] = []
    excluded_refs: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    implementation_counts: Counter[str] = Counter()
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            failures = ["row_not_mapping"]
            record_id = f"non_mapping:{index}"
            split = "unknown"
            implementation = "unknown"
        else:
            row = dict(raw)
            result = validate_pg331_source_row(row)
            failures = [str(item) for item in list(result.get("failures") or [])]
            record_id = str(row.get("record_id", ""))
            split = str(row.get("split", "unknown"))
            meta = row.get("source_meta")
            implementation = str(meta.get("implementation", "unknown")) if isinstance(meta, Mapping) else "unknown"
        split_counts[split] += 1
        implementation_counts[implementation] += 1
        ref = {"record_id_sha256": _safe_ref(record_id), "row_sha256": _sha256_bytes(_canonical(raw)), "split": split, "implementation_sha256": _safe_ref(implementation)}
        if not failures:
            valid_refs.append(ref)
        else:
            reasons.update(failures)
            excluded_refs.append({**ref, "failure_count": len(set(failures))})
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_valid_rows_only",
        "input": {"dataset_sha256": dataset_sha256, "source_audit_sha256": source_audit_sha256, "source_audit_status": str(audit.get("status", "not_provided")), "source_audit_schema": str(audit.get("schema_version", ""))},
        "counts": {"input_rows": len(values), "valid_rows": len(valid_refs), "excluded_rows": len(excluded_refs), "split_counts": dict(sorted(split_counts.items())), "implementation_counts": {_safe_ref(key): value for key, value in sorted(implementation_counts.items())}},
        "valid_row_refs": valid_refs,
        "excluded_row_refs": excluded_refs,
        "excluded_reason_counts": dict(sorted(reasons.items())),
        "split_relabelled": False,
        "promotion": promotion,
        "training_eligible": False,
        "raw_material_available": False,
    }
    result["selection_sha256"] = _sha256_bytes(_canonical(result))
    return result


def materialize_valid_rows(dataset: Mapping[str, Any], source_audit: Mapping[str, Any], *, dataset_sha256: str = "", source_audit_sha256: str = "") -> dict[str, Any]:
    """Explicitly emit only validated, already-de-identified abstract rows.

    This is intentionally separate from the default hash manifest.  The
    retained rows are needed by later *diagnostic* information audits, but no
    training authorization is conveyed and split values are copied verbatim.
    """
    manifest = select_audited_rows(dataset, source_audit, dataset_sha256=dataset_sha256, source_audit_sha256=source_audit_sha256)
    values = list(dict(dataset).get("records") or [])
    selected: list[dict[str, Any]] = []
    for raw in values:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        validation = validate_pg331_source_row(row)
        if not validation.get("valid"):
            continue
        context = [str(token) for token in list(row.get("context_tokens") or [])]
        forbidden = ("payload", "raw_", "response_body=", "response_body_text=", "oracle=", "evaluator=")
        if any(any(item in token.casefold() for item in forbidden) for token in context):
            raise ValueError("validated row has forbidden context token")
        # Materialization is an information-audit artifact, never an implicit
        # training grant.  Preserve the split and abstract tokens, but clear
        # the per-row authorization bit and re-hash the changed row.
        row["training_eligible"] = False
        promotion = dict(row.get("promotion") or {})
        promotion["training_eligible"] = False
        promotion["memory_promotion_allowed"] = False
        promotion["payload_catalog_promotion_allowed"] = False
        promotion["vulnerability_claim_allowed"] = False
        row["promotion"] = promotion
        row.pop("record_sha256", None)
        row["record_sha256"] = sha256_json(row)
        selected.append(row)
    result: dict[str, Any] = {
        "schema_version": "pg331-audited-row-materialization-v1",
        "status": "diagnostic_valid_rows_materialized",
        "input": dict(manifest["input"]),
        "counts": {"input_rows": int(manifest["counts"]["input_rows"]), "materialized_valid_rows": len(selected), "excluded_rows": int(manifest["counts"]["excluded_rows"]), "split_counts": dict(manifest["counts"]["split_counts"]), "implementation_counts": dict(manifest["counts"]["implementation_counts"])},
        "records": selected,
        "excluded_row_refs": list(manifest["excluded_row_refs"]),
        "excluded_reason_counts": dict(manifest["excluded_reason_counts"]),
        "split_relabelled": False,
        "promotion": dict(manifest["promotion"]),
        "training_eligible": False,
        "raw_material_available": False,
    }
    result["materialization_sha256"] = _sha256_bytes(_canonical(result))
    return result


def _load(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return dict(value), _sha256_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="select valid PG-331 rows into a privacy-safe diagnostic manifest")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--materialize-valid-output", type=Path, help="explicitly write validated abstract rows to a separate diagnostic file")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    dataset, dataset_hash = _load(args.dataset)
    audit, audit_hash = _load(args.source_audit)
    selected = select_audited_rows(dataset, audit, dataset_sha256=dataset_hash, source_audit_sha256=audit_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.materialize_valid_output is not None:
        materialized = materialize_valid_rows(dataset, audit, dataset_sha256=dataset_hash, source_audit_sha256=audit_hash)
        args.materialize_valid_output.parent.mkdir(parents=True, exist_ok=True)
        args.materialize_valid_output.write_text(json.dumps(materialized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected if args.json else selected["counts"], ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
