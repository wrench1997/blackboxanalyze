"""Project PG-309 contexts into the PG-313 probe-variant target."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg301_payload_assembly import target_map  # noqa: E402
from app.pg313_probe_variant import audit_probe_records, probe_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg309_balanced_counterfactual_dataset_v1.json"
SOURCE_AUDIT = RESEARCH / "pg309_balanced_counterfactual_dataset_audit_v1.json"
OUT = RESEARCH / "pg313_probe_variant_dataset_v1.json"
AUDIT = RESEARCH / "pg313_probe_variant_dataset_audit_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _project(row: dict[str, Any], *, record_id: str | None = None, source: str | None = None) -> dict[str, Any]:
    projected = copy.deepcopy(row)
    projected["schema_version"] = "pg313-probe-variant-record-v1"
    projected["record_id"] = record_id or str(row.get("record_id", ""))
    if source:
        projected["source"] = source
    context = [str(token) for token in projected.get("context_tokens") or []]
    target = probe_target_for_context(context)
    projected["target_tokens"] = target
    values = target_map(target)
    projected["question"] = values.get("question", "none")
    projected["safe_to_send"] = values.get("safe_to_send") == "1"
    projected["probe_variant_target"] = values.get("probe_variant_ref", "none")
    projected["encoding_chain_target"] = values.get("encoding_chain_ref", "none")
    projected["raw_payload_stored"] = False
    projected["raw_response_body_stored"] = False
    projected["oracle_target_off_input"] = True
    projected["memory_promotion_allowed"] = False
    projected["record_sha256"] = ""
    projected["record_sha256"] = _digest(projected)
    return projected


def main() -> int:
    source = _load(SOURCE)
    source_audit = _load(SOURCE_AUDIT)
    if source_audit.get("status") != "passed":
        raise RuntimeError("PG-313 requires a passed PG-309 audit")
    records = [_project(dict(row)) for row in source.get("records", [])]
    safe_train = [row for row in records if row.get("split") == "train" and row.get("training_eligible") and row.get("safe_to_send")]
    generated: list[dict[str, Any]] = []
    for index, row in enumerate(safe_train[:12]):
        for action in ("candidate_request", "reference_request", "negative_control"):
            clone = copy.deepcopy(row)
            if index >= 8:
                clone["split"] = "implementation_holdout"
                clone["training_eligible"] = False
            clone["context_tokens"] = [
                f"history_action={action}" if str(token).startswith("history_action=") else token
                for token in clone.get("context_tokens", [])
            ]
            generated.append(_project(clone, record_id=f"pg313:variant:{index:03d}:{action}", source="pg313_probe_variant_counterfactual"))
    records.extend(generated)
    audit_base = audit_probe_records(records)
    counts = {
        "total": len(records),
        "train": sum(int(row.get("split") == "train") for row in records),
        "implementation_holdout": sum(int(row.get("split") == "implementation_holdout") for row in records),
        "real_live_holdout": sum(int(row.get("split") == "real_live_holdout") for row in records),
        "hard_negative_eval": sum(int(row.get("split") == "hard_negative_eval") for row in records),
        "training_eligible": sum(int(row.get("training_eligible")) for row in records),
        "generated_variant_rows": len(generated),
        "candidate_variant_rows": sum(int(row.get("probe_variant_target") == "source_attested_candidate") for row in generated),
        "reference_variant_rows": sum(int(row.get("probe_variant_target") == "reference_canary") for row in generated),
        "negative_variant_rows": sum(int(row.get("probe_variant_target") == "negative_control") for row in generated),
    }
    checks = dict(audit_base.get("checks") or {})
    checks.update({"source_audit_pass": True, "variant_rows_present": len(generated) > 0, "variant_classes_complete": counts["candidate_variant_rows"] > 0 and counts["reference_variant_rows"] > 0 and counts["negative_variant_rows"] > 0, "promotion_blocked": True})
    dataset = {"schema_version": "pg313-probe-variant-dataset-v1", "purpose": "causal next-token abstract probe variant and encoding-chain selection", "source": {"dataset": str(SOURCE.relative_to(ROOT)), "dataset_sha256": source.get("dataset_sha256"), "audit": str(SOURCE_AUDIT.relative_to(ROOT)), "audit_sha256": source_audit.get("audit_sha256")}, "records": records, "counts": counts, "contract": {"causal_next_token_targets": True, "probe_variant_refs_bounded": True, "encoding_chain_refs_bounded": True, "deterministic_binder": True, "route_family_not_in_context": True, "literal_payload_excluded": True, "response_bodies_excluded": True, "oracle_target_off_input": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _digest(dataset)
    audit = {"schema_version": "pg313-probe-variant-dataset-audit-v1", "dataset": str(OUT.relative_to(ROOT)), "dataset_sha256": dataset["dataset_sha256"], "checks": checks, "base_audit": audit_base, "status": "passed" if all(checks.values()) else "failed", "audit_sha256": ""}
    audit["audit_sha256"] = _digest(audit)
    OUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "counts": counts, "dataset": str(OUT.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
