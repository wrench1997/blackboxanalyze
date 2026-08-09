"""Build PG-321 role-conditioned abstract probe-variant lattice."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg301_payload_assembly import OBSERVATION_KEYS, canonical_assembly_context, target_map  # noqa: E402
from app.pg313_probe_variant import probe_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
OUTPUT = RESEARCH / "pg321_variant_role_lattice_dataset_v1.json"
AUDIT = RESEARCH / "pg321_variant_role_lattice_dataset_audit_v1.json"
SURFACES = (
    ("GET", "query_param", "url_percent"),
    ("GET", "query_param", "identity"),
    ("GET", "path_segment", "url_percent"),
    ("GET", "header_value", "identity"),
    ("POST", "form_field", "form_urlencoded"),
    ("POST", "form_field", "json_string"),
    ("POST", "header_value", "identity"),
    ("POST", "path_segment", "url_percent"),
)
ROLES = ("candidate_request", "candidate_probe", "reference_request", "reference_probe", "negative_control", "negative_probe", "candidate_failed")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _context(method: str, field: str, encoding: str, role: str) -> list[str]:
    values = {key: "1" for key in OBSERVATION_KEYS}
    values["feedback_state"] = "observable_no_effect" if role == "candidate_failed" else "negative_control_clear"
    history = role
    failure = "effect_not_confirmed" if role == "candidate_failed" else "none"
    raw = ["[BOS]"] + [f"{key}={values[key]}" for key in OBSERVATION_KEYS] + [f"surface_method={method}", f"surface_field_role={field}", f"surface_encoding={encoding}", f"history_action={history}", f"failure_class={failure}", "step_budget=present", "[EOS]"]
    return canonical_assembly_context(raw)


def main() -> int:
    records: list[dict[str, Any]] = []
    ordinal = 0
    for surface_index, (method, field, encoding) in enumerate(SURFACES):
        split = "train" if surface_index < 6 else "variant_holdout"
        for role in ROLES:
            context = _context(method, field, encoding, role)
            target = probe_target_for_context(context)
            values = target_map(target)
            row = {"schema_version": "pg321-variant-role-lattice-v1", "record_id": f"pg321:variant:{ordinal}", "split": split, "training_eligible": split == "train", "surface_meta": {"method": method, "field_role": field, "encoding": encoding}, "role_meta": role, "context_tokens": context, "target_tokens": target, "expected_variant": values.get("probe_variant_ref"), "expected_encoding_chain": values.get("encoding_chain_ref"), "expected_safe": values.get("safe_to_send") == "1", "raw_payload_stored": False, "raw_response_body_stored": False, "oracle_target_off_input": True, "record_sha256": ""}
            row["record_sha256"] = _digest(row)
            records.append(row)
            ordinal += 1
    holdout = [row for row in records if row["split"] == "variant_holdout"]
    candidates = [row for row in records if row["role_meta"] in {"candidate_request", "candidate_probe"}]
    refs = [row for row in records if row["role_meta"] in {"reference_request", "reference_probe"}]
    negatives = [row for row in records if row["role_meta"] in {"negative_control", "negative_probe"}]
    failures = [row for row in records if row["role_meta"] == "candidate_failed"]
    forbidden = {"payload", "url", "route", "family", "response", "response_body", "source_code", "sql", "xss", "xxe"}
    checks = {"records_present": bool(records), "train_present": any(row["split"] == "train" for row in records), "holdout_present": bool(holdout), "role_balance": len(candidates) == len(refs) == len(negatives) == 16 and len(failures) == 8, "candidate_variant": all(row["expected_variant"] == "source_attested_candidate" for row in candidates), "reference_variant": all(row["expected_variant"] == "reference_canary" for row in refs), "negative_variant": all(row["expected_variant"] == "negative_control" for row in negatives), "failure_safe_zero": all(not row["expected_safe"] and row["expected_variant"] == "none" for row in failures), "forbidden_context_absent": not any(str(token).split("=", 1)[0] in forbidden for row in records for token in row["context_tokens"]), "forbidden_target_absent": not any(str(token).split("=", 1)[0] in forbidden or "<" in str(token) or ">" in str(token) for row in records for token in row["target_tokens"]), "raw_excluded": all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in records), "abstract_target_shape": all(len(row["target_tokens"]) == 14 for row in records)}
    dataset = {"schema_version": "pg321-variant-role-lattice-dataset-v1", "source": "generated_from_visible_pg313_probe_variant_rule", "records": records, "counts": {"total": len(records), "train": sum(int(row["split"] == "train") for row in records), "variant_holdout": len(holdout), "candidate": len(candidates), "reference": len(refs), "negative": len(negatives), "failure": len(failures)}, "contract": {"decoder_only_next_token": True, "variant_role_visible_only_as_history": True, "family_hidden": True, "raw_payload_excluded": True, "raw_response_excluded": True, "holdout_surface_split": True, "training_promotion_allowed": False, "memory_promotion_allowed": False}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _digest(dataset)
    audit = {"schema_version": "pg321-variant-role-lattice-audit-v1", "checks": checks, "status": "passed" if all(checks.values()) else "failed", "audit_sha256": ""}
    audit["audit_sha256"] = _digest(audit)
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": dataset["schema_version"], "counts": dataset["counts"], "audit": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
