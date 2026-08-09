"""Build a family-neutral observation-priority lattice for PG-320.

The lattice is deliberately generated from the visible six-slot observation
vector.  It contains no route, family, payload, response, or oracle result;
its only supervision is the deterministic priority rule already used by the
Rule-IR target builder.  It is an augmentation for question composition, not
vulnerability gold.
"""

from __future__ import annotations

import hashlib
import itertools
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
OUTPUT = RESEARCH / "pg320_observation_lattice_dataset_v1.json"
AUDIT = RESEARCH / "pg320_observation_lattice_dataset_audit_v1.json"
MISSING = tuple(itertools.combinations(OBSERVATION_KEYS, 2))
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
HISTORY = ("none", "observe", "ask_anchor", "ask_anchor_retry", "reference_request", "negative_control")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _context(method: str, field: str, encoding: str, history: str, missing: tuple[str, ...]) -> list[str]:
    values = {key: "1" for key in OBSERVATION_KEYS}
    values["feedback_state"] = "negative_control_clear"
    for key in missing:
        values[key] = "unknown"
    raw = ["[BOS]"] + [f"{key}={values[key]}" for key in OBSERVATION_KEYS] + [f"surface_method={method}", f"surface_field_role={field}", f"surface_encoding={encoding}", f"history_action={history}", "failure_class=none", "step_budget=present", "[EOS]"]
    return canonical_assembly_context(raw)


def main() -> int:
    records: list[dict[str, Any]] = []
    ordinal = 0
    for surface_index, (method, field, encoding) in enumerate(SURFACES):
        split = "train" if surface_index < 6 else "lattice_holdout"
        for history in HISTORY:
            for missing in ((), *MISSING):
                context = _context(method, field, encoding, history, missing)
                target = probe_target_for_context(context)
                values = target_map(target)
                role = "complete" if not missing else "ask"
                row = {"schema_version": "pg320-observation-lattice-v1", "record_id": f"pg320:lattice:{ordinal}", "split": split, "training_eligible": split == "train", "surface_meta": {"method": method, "field_role": field, "encoding": encoding, "history": history}, "missing_slots": list(missing), "lattice_role": role, "context_tokens": context, "target_tokens": target, "question_expected": values.get("question", "none"), "safe_expected": values.get("safe_to_send") == "1", "counterfactual_group": f"pg320:{method}:{field}:{encoding}:{history}", "raw_payload_stored": False, "raw_response_body_stored": False, "oracle_target_off_input": True, "record_sha256": ""}
                row["record_sha256"] = _digest(row)
                records.append(row)
                ordinal += 1
    ask = [row for row in records if row["lattice_role"] == "ask"]
    complete = [row for row in records if row["lattice_role"] == "complete"]
    forbidden = {"payload", "url", "route", "family", "response", "response_body", "source_code", "sql", "xss", "xxe"}
    checks = {"records_present": bool(records), "surface_count": len(SURFACES) == 8, "history_count": len(HISTORY) == 6, "lattice_holdout_present": any(row["split"] == "lattice_holdout" for row in records), "ask_rows": len(ask) == len(SURFACES) * len(HISTORY) * len(MISSING), "complete_rows": len(complete) == len(SURFACES) * len(HISTORY), "ask_has_two_missing": all(len(row["missing_slots"]) == 2 for row in ask), "ask_questions": all(row["question_expected"] != "none" and not row["safe_expected"] for row in ask), "complete_no_question": all(row["question_expected"] == "none" and row["safe_expected"] for row in complete), "forbidden_context_absent": not any(str(token).split("=", 1)[0] in forbidden for row in records for token in row["context_tokens"]), "forbidden_target_absent": not any(str(token).split("=", 1)[0] in forbidden or "<" in str(token) or ">" in str(token) for row in records for token in row["target_tokens"]), "raw_excluded": all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in records), "abstract_target_only": all(len(row["target_tokens"]) == 14 for row in records)}
    dataset = {"schema_version": "pg320-observation-lattice-dataset-v1", "source": "generated_from_visible_pg301_observation_priority_rule", "records": records, "counts": {"total": len(records), "train": sum(int(row["split"] == "train") for row in records), "lattice_holdout": sum(int(row["split"] == "lattice_holdout") for row in records), "ask": len(ask), "complete": len(complete), "surface_count": len(SURFACES), "history_count": len(HISTORY), "missing_combination_count": len(MISSING)}, "contract": {"decoder_only_next_token": True, "question_priority_from_visible_slots": True, "family_hidden": True, "raw_payload_excluded": True, "raw_response_excluded": True, "lattice_surface_holdout": True, "training_promotion_allowed": False, "memory_promotion_allowed": False}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _digest(dataset)
    audit = {"schema_version": "pg320-observation-lattice-audit-v1", "checks": checks, "status": "passed" if all(checks.values()) else "failed", "audit_sha256": ""}
    audit["audit_sha256"] = _digest(audit)
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": dataset["schema_version"], "counts": dataset["counts"], "audit": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
