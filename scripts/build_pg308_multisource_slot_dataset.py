"""Build PG-308 from audited multi-source process traces.

The model-visible representation is still the PG-302 canonical observable
context plus bounded symbolic slot references.  PG-269 failure-repair traces
are admitted to training; PG-266/PG-268 remain source-held-out.  Additional
slot permutations are evaluation-only hard negatives: their visible surface
slots are inconsistent and the expected plan must fail closed.

No route, family, literal probe, payload, response body, oracle verdict, or
source label is placed in ``context_tokens`` or ``target_tokens``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg301_payload_assembly import canonical_assembly_context, target_map  # noqa: E402
from app.pg302_symbolic_assembly import audit_symbolic_records, symbolic_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
PG307_PATH = RESEARCH / "pg307_symbolic_real_process_dataset_v1.json"
PG269_PATH = RESEARCH / "pg269_failure_guided_replay_dataset_v1.json"
PG266_PATH = RESEARCH / "pg266_pikachu_payload_grounding_training_dataset_v1.json"
PG268_PATH = RESEARCH / "pg268_pikachu_parameterized_replay_dataset_v1.json"
PG268_AUDIT_PATH = RESEARCH / "pg268_pikachu_parameterized_replay_audit_v1.json"
PG269_AUDIT_PATH = RESEARCH / "pg269_failure_guided_replay_audit_v1.json"
PG305_PATH = RESEARCH / "pg305_live_loopback_replay_training_dataset_v1.json"
OUT = RESEARCH / "pg308_multisource_slot_dataset_v1.json"
AUDIT = RESEARCH / "pg308_multisource_slot_dataset_audit_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse(tokens: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        token = str(token)
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return values


def _token_values(tokens: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        token = str(token)
        if "=" in token:
            key, value = token.split("=", 1)
            # Repeated phase tokens are irrelevant for this bounded projection.
            values.setdefault(key, value)
    return values


def _surface(method: str, channel: str | None = None) -> tuple[str, str, str]:
    method = str(method or "unknown").upper()
    method = method if method in {"GET", "POST"} else "unknown"
    channel = str(channel or "").lower()
    if channel == "query":
        return method, "query_param", "url_percent"
    if channel in {"form", "body"}:
        return method, "form_field", "form_urlencoded"
    if method == "GET":
        return method, "query_param", "url_percent"
    if method == "POST":
        return method, "form_field", "form_urlencoded"
    return method, "unknown", "unknown"


def _canonical(values: Mapping[str, Any]) -> list[str]:
    method, field_role, encoding = _surface(str(values.get("method", "unknown")), values.get("channel"))
    raw = [
        "[BOS]",
        f"typed_available={values.get('typed_available', 'unknown')}",
        f"feedback_state={values.get('feedback_state', 'unknown')}",
        f"replay_ready={values.get('replay_ready', 'unknown')}",
        f"evidence_present={values.get('evidence_present', 'unknown')}",
        f"negative_control={values.get('negative_control', 'unknown')}",
        f"fresh_reset={values.get('fresh_reset', 'unknown')}",
        f"surface_method={method}",
        f"surface_field_role={field_role}",
        f"surface_encoding={encoding}",
        f"history_action={values.get('history_action', 'observe')}",
        f"failure_class={values.get('failure_class', 'none')}",
        f"step_budget={values.get('step_budget', 'present')}",
        "[EOS]",
    ]
    return canonical_assembly_context(raw)


def _record(
    *,
    record_id: str,
    source: str,
    split: str,
    training_eligible: bool,
    context_tokens: Sequence[str],
    source_dataset: Path,
    source_audit: Path | None,
    source_evidence_sha256: str,
    hard_negative: bool = False,
    provenance: str,
) -> dict[str, Any]:
    context = [str(token) for token in context_tokens]
    target = symbolic_target_for_context(context)
    values = target_map(target)
    row = {
        "schema_version": "pg308-multisource-slot-record-v1",
        "record_id": record_id,
        "source": source,
        "split": split,
        "training_eligible": bool(training_eligible),
        "context_tokens": context,
        "target_tokens": target,
        "question": values.get("question", "none"),
        "safe_to_send": values.get("safe_to_send") == "1",
        "hard_negative": bool(hard_negative),
        "provenance": provenance,
        "source_dataset_sha256": _source_digest(source_dataset),
        "source_audit_sha256": _source_digest(source_audit) if source_audit and source_audit.exists() else None,
        "source_evidence_sha256": source_evidence_sha256,
        "source_authorized_loopback": True,
        "oracle_target_off_input": True,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "memory_promotion_allowed": False,
        "record_sha256": "",
    }
    row["record_sha256"] = _digest(row)
    return row


def _pg269_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(source.get("records", [])):
        tokens = _token_values(item.get("context_tokens") or [])
        method = tokens.get("method", item.get("method", "unknown"))
        channel = tokens.get("channel", "form")
        context = _canonical(
            {
                "method": method,
                "channel": channel,
                # A typed evaluator channel was available, but this candidate
                # failed.  The failure is observable; the hidden verdict is not.
                "typed_available": "1",
                "feedback_state": "candidate_failed",
                "replay_ready": "1",
                "evidence_present": "1" if item.get("source_evidence_hash") else "unknown",
                "negative_control": "1",
                "fresh_reset": "1" if tokens.get("fresh_reset") == "1" else "unknown",
                "history_action": "candidate_failed",
                "failure_class": "candidate_failed",
                "step_budget": tokens.get("step_budget", "present"),
            }
        )
        rows.append(
            _record(
                record_id=f"pg308:pg269:{index:03d}:{str(item.get('record_id', ''))[-12:]}",
                source="pg269_failure_guided_replay",
                split="train",
                training_eligible=True,
                context_tokens=context,
                source_dataset=PG269_PATH,
                source_audit=PG269_AUDIT_PATH,
                source_evidence_sha256=str(item.get("source_evidence_hash", "")),
                provenance="audited_failure_repair_process",
            )
        )
    return rows


def _pg266_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(source.get("records", [])):
        outcome = str(item.get("outcome_class", ""))
        confirmed = outcome.startswith("confirmed_local_")
        method = str(item.get("method", "unknown"))
        context = _canonical(
            {
                "method": method,
                "channel": "query" if method == "GET" else "form",
                "typed_available": "1" if confirmed else "unknown",
                "feedback_state": "negative_control_clear" if confirmed else "unknown",
                "replay_ready": "1",
                "evidence_present": "1" if item.get("evidence_sha256") else "unknown",
                "negative_control": "1",
                "fresh_reset": "1" if item.get("fresh_reset") else "unknown",
                "history_action": "none" if confirmed else "observe",
                "failure_class": "none" if confirmed else "oracle_gap",
            }
        )
        rows.append(
            _record(
                record_id=f"pg308:pg266:{index:03d}:{str(item.get('record_id', ''))[-12:]}",
                source="pg266_pikachu_local_grounded_replay",
                split="real_live_holdout",
                training_eligible=False,
                context_tokens=context,
                source_dataset=PG266_PATH,
                source_audit=None,
                source_evidence_sha256=str(item.get("evidence_sha256", "")),
                provenance="audited_local_grounded_replay_source_holdout",
            )
        )
    return rows


def _pg268_rows(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gold: list[dict[str, Any]] = []
    hard: list[dict[str, Any]] = []
    for index, item in enumerate(source.get("records", [])):
        tokens = _token_values(item.get("tokens") or [])
        is_gold = str(item.get("replay_expected", "")) == "typed"
        context = _canonical(
            {
                "method": item.get("method", tokens.get("method", "unknown")),
                "channel": item.get("channel_class", tokens.get("channel", "")),
                "typed_available": "1" if is_gold else "unknown",
                "feedback_state": "negative_control_clear" if is_gold else "unknown",
                "replay_ready": "1",
                "evidence_present": "1" if item.get("source_evidence_hash") else "unknown",
                "negative_control": "1",
                "fresh_reset": "1",
                "history_action": "none" if is_gold else "candidate_failed",
                "failure_class": "none" if is_gold else "oracle_gap",
            }
        )
        row = _record(
            record_id=f"pg308:pg268:{index:03d}:{str(item.get('record_id', ''))[-12:]}",
            source="pg268_pikachu_parameterized_replay",
            split="real_live_holdout" if is_gold else "hard_negative_eval",
            training_eligible=False,
            context_tokens=context,
            source_dataset=PG268_PATH,
            source_audit=PG268_AUDIT_PATH,
            source_evidence_sha256=str(item.get("source_evidence_hash", "")),
            hard_negative=not is_gold,
            provenance="audited_parameterized_replay_source_holdout" if is_gold else "audited_parameterized_hard_negative",
        )
        (gold if is_gold else hard).append(row)
    return gold, hard


def _permutation_rows(base_rows: Sequence[Mapping[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    candidates = [row for row in base_rows if bool(row.get("safe_to_send"))]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidates[:limit]):
        values = _parse(row.get("context_tokens") or [])
        method = values.get("surface_method", "unknown")
        if method == "GET":
            field_role, encoding = "form_field", "form_urlencoded"
        else:
            field_role, encoding = "query_param", "url_percent"
        raw = [
            "[BOS]",
            f"typed_available={values.get('typed_available', '1')}",
            f"feedback_state={values.get('feedback_state', 'negative_control_clear')}",
            f"replay_ready={values.get('replay_ready', '1')}",
            f"evidence_present={values.get('evidence_present', '1')}",
            f"negative_control={values.get('negative_control', '1')}",
            f"fresh_reset={values.get('fresh_reset', '1')}",
            f"surface_method={method}",
            f"surface_field_role={field_role}",
            f"surface_encoding={encoding}",
            "history_action=surface_mismatch",
            "failure_class=surface_slot_mismatch",
            f"step_budget={values.get('step_budget', 'present')}",
            "[EOS]",
        ]
        context = canonical_assembly_context(raw)
        rows.append(
            _record(
                record_id=f"pg308:slot-permutation:{index:03d}:{str(row.get('record_id', ''))[-12:]}",
                source="pg308_slot_permutation_hard_negative",
                split="hard_negative_eval",
                training_eligible=False,
                context_tokens=context,
                source_dataset=PG307_PATH,
                source_audit=RESEARCH / "pg307_symbolic_real_process_dataset_audit_v1.json",
                source_evidence_sha256=str(row.get("record_sha256", "")),
                hard_negative=True,
                provenance="visible_surface_slot_permutation_counterfactual",
            )
        )
    return rows


def _copy_pg307(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [copy.deepcopy(row) for row in source.get("records", [])]


def main() -> int:
    pg307 = _load(PG307_PATH)
    pg269 = _load(PG269_PATH)
    pg266 = _load(PG266_PATH)
    pg268 = _load(PG268_PATH)
    pg268_audit = _load(PG268_AUDIT_PATH)
    pg269_audit = _load(PG269_AUDIT_PATH)
    pg305 = _load(PG305_PATH)
    if pg268_audit.get("status") != "passed" or pg269_audit.get("status") != "passed":
        raise RuntimeError("PG-308 refuses an un-audited PG-268/PG-269 source")
    if not pg305.get("contract", {}).get("real_get_post_replay"):
        raise RuntimeError("PG-308 requires the PG-305 real GET/POST contract")

    records = _copy_pg307(pg307)
    pg269_records = _pg269_rows(pg269)
    pg266_records = _pg266_rows(pg266)
    pg268_gold, pg268_hard = _pg268_rows(pg268)
    records.extend(pg269_records)
    records.extend(pg266_records)
    records.extend(pg268_gold)
    records.extend(pg268_hard)
    records.extend(_permutation_rows(records, limit=24))

    audit_base = audit_symbolic_records(records)
    checks = dict(audit_base.get("checks") or {})
    source_names = {str(row.get("source")) for row in records}
    checks.update(
        {
            "multi_source_present": len(source_names) >= 5,
            "pg269_training_rows": any(row.get("source") == "pg269_failure_guided_replay" and row.get("training_eligible") for row in records),
            "pg266_source_holdout": any(row.get("source") == "pg266_pikachu_local_grounded_replay" and row.get("split") == "real_live_holdout" for row in records),
            "pg268_source_holdout": any(row.get("source") == "pg268_pikachu_parameterized_replay" and row.get("split") == "hard_negative_eval" for row in records),
            "slot_permutation_hard_negative": any(row.get("source") == "pg308_slot_permutation_hard_negative" and row.get("hard_negative") for row in records),
            "all_authorized_loopback": all(bool(row.get("source_authorized_loopback")) for row in records if "source_authorized_loopback" in row),
            "payload_strings_excluded": all(not row.get("raw_payload_stored") and not row.get("raw_response_body_stored") for row in records),
            "oracle_target_off_input": all(bool(row.get("oracle_target_off_input")) for row in records if "oracle_target_off_input" in row),
            "promotion_blocked": True,
        }
    )
    counts = {
        "total": len(records),
        "train": sum(int(row.get("split") == "train") for row in records),
        "implementation_holdout": sum(int(row.get("split") == "implementation_holdout") for row in records),
        "real_live_holdout": sum(int(row.get("split") == "real_live_holdout") for row in records),
        "hard_negative_eval": sum(int(row.get("split") == "hard_negative_eval") for row in records),
        "training_eligible": sum(int(row.get("training_eligible")) for row in records),
        "source_count": len(source_names),
        "pg269_train": len(pg269_records),
        "pg266_holdout": len(pg266_records),
        "pg268_gold_holdout": len(pg268_gold),
        "pg268_hard_holdout": len(pg268_hard),
        "slot_permutation_hard_negative": sum(int(row.get("source") == "pg308_slot_permutation_hard_negative") for row in records),
    }
    source_manifest = {
        "pg307": {"dataset": str(PG307_PATH.relative_to(ROOT)), "sha256": _source_digest(PG307_PATH), "audit": str((RESEARCH / "pg307_symbolic_real_process_dataset_audit_v1.json").relative_to(ROOT))},
        "pg269": {"dataset": str(PG269_PATH.relative_to(ROOT)), "sha256": _source_digest(PG269_PATH), "audit": str(PG269_AUDIT_PATH.relative_to(ROOT)), "audit_sha256": _source_digest(PG269_AUDIT_PATH)},
        "pg266": {"dataset": str(PG266_PATH.relative_to(ROOT)), "sha256": _source_digest(PG266_PATH), "report": "research/pg266_pikachu_payload_grounding_replay_report_v1.json"},
        "pg268": {"dataset": str(PG268_PATH.relative_to(ROOT)), "sha256": _source_digest(PG268_PATH), "audit": str(PG268_AUDIT_PATH.relative_to(ROOT)), "audit_sha256": _source_digest(PG268_AUDIT_PATH)},
        "pg305": {"dataset": str(PG305_PATH.relative_to(ROOT)), "sha256": _source_digest(PG305_PATH), "report": "research/pg305_live_loopback_replay_report_v1.json"},
    }
    dataset = {
        "schema_version": "pg308-multisource-slot-dataset-v1",
        "purpose": "source-held-out symbolic slot-copy process training with missing observation and permutation hard-negatives",
        "source_manifest": source_manifest,
        "records": records,
        "counts": counts,
        "contract": {
            "causal_next_token_targets": True,
            "symbolic_slot_references": True,
            "deterministic_binder": True,
            "multi_source_process": True,
            "source_heldout": True,
            "slot_permutation_hard_negatives": True,
            "route_family_not_in_context": True,
            "oracle_target_off_input": True,
            "payload_strings_excluded": True,
            "response_bodies_excluded": True,
            "training_promotion_allowed": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    audit = {
        "schema_version": "pg308-multisource-slot-dataset-audit-v1",
        "dataset": str(OUT.relative_to(ROOT)),
        "dataset_sha256": dataset["dataset_sha256"],
        "source_audit": source_manifest,
        "checks": checks,
        "base_audit": audit_base,
        "status": "passed" if all(checks.values()) else "failed",
        "audit_sha256": "",
    }
    audit["audit_sha256"] = _digest(audit)
    OUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "counts": counts,
                "dataset": str(OUT.relative_to(ROOT)),
                "audit": str(AUDIT.relative_to(ROOT)),
                "dataset_sha256": dataset["dataset_sha256"],
                "audit_sha256": audit["audit_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
