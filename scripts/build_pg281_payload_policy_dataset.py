"""Build PG-281's abstract payload-policy dataset.

This is a static, payload-free transformation.  It keeps the useful part of
the local replay (method/channel/field shape, failure state and repair stage)
and turns it into an abstract probe plan.  Literal payloads, response bodies
and oracle facts never enter the model context.  Positive catalog rows are
paired with deliberately incomplete, evaluation-only rows so a policy can be
tested for safe abstention before a real evaluator is available.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PG266 = RESEARCH / "pg266_pikachu_payload_grounding_catalog_v1.json"
PG269 = RESEARCH / "pg269_failure_guided_replay_dataset_v1.json"
PG266_AUDIT = RESEARCH / "pg266_pikachu_payload_grounding_replay_report_v1.json"
PG269_AUDIT = RESEARCH / "pg269_failure_guided_replay_audit_v1.json"
OUT = RESEARCH / "pg281_payload_policy_dataset_v1.json"
HARD = RESEARCH / "pg281_payload_policy_hard_negative_v1.json"

FORBIDDEN_CONTEXT = ("payload", "oracle", "response_body", "echo_excerpt", "confirmed_positive", "body_sha256")
UNSEEN_FAMILIES = {"redirect", "xxe", "serialization", "infoleak", "other"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def family_to_probe(value: str) -> str:
    value = value.casefold()
    if value.startswith("sql"):
        return "sql"
    if value.startswith("xss") or value == "dom":
        return "xss"
    if value in {"redirect", "location"}:
        return "redirect"
    if value in {"logic", "authorization", "access"}:
        return "logic"
    if value in {"file", "xxe", "serialization"}:
        return "file"
    return "other"


def split_for(group_id: str, family: str, source: str) -> str:
    if source == "pg269" and family in UNSEEN_FAMILIES:
        return "family_holdout"
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 5
    return "route_dev" if bucket == 0 else "train"


def clean_context(tokens: list[str]) -> list[str]:
    result = [str(token) for token in tokens]
    if any(any(term in token.casefold() for term in FORBIDDEN_CONTEXT) for token in result):
        raise ValueError(f"forbidden model-context token: {result}")
    return result


def plan_tokens(*, probe_class: str, channel: str, encoding: str, final_action: str, safe_to_send: bool, family_agnostic: bool = True) -> list[str]:
    return [
        "[TARGET_BOS]",
        "plan=negative_control",
        "plan=reference_probe",
        "plan=candidate_probe" if safe_to_send else "plan=abstain",
        f"probe_class={probe_class}",
        f"channel={channel}",
        f"encoding={encoding}",
        "family_agnostic=1" if family_agnostic else "family_agnostic=0",
        f"final_action={final_action}",
        f"safe_to_send={int(safe_to_send)}",
        "[TARGET_EOS]",
    ]


def entry_from_pg266(row: dict[str, Any]) -> dict[str, Any]:
    route = dict(row.get("route") or {})
    ai = dict(row.get("ai") or {})
    wire = dict(ai.get("wire") or {})
    oracle = dict(row.get("oracle") or {})
    family = str(route.get("family") or "other")
    probe = family_to_probe(str(route.get("rule_ir") or family))
    method = str(route.get("method") or wire.get("method") or "GET").upper()
    channel = "form" if method == "POST" else "query"
    encoding = "url_percent" if "%" in str(wire.get("request_line") or "") else "plain"
    confirmed = bool(oracle.get("confirmed_positive"))
    route_id = str(route.get("id") or row.get("record_id") or "unknown")
    context = clean_context([
        "[BOS]", "phase=surface", "task=probe_policy", f"method={method}", f"channel={channel}",
        f"field_count={len(list(route.get('fields') or []))}", "fresh_reset=1", "source_attested=1",
        "reference_sent=1", "negative_sent=1", "candidate_sent=1", "repair_attempted=0",
        f"evidence_state={'typed' if confirmed else 'gap'}", "family_hidden=1", "[CTX_END]",
    ])
    target = {
        "probe_class": probe,
        "channel": channel,
        "encoding": encoding,
        "final_action": "replay_confirmed" if confirmed else "abstain",
        "safe_to_send": confirmed,
        "oracle_required": True,
    }
    return {
        "record_id": f"pg281:pg266:{route_id}",
        "group_id": f"pg266:{route_id}",
        "source": "pg266",
        "family": family,
        "method": method,
        "split": split_for(f"pg266:{route_id}", family, "pg266"),
        "context_tokens": context,
        "target_tokens": plan_tokens(probe_class=probe, channel=channel, encoding=encoding, final_action=target["final_action"], safe_to_send=target["safe_to_send"]),
        "target": target,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_in_context": False,
        "training_eligible": True,
        "memory_promotion_allowed": False,
        "source_evidence_hash": str((row.get("evidence") or {}).get("source_sha256") or ""),
    }


def entry_from_pg269(row: dict[str, Any]) -> dict[str, Any]:
    labels = dict(row.get("labels") or {})
    family = str(labels.get("family_class") or "other")
    probe = family_to_probe(str(labels.get("rule_ir_class") or family))
    context = clean_context(["ir_layer=probe_policy_v1", *list(row.get("context_tokens") or [])])
    method = next((token.split("=", 1)[1] for token in context if token.startswith("method=")), "GET")
    channel = next((token.split("=", 1)[1] for token in context if token.startswith("channel=")), "unknown")
    safe = str(labels.get("final_belief")) == "confirmed_effect"
    target = {
        "probe_class": probe,
        "channel": channel,
        "encoding": "unknown",
        "final_action": "replay_confirmed" if safe else "abstain",
        "safe_to_send": safe,
        "oracle_required": True,
    }
    group = str(row.get("route") or row.get("record_id") or "unknown")
    return {
        "record_id": f"pg281:pg269:{row.get('record_id')}",
        "group_id": f"pg269:{group}",
        "source": "pg269",
        "family": family,
        "method": str(method).upper(),
        "split": split_for(f"pg269:{group}", family, "pg269"),
        "context_tokens": context,
        "target_tokens": plan_tokens(probe_class=probe, channel=channel, encoding="unknown", final_action=target["final_action"], safe_to_send=target["safe_to_send"]),
        "target": target,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_in_context": False,
        "training_eligible": True,
        "memory_promotion_allowed": False,
        "source_evidence_hash": str(row.get("source_evidence_hash") or ""),
    }


def hard_negative_from(entry: dict[str, Any], index: int) -> dict[str, Any]:
    target = dict(entry["target"])
    target.update({"final_action": "abstain", "safe_to_send": False, "oracle_required": True})
    context = [token for token in entry["context_tokens"] if not token.startswith("evidence_state=")]
    context.insert(-1, "evidence_state=gap")
    context = clean_context(context)
    return {
        "hard_negative_id": f"pg281:hard-negative:{index:03d}:{entry['record_id']}",
        "source_record_id": entry["record_id"],
        "group_id": entry["group_id"],
        "family": entry["family"],
        "split": "payload_policy_hard_negative",
        "context_tokens": context,
        "target_tokens": plan_tokens(probe_class=str(target["probe_class"]), channel=str(target["channel"]), encoding=str(target["encoding"]), final_action=str(target["final_action"]), safe_to_send=bool(target["safe_to_send"])),
        "target": target,
        "reason": "typed evidence is missing; an intelligent payload policy must abstain rather than emit a candidate.",
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_in_context": False,
        "source_evidence_hash": entry["source_evidence_hash"],
    }


def main() -> None:
    pg266 = json.loads(PG266.read_text(encoding="utf-8"))
    pg269 = json.loads(PG269.read_text(encoding="utf-8"))
    pg266_audit = json.loads(PG266_AUDIT.read_text(encoding="utf-8"))
    pg269_audit = json.loads(PG269_AUDIT.read_text(encoding="utf-8"))
    if pg266.get("status") != "completed_human_review_catalog" or not pg269.get("records"):
        raise RuntimeError("PG-281 requires the audited local PG-266/PG-269 catalogs")
    if pg266_audit.get("status") != "completed_local_payload_grounding_replay" or pg269_audit.get("status") != "passed":
        raise RuntimeError("PG-281 refuses an unaudited source catalog")
    records = [entry_from_pg266(row) for row in list(pg266.get("entries") or [])]
    records.extend(entry_from_pg269(row) for row in list(pg269.get("records") or []))
    hard_negatives = [hard_negative_from(row, index) for index, row in enumerate(records[:12], 1)]
    payload = {
        "schema_version": "pg281-abstract-payload-policy-dataset-v1",
        "purpose": "训练 AI 选择安全的抽象 payload probe plan，并在 typed evidence 缺失时 abstain；不训练原始 payload 字符串。",
        "source": {
            "pg266_catalog": str(PG266.relative_to(ROOT)),
            "pg266_catalog_sha256": str(pg266.get("catalog_sha256") or sha({key: value for key, value in pg266.items() if key != "catalog_sha256"})),
            "pg266_audit": str(PG266_AUDIT.relative_to(ROOT)),
            "pg266_audit_sha256": str(pg266_audit.get("report_sha256") or ""),
            "pg269_dataset": str(PG269.relative_to(ROOT)),
            "pg269_dataset_sha256": str(pg269.get("dataset_sha256") or ""),
            "pg269_audit": str(PG269_AUDIT.relative_to(ROOT)),
            "pg269_audit_sha256": str(pg269_audit.get("audit_sha256") or ""),
            "loopback_only": True,
            "external_network": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "split_contract": {
            "grouped_by_route": True,
            "family_holdout": sorted(UNSEEN_FAMILIES),
            "hard_negative_lane": "evaluation_only",
            "family_hidden_in_context": True,
            "oracle_in_context": False,
        },
        "records": records,
        "hard_negative_records": hard_negatives,
        "counts": {
            "total": len(records),
            "train": sum(row["split"] == "train" for row in records),
            "route_dev": sum(row["split"] == "route_dev" for row in records),
            "family_holdout": sum(row["split"] == "family_holdout" for row in records),
            "hard_negative": len(hard_negatives),
            "confirmed_effect_targets": sum(bool(row["target"]["safe_to_send"]) for row in records),
            "abstain_targets": sum(not bool(row["target"]["safe_to_send"]) for row in records),
        },
        "label_contract": {
            "probe_class": ["sql", "xss", "redirect", "logic", "file", "other"],
            "channel": ["query", "form", "unknown"],
            "encoding": ["plain", "url_percent", "unknown"],
            "final_action": ["replay_confirmed", "abstain"],
            "safe_to_send_is_not_vulnerability_claim": True,
            "payload_values_out_of_context": True,
        },
        "training_contract": {
            "teacher_targets_are_abstract": True,
            "hard_negative_training_eligible": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "real_docker_required_for_live_replay": True,
        },
    }
    payload["dataset_sha256"] = sha(payload)
    hard_payload = {"schema_version": "pg281-payload-policy-hard-negative-v1", "records": hard_negatives, "training_eligible": False, "memory_promotion_allowed": False}
    hard_payload["dataset_sha256"] = sha(hard_payload)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HARD.write_text(json.dumps(hard_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(OUT.relative_to(ROOT)), "dataset_sha256": payload["dataset_sha256"], "counts": payload["counts"], "hard_negative_sha256": hard_payload["dataset_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
