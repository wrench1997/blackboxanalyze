"""PG-280 shared Rule-IR ontology and family-OOD hard-negative builder.

This is a deterministic transformation of the audited PG-279 replay records;
it performs no HTTP requests, starts no container, and trains no model.  The
new tokens are an explicit, family-agnostic slot ontology layer.  They are
derived from the missing-slot schema, never from an evaluator/oracle or final
outcome.  Raw payloads, response bodies and oracle fields remain outside model
context.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "pg279_remote_replay_dataset_v1.json"
SOURCE_AUDIT = ROOT / "research" / "pg279_remote_replay_dataset_audit_v1.json"
OUTPUT = ROOT / "research" / "pg280_shared_ontology_dataset_v1.json"
HARD_NEGATIVES = ROOT / "research" / "pg280_family_ood_hard_negative_v1.json"
DOCKER_PROBE = ROOT / "research" / "pg280_remote_docker_probe_v1.json"


FAMILY_SURFACE = {
    "dom_effect": "render_surface",
    "sql_differential": "query_shape",
    "redirect_contract": "navigation_state",
    "logic_access": "authorization_state",
}
SLOT_MEASURE = {
    "dom_render_channel": "channel",
    "dom_control_alignment": "alignment",
    "sql_response_shape": "shape",
    "sql_baseline_delta": "delta",
    "redirect_status_hop": "hop",
    "redirect_location_scope": "scope",
    "logic_outcome_transition": "transition",
    "logic_invariant_control": "control",
}
SLOT_ROLE = {
    "dom_render_channel": "effect",
    "dom_control_alignment": "control",
    "sql_response_shape": "effect",
    "sql_baseline_delta": "control",
    "redirect_status_hop": "effect",
    "redirect_location_scope": "control",
    "logic_outcome_transition": "effect",
    "logic_invariant_control": "control",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((value / total) * math.log2(value / total) for value in counts.values())


def inject_ontology(tokens: list[str], slot: str, family: str) -> list[str]:
    role = SLOT_ROLE[slot]
    measure = SLOT_MEASURE[slot]
    surface = FAMILY_SURFACE[family]
    ontology = [
        "ir_layer=shared_slot_ontology",
        "ir_family_agnostic=1",
        f"ir_role={role}",
        f"ir_surface={surface}",
        f"ir_measure={measure}",
    ]
    original = list(tokens)
    try:
        end = original.index("[CTX_END]")
    except ValueError:
        end = len(original)
    return [*original[:end], *ontology, *original[end:]]


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if source_audit.get("status") != "passed":
        raise RuntimeError("PG-279 dataset audit must pass before PG-280 transformation")
    records: list[dict[str, Any]] = []
    for original in source.get("records", []):
        row = json.loads(json.dumps(original, ensure_ascii=False))
        family = str(row["family"])
        slot = str(row["missing_observation_slot"])
        row["pre_question_context_tokens"] = inject_ontology(row["pre_question_context_tokens"], slot, family)
        row["post_observation_context_tokens"] = inject_ontology(row["post_observation_context_tokens"], slot, family)
        row["shared_slot_ontology"] = {
            "layer": "rule_ir_slot_ontology_v1",
            "family_agnostic": True,
            "role": SLOT_ROLE[slot],
            "surface": FAMILY_SURFACE[family],
            "measure": SLOT_MEASURE[slot],
            "derived_from": "missing_observation_slot_schema",
            "from_oracle": False,
            "from_final_outcome": False,
        }
        row["training_lane"] = "remote_controlled_replay_with_shared_slot_ontology"
        row["memory_promotion_allowed"] = False
        records.append(row)

    # A family-OOD hard-negative is a held-out implementation row whose
    # matched negative is deliberately retained as a separate diagnostic lane.
    # It is never appended to the training rows: it tests abstain/reject under
    # an unseen family, rather than teaching the answer by leakage.
    hard_negatives: list[dict[str, Any]] = []
    by_record = {str(row["record_id"]): row for row in records}
    for row in records:
        if row.get("split") != "implementation_holdout" or row.get("labels", {}).get("expected_positive") is True:
            continue
        opposite = by_record.get(str(row.get("paired_opposite_record_id")))
        hard_negatives.append({
            "schema_version": "pg280-family-ood-hard-negative-v1",
            "hard_negative_id": f"pg280:ood-hard-negative:{row['record_id']}",
            "source_record_id": row["record_id"],
            "matched_positive_record_id": str((opposite or {}).get("record_id", "")),
            "family": row["family"],
            "implementation": row["implementation"],
            "split": "family_ood_holdout",
            "shared_slot_ontology": row["shared_slot_ontology"],
            "context_tokens": row["post_observation_context_tokens"],
            "target": {"action": "abstain", "belief": "rejected", "question": "explain_failure", "slot": row["missing_observation_slot"]},
            "reason": "matched-negative response under an implementation/family holdout; do not promote from final label",
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "oracle_in_context": False,
            "source_evidence_hash": row["evidence_hash"],
        })

    # The raw/coarse view is intentionally left unchanged.  This keeps the
    # information-theoretic collision as a visible diagnostic while the
    # enriched ontology view is the trainable representation.
    coarse_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        coarse_groups[sha(row["coarse_pre_question_context_tokens"])].append(row)
    label_counts = [Counter(str(item["targets"]["pre_question"]["slot"]) for item in group) for group in coarse_groups.values()]
    total = len(records)
    conditional_entropy = sum((sum(counts.values()) / total) * entropy(counts) for counts in label_counts)
    bayes_error = sum((sum(counts.values()) - max(counts.values())) for counts in label_counts) / total
    ambiguity_rows = sum(sum(counts.values()) for counts in label_counts if len(counts) > 1)

    payload: dict[str, Any] = {
        "schema_version": "pg280-shared-ontology-dataset-v1",
        "purpose": "Shared Rule-IR slot ontology plus family-OOD hard-negative lane; preserves missing-observation identifiability evidence.",
        "source": {
            "dataset": SOURCE.relative_to(ROOT).as_posix(),
            "dataset_sha256": source["dataset_sha256"],
            "audit": SOURCE_AUDIT.relative_to(ROOT).as_posix(),
            "audit_sha256": source_audit["audit_sha256"],
            "remote_host": "112.111.7.91:60228",
            "loopback_only": True,
            "external_network": False,
            "real_application_gold_rows": 0,
            "remote_docker_available": False,
        },
        "records": records,
        "hard_negative_records": hard_negatives,
        "projection_collision_audit": source.get("projection_collision_audit", {}),
        "counts": {
            "total": len(records),
            "train": sum(row.get("split") == "implementation_train" for row in records),
            "holdout": sum(row.get("split") == "implementation_holdout" for row in records),
            "family_ood_hard_negative": len(hard_negatives),
        },
        "shared_slot_ontology": {
            "roles": sorted(set(SLOT_ROLE.values())),
            "surfaces": sorted(set(FAMILY_SURFACE.values())),
            "measures": sorted(set(SLOT_MEASURE.values())),
            "tokens": ["ir_layer=shared_slot_ontology", "ir_family_agnostic=1", "ir_role=<effect|control>", "ir_surface=<abstract_surface>", "ir_measure=<abstract_measure>"],
            "target_oracle_leakage": False,
        },
        "identifiability": {
            "view": "coarse_pre_question_context_tokens",
            "target": "missing_observation_slot",
            "collision_groups": len(coarse_groups),
            "ambiguous_rows": ambiguity_rows,
            "conditional_entropy_bits": round(conditional_entropy, 6),
            "bayes_error_lower_bound": round(bayes_error, 6),
            "interpretation": "同一可见 context 对应多个 slot 时，精确 slot/最终答案不是训练量不足，而是当前观测下不可识别；安全 ASK/未决 belief 仍可学习。",
            "final_only_pre_supervision_rows": 0,
            "process_pre_supervision_rows": len(records),
        },
        "data_contract": {
            "shared_slot_ontology_required": True,
            "family_ood_hard_negative_required": True,
            "hard_negatives_training_eligible": False,
            "raw_payload_in_context": False,
            "raw_response_body_in_context": False,
            "oracle_in_context": False,
            "real_application_gold_required_for_promotion": True,
            "promotion_blocked": True,
        },
        "training_contract": {
            "compare": ["final_only_sft", "enriched_process_sft"],
            "final_only_must_not_claim_ask_capability": True,
            "process_must_include_pre_question_and_failure_repair": True,
            "family_ood_hard_negative_lane": "evaluation_only",
            "memory_promotion_blocked": True,
        },
    }
    payload["dataset_sha256"] = sha(payload)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HARD_NEGATIVES.write_text(json.dumps({"schema_version": "pg280-family-ood-hard-negative-v1", "dataset_sha256": payload["dataset_sha256"], "records": hard_negatives, "training_eligible": False, "memory_promotion_allowed": False, "audit_required": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DOCKER_PROBE.write_text(json.dumps({"schema_version": "pg280-remote-docker-probe-v1", "remote_host": "112.111.7.91:60228", "command": "command -v docker; docker version; docker ps", "status": "unavailable", "docker_binary": False, "docker_server": False, "running_containers": [], "external_network": False, "scope_authorized": True, "training_or_replay_started": False, "interpretation": "远程主机没有 Docker；PG-280 不得把 loopback fixture 当真实应用 gold。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed_pg280_shared_ontology_build", "dataset": OUTPUT.relative_to(ROOT).as_posix(), "dataset_sha256": payload["dataset_sha256"], "rows": len(records), "hard_negative_rows": len(hard_negatives), "identifiability": payload["identifiability"], "remote_docker_available": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
