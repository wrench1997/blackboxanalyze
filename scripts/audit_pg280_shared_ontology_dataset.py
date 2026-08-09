"""Independent audit for PG-280 ontology and identifiability data."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg280_shared_ontology_dataset_v1.json"
HARD_NEGATIVES = ROOT / "research" / "pg280_family_ood_hard_negative_v1.json"
DOCKER_PROBE = ROOT / "research" / "pg280_remote_docker_probe_v1.json"
AUDIT = ROOT / "research" / "pg280_shared_ontology_dataset_audit_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    return -sum((value / total) * math.log2(value / total) for value in counts.values()) if total else 0.0


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    hard = json.loads(HARD_NEGATIVES.read_text(encoding="utf-8"))
    docker = json.loads(DOCKER_PROBE.read_text(encoding="utf-8"))
    rows = [dict(row) for row in list(data.get("records") or []) if isinstance(row, dict)]
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    check("dataset_hash", data.get("dataset_sha256") == sha({key: value for key, value in data.items() if key != "dataset_sha256"}))
    check("row_quota", len(rows) == 288 and int(data.get("counts", {}).get("train", 0)) == 192 and int(data.get("counts", {}).get("holdout", 0)) == 96)
    check("family_quota", set(str(row.get("family")) for row in rows) == {"dom_effect", "sql_differential", "redirect_contract", "logic_access"})
    check("ontology_tokens", all({"ir_layer=shared_slot_ontology", "ir_family_agnostic=1"}.issubset(set(row.get("pre_question_context_tokens") or [])) and {"ir_layer=shared_slot_ontology", "ir_family_agnostic=1"}.issubset(set(row.get("post_observation_context_tokens") or [])) for row in rows))
    check("ontology_metadata", all(row.get("shared_slot_ontology", {}).get("family_agnostic") is True and row.get("shared_slot_ontology", {}).get("from_oracle") is False and row.get("shared_slot_ontology", {}).get("from_final_outcome") is False for row in rows))
    check("context_firewall", data.get("data_contract", {}).get("raw_payload_in_context") is False and data.get("data_contract", {}).get("raw_response_body_in_context") is False and data.get("data_contract", {}).get("oracle_in_context") is False)
    check("remote_scope", data.get("source", {}).get("remote_host") == "112.111.7.91:60228" and data.get("source", {}).get("loopback_only") is True and data.get("source", {}).get("external_network") is False)
    check("docker_probe_honest", docker.get("status") == "unavailable" and docker.get("docker_binary") is False and docker.get("training_or_replay_started") is False)
    check("hard_negative_hash", hard.get("dataset_sha256") == data.get("dataset_sha256"))
    hard_rows = [dict(row) for row in list(hard.get("records") or []) if isinstance(row, dict)]
    check("hard_negative_quota", len(hard_rows) == 48 and int(data.get("counts", {}).get("family_ood_hard_negative", 0)) == len(hard_rows))
    check("hard_negative_holdout_only", all(row.get("split") == "family_ood_holdout" and row.get("training_eligible") is False and row.get("memory_promotion_allowed") is False for row in hard_rows))
    check("hard_negative_firewall", all(row.get("raw_payload_strings_stored") is False and row.get("raw_response_bodies_stored") is False and row.get("oracle_in_context") is False for row in hard_rows))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[sha(row.get("coarse_pre_question_context_tokens") or [])].append(row)
    label_counts = [Counter(str(item.get("targets", {}).get("pre_question", {}).get("slot", "")) for item in group) for group in groups.values()]
    total = len(rows)
    conditional_entropy = sum((sum(counts.values()) / total) * entropy(counts) for counts in label_counts)
    bayes_error = sum(sum(counts.values()) - max(counts.values()) for counts in label_counts) / total
    check("identifiability_entropy_positive", conditional_entropy > 0.0 and bayes_error >= 0.49)
    check("identifiability_declared", float(data.get("identifiability", {}).get("conditional_entropy_bits", 0.0)) > 0.0 and float(data.get("identifiability", {}).get("bayes_error_lower_bound", 0.0)) >= 0.49)
    check("final_only_has_no_pre_supervision", int(data.get("identifiability", {}).get("final_only_pre_supervision_rows", -1)) == 0)
    check("process_has_pre_supervision", int(data.get("identifiability", {}).get("process_pre_supervision_rows", 0)) == len(rows))

    audit = {
        "audit_id": "pg280-shared-ontology-dataset-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "audit_checks": {name: name not in failures for name in ["dataset_hash", "row_quota", "family_quota", "ontology_tokens", "ontology_metadata", "context_firewall", "remote_scope", "docker_probe_honest", "hard_negative_hash", "hard_negative_quota", "hard_negative_holdout_only", "hard_negative_firewall", "identifiability_entropy_positive", "identifiability_declared", "final_only_has_no_pre_supervision", "process_has_pre_supervision"]},
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "hard_negative_dataset": HARD_NEGATIVES.relative_to(ROOT).as_posix(),
        "docker_probe": DOCKER_PROBE.relative_to(ROOT).as_posix(),
        "rows": len(rows),
        "family_ood_hard_negative_rows": len(hard_rows),
        "conditional_entropy_bits_recomputed": round(conditional_entropy, 6),
        "bayes_error_lower_bound_recomputed": round(bayes_error, 6),
        "failures": failures,
        "interpretation": "PG-280 证明粗粒度缺失观测存在正的条件熵和至少 49% 的精确 slot Bayes 错误下界；这不妨碍模型学习安全 ASK/未决 belief，但最终答案不能从不可见信息中凭训练量产生。final-only 没有 pre-question 监督，因此不能声称主动提问能力。",
    }
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
