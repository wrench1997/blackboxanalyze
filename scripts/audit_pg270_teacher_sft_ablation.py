"""Independent audit for the PG-270 teacher-signal ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "research" / "pg270_teacher_sft_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg270_teacher_sft_ablation_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg270_teacher_sft_ablation_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg270_teacher_sft_ablation_trace_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg270-teacher-sft" / "teacher_sft_ablation.pt"

FORBIDDEN_CONTEXT = ("oracle", "payload", "response", "echo", "body_sha", "confirmed_positive", "outcome_class")
REQUIRED_COMPONENTS = {"scope_and_safety", "information_completeness", "probe_utility", "failure_diagnosis", "repair_quality", "oracle_and_evidence_alignment", "calibrated_abstain"}
UNSEEN_FAMILIES = {"redirect", "xxe", "serialization", "infoleak", "other"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def check(name: str, value: bool, failures: list[str]) -> bool:
    if not value:
        failures.append(name)
    return value


def main() -> None:
    failures: list[str] = []
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    records = list(dataset.get("records", []))
    preferences = list(dataset.get("preferences", []))
    process = list(dataset.get("process_rewards", []))
    splits = {"train", "route_dev", "family_holdout"}
    ids = {row.get("record_id") for row in records}
    rows_by_id = {row.get("record_id"): row for row in records}

    check("report_status", report.get("status") == "candidate_ablation_completed", failures)
    check("dataset_sha_valid", dataset.get("dataset_sha256") == sha({key: value for key, value in dataset.items() if key != "dataset_sha256"}), failures)
    check("report_sha_valid", report.get("report_sha256") == sha({key: value for key, value in report.items() if key != "report_sha256"}), failures)
    check("protocol_sha_valid", protocol.get("protocol_sha256") == sha({key: value for key, value in protocol.items() if key != "protocol_sha256"}), failures)
    check("trace_sha_valid", trace.get("trace_sha256") == sha({key: value for key, value in trace.items() if key != "trace_sha256"}), failures)
    check("record_count_40", len(records) == 40, failures)
    check("preference_count_matches", len(preferences) == len(records), failures)
    check("process_count_matches", len(process) == len(records), failures)
    check("split_disjoint", len({row.get("record_id") for row in records}) == len(records), failures)
    check("checkpoint_exists", CHECKPOINT_PATH.exists() and CHECKPOINT_PATH.stat().st_size > 0, failures)
    check("evaluation_only", trace.get("evaluation_only") is True and trace.get("training_eligible") is False, failures)
    check("promotion_blocked", report.get("promotion", {}).get("training_allowed") is False and report.get("promotion", {}).get("memory_promotion_allowed") is False, failures)
    check("claim_blocked", report.get("capability_gate", {}).get("claim_allowed") is False and report.get("formal_claim", {}).get("allowed") is False, failures)

    for row in records:
        record_id = row.get("record_id")
        context = [str(token) for token in row.get("context_tokens", [])]
        target = [str(token) for token in row.get("target_tokens", [])]
        check(f"record_id_present:{record_id}", bool(record_id) and record_id in ids, failures)
        check(f"split_known:{record_id}", row.get("split") in splits, failures)
        check(f"context_end:{record_id}", context[-1:] == ["[CTX_END]"], failures)
        check(f"target_markers:{record_id}", target[:1] == ["[TARGET_BOS]"] and target[-1:] == ["[TARGET_EOS]"], failures)
        check(f"context_raw_free:{record_id}", not any(any(term in token.casefold() for term in FORBIDDEN_CONTEXT) for token in context), failures)
        check(f"teacher_components_complete:{record_id}", set(row.get("teacher_components", {})) == REQUIRED_COMPONENTS, failures)
        check(f"teacher_score_range:{record_id}", 0.0 <= float(row.get("teacher_score", -1)) <= 1.0, failures)

    family_holdout_ids = {row["record_id"] for row in records if row["split"] == "family_holdout"}
    check("family_holdout_nonempty", bool(family_holdout_ids), failures)
    check("family_holdout_only_expected_families", all(rows_by_id[record_id]["labels"]["family_class"] in UNSEEN_FAMILIES for record_id in family_holdout_ids), failures)
    check("family_not_in_train_dev", all(row["labels"]["family_class"] not in UNSEEN_FAMILIES for row in records if row["split"] != "family_holdout"), failures)

    for pair in preferences:
        record_id = pair.get("record_id")
        source = rows_by_id.get(record_id)
        check(f"pair_source:{pair.get('pair_id')}", source is not None and pair.get("split") == source.get("split"), failures)
        check(f"pair_chosen_matches_sft:{pair.get('pair_id')}", pair.get("chosen_target_tokens") == source.get("target_tokens") if source else False, failures)
        check(f"pair_rejected_differs:{pair.get('pair_id')}", pair.get("rejected_target_tokens") != pair.get("chosen_target_tokens"), failures)
        check(f"pair_score_order:{pair.get('pair_id')}", float(pair.get("chosen_teacher_score", -1)) > float(pair.get("rejected_teacher_score", 2)), failures)
        # ``oracle_gap`` is an abstract teacher label and is intentionally
        # allowed in the target.  Raw probes/response material remain banned.
        target_forbidden = ("payload", "response_body", "echo_excerpt", "body_sha256", "confirmed_positive")
        check(f"pair_raw_free:{pair.get('pair_id')}", not any(any(term in token.casefold() for term in target_forbidden) for token in pair.get("rejected_target_tokens", [])), failures)

    for episode in process:
        check(f"process_source:{episode.get('record_id')}", episode.get("record_id") in ids, failures)
        check(f"process_steps:{episode.get('record_id')}", bool(episode.get("step_scores")), failures)
        for step in episode.get("step_scores", []):
            check(f"process_score_range:{episode.get('record_id')}", 0.0 <= float(step.get("score", -1)) <= 1.0, failures)
            check(f"process_components:{episode.get('record_id')}", set(step.get("components", {})) == REQUIRED_COMPONENTS, failures)

    cuda_assignment = report.get("source", {}).get("cuda_assignment", {})
    check("a800_visible_single_device", cuda_assignment.get("cuda_visible_devices") == "0" and cuda_assignment.get("visible_device_count") == 1 and cuda_assignment.get("current_device") == 0, failures)
    check("a800_name", "A800" in str(cuda_assignment.get("device_name", "")), failures)
    checks = report.get("capability_gate", {}).get("checks", {})
    check("reported_gate_all_true", bool(checks) and all(bool(value) for value in checks.values()), failures)
    plain = report.get("evaluations", {}).get("plain_sft", {}).get("family_holdout", {})
    guided = report.get("evaluations", {}).get("guided_sft", {}).get("family_holdout", {})
    check("guided_token_accuracy_not_lower", float(guided.get("token_accuracy", 0)) >= float(plain.get("token_accuracy", 1)), failures)
    check("guided_next_action_not_lower", float(guided.get("next_action_accuracy", 0)) >= float(plain.get("next_action_accuracy", 1)), failures)

    audit = {
        "audit_id": "pg270-teacher-sft-ablation-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "all_required_fields_complete": not failures,
        "audit_checks": {
            "report_status": "report_status" not in failures,
            "dataset_report_protocol_trace_hashes": all(not name.endswith("_sha_valid") for name in failures),
            "source_40_records": "record_count_40" not in failures,
            "disjoint_grouped_split": "split_disjoint" not in failures and "family_not_in_train_dev" not in failures,
            "context_target_raw_separation": not any("context_raw_free" in name for name in failures),
            "preference_pairs_valid": not any(name.startswith("pair_") for name in failures),
            "process_rewards_valid": not any(name.startswith("process_") for name in failures),
            "a800_gpu0_only": "a800_visible_single_device" not in failures and "a800_name" not in failures,
            "guided_not_worse": "guided_token_accuracy_not_lower" not in failures and "guided_next_action_not_lower" not in failures,
            "promotion_blocked": "promotion_blocked" not in failures and "claim_blocked" not in failures,
        },
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "trace": str(TRACE_PATH.relative_to(ROOT)),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "record_count": len(records),
        "preference_pair_count": len(preferences),
        "process_reward_count": len(process),
        "failures": failures,
    }
    audit["audit_sha256"] = sha(audit)
    out = ROOT / "research" / "pg270_teacher_sft_ablation_audit_v1.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
