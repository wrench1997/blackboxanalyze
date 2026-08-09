"""Merge PG-236 per-seed replay batches without hiding replica evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
INPUTS = (
    RESEARCH / "pg236_pikachu_independent_replay_dataset_seed23631.json",
    RESEARCH / "pg236_pikachu_independent_replay_dataset_seed23632.json",
)
TRACES = (
    RESEARCH / "pg236_pikachu_independent_replay_trace_seed23631.json",
    RESEARCH / "pg236_pikachu_independent_replay_trace_seed23632.json",
)
DATASET = RESEARCH / "pg236_pikachu_independent_replay_dataset_v1.json"
REPORT = RESEARCH / "pg236_pikachu_independent_replay_report_v1.json"
TRACE = RESEARCH / "pg236_pikachu_independent_replay_trace_v1.json"
PROTOCOL = RESEARCH / "pg236_pikachu_independent_replay_protocol_v1.json"
MARKDOWN = RESEARCH / "pg236_pikachu_independent_replay_report_v1.md"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def main() -> int:
    batches = [json.loads(path.read_text(encoding="utf-8-sig")) for path in INPUTS]
    records = [dict(record) for batch in batches for record in batch.get("records", [])]
    groups = Counter(str(record.get("trajectory_hash", "")) for record in records)
    for record in records:
        record["replicate_group_hash"] = str(record.get("trajectory_hash", ""))
        record["replicate_group_size"] = int(groups[record["replicate_group_hash"]])
    traces = [json.loads(path.read_text(encoding="utf-8-sig")) for path in TRACES]
    trace_rows = [row for trace in traces for row in trace.get("rows", [])]
    counts = {
        "seed_count": len(batches),
        "fresh_container_count": sum(int(batch.get("counts", {}).get("fresh_container_count", 0)) for batch in batches),
        "raw_record_count": len(records),
        "unique_template_count": len(groups),
        "replicated_record_count": sum(1 for size in groups.values() if size > 1 for _ in range(size)),
        "get_record_count": sum(int(record["method"] == "GET") for record in records),
        "post_record_count": sum(int(record["method"] == "POST") for record in records),
        "family_counts": dict(Counter(str(record["family_class"]) for record in records)),
        "reference_sent_count": len(records),
        "negative_control_count": len(records),
        "oracle_available_count": 0,
        "model_self_error_count": 0,
    }
    dataset = {
        "schema_version": "pg236-pikachu-independent-replay-dataset-v1",
        "source_batches": [str(path.relative_to(ROOT)) for path in INPUTS],
        "records": records,
        "counts": counts,
        "cross_seed_policy": {"seed_holdout_required": True, "replicate_template_hash_retained": True, "template_duplicates_not_counted_as_new_surface": True},
        "contract": {"fresh_reset_per_route_episode": True, "candidate_reference_negative": True, "projection_only": True, "typed_oracle_unavailable_is_silver": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False},
    }
    dataset["dataset_sha256"] = _digest(dataset)
    report = {"protocol_id": "pg-pk-236-pikachu-independent-replay-v1", "schema_version": "pg236-pikachu-independent-replay-v1", "status": "completed_two_seed_pikachu_independent_replay_merge", "counts": counts, "dataset_file": str(DATASET.relative_to(ROOT)), "source_batches": dataset["source_batches"], "promotion": {"training_promotion_allowed": True, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "honesty": {"cross_seed_templates_are_replicates_not_new_families": True, "typed_oracle_unavailable": True, "general_web_capability_not_established": True}, "safety": {"loopback_only": True, "external_network": False, "fresh_container_per_route_episode": True, "database_write": False, "script_execution": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = _digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg236-pikachu-independent-replay-protocol-v1", "seed_count": 2, "methods": ["GET", "POST"], "families": ["sql", "dom", "redirect"], "fresh_reset_per_route_episode": True, "cross_seed_template_grouping": True, "candidate_reference_negative": True, "typed_oracle": False, "silver_abstention_only": True, "memory_promotion_blocked": True, "raw_payload_and_response_excluded": True}
    protocol["protocol_sha256"] = _digest(protocol)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE.write_text(json.dumps({"schema_version": "pg236-pikachu-independent-replay-trace-v1", "rows": trace_rows, "cross_seed_template_group_count": len(groups), "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-236 independent Pikachu replay", "", f"seeds={counts['seed_count']}; fresh={counts['fresh_container_count']}; raw={counts['raw_record_count']}; unique_templates={counts['unique_template_count']}; GET={counts['get_record_count']}; POST={counts['post_record_count']}", f"families={counts['family_counts']}", "跨 seed 的相同 token 模板保留为 replicate group，不伪装成新表面；后续训练按 seed 留出。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

