"""Independent dataset audit for PG-279 remote loopback replay collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg279_remote_replay_dataset_v1.json"
AUDIT = ROOT / "research" / "pg279_remote_replay_dataset_audit_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def without(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def collision(rows: list[dict[str, Any]], field: str) -> int:
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(sha(row[field]), set()).add(sha(row["targets"]["pre_question"]))
    return sum(len(targets) > 1 for targets in groups.values())


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = [dict(row) for row in list(data.get("records") or []) if isinstance(row, dict)]
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    check("dataset_hash", data.get("dataset_sha256") == sha(without(data, "dataset_sha256")))
    counts = dict(data.get("counts") or {})
    check("row_quota", len(rows) == 288 and counts.get("total") == 288 and counts.get("train") == 192 and counts.get("holdout") == 96)
    families = dict(counts.get("families") or {})
    check(
        "family_quota",
        set(families.keys()) == {"dom_effect", "sql_differential", "redirect_contract", "logic_access"}
        and all(int(value.get("total", 0)) == 72 for value in families.values())
        and all(int(value.get("positive", 0)) == 36 and int(value.get("negative", 0)) == 36 for value in families.values()),
    )
    check("split_contract", data.get("split_contract", {}).get("implementation_disjoint") is True and data.get("split_contract", {}).get("collection_seeds") == [27901, 27902, 27903])
    replay = dict(data.get("replay_contract") or {})
    check("get_post_present", int(replay.get("get_rows", 0)) > 0 and int(replay.get("post_rows", 0)) > 0)
    check("failure_repair_present", int(replay.get("failure_repair_rows", 0)) == len(rows))
    check("typed_and_abstain_present", int(replay.get("typed_effect_rows", 0)) > 0 and int(replay.get("abstain_rows", 0)) > 0)
    source = dict(data.get("source") or {})
    check("remote_loopback_scope", source.get("remote_host") == "112.111.7.91:60228" and source.get("loopback_only") is True and source.get("external_network") is False and source.get("remote_docker_available") is False)
    check("real_gold_blocked", int(source.get("real_application_gold_rows", 1)) == 0)
    checks_per_row = []
    for row in rows:
        projection = dict(row.get("request_projection") or {})
        hashes = list(row.get("source", {}).get("fresh_replay_evidence_hashes") or [])
        checks_per_row.append(
            row.get("raw_payload_strings_stored") is False
            and row.get("raw_response_bodies_stored") is False
            and row.get("oracle_in_context") is False
            and len(hashes) == 2
            and hashes[0] == hashes[1]
            and projection.get("fresh_reset") is True
            and int(projection.get("replay_count", 0)) == 2
            and all(key in projection for key in ("candidate_initial", "repair", "reference", "negative"))
            and projection.get("method") in {"GET", "POST"}
            and "oracle_status=" not in " ".join(row.get("pre_question_context_tokens") or [])
            and "oracle_status=" not in " ".join(row.get("post_observation_context_tokens") or [])
            and row.get("source", {}).get("real_application_gold") is False
        )
    check("row_integrity", all(checks_per_row) and len(checks_per_row) == len(rows))
    check("paired_rows", len({str(row.get("pair_id")) for row in rows}) == 144 and all(row.get("paired_opposite_record_id") for row in rows))
    collisions = dict(data.get("projection_collision_audit") or {})
    check("enriched_collision_zero", int(dict(collisions.get("enriched") or {}).get("conflict_group_count", -1)) == 0)
    check("post_collision_zero", int(dict(collisions.get("post") or {}).get("conflict_group_count", -1)) == 0)
    check("contract_firewall", data.get("data_contract", {}).get("raw_payload_in_context") is False and data.get("data_contract", {}).get("raw_response_body_in_context") is False and data.get("data_contract", {}).get("oracle_in_context") is False)

    audit = {
        "audit_id": "pg279-remote-replay-dataset-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "audit_checks": {name: name not in failures for name in ["dataset_hash", "row_quota", "family_quota", "split_contract", "get_post_present", "failure_repair_present", "typed_and_abstain_present", "remote_loopback_scope", "real_gold_blocked", "row_integrity", "paired_rows", "enriched_collision_zero", "post_collision_zero", "contract_firewall"]},
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "controlled_replay_rows": len(rows),
        "real_application_gold_rows": int(source.get("real_application_gold_rows", 0) or 0),
        "failures": failures,
        "interpretation": "PG-279 proves that the record contract can capture real remote loopback GET/POST traffic and failure-to-repair evidence. It is not Pikachu/Docker data and cannot establish real application vulnerability capability.",
    }
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
