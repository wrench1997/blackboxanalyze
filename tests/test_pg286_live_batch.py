from __future__ import annotations

from app.pg286_live_batch import audit_pg286_live_batch


def _row(index: int, method: str = "GET", modality: str = "dom_effect") -> dict:
    row = {
        "record_id": f"pg286:batch:{index:03d}",
        "surface": {"surface_id": f"surface-{index}", "method": method},
        "reset": {"reset_id": f"reset-{index}"},
        "context_tokens": ["[BOS]", "ir_family_agnostic=1", "field_role=text", "evidence_status=complete", "[CTX_END]"],
        "evaluator_status": "confirmed_effect",
        "typed_effect_type": modality,
        "operator_reviewed": True,
        "hard_negative": False,
        "training_eligible": False,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "decision": "eligible_for_cross_seed_review",
    }
    import hashlib
    import json

    row["record_sha256"] = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return row


def _hard(index: int) -> dict:
    row = _row(index, "GET")
    row["record_id"] = f"pg286:hard:{index:03d}"
    row["hard_negative"] = True
    row["decision"] = "quarantine"
    row["evaluator_status"] = "blocked"
    row["operator_reviewed"] = False
    import hashlib
    import json

    row.pop("record_sha256")
    row["record_sha256"] = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return row


def test_batch_gate_blocks_empty_remote_batch():
    result = audit_pg286_live_batch([], remote_docker_status="unavailable")
    assert result["status"] == "blocked"
    assert result["training_eligible_rows"] == 0
    assert "remote_docker_available" in result["blocking_reasons"]


def test_batch_gate_requires_get_post_reset_modality_and_audit():
    rows = [_row(index, "GET" if index < 3 else "POST", "dom_effect" if index < 3 else "sql_ast_shape") for index in range(6)]
    result = audit_pg286_live_batch(rows, hard_negative_records=[_hard(1)], independent_audit_pass=True, remote_docker_status="available")
    assert result["status"] == "ready_for_remote_a800_training"
    assert result["eligible_record_count"] == 6
    assert result["fresh_reset_count"] == 6
    assert result["get_count"] == 3
    assert result["post_count"] == 3
    assert result["typed_modality_count"] == 2
    assert result["hard_negative_count"] == 1


def test_batch_gate_rejects_context_label_or_hash_tampering():
    row = _row(1)
    row["context_tokens"].append("family=sql")
    result = audit_pg286_live_batch([row], independent_audit_pass=True, remote_docker_status="available")
    assert result["status"] == "blocked"
    assert "context_family_agnostic" in result["blocking_reasons"]
    row2 = _row(2)
    row2["record_sha256"] = "0" * 64
    result2 = audit_pg286_live_batch([row2], independent_audit_pass=True, remote_docker_status="available")
    assert "record_integrity" in result2["blocking_reasons"]
