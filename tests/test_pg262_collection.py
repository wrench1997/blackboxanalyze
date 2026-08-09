import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str):
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_pg262_audit_has_complete_fresh_dual_channel_records():
    audit = _read("pg262_targeted_paired_trace_collection_audit_v1.json")
    assert audit["all_required_fields_complete"] is True
    assert audit["audited_record_count"] == 20
    assert audit["fresh_reset_count"] == 20
    assert audit["ai_send_count"] == 20
    assert audit["reference_send_count"] == 20
    assert audit["negative_send_count"] == 20
    assert audit["evidence_hash_count"] == 20
    assert audit["raw_payload_strings_stored"] is False
    assert audit["raw_response_bodies_stored"] is False
    assert not audit["missing_records"]
    assert {row["method"] for row in audit["records"]} == {"GET", "POST"}
    assert all(row["fresh_reset"] and row["reset_completed"] for row in audit["records"])
    assert all(row["required_fields_complete"] for row in audit["records"])


def test_pg262_collection_stays_out_of_training_until_pg263():
    report = _read("pg262_targeted_paired_trace_collection_report_v1.json")
    assert report["collection_audit"]["all_required_fields_complete"] is True
    assert report["collection_audit"]["audited_record_count"] == 20
    assert report["promotion"]["training_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["counts"]["source_counts"] == {
        "pg262_pikachu_sql_paired": 8,
        "pg262_pikachu_xss_paired": 6,
        "pg262_pikachu_boolean_paired": 3,
        "pg262_pikachu_widebyte_paired": 3,
    }
