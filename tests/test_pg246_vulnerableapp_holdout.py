import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_pg246_independent_dom_holdout_has_typed_positive_negative_and_replay() -> None:
    report = _load("pg246_vulnerableapp_independent_dom_holdout_report_v1.json")
    dataset = _load("pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json")
    trace = _load("pg246_vulnerableapp_independent_dom_holdout_trace_v1.json")
    protocol = _load("pg246_vulnerableapp_independent_dom_holdout_protocol_v1.json")

    assert report["status"] == "completed_independent_implementation_route_holdout"
    assert report["source"]["independent_from_pikachu"] is True
    assert report["counts"]["initial_episode_count"] == 18
    assert report["counts"]["fresh_container_count"] == 36
    assert report["counts"]["get_count"] == 12
    assert report["counts"]["post_count"] == 6
    assert report["counts"]["ai_send_count"] == 12
    assert report["counts"]["reference_send_count"] == 12
    assert report["counts"]["confirmed_positive_count"] == 6
    assert report["counts"]["model_missed_positive_count"] == 0
    assert report["counts"]["model_false_positive_count"] == 0
    assert report["counts"]["post_405_abstain_count"] == 6
    assert report["counts"]["replay_match_count"] == 18
    assert report["counts"]["external_network_count"] == 0
    assert report["route_holdout"]["positive_recall"] == 1.0
    assert report["route_holdout"]["secure_false_accept_count"] == 0
    assert report["route_holdout"]["post_abstain_recall"] == 1.0
    assert report["promotion"]["training_eligible"] is True
    assert report["promotion"]["training_artifact_generated"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["parameter_discovery"]["raw_source_persisted"] is False

    assert dataset["counts"] == {"records": 54, "gold": 12, "hard_negative": 42, "silver": 0, "quarantine": 0, "reject": 0}
    assert dataset["contract"]["independent_generator"] is True
    assert dataset["contract"]["fresh_reset_replay"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    assert trace["training_eligible"] is True
    assert protocol["route_and_implementation_holdout_required"] is True
    assert protocol["canary_replay_required"] is True
    assert protocol["unsupported_post_is_abstain_not_positive"] is True

    for record in dataset["records"]:
        assert record["lane"] in {"gold", "hard_negative"}
        assert record["raw_payload_strings_stored"] is False
        assert record["raw_response_bodies_stored"] is False
        assert record["source_implementation"] == "owasp-vulnerableapp-java-spring"
        assert record["generator_id"] == "pg246-vulnerableapp-dom-surface-generator-v1"


def test_pg246_persisted_artifacts_do_not_contain_runtime_dom_payloads() -> None:
    files = [
        "pg246_vulnerableapp_independent_dom_holdout_report_v1.json",
        "pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json",
        "pg246_vulnerableapp_independent_dom_holdout_trace_v1.json",
        "pg246_vulnerableapp_independent_dom_holdout_protocol_v1.json",
    ]
    serialized = "\n".join((ROOT / "research" / name).read_text(encoding="utf-8-sig") for name in files).casefold()
    for forbidden in ("<svg", "onerror=", "onload=", "document.body.dataset.pg246", "<script"):
        assert forbidden not in serialized


def test_pg246_lane_repair_is_audited_without_network_replay() -> None:
    report = _load("pg246_vulnerableapp_independent_dom_holdout_report_v1.json")
    dataset = _load("pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json")
    repair = report["data_repair"]
    assert repair["repair_id"] == "pg246-lane-reclassification-v1"
    assert repair["network_replay_performed"] is False
    assert repair["oracle_evidence_changed"] is False
    assert dataset["data_repair"]["lane_rule"].startswith("negative_control_confirmed")
