import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pg247_final_judge_passes_without_pikachu_training_leakage() -> None:
    report = _load("pg247_vulnerableapp_capacity_training_report_v1.json")
    dataset = _load("pg247_vulnerableapp_capacity_training_dataset_v1.json")
    protocol = _load("pg247_vulnerableapp_capacity_training_protocol_v1.json")
    selected = report["selected"]
    holdout = report["selected"]["metrics"]["seed_holdout"]
    judge = report["independent_final_judge"]

    assert report["status"] == "completed_vulnerableapp_capacity_training_with_pikachu_implementation_holdout"
    assert report["training_eligible"] is True
    assert selected["hidden_dim"] == 256
    assert holdout["positive_send_recall"] == 1.0
    assert holdout["abstain_recall"] == 1.0
    assert holdout["false_send_count"] == 0
    assert holdout["missed_send_count"] == 0
    assert report["catastrophic_forgetting_canary"]["pass"] is True
    assert report["catastrophic_forgetting_canary"]["deltas"]["false_send_count"] == 0
    assert report["catastrophic_forgetting_canary"]["deltas"]["missed_send_count"] == 0
    assert judge["pass"] is True
    assert judge["decision"] == "candidate_eligible_for_next_replay"
    assert judge["model_output_is_candidate_only"] is True
    assert judge["oracle_or_reference_is_not_model_input"] is True
    assert report["action_send_probability_threshold"] == 0.9
    assert protocol["implementation_holdout"] == "all sources containing pikachu"
    assert protocol["canary_replay_required_after_update"] is True
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False

    artifact = ROOT / selected["artifact"]
    assert artifact.exists()
    assert _sha256(artifact) == selected["artifact_sha256"]


def test_pg247_split_and_persistence_contract_is_fail_closed() -> None:
    report = _load("pg247_vulnerableapp_capacity_training_report_v1.json")
    dataset = _load("pg247_vulnerableapp_capacity_training_dataset_v1.json")
    rows = dataset["records"]
    eligible = [row for row in rows if row.get("lane") not in {"quarantine", "reject"}]
    is_holdout = lambda row: "pikachu" in str(row.get("source", "")) or (
        row.get("source") == "pg246_vulnerableapp_source_independent" and int(row.get("seed", 0) or 0) == 24603
    )
    train = [row for row in eligible if not is_holdout(row)]
    holdout = [row for row in eligible if is_holdout(row)]

    assert len(train) == report["counts"]["train_rows"]
    assert len(holdout) == report["counts"]["holdout_rows"]
    assert all("pikachu" not in str(row.get("source", "")) for row in train)
    assert all(not (row.get("source") == "pg246_vulnerableapp_source_independent" and int(row.get("seed", 0) or 0) == 24603) for row in train)
    assert any(row.get("payload_grounded_eligible") is True for row in holdout)
    assert any(row.get("payload_grounded_eligible") is False for row in holdout)
    assert dataset["contract"]["canary_never_used_as_oracle_feature"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    assert report["honesty"]["final_judge_is_not_model_self_report"] is True

    serialized = json.dumps(dataset, ensure_ascii=False).lower()
    for forbidden in ("<svg", "onerror=", "onload=", "document.body.dataset.pg246", "<script"):
        assert forbidden not in serialized

