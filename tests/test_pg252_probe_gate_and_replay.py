import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_pg251_target_mismatch_is_recorded_as_blocked() -> None:
    report = _load("pg251_preprobe_action_capacity_training_report_v1.json")
    assert report["training_eligible"] is False
    assert report["independent_final_judge"]["decision"] == "blocked"
    assert report["evaluation_splits"]["pikachu_preprobe_holdout"]["metrics"]["positive_send_recall"] == 0.0


def test_pg252_probe_gate_passes_without_claiming_vulnerability() -> None:
    report = _load("pg252_probe_gate_capacity_training_report_v1.json")
    judge = report["independent_final_judge"]
    assert report["training_eligible"] is True
    assert judge["pass"] is True
    assert judge["target_semantics"] == "safe probe availability, not vulnerability success"
    assert report["evaluation_splits"]["pikachu_preprobe_holdout"]["metrics"]["false_send_count"] == 0
    assert report["evaluation_splits"]["pikachu_preprobe_holdout"]["metrics"]["missed_send_count"] == 0
    assert report["evaluation_splits"]["vulnerableapp_preprobe_ood"]["capability_pass"] is True
    assert report["promotion"]["vulnerability_claim_allowed"] is False


def test_pg250_real_replay_has_ephemeral_wires_and_independent_reference() -> None:
    report = _load("pg250_pikachu_pg249_payload_replay_report_v1.json")
    counts = report["counts"]
    comparison = report["reference_comparison"]
    assert counts["fresh_container_count"] == 2
    assert counts["get_route_count"] == 22
    assert counts["post_route_count"] == 4
    assert counts["candidate_send_count"] == 12
    assert counts["unknown_oracle_abstain_count"] == 14
    assert counts["false_positive_count"] == 0
    assert comparison["pair_count"] == 12
    assert comparison["reference_is_independent_catalog_candidate"] is True
    assert comparison["raw_wires_persisted"] is False
    assert report["promotion"]["training_eligible"] is False

