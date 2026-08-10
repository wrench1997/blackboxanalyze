import json
from pathlib import Path

from scripts.run_pg388_logic_composed_candidate import (
    DEFAULT_AUDIT,
    DEFAULT_DATASET,
    SLOT_ORDER,
    load_rows,
    run_candidate,
)


def test_pg388_composed_loader_keeps_11_slots_and_blocks_capability_contract():
    train, holdout, info = load_rows(DEFAULT_DATASET, DEFAULT_AUDIT)
    assert len(train) == 420
    assert len(holdout) == 420
    assert tuple(train[0]["target_values"]) == SLOT_ORDER
    assert info["source_contract"]["row_bound_typed_evidence"] is False


def test_pg388_composed_cpu_smoke_is_candidate_only_and_bounded():
    report = run_candidate(cpu_smoke=True, row_limit=8, epochs=1, d_model=32, layers=1, experts=2, expert_hidden=64, microbatch=4)
    assert report["status"] == "cpu_composed_candidate_only"
    assert report["train_count"] == 8
    assert report["holdout_count"] == 8
    assert report["execution"]["optimizer_started"] is True
    assert report["execution"]["device"] == "cpu"
    assert report["execution"]["gpu_touched"] is False
    assert report["training_eligible"] == 0
    assert report["capability_training_allowed"] is False
    assert all(value is False for value in report["promotion"].values())
    assert len(report["seeds"]) == 3
    assert all(set(seed["holdout"]["per_slot"]) == set(SLOT_ORDER) for seed in report["seeds"])


def test_pg388_composed_report_projection_has_no_raw_markers(tmp_path: Path):
    report = run_candidate(cpu_smoke=True, row_limit=4, epochs=1, d_model=32, layers=1, experts=2, expert_hidden=64, microbatch=2)
    serialized = json.dumps(report, ensure_ascii=False).casefold()
    for marker in ("raw_payload=", "response_body=", "wire=", "http://", "https://", "oracle_answer="):
        assert marker not in serialized


def test_pg388_full_e8_artifact_remains_candidate_only_and_abstract():
    path = Path("research/pg388_logic_composed_candidate_cpu_full_e8_v1.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "cpu_composed_candidate_only"
    assert report["train_count"] == 420
    assert report["holdout_count"] == 420
    assert report["execution"]["device"] == "cpu"
    assert report["execution"]["gpu_touched"] is False
    assert report["training_eligible"] == 0
    serialized = json.dumps(report, ensure_ascii=False).casefold()
    for marker in ("raw_payload=", "response_body=", "wire=", "http://", "https://", "oracle_answer="):
        assert marker not in serialized
