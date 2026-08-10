import json
from pathlib import Path

from scripts.run_pg389_js_chain_candidate import DEFAULT_AUDIT, DEFAULT_DATASET, SLOT_ORDER, load_rows, run_candidate


def test_pg389_loader_keeps_abstract_chain_slots_and_split():
    train, holdout, info = load_rows(DEFAULT_DATASET, DEFAULT_AUDIT)
    assert len(train) == 144
    assert len(holdout) == 144
    assert tuple(train[0]["target_values"]) == SLOT_ORDER
    assert info["source_contract"]["typed_evidence"] is False


def test_pg389_cpu_candidate_is_bounded_and_fail_closed():
    report = run_candidate(cpu_smoke=True, row_limit=8, epochs=1, d_model=32, layers=1, experts=2, expert_hidden=64, microbatch=4)
    assert report["status"] == "cpu_js_chain_candidate_only"
    assert report["train_count"] == 8
    assert report["holdout_count"] == 8
    assert report["execution"]["optimizer_started"] is True
    assert report["execution"]["gpu_touched"] is False
    assert report["training_eligible"] == 0
    assert report["capability_training_allowed"] is False
    assert all(value is False for value in report["promotion"].values())
    assert len(report["seeds"]) == 3
    assert all(set(seed["holdout"]["per_slot"]) == set(SLOT_ORDER) for seed in report["seeds"])


def test_pg389_report_has_no_raw_chain_material(tmp_path: Path):
    report = run_candidate(cpu_smoke=True, row_limit=4, epochs=1, d_model=32, layers=1, experts=2, expert_hidden=64, microbatch=2)
    output = tmp_path / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    serialized = output.read_text(encoding="utf-8").casefold()
    for marker in ("raw_payload=", "response_body=", "wire=", "http://", "https://", "oracle_answer=", "raw_value="):
        assert marker not in serialized
