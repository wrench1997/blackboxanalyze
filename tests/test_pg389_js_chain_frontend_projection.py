import json
from pathlib import Path

from scripts.project_pg389_js_chain_frontend_summary import build_summary


def test_pg389_frontend_summary_is_bounded_and_candidate_only():
    summary = build_summary()
    assert summary["status"] == "diagnostic_candidate_projection"
    assert summary["candidate_status"] == "local_cuda_js_chain_candidate_only"
    assert summary["train_count"] == 144
    assert summary["holdout_count"] == 144
    assert summary["holdout"]["composition_exact"] == 1.0
    assert summary["holdout"]["ask_recall"] == 1.0
    assert summary["holdout"]["negative_false_allow"] == 0
    assert summary["execution"]["device"] == "cuda:0"
    assert summary["training_allowed"] is False
    assert all(value is False for value in summary["promotion"].values())


def test_pg389_frontend_summary_does_not_copy_raw_material():
    summary = build_summary()
    serialized = json.dumps(summary, ensure_ascii=False).casefold()
    for marker in ("raw_payload=", "response_body=", "wire=", "http://", "https://", "oracle_answer="):
        assert marker not in serialized
