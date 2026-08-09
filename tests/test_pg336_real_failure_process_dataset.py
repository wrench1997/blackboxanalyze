import json
from pathlib import Path

from scripts.audit_pg336_real_failure_process_dataset import AXES, audit
from scripts.build_pg336_real_failure_process_dataset import build_dataset
from scripts.build_pg336_real_failure_process_vocabulary import build as build_vocabulary


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "research" / "pg325_sql_family_holdout_trace_v1.json"


def _data():
    source = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    return build_dataset(source, source_trace_sha256="a" * 64)


def test_real_failure_and_ask_rows_are_present_without_route_or_wire_literals():
    data = _data()
    assert data["counts"] == {
        "total": 180,
        "probe_observed": 27,
        "failure_repair": 9,
        "negative_review": 9,
        "ask_preflight": 135,
        "train": 60,
        "seed_holdout": 120,
        "get": 120,
        "post": 60,
    }
    assert data["source"]["real_failure_trace_count"] == 9
    assert data["source"]["independent_implementation_holdout"] is False
    for row in data["records"]:
        assert set(row["field_capture_manifest"]) == set(AXES)
        assert row["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True}
        joined = " ".join(row["context_tokens"] + row["target_tokens"]).casefold()
        assert "payload=" not in joined
        assert "response_body=" not in joined
        assert "route=" not in joined
        assert "oracle=" not in joined


def test_failure_changes_action_and_ask_negative_are_safe():
    data = _data()
    failures = [row for row in data["records"] if row["diagnostic_kind"] == "failure_repair"]
    asks = [row for row in data["records"] if row["diagnostic_kind"] == "ask_preflight"]
    negatives = [row for row in data["records"] if row["diagnostic_kind"] == "negative_review"]
    assert len(failures) == 9
    assert all("next_action=repair_abstract_plan" in row["target_tokens"] and "action_changed=1" in row["target_tokens"] for row in failures)
    assert len(asks) == 135 and all("next_action=ask_typed" in row["target_tokens"] and "safe_to_send=0" in row["target_tokens"] for row in asks)
    assert len(negatives) == 9 and all("next_action=abstain" in row["target_tokens"] and "safe_to_send=0" in row["target_tokens"] for row in negatives)


def test_audit_is_diagnostic_only_and_promotion_closed():
    report = audit(_data())
    assert report["status"] == "diagnostic_only"
    assert all(report["checks"].values())
    assert report["scientific_gate"]["accepted_training_rows"] == 0
    assert report["scientific_gate"]["independent_implementation_holdout"] is False
    assert all(value is False for value in report["promotion"].values())


def test_vocabulary_keeps_context_and_target_spaces_separate():
    data = _data()
    vocab = build_vocabulary(data)
    assert vocab["append_only"] is True
    assert "[BOS]" in vocab["context_tokens"]
    assert "[TARGET_BOS]" in vocab["target_tokens"]
    assert "[TARGET_BOS]" not in vocab["context_tokens"]
    assert vocab["training_eligibility"]["allowed"] is False
    assert all(value is False for value in vocab["promotion"].values())
