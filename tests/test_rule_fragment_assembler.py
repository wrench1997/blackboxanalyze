import json
from pathlib import Path

import pytest

from app.probe_binding_attestation import add_binding_attestation
from app.rule_fragment_assembler import assemble_rule_fragments, fragment_from_row


ROOT = Path(__file__).resolve().parents[1]


def _row(role: str = "train") -> dict:
    dataset = json.loads((ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json").read_text(encoding="utf-8"))
    row = next(item for item in dataset["rows"] if item.get("role") == role)
    value = dict(row)
    value["model_input"] = add_binding_attestation(value["model_input"])
    return value


def test_fragment_assembly_is_order_invariant_and_non_executable():
    get_row = _row()
    dataset = json.loads((ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json").read_text(encoding="utf-8"))
    post_source = next(item for item in dataset["rows"] if item.get("role") == "train" and item.get("method") == "POST" and item["model_input"]["delta_pattern"] == get_row["model_input"]["delta_pattern"])
    post_row = dict(post_source)
    post_row["model_input"] = add_binding_attestation(post_row["model_input"])
    left = fragment_from_row(get_row)
    right = fragment_from_row(post_row)
    forward = assemble_rule_fragments([left, right], supported_slots=["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"])
    reverse = assemble_rule_fragments([right, left], supported_slots=["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"])
    assert forward["decision"] == "await_typed_oracle"
    assert forward["executable"] is False
    assert forward["promotion_eligible"] is False
    assert forward["canonical_sha256"] == reverse["canonical_sha256"]
    assert forward["atoms"] == ["effect_present", "probe_binding_valid", "get_post_repeat", "negative_control_clear"]


def test_fragment_assembly_rejects_duplicate_and_invalid_binding_evidence():
    row = _row()
    fragment = fragment_from_row(row)
    duplicate = assemble_rule_fragments([fragment, fragment], supported_slots=["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"])
    assert duplicate["decision"] == "abstain"
    assert duplicate["reason"] == "duplicate_evidence"
    invalid = dict(fragment)
    invalid["binding_sha256"] = "0" * 64
    result = assemble_rule_fragments([fragment, invalid], supported_slots=["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"])
    assert result["decision"] == "abstain"
    assert result["reason"] == "invalid_fragment_binding"


def test_fragment_assembler_rejects_raw_or_evaluator_model_input():
    row = _row()
    row["model_input"] = dict(row["model_input"])
    row["model_input"]["family"] = "forbidden"
    with pytest.raises(ValueError, match="evaluator or raw field"):
        fragment_from_row(row)
