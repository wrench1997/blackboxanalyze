import torch

from app.shared_family_representation import (
    OOD_INVARIANT_FEATURE_INDICES,
    SHARED_FEATURE_DIM,
    SharedFamilyRouter,
    shared_model_input,
)


def test_shared_model_input_strips_oracle_and_labels():
    row = {
        "payload": {"method": "GET", "path": "/query?mode=plain", "probe": "opaque"},
        "response_projection": {"status_code": 200, "body_length": 48, "headers": {"content-type": "application/json"}, "json_shape": {"key_count": 2, "type": "object"}},
        "family": "injection",
        "semantic": {"family": "injection", "expected_oracle": "secret"},
        "oracle_projection": {"interpreter_boundary": True, "secret": "hidden"},
        "rule_ir_result": True,
        "candidate_status": "positive",
    }
    vector = shared_model_input(row)
    assert len(vector) == SHARED_FEATURE_DIM
    assert "hidden" not in str(vector)


def test_shared_router_decode_has_abstain_and_rule_ir_contract():
    model = SharedFamilyRouter().eval()
    outputs = model.decode(torch.zeros(2, SHARED_FEATURE_DIM), abstain_threshold=0.0, margin_threshold=0.0, temperature=1.5)
    assert len(outputs) == 2
    assert all("rule_ir" in output and "candidate_family" in output for output in outputs)


def test_shared_ood_contract_excludes_surface_and_query_geometry():
    indices = set(OOD_INVARIANT_FEATURE_INDICES)
    assert indices
    # Content type/parser shape and URL/query geometry are handled by the
    # family-specific surface oracle; they must not turn a valid new surface
    # into a hard shared-router OOD abstain.
    assert indices.isdisjoint(set(range(22, 33)) | set(range(35, 46)))
    assert indices.isdisjoint({46, 47, 50, 67, 68, 69, 70, 71, 96, 98, 100, 113, 114, 115, 116, 117, 118, 119})
