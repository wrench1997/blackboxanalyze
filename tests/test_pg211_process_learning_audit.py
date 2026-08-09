import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "pg211_process_learning_audit", ROOT / "scripts" / "run_pg211_process_learning_audit.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit = _MODULE.audit


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg211_distinguishes_real_send_from_process_learning() -> None:
    report = _load("research/pg210_ai_reference_payload_validation_report_v1.json")
    view = _load("research/pg210_request_anatomy_view_v1.json")
    result = audit(report, view)
    assert result["status"] == "attached_but_not_learned"
    assert result["counts"]["ai_sent_count"] == 12
    assert result["counts"]["route_request_binding_count"] == 12
    assert result["counts"]["independent_effect_agreement_count"] == 12
    assert result["counts"]["unique_model_decision_signature_count"] == 1
    assert result["counts"]["feedback_policy_uses_evaluator_count"] == 0
    assert result["counts"]["history_feature_present_count"] == 0
    assert result["model"]["online_weight_update"] is False
    assert result["gates"]["process_learning_proven"] is False


def test_pg211_audit_never_persists_raw_payload_or_body() -> None:
    result = audit(
        _load("research/pg210_ai_reference_payload_validation_report_v1.json"),
        _load("research/pg210_request_anatomy_view_v1.json"),
    )
    serialized = json.dumps(result, ensure_ascii=False).casefold()
    assert "raw_payload_strings_stored\": true" not in serialized
    assert "raw_response_bodies_stored\": true" not in serialized
    assert result["training_eligible"] is False
    assert result["vulnerability_claim_allowed"] is False
