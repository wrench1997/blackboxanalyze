from __future__ import annotations

from app.failure_guided_scheduler import failure_signature
from app.layered_ir_tokenizer import layered_compress, summarize_rule_ir, tokenize_action_manifest, validate_layered_compression


def _manifest(method: str) -> dict[str, object]:
    value: dict[str, object] = {
        "method": method,
        "route_template_id": "pg129-route-demo",
        "placement": "query" if method == "GET" else "json",
        "encoding_chain": ["url_percent"],
        "probe_ref": "abstract-probe-demo",
        "probe_sha256": "0" * 64,
        "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True},
    }
    if method == "POST":
        value["form_field_names"] = ["abstract_probe"]
    return value


def test_layered_source_ir_token_contract_is_raw_free_and_hashed() -> None:
    signature = failure_signature({"method": "GET", "role": "candidate", "candidate_signal": True, "positive": False, "positive_authority": False, "typed_available": True, "probe_round": 1, "max_probe_rounds": 2})
    compressed = layered_compress(
        html_snapshot='<form method="post"><input name="abstract_probe"></form><script>fetch("local")</script>',
        javascript_snapshot='if (document.querySelector("form")) { fetch("/local"); }',
        action_manifests=[_manifest("GET"), _manifest("POST")],
        response_projection={"status_class": "2xx", "shape_class": "decision-v7", "transition_delta": "none"},
        failure_signature=signature,
    )
    checked = validate_layered_compression(compressed)
    encoded = str(checked)
    assert "abstract_probe" not in encoded
    assert "fetch(\"/local\")" not in encoded
    assert checked["layers"]["rule_ir"]["oracle_authority_included"] is False
    assert checked["layers"]["ir_tokens"]["token_count"] >= 6
    phases = {slot["value"] for slot in checked["layers"]["rule_ir"]["slots"] if slot["slot_id"] == "failure.recovery_phase"}
    assert phases == {"failure_adjusted"}


def test_forward_token_returns_to_baseline_phase() -> None:
    signature = failure_signature({"method": "POST", "role": "candidate", "candidate_signal": True, "positive": True, "positive_authority": True, "typed_available": True}, prior_records=[{"method": "GET", "role": "candidate", "candidate_signal": True}])
    compressed = layered_compress(
        html_snapshot="<main>ok</main>",
        javascript_snapshot="return;",
        action_manifests=[_manifest("POST")],
        response_projection={"status_class": "2xx", "shape_class": "decision-v7", "transition_delta": "visibility"},
        failure_signature=signature,
    )
    phase = next(slot["value"] for slot in compressed["layers"]["rule_ir"]["slots"] if slot["slot_id"] == "failure.recovery_phase")
    assert phase == "forward_baseline"


def test_ir_transport_slot_uses_observed_method_history_not_only_current_manifest() -> None:
    signature = failure_signature({"method": "POST", "role": "candidate", "candidate_signal": True, "positive": True, "positive_authority": True, "typed_available": True}, prior_records=[{"method": "GET", "role": "candidate", "candidate_signal": True}])
    ir = summarize_rule_ir(
        [tokenize_action_manifest(_manifest("POST"))],
        {"transition_delta": "none"},
        signature,
    )
    slot = next(slot for slot in ir["slots"] if slot["slot_id"] == "transport.methods_seen")
    assert slot["value"] == "GET+POST"
