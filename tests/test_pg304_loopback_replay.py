import hashlib

import pytest

from app.pg284_evaluator_contract import sha256_json
from app.pg302_symbolic_assembly import symbolic_target_for_context
from app.pg304_loopback_replay import evaluate_loopback_batch


def _context(method="GET", field="query_param", encoding="url_percent"):
    return [
        f"surface_method={method}", f"surface_field_role={field}", f"surface_encoding={encoding}",
        "typed_available=1", "feedback_state=observable_progress", "replay_ready=1", "evidence_present=1", "negative_control=1", "fresh_reset=1", "history_action=observe", "failure_class=none", "step_budget=present",
    ]


def _episode(surface_id, method, *, negative=False):
    field, encoding = ("query_param", "url_percent") if method == "GET" else ("form_field", "form_urlencoded")
    context = _context(method, field, encoding)
    unsigned = {"effect_type": "result_shape", "typed_effect_confirmed": not negative, "negative_control_clean": not negative, "reference_agreement": not negative, "replay_consistent": not negative, "non_destructive": True, "evaluator_id": "test"}
    shape = lambda name: sha256_json({"shape": name})
    projection = lambda name, backend=False, marker="none": {"status_class": "2xx", "shape_sha256": shape(name), "redirect_hops": 0, "backend_observed": backend, "effect_marker": marker}
    return {
        "context_tokens": context,
        "plan_tokens": symbolic_target_for_context(context),
        "surface": {"surface_id": surface_id, "method": method, "path": f"/{surface_id}", "channel": "query" if method == "GET" else "form", "field_count": 1, "authorization": "operator_allowlisted_remote_docker", "source_attestation_sha256": sha256_json({"id": surface_id})},
        "reset": {"reset_id": f"reset-{surface_id}", "fresh_target": True, "container_recreated": True, "container_restart_used": False, "volume_mount_count": 0, "database_health_gate": "healthy", "state_change_allowed": False},
        "reference": projection("ref"), "negative": projection("neg"), "candidate": projection("cand", not negative, "typed" if not negative else "none"), "replay": projection("cand", not negative, "typed" if not negative else "none"),
        "typed_evidence": {**unsigned, "evidence_sha256": sha256_json(unsigned)},
        "remote_probe": {"status": "available", "loopback_only": True, "external_network": False},
        "hard_negative": negative,
    }


def test_loopback_fixture_contract_confirms_only_bounded_positive_pair():
    result = evaluate_loopback_batch([_episode("get", "GET"), _episode("post", "POST"), _episode("neg", "GET", negative=True)])
    assert result["pair_contract"]["get_post_pair"] is True
    assert result["metrics"]["typed_positive_count"] == 2
    assert result["episodes"][2]["typed_effect_confirmed"] is False
    assert result["metrics"]["training_eligible_count"] == 0
    assert result["checks"]["wire_emission"] is False


def test_available_remote_probe_must_be_loopback_only():
    episode = _episode("get", "GET")
    episode["remote_probe"]["external_network"] = True
    with pytest.raises(ValueError, match="loopback_only"):
        evaluate_loopback_batch([episode], require_get_post_pair=False)


def test_raw_material_is_rejected():
    episode = _episode("get", "GET")
    episode["candidate"]["response_body"] = "not stored"
    with pytest.raises(ValueError, match="raw request/response"):
        evaluate_loopback_batch([episode], require_get_post_pair=False)
