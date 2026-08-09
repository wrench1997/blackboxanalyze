import hashlib

import pytest

from app.trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _step(method: str = "GET", *, decision: str = "abstain", negative_pair: bool = True) -> dict:
    action = {
        "method": method,
        "route_template_id": "fixture-route",
        "placement": "query" if method == "GET" else "form",
        "encoding_chain": ["identity"],
        "probe_ref": "inert-marker",
        "probe_sha256": _hash("probe" + method),
        "safety": {
            "no_external_network": True,
            "does_not_execute": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }
    if method == "POST":
        action["form_field_names"] = ["probe"]
    oracle = {
        "modality": "negative_control",
        "positive": False,
        "positive_authority": False,
        **({"negative_control_pair_id": "control-1"} if negative_pair else {}),
    }
    body = {
        "action_manifest": action,
        "baseline_projection": {"status_class": "2xx"},
        "response_projection": {"status_class": "2xx"},
        "oracle_projection": oracle,
        "belief_before": {"xss": .5, "none": .5},
        "belief_after": {"xss": .4, "none": .6},
        "decision": decision,
        "next_action": "try_post" if method == "GET" else "stop",
    }
    return {
        "episode_id": "episode-1",
        "step_id": "step-" + method.lower(),
        "parent_step_id": None,
        "sampling_seed": 1,
        "target_instance_id": "target-1",
        "hypothesis": "xss",
        **body,
        "fresh_reset": {"completed": True, "fresh_target": True, "evaluator_state_hidden": True},
        "evidence_sha256": _hash("evidence" + method),
        "echo": {"sha256": sha256_json(body)},
    }


def test_trace_echo_binds_the_model_decision_to_observed_projection():
    normalized = validate_trace_step(_step())
    assert normalized["online_weight_update"] is False
    assert normalized["long_term_memory_write"] is False
    assert normalized["echo"]["sha256"]


def test_episode_requires_get_post_and_negative_pair_before_accepting_evaluation():
    first = _step("GET")
    second = _step("POST")
    second["parent_step_id"] = first["step_id"]
    report = evaluate_episode([first, second])
    assert report["status"] == "accepted_evaluation"
    assert report["training_candidate"] is False

    blocked = evaluate_episode([_step("GET", negative_pair=False)])
    assert blocked["status"] == "trace_only"
    assert "missing_get_or_post_step" in blocked["reasons"]
    assert "missing_negative_control_pair" in blocked["reasons"]


def test_weak_oracle_cannot_be_confirmed_positive():
    step = _step(decision="confirmed_positive")
    step["oracle_projection"]["positive"] = True
    step["oracle_projection"]["positive_authority"] = False
    body = {
        key: step[key]
        for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action")
    }
    step["echo"] = {"sha256": sha256_json(body)}
    with pytest.raises(ValueError, match="confirmed_positive"):
        validate_trace_step(step)


def test_optional_causal_triplet_fields_are_validated_and_echo_bound():
    step = _step()
    step["neutral_projection"] = {"status_class": "2xx", "body_length_bucket": "0"}
    step["negative_probe_projection"] = {"status_class": "2xx", "body_length_bucket": "1-255"}
    step["neutral_oracle_projection"] = {"modality": "typed_negative", "positive": False, "positive_authority": True}
    step["negative_oracle_projection"] = {"modality": "typed_negative", "positive": False, "positive_authority": True}
    body = {
        key: step[key]
        for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action", "neutral_projection", "negative_probe_projection", "neutral_oracle_projection", "negative_oracle_projection")
    }
    step["echo"] = {"sha256": sha256_json(body)}
    normalized = validate_trace_step(step)
    assert normalized["neutral_projection"]["body_length_bucket"] == "0"
    assert normalized["negative_probe_projection"]["body_length_bucket"] == "1-255"
    assert normalized["negative_oracle_projection"]["positive"] is False
