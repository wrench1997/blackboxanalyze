from __future__ import annotations

import importlib


def test_failure_projection_is_abstract_and_changes_action():
    module = importlib.import_module("scripts.run_pg337_dvwa_failure_repair_replay")
    value = module._failure_projection(previous="candidate_failed", next_action="repair", outcome="recovered")
    assert value["failure_class"] == "candidate_without_typed_effect"
    assert value["previous_action"] != value["next_action"]
    assert value["repair_outcome"] == "recovered"
    assert "payload" not in repr(value).casefold()


def test_negative_target_is_safe_abstain():
    module = importlib.import_module("scripts.run_pg337_dvwa_failure_repair_replay")
    value = module._target(role="negative")
    assert value["question"] == "ask_failure"
    assert value["next_action"] == "abstain"
    assert value["safe_to_send"] is False
    assert value["probe_variant_ref"] == "negative_control"


def test_candidate_target_requires_repair_observation():
    module = importlib.import_module("scripts.run_pg337_dvwa_failure_repair_replay")
    value = module._target(role="candidate")
    assert value["question"] == "ask_failure"
    assert value["next_action"] == "repair"
    assert value["repair_action"] == "observe"
    assert value["safe_to_send"] is False
