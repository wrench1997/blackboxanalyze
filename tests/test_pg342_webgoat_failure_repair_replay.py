from __future__ import annotations

import os

import pytest

from scripts.run_pg342_webgoat_failure_repair_replay import (
    _failure,
    _route_ref,
    _target,
    collect_pg342,
)


def test_failure_contract_changes_action_for_candidate_and_abstains_negative() -> None:
    candidate = _failure(expected_method="GET", failed_method="POST", role="candidate", repaired=True)
    negative = _failure(expected_method="GET", failed_method="POST", role="negative", repaired=False)
    assert candidate["previous_action"] != candidate["next_action"]
    assert candidate["repair_outcome"] == "recovered"
    assert negative["next_action"] == "abstain"
    assert negative["repair_outcome"] == "abstained"


def test_targets_are_ask_failure_and_safe_by_default() -> None:
    for role in ("candidate", "reference", "negative"):
        target = _target(role=role)
        assert target["question"] == "ask_failure"
        assert target["safe_to_send"] is False
    assert _target(role="negative")["next_action"] == "abstain"
    assert _target(role="candidate")["next_action"] == "repair"


def test_route_refs_are_one_way_and_live_runner_is_opt_in() -> None:
    route = {"route_id": "webgoat-shape-get", "expected_method": "GET", "surface_id": "method_shape_get"}
    ref = _route_ref(route)
    assert len(ref) == 64
    assert "/WebGoat" not in ref
    old_local = os.environ.pop("PG342_LOCAL_DOCKER_EVAL", None)
    old_live = os.environ.pop("PG342_WEBGOAT_FAILURE_REPAIR_LIVE", None)
    try:
        with pytest.raises(RuntimeError):
            collect_pg342(seeds=(34201,))
    finally:
        if old_local is not None:
            os.environ["PG342_LOCAL_DOCKER_EVAL"] = old_local
        if old_live is not None:
            os.environ["PG342_WEBGOAT_FAILURE_REPAIR_LIVE"] = old_live
