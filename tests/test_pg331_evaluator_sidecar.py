from __future__ import annotations

from copy import deepcopy

import pytest

from app.pg331_evaluator_sidecar import (
    build_pg331_evaluator_record,
    build_pg331_evaluator_sidecar,
    sha256_json,
    validate_pg331_evaluator_record,
    validate_pg331_evaluator_sidecar,
)


def _digest(value: object) -> str:
    return sha256_json(value)


def _reset() -> dict[str, object]:
    return {
        "reset_id": "pg331-reset-seed-route",
        "fresh_reset": True,
        "target_instance_digest": _digest("container-id"),
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "state_clean": True,
        "volume_mount_count": 0,
        "container_restart_used": False,
    }


def _projection(*, typed: bool, dom: bool = False) -> dict[str, object]:
    return {
        "status_class": "2xx",
        "content_type_class": "html",
        "body_shape": "html",
        "shape_sha256": _digest({"shape": typed, "dom": dom}),
        "challenge_state_available": dom,
        "challenge_state_delta": dom,
        "sink_present": dom,
        "response_shape_changed": typed,
        "non_destructive": True,
    }


def _role(role: str, *, typed: bool, evidence: str, dom: bool = False, sent: bool = True) -> dict[str, object]:
    return {
        "sent": sent,
        "available": True,
        "executed": sent,
        "typed_effect_confirmed": typed,
        "effect_class": "dom_effect" if dom and typed else "result_shape" if typed else "none",
        "projection": _projection(typed=typed, dom=dom),
        "evidence_sha256": _digest({"role": role, "evidence": evidence}),
        "non_destructive": True,
    }


def _triplet() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return (
        _role("candidate", typed=True, evidence="candidate", dom=True),
        _role("reference", typed=True, evidence="reference", dom=True),
        _role("negative", typed=False, evidence="negative", dom=False),
    )


def test_complete_triplet_is_role_bound_and_valid() -> None:
    candidate, reference, negative = _triplet()
    sidecar = build_pg331_evaluator_sidecar(
        record_id="pg331:seed:route",
        reset=_reset(),
        candidate=candidate,
        reference=reference,
        negative=negative,
        replay_consistent=True,
        reference_agreement=True,
        negative_control_clean=True,
        evaluator_id="pg331-fixture-evaluator",
    )
    assert set(sidecar["roles"]) == {"candidate", "reference", "negative"}
    assert sidecar["roles"]["candidate"]["evidence_scope"] == "record_role_bound"
    assert sidecar["roles"]["candidate"]["evidence_sha256"] != sidecar["roles"]["reference"]["evidence_sha256"]
    assert sidecar["typed_effect_confirmed"] is True
    assert sidecar["confirmed_positive"] is True
    assert validate_pg331_evaluator_sidecar(sidecar)["valid"] is True


def test_record_keeps_sidecar_off_context_and_never_promotes() -> None:
    candidate, reference, negative = _triplet()
    record = build_pg331_evaluator_record(
        record_id="pg331:seed:route",
        reset=_reset(),
        candidate=candidate,
        reference=reference,
        negative=negative,
        replay_consistent=True,
        reference_agreement=True,
        negative_control_clean=True,
        evaluator_id="pg331-fixture-evaluator",
    )
    assert record["model_context"]["typed_available"] is True
    assert "roles" not in record["model_context"]
    assert record["raw_payload_stored"] is False
    assert record["raw_response_stored"] is False
    assert record["training_eligible"] is False
    assert record["promotion"]["vulnerability_claim_allowed"] is False
    assert validate_pg331_evaluator_record(record)["valid"] is True


def test_missing_source_evidence_is_diagnostic_not_confirmed() -> None:
    candidate, reference, negative = _triplet()
    candidate = deepcopy(candidate)
    candidate.pop("evidence_sha256")
    sidecar = build_pg331_evaluator_sidecar(
        record_id="pg331:missing-evidence",
        reset=_reset(),
        candidate=candidate,
        reference=reference,
        negative=negative,
        replay_consistent=True,
        reference_agreement=True,
        negative_control_clean=True,
    )
    assert sidecar["roles"]["candidate"]["source_evidence_hash_valid"] is False
    assert sidecar["checks"]["evidence_hashes"] is False
    assert sidecar["confirmed_positive"] is False
    assert validate_pg331_evaluator_sidecar(sidecar)["valid"] is True


def test_negative_typed_effect_blocks_confirmation_and_records_violation() -> None:
    candidate, reference, negative = _triplet()
    negative = deepcopy(negative)
    negative["typed_effect_confirmed"] = True
    negative["effect_class"] = "result_shape"
    sidecar = build_pg331_evaluator_sidecar(
        record_id="pg331:negative-violation",
        reset=_reset(),
        candidate=candidate,
        reference=reference,
        negative=negative,
        replay_consistent=True,
        reference_agreement=True,
    )
    assert sidecar["checks"]["negative_control_clean"] is False
    assert sidecar["confirmed_positive"] is False
    assert "negative_control_clean" in sidecar["reasons"]


def test_missing_control_cannot_be_learned_as_clean_context() -> None:
    candidate, reference, negative = _triplet()
    negative = deepcopy(negative)
    negative["sent"] = False
    record = build_pg331_evaluator_record(
        record_id="pg331:missing-control-context",
        reset=_reset(),
        candidate=candidate,
        reference=reference,
        negative=negative,
        replay_consistent=False,
        reference_agreement=True,
    )
    assert record["model_context"]["negative_control"] is False
    assert record["model_context"]["typed_available"] is False
    assert record["model_context"]["replay_ready"] is False


def test_raw_projection_is_rejected_before_it_can_reach_context() -> None:
    candidate, reference, negative = _triplet()
    candidate = deepcopy(candidate)
    candidate["projection"] = {"response_body": "literal body"}
    with pytest.raises(ValueError, match="raw"):
        build_pg331_evaluator_record(
            record_id="pg331:raw",
            reset=_reset(),
            candidate=candidate,
            reference=reference,
            negative=negative,
        )


def test_reset_missing_external_network_does_not_become_fresh() -> None:
    candidate, reference, negative = _triplet()
    reset = _reset()
    reset.pop("external_network")
    sidecar = build_pg331_evaluator_sidecar(
        record_id="pg331:reset-gap",
        reset=reset,
        candidate=candidate,
        reference=reference,
        negative=negative,
        replay_consistent=True,
        reference_agreement=True,
        negative_control_clean=True,
    )
    assert sidecar["checks"]["fresh_reset"] is False
    assert sidecar["confirmed_positive"] is False


def test_hash_tampering_is_detected() -> None:
    candidate, reference, negative = _triplet()
    sidecar = build_pg331_evaluator_sidecar(
        record_id="pg331:tamper",
        reset=_reset(),
        candidate=candidate,
        reference=reference,
        negative=negative,
    )
    sidecar["roles"]["negative"]["typed_effect_confirmed"] = True
    result = validate_pg331_evaluator_sidecar(sidecar)
    assert result["valid"] is False
    assert "evidence_hash_mismatch" in result["failures"]


def test_disposable_state_delta_is_distinguished_from_external_write() -> None:
    candidate, reference, negative = _triplet()
    for role in (candidate, reference, negative):
        role["projection"] = {
            **role["projection"],
            "database_touched": True,
            "disposable_state_delta": True,
            "state_delta_class": "disposable_evaluator_state",
            "external_network_blocked": True,
        }
    sidecar = build_pg331_evaluator_sidecar(
        record_id="pg331:stateful-disposable",
        reset=_reset(),
        candidate=candidate,
        reference=reference,
        negative=negative,
        replay_consistent=True,
        reference_agreement=True,
        negative_control_clean=True,
        evaluator_id="pg331-stateful-fixture",
    )
    assert sidecar["checks"]["non_destructive"] is True
    assert sidecar["confirmed_positive"] is True
    assert validate_pg331_evaluator_sidecar(sidecar)["valid"] is True
