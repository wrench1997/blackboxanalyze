import json

from scripts.audit_pg335_real_process_token_dataset import audit
from scripts.build_pg335_real_process_token_dataset import AXES, build_dataset


def _row(role, split="train"):
    return {
        "record_id": f"fixture:{role}",
        "split": split,
        "context_tokens": ["[BOS]", "axis_begin=document_structure", "doc_title_shape=alpha", "axis_end=document_structure", "axis_begin=navigation", "nav_link_count=one", "axis_end=navigation", "axis_begin=request_transport", "method=GET", "axis_end=request_transport", "axis_begin=response_transport", "content_type=html", "axis_end=response_transport", "axis_begin=javascript_surface", "js_source_shape=empty", "axis_end=javascript_surface", "axis_begin=failure_feedback", "failure_class=none", "axis_end=failure_feedback", "axis_begin=belief_and_replay", "replay_state=ready", "axis_end=belief_and_replay", "[CTX_END]"],
        "target_tokens": ["[TARGET_BOS]", "question=none", "next_action=send_probe", "probe_variant_ref=negative_control" if role == "negative" else "probe_variant_ref=source_attested_candidate", "safe_to_send=1", "[TARGET_EOS]"],
        "field_capture_manifest": {axis: {"presence": "observed"} for axis in AXES},
        "evaluator_sidecar": {"typed_available": True, "fresh_reset": True, "evidence_hash": "a" * 64},
        "record_sha256": "b" * 64,
    }


def test_real_source_masks_preserve_axes_and_remove_literals():
    data = build_dataset({"records": [_row("candidate"), _row("negative"), _row("holdout", "implementation_holdout")]})
    assert data["counts"]["source_rows"] == 3
    assert set(data["counts"]["axis_masks"]) == set(AXES)
    assert all(not any(str(token).startswith(("family=", "implementation=", "route=", "source=", "payload=", "oracle=")) for token in row["context_tokens"]) for row in data["records"])
    masked = next(row for row in data["records"] if row["diagnostic_kind"] == "ask" and row["axis_mask"] == "document_structure")
    assert masked["target_projection"]["next_action"] == "ask_typed"
    assert masked["field_capture_manifest"]["document_structure"]["presence"] == "not_observed"


def test_audit_requires_ask_failure_negative_and_no_promotion():
    data = build_dataset({"records": [_row("candidate"), _row("negative"), _row("holdout", "implementation_holdout")]})
    report = audit(data)
    # The miniature unit fixture has only three source rows, so the real-row
    # quota correctly keeps its overall status blocked; process checks still
    # have to pass independently.
    assert report["status"] in {"diagnostic_only", "blocked"}
    assert report["checks"]["ask_recall"]
    assert report["checks"]["failure_action_change"]
    assert report["checks"]["negative_abstain"]
    assert all(value is False for value in report["promotion"].values())
