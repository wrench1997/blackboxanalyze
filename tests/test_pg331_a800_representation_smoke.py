from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from scripts.run_pg331_a800_representation_smoke import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, _predictive_metrics, _required_context_window, evaluate_representation_gate


def _row() -> dict[str, object]:
    return {"split": "train", "context_tokens": ["document_presence=observed", "javascript_presence=not_observed", "request_transport_presence=unknown"], "target_tokens": ["question=ask_typed"], "field_capture_manifest": {name: {"field": "not_observed" if name == "javascript" else "observed"} for name in ("document", "navigation", "request", "response", "javascript", "failure", "belief")}, "raw_payload_stored": False, "raw_response_body_stored": False, "oracle_answer_in_context": False, "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True}}


def _device() -> dict[str, object]: return {"cuda_available": True, "visible_device_count": 1, "current_device": 0, "name": "NVIDIA A800-SXM4-80GB"}


def test_context_only_gate_allows_diagnostic_rows_but_keeps_information_promotion_closed() -> None:
    result = evaluate_representation_gate(now=datetime(2026, 8, 8, 9, tzinfo=ZoneInfo("Asia/Shanghai")), env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"}, dataset={"records": [_row()]}, information_audit={"status": "incomplete"}, vocabulary={"context_tokens": _row()["context_tokens"]}, device=_device(), locks={key: "a" * 64 for key in ("dataset", "information", "vocabulary", "rules", "script", "model")})
    assert result["status"] == "ready_representation_pretrain_candidate"
    assert result["information_promotion_gate_passed"] is False
    assert all(value is False for value in result["promotion"].values())


def test_raw_context_or_wrong_device_blocks_before_cuda_training() -> None:
    row = _row(); row["context_tokens"] = ["document_presence=observed", "response_body=bad"]
    result = evaluate_representation_gate(now=datetime(2026, 8, 8, 9, tzinfo=ZoneInfo("Asia/Shanghai")), env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"}, dataset={"records": [row]}, information_audit={}, vocabulary={"context_tokens": ["document_presence=observed", "response_body=bad"]}, device={"cuda_available": True, "visible_device_count": 2, "current_device": 0, "name": "NVIDIA A800"}, locks={})
    assert result["representation_training_allowed"] is False
    assert "single_visible_a800_gpu0" in result["failures"]
    assert any("context_firewall" in item for item in result["failures"])


def test_implementation_holdout_is_counted_but_not_read_into_context_training() -> None:
    train = _row(); train["split"] = "train"
    holdout = {"split": "implementation_holdout", "context_tokens": ["response_body=must-not-be-read"], "target_tokens": ["payload=must-not-be-read"]}
    result = evaluate_representation_gate(now=datetime(2026, 8, 8, 9, tzinfo=ZoneInfo("Asia/Shanghai")), env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"}, dataset={"records": [train, holdout]}, information_audit={}, vocabulary={"context_tokens": train["context_tokens"]}, device=_device(), locks={key: "a" * 64 for key in ("dataset", "information", "vocabulary", "rules", "script", "model")})
    assert result["representation_training_allowed"] is True
    assert result["context_row_count"] == 1
    assert result["split_counts"] == {"train": 1, "implementation_holdout": 1}
    assert result["source_implementation_holdout_recorded"] is True


def test_unknown_context_token_blocks_fixed_vocabulary_training() -> None:
    row = _row(); row["split"] = "train"; row["context_tokens"].append("new_axis=unknown")
    result = evaluate_representation_gate(now=datetime(2026, 8, 8, 9, tzinfo=ZoneInfo("Asia/Shanghai")), env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"}, dataset={"records": [row]}, information_audit={}, vocabulary={"context_tokens": _row()["context_tokens"]}, device=_device(), locks={key: "a" * 64 for key in ("dataset", "information", "vocabulary", "rules", "script", "model")})
    assert result["representation_training_allowed"] is False
    assert result["unknown_context_token_count"] == 1
    assert "context_vocabulary_locked" in result["failures"]


def test_abstract_response_body_shape_and_length_are_not_raw_side_channels() -> None:
    row = _row(); row["context_tokens"] += ["response_body_length=medium", "response_body_shape=html"]
    result = evaluate_representation_gate(now=datetime(2026, 8, 8, 9, tzinfo=ZoneInfo("Asia/Shanghai")), env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"}, dataset={"records": [row]}, information_audit={}, vocabulary={"context_tokens": row["context_tokens"]}, device=_device(), locks={key: "a" * 64 for key in ("dataset", "information", "vocabulary", "rules", "script", "model")})
    assert result["representation_training_allowed"] is True


def test_predictive_entropy_is_meaned_across_all_next_tokens() -> None:
    import torch

    class _Model:
        def __call__(self, ids):
            # Two prediction positions: the first confident, the second flat.
            return torch.tensor([[[8.0, 0.0], [0.0, 0.0]]]), torch.tensor(0.0)

    rows = [{"context_tokens": ["a", "b", "a"]}]
    metrics = _predictive_metrics(_Model(), rows, {"a": 0, "b": 1}, torch.device("cpu"))
    assert metrics["next_token_count"] == 2
    assert 0.3 < metrics["mean_predictive_entropy_nats"] < 0.5


def test_capacity_uses_longest_holdout_context_without_truncation() -> None:
    requirement = _required_context_window([{"context_tokens": ["a", "b"]}], [{"context_tokens": ["a", "b", "c", "d", "e"]}])
    assert requirement == {"train_context_max_length": 2, "holdout_context_max_length": 5, "required_max_length": 5}


def test_stability_defaults_are_conservative() -> None:
    assert DEFAULT_LEARNING_RATE == 1e-4
    assert DEFAULT_EPOCHS == 1
