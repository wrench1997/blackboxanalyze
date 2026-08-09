import hashlib
import json
from pathlib import Path

import torch

from app.neural_active_probe_decoder import NeuralActiveProbeSetDecoder, signature_to_tokens


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg102_guarded_decoder_passes_diagnostic_but_keeps_raw_failure_visible():
    report = _load("pg102_neural_active_probe_decoder_report_v1.json")
    assert report["status"] == "passed_guarded_diagnostic"
    guarded = report["metrics"]["guarded_neural"]
    pg42 = guarded["pg42"]
    pg35 = guarded["pg35_third_implementation"]
    assert pg42["count"] == 360
    assert pg42["known_confirm_recall"] == 0.833333
    assert pg42["false_accept_count"] == 0
    assert pg42["unknown_family_strict_abstain"] is True
    assert pg42["not_all_abstain"] is True
    assert pg35["count"] == 162
    assert pg35["known_confirm_recall"] == 0.875
    assert pg35["false_accept_count"] == 0
    raw = report["raw_failure_visible"]
    assert raw["failure_present"] is True
    assert raw["unknown_misname_count_pg42"] == 36
    assert raw["false_accept_count_pg42"] == 36
    assert raw["raw_model_would_be_promotable"] is False
    assert report["metrics"]["order_permutation"]["invariant"] is True
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg102_artifacts_keep_model_input_bounded_and_checkpoint_attested():
    report = _load("pg102_neural_active_probe_decoder_report_v1.json")
    dataset = _load("pg102_neural_active_probe_visible_dataset_v1.json")
    trace = _load("pg102_neural_active_probe_trace_v1.json")
    assert dataset["training_eligible"] is False
    assert trace["training_eligible"] is False
    assert dataset["model_input_contract"]["oracle_is_label_not_feature"] is True
    assert dataset["model_input_contract"]["family_label_in_features"] is False
    assert dataset["model_input_contract"]["raw_probe_strings_stored"] is False
    assert dataset["model_input_contract"]["raw_response_bodies_stored"] is False
    assert len(dataset["rows"]) == 618
    assert len(trace["steps"]) == 618
    assert trace["evaluator_labels_in_trace"] is False
    assert all("evaluator_label" not in row for row in dataset["rows"])
    assert all(not row["raw_probe_strings_stored"] and not row["raw_response_body_stored"] for row in dataset["rows"])
    for row in dataset["rows"]:
        assert row["model_input"]["probe_order"] == [f"p{i}" for i in range(9)]
        assert len(row["model_input"]["delta_pattern"]) == 9
        assert len(row["model_input"]["geometry_sign_pattern"]) == 9
        assert len(row["evidence_sha256"]) == 64
    checkpoint = ROOT / "artifacts" / "pg102-neural-active-probe" / "model.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == report["model"]["checkpoint_sha256"] == dataset["checkpoint_sha256"]
    assert report["model"]["calibration"]["dev_row_count"] == 16
    assert report["model"]["calibration"]["distance_ceiling"] >= 0.0
    text = json.dumps({"report": report, "dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("<script", "onerror", "union select", "markup_candidate", "operator_like", "template_candidate"):
        assert forbidden not in text


def test_pg102_deepsets_forward_is_order_invariant_and_unseen_slot_is_guarded():
    dataset = _load("pg102_neural_active_probe_visible_dataset_v1.json")
    signature = next(row["model_input"] for row in dataset["rows"] if row["role"] == "train")
    model = NeuralActiveProbeSetDecoder(("xss", "injection"), hidden_dim=16)
    tokens = signature_to_tokens(signature)
    with torch.inference_mode():
        first = model(tokens.unsqueeze(0))
        reversed_output = model(tokens.flip(0).unsqueeze(0))
    assert torch.allclose(first, reversed_output, atol=1e-5, rtol=0.0)
    assert tokens.shape == (9, 27)
    assert model.token_dim == 27

