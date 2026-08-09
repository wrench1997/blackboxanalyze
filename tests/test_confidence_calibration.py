from app.confidence_calibration import (
    accept_with_evidence,
    binary_brier_score,
    evidence_fused_confidence,
    expected_calibration_error,
    family_oracle_support,
    fit_temperature,
    temperature_scale,
)


def test_temperature_scaling_is_deterministic_and_fits_nll():
    probabilities = [[0.90, 0.10], [0.80, 0.20], [0.40, 0.60], [0.30, 0.70]]
    labels = [0, 0, 1, 1]
    fit = fit_temperature(probabilities, labels)
    assert fit["temperature"] > 0
    assert fit["nll_after"] <= fit["nll_before"]
    assert temperature_scale(probabilities, fit["temperature"]) == temperature_scale(probabilities, fit["temperature"])


def test_calibration_metrics_and_family_specific_oracle_gate():
    confidence = [0.95, 0.10, 0.80, 0.20]
    labels = [True, False, True, False]
    assert expected_calibration_error(confidence, labels) >= 0.0
    assert binary_brier_score(confidence, labels) < 0.05
    positive = family_oracle_support("xss", {"marker_reflected": True, "marker_in_attribute": True})
    negative = family_oracle_support("xss", {"marker_reflected": False})
    assert positive > negative
    assert family_oracle_support("xss", {"marker_reflected": True, "marker_in_html_text": True}) < 0.50
    assert family_oracle_support("xss", {"marker_reflected": True, "marker_in_json_value": True}) < 0.50
    text_support = family_oracle_support("xss", {"marker_reflected": True, "marker_in_html_text": True})
    assert not accept_with_evidence(
        calibrated_confidence=evidence_fused_confidence(0.95, text_support),
        oracle_support=text_support,
    )
    assert family_oracle_support("injection", {"controlled_differential": True, "interpreter_boundary": True, "modality": "syntax_error"}) > 0.80
    fused = evidence_fused_confidence(0.9, positive)
    assert accept_with_evidence(calibrated_confidence=fused, oracle_support=positive)
    assert not accept_with_evidence(calibrated_confidence=fused, oracle_support=negative)
