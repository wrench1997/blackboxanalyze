from app.surface_role_gate import assess_surface_role_gate


def test_surface_role_gate_stays_diagnostic_when_positive_recall_is_missing():
    result = assess_surface_role_gate(
        [
            {"candidate_role": "plain_control", "abstained": False},
            {"candidate_role": "plain_control", "abstained": True},
            {"candidate_role": "json_echo", "abstained": False},
        ],
        ["reflected_attribute", "reflected_text", "json_echo"],
        validation_precision=1.0,
        training_acceptance=True,
    )
    assert result["enabled"] is False
    assert result["status"] == "diagnostic_only"
    assert "positive_recall_below_gate" in result["reasons"]


def test_surface_role_gate_accepts_high_precision_high_recall_head():
    result = assess_surface_role_gate(
        [
            {"candidate_role": "reflected_attribute", "abstained": False},
            {"candidate_role": "reflected_text", "abstained": False},
            {"candidate_role": "json_echo", "abstained": False},
        ],
        ["reflected_attribute", "reflected_text", "json_echo"],
        validation_precision=1.0,
        training_acceptance=True,
    )
    assert result["enabled"] is True
    assert result["status"] == "enabled"
