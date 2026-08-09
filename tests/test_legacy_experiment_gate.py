import os

import pytest

from app.legacy_experiment_gate import LEGACY_DIAGNOSTIC_ENV, assert_legacy_training_blocked, legacy_status


def test_legacy_artifacts_are_not_training_eligible(monkeypatch):
    monkeypatch.delenv(LEGACY_DIAGNOSTIC_ENV, raising=False)
    status = legacy_status("research/pikachu_payload_catalog_v1.json")
    assert status["status"] == "legacy_diagnostic_only"
    assert status["training_eligible"] is False
    assert status["memory_promotion"] is False
    with pytest.raises(RuntimeError, match="typed payload-success acceptance"):
        assert_legacy_training_blocked(["research/pikachu_payload_catalog_v1.json"])


def test_legacy_diagnostic_bypass_is_explicit(monkeypatch):
    monkeypatch.setenv(LEGACY_DIAGNOSTIC_ENV, "1")
    assert_legacy_training_blocked(["research/pikachu_payload_catalog_v1.json"])
