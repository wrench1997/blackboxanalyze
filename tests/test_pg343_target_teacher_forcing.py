from __future__ import annotations

from scripts.audit_pg343_target_teacher_forcing import SLOTS


def test_teacher_forcing_audit_uses_abstract_slots_only() -> None:
    assert SLOTS[:3] == ("question", "next_action", "repair_action")
    assert "payload" not in SLOTS
    assert "response_body" not in SLOTS
