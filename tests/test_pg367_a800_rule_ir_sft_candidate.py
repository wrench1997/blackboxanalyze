from __future__ import annotations

from scripts.run_pg367_a800_rule_ir_sft_candidate import CRITICAL_PREFIXES, _weights


def test_sft_weights_keep_critical_slots_explicit() -> None:
    vocab = {"next_action=repair": 0, "request_method=get": 1, "[TARGET_BOS]": 2}
    weights = _weights(vocab, critical=3.0, context=0.25)
    assert weights["next_action=repair"] == 3.0
    assert weights["request_method=get"] == 0.25
    assert all(prefix for prefix in CRITICAL_PREFIXES)
