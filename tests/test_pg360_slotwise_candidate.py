from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.run_pg360_a800_slotwise_candidate import _balanced_slot_rows, _guard_prediction, _slot_prediction, _slot_value, _sqrt_balanced_slot_rows


def test_slot_value_reads_only_requested_abstract_slot() -> None:
    target = ["[TARGET_BOS]", "question=none", "next_action=repair", "[TARGET_EOS]"]
    assert _slot_value(target, "next_action") == "next_action=repair"
    assert _slot_value(target, "oracle_ref") == "oracle_ref=unknown"


def test_slot_value_balancer_equalizes_each_slot_without_cross_slot_mixing() -> None:
    rows = [
        {"slot": "next_action", "target_tokens": ["[TARGET_BOS]", "next_action=ask_typed", "[TARGET_EOS]"]},
        {"slot": "next_action", "target_tokens": ["[TARGET_BOS]", "next_action=repair", "[TARGET_EOS]"]},
        {"slot": "next_action", "target_tokens": ["[TARGET_BOS]", "next_action=repair", "[TARGET_EOS]"]},
        {"slot": "safe_to_send", "target_tokens": ["[TARGET_BOS]", "safe_to_send=0", "[TARGET_EOS]"]},
    ]
    balanced, counts = _balanced_slot_rows(rows, 36001)
    assert counts["next_action"] == {"next_action=ask_typed": 2, "next_action=repair": 2}
    assert counts["safe_to_send"] == {"safe_to_send=0": 1}
    assert len(balanced) == 5


def test_sqrt_balancer_increases_rare_values_without_flattening_prior() -> None:
    rows = [
        *[{"slot": "next_action", "target_tokens": ["[TARGET_BOS]", "next_action=ask_typed", "[TARGET_EOS]"]} for _ in range(9)],
        *[{"slot": "next_action", "target_tokens": ["[TARGET_BOS]", "next_action=repair", "[TARGET_EOS]"]} for _ in range(1)],
    ]
    sampled, counts = _sqrt_balanced_slot_rows(rows, 36001)
    assert counts["next_action"]["next_action=ask_typed"] == 9
    assert counts["next_action"]["next_action=repair"] == 3
    assert len(sampled) == 12


def test_rule_ir_guard_forces_ask_when_typed_observation_is_missing() -> None:
    row = {"context_tokens": ["belief_typed_available=unknown", "belief_fresh_reset=unknown"]}
    proposal = {
        "question": "question=none",
        "next_action": "next_action=select_probe_variant",
        "repair_action": "repair_action=none",
        "transport_ref": "transport_ref=get_query",
        "field_role_ref": "field_role_ref=query_text",
        "encoding_ref": "encoding_ref=identity",
        "probe_variant_ref": "probe_variant_ref=source_attested_candidate",
        "payload_shape_ref": "payload_shape_ref=query_marker",
        "oracle_ref": "oracle_ref=typed_effect",
        "safe_to_send": "safe_to_send=1",
    }
    guarded = _guard_prediction(row, proposal)
    assert guarded["question"] == "question=ask_typed"
    assert guarded["next_action"] == "next_action=ask_typed"
    assert guarded["safe_to_send"] == "safe_to_send=0"


def test_rule_ir_guard_changes_failure_to_one_variable_repair() -> None:
    row = {"context_tokens": ["failure_failure_class=blocked_variant", "belief_typed_available=present", "belief_fresh_reset=present", "belief_replay_ready=present", "belief_evidence_present=present", "belief_reference_present=present", "belief_candidate_present=present"]}
    proposal = {"question": "question=none", "next_action": "next_action=select_probe_variant", "repair_action": "repair_action=none", "probe_variant_ref": "probe_variant_ref=source_attested_candidate", "safe_to_send": "safe_to_send=1"}
    guarded = _guard_prediction(row, proposal)
    assert guarded["question"] == "question=ask_failure"
    assert guarded["next_action"] == "next_action=repair"
    assert guarded["repair_action"] == "repair_action=one_variable"
    assert guarded["safe_to_send"] == "safe_to_send=0"


def test_slot_prediction_keeps_schema_query_prefix_aligned_with_training() -> None:
    vocabulary = {
        "[UNK]": 0,
        "page=surface": 1,
        "[SLOT_QUERY_BOS]": 2,
        "slot_query=safe_to_send": 3,
        "[SLOT_QUERY_EOS]": 4,
        "[TARGET_BOS]": 5,
        "safe_to_send=0": 6,
        "safe_to_send=1": 7,
    }

    class RecordingModel:
        config = SimpleNamespace(max_length=32)

        def __init__(self) -> None:
            self.inputs: list[list[int]] = []

        def __call__(self, input_ids: torch.Tensor, *, valid_mask: torch.Tensor):
            self.inputs.append(input_ids.detach().cpu().tolist()[0])
            logits = torch.zeros((1, input_ids.shape[1], len(vocabulary)), dtype=torch.float32)
            logits[:, -1, vocabulary["safe_to_send=0"]] = 1.0
            return logits, torch.tensor(0.0)

    model = RecordingModel()
    row = {"slot": "safe_to_send", "context_tokens": ["page=surface"]}
    assert _slot_prediction(model, row, vocabulary, torch.device("cpu")) == "safe_to_send=0"
    assert model.inputs == [[1, 2, 3, 4, 5]]
