from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg351_ask_oracle_composition import audit
from scripts.build_pg351_ask_oracle_composition_dataset import build
from scripts.run_pg351_a800_ask_oracle_composition_candidate import TARGET_KEY_ORDER, _balanced_action_rows, _coverage, _generate_constrained_target, _rows, evaluate_gate


ROOT = Path(__file__).resolve().parents[1]


def _dataset() -> dict:
    typed = json.loads((ROOT / "research" / "pg350_oracle_slot_source_rows_v1.json").read_text(encoding="utf-8"))
    ask = json.loads((ROOT / "research" / "pg348_dynamic_context_dataset_v1.json").read_text(encoding="utf-8"))
    return build(typed, ask, typed_sha256="a" * 64, ask_sha256="b" * 64)


def test_pg351_runner_sees_ask_and_rule_ir_on_both_splits_without_raw() -> None:
    dataset = _dataset()
    train, train_failures = _rows(dataset, "train")
    holdout, holdout_failures = _rows(dataset, "implementation_holdout")
    assert not train_failures and not holdout_failures
    for rows in (train, holdout):
        coverage = _coverage(rows)
        assert all(coverage[key] for key in ("ask_present", "repair_present", "abstain_present", "positive_present"))
    assert len(train) == 1152 and len(holdout) == 680


def test_pg351_gate_is_remote_candidate_only_and_fail_closed_on_bad_device() -> None:
    dataset = _dataset()
    audit_result = audit(dataset, dataset_sha256="c" * 64)
    train, train_failures = _rows(dataset, "train")
    holdout, holdout_failures = _rows(dataset, "implementation_holdout")
    base = evaluate_gate(
        dataset=dataset,
        audit=audit_result,
        env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"},
        device={"cuda_available": True, "visible_device_count": 1, "current_device": 0, "name": "NVIDIA A800-SXM4-80GB"},
        locks={key: "a" * 64 for key in ("dataset", "audit", "rules", "script", "model")},
        train_rows=train,
        train_failures=train_failures,
        holdout_rows=holdout,
        holdout_failures=holdout_failures,
        now=__import__("datetime").datetime(2026, 8, 9, 6, 0),
    )
    assert base["training_allowed"] is True
    bad = evaluate_gate(
        dataset=dataset,
        audit=audit_result,
        env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "1"},
        device={"cuda_available": True, "visible_device_count": 1, "current_device": 0, "name": "NVIDIA A800-SXM4-80GB"},
        locks={key: "a" * 64 for key in ("dataset", "audit", "rules", "script", "model")},
        train_rows=train,
        train_failures=train_failures,
        holdout_rows=holdout,
        holdout_failures=holdout_failures,
        now=__import__("datetime").datetime(2026, 8, 9, 6, 0),
    )
    assert bad["training_allowed"] is False
    assert "cuda_visible_devices_zero" in bad["failures"]


def test_constrained_rule_ir_decoder_keeps_field_order_without_answer_values() -> None:
    import torch

    tokens = ["[UNK]", "ctx", "[TARGET_BOS]", "[TARGET_EOS]"]
    values = {
        "question": ["ask_typed", "none"],
        "ask_reason": ["typed_evidence"],
        "next_action": ["ask_typed", "repair"],
        "repair_action": ["observe", "method"],
        "safe_to_send": ["0", "1"],
        "transport_ref": ["unknown"],
        "field_role_ref": ["unknown"],
        "encoding_ref": ["unknown"],
        "probe_variant_ref": ["none"],
        "payload_shape_ref": ["unknown"],
        "oracle_ref": ["unknown"],
        "negative_control_presence_ref": ["not_observed"],
    }
    for key in TARGET_KEY_ORDER:
        tokens.extend(f"{key}={value}" for value in values[key])
    vocabulary = {token: index for index, token in enumerate(dict.fromkeys(tokens))}

    class StubModel:
        config = type("Config", (), {"max_length": 128})()

        def __call__(self, input_ids, valid_mask=None):
            # Deliberately make a forbidden token globally dominant.  The
            # grammar mask must still select a value from the current slot.
            logits = torch.zeros((1, input_ids.shape[1], len(vocabulary)))
            logits[0, -1, vocabulary["[TARGET_BOS]"]] = 100.0
            return logits, torch.tensor(0.0)

    output = _generate_constrained_target(StubModel(), ["ctx"], vocabulary, torch.device("cpu"))
    assert output[0] == "[TARGET_BOS]"
    assert output[-1] == "[TARGET_EOS]"
    assert [token.split("=", 1)[0] for token in output[1:-1]] == list(TARGET_KEY_ORDER)


def test_action_balancing_changes_frequency_only() -> None:
    dataset = _dataset()
    rows, failures = _rows(dataset, "train")
    assert not failures
    balanced, counts = _balanced_action_rows(rows, seed=35101)
    assert len(set(counts.values())) == 1
    assert len(balanced) == sum(counts.values())
    assert {id(row) for row in balanced}.issubset({id(row) for row in rows})
