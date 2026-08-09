from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from app.pg295_causal_moe import CausalMoEConfig
from scripts.build_pg362_full_rule_ir_dataset import SLOTS
from scripts.run_pg375_composed_rule_ir_candidate import (
    ComposedRuleIRModel,
    _device_gate,
    _pad_lm,
    audit_pg375_contract,
    build_train_vocabulary,
    load_pg364_dataset,
    run_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def _target(*, safe: str = "1", next_action: str = "select_probe_variant") -> tuple[list[str], dict[str, str]]:
    values = {
        "question": "none",
        "ask_reason": "none",
        "next_action": next_action,
        "repair_action": "none" if next_action != "repair" else "method",
        "transport_ref": "get_query",
        "field_role_ref": "query_term",
        "encoding_ref": "identity",
        "syntax_category_ref": "marker",
        "probe_variant_ref": "source_attested_candidate",
        "safe_to_send": safe,
        "payload_shape_ref": "html_text_marker",
        "oracle_ref": "typed_effect" if safe == "1" else "negative_no_effect",
        "negative_control_presence_ref": "matched_triplet",
    }
    return ["[TARGET_BOS]", *[f"{slot}={values[slot]}" for slot in SLOTS], "[TARGET_EOS]"], values


def _row(split: str, context: list[str], *, safe: str = "1", next_action: str = "select_probe_variant") -> dict[str, object]:
    target, values = _target(safe=safe, next_action=next_action)
    return {
        "context_tokens": context,
        "target_tokens": target,
        "split": split,
        "_target_values": values,
    }


def _fixture() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    # Holdout contexts use only train-known tokens but a different ordering,
    # so the strict split gate can distinguish composition from memorisation.
    train = [
        _row("train", ["document_presence=observed", "request_method=get", "surface=alpha"]),
        _row("train", ["document_presence=observed", "request_method=get", "surface=beta"], safe="0", next_action="repair"),
    ]
    holdout = [
        _row("implementation_holdout", ["document_presence=observed", "surface=alpha", "request_method=get"]),
        _row("implementation_holdout", ["request_method=get", "document_presence=observed", "surface=beta"], safe="0", next_action="repair"),
    ]
    dataset = {
        "status": "diagnostic_candidate_only",
        "split_contract": {"train_group_hashes": ["train-group"], "holdout_group_hashes": ["holdout-group"]},
        "vocabulary": {},
    }
    return train, holdout, dataset


def test_pg375_strict_gate_blocks_existing_pg364_before_optimizer(tmp_path: Path) -> None:
    train, holdout, info, contract = load_pg364_dataset()
    assert contract["status"] == "blocked"
    assert contract["vocabulary_gaps"]["unknown_token_count"] == 1
    assert contract["cross_split"]["exact_sequence_overlap_count"] == 152
    result = run_candidate(train_rows=train, holdout_rows=holdout, dataset=info["dataset"], device="cpu", checkpoint_dir=tmp_path)
    assert result["status"] == "blocked_data_contract"
    assert result["training"]["optimizer_started"] is False
    assert result["execution"]["checkpoint_written"] is False
    assert list(tmp_path.iterdir()) == []


def test_pg375_derived_clean_split_still_blocks_capability_without_source_gate(tmp_path: Path) -> None:
    dataset = json.loads((ROOT / "research" / "pg375_strict_filtered_rule_ir_dataset_v1.json").read_text(encoding="utf-8"))
    train, holdout, _info, contract = load_pg364_dataset(ROOT / "research" / "pg375_strict_filtered_rule_ir_dataset_v1.json", None)
    assert contract["status"] == "blocked"
    assert "capability_training_not_authorized" in contract["failures"]
    result = run_candidate(train_rows=train, holdout_rows=holdout, dataset=dataset, device="cpu", checkpoint_dir=tmp_path)
    assert result["status"] == "blocked_data_contract"
    assert result["training"]["optimizer_started"] is False
    assert list(tmp_path.iterdir()) == []


def test_pg375_target_mask_covers_all_target_labels() -> None:
    train, _, _ = _fixture()
    vocabulary = build_train_vocabulary(train)
    ids, valid, target_mask = _pad_lm([train[0]], vocabulary, torch.device("cpu"))
    context_len = len(train[0]["context_tokens"])
    total_len = len(train[0]["context_tokens"]) + len(train[0]["target_tokens"])
    assert ids.shape[1] == total_len
    assert bool(valid[0, :total_len].all())
    assert int(target_mask[0].sum()) == len(train[0]["target_tokens"])
    assert bool(target_mask[0, context_len - 1])
    assert bool(target_mask[0, total_len - 2])


def test_pg375_composed_decoder_has_all_slot_logits() -> None:
    train, _, _ = _fixture()
    vocabulary = build_train_vocabulary(train)
    from scripts.run_pg375_composed_rule_ir_candidate import _slot_values_from_rows, _pad_context

    classes = _slot_values_from_rows(train)
    model = ComposedRuleIRModel(
        vocab_size=len(vocabulary),
        config=CausalMoEConfig(d_model=16, n_heads=2, n_layers=1, experts=2, expert_hidden=32, max_length=128),
        slot_classes=classes,
        slot_decoder_layers=1,
        slot_decoder_heads=2,
    )
    context_ids, context_mask = _pad_context([train[0]], vocabulary, torch.device("cpu"))
    with torch.inference_mode():
        output = model(context_ids, context_mask)
    assert set(output["composition"]) == set(SLOTS)
    assert all(output["composition"][slot].shape[0] == 1 for slot in SLOTS)
    assert set(output["slot_aux"]) == set(SLOTS)
    assert output["ask"].shape == (1, 2)
    assert output["repair"].shape == (1, 4)
    assert output["negative"].shape == (1, 2)


def test_pg375_cpu_candidate_reports_composition_and_safety_metrics() -> None:
    train, holdout, dataset = _fixture()
    result = run_candidate(
        train_rows=train,
        holdout_rows=holdout,
        dataset=dataset,
        seeds=(37501,),
        device="cpu",
        config=CausalMoEConfig(d_model=16, n_heads=2, n_layers=1, experts=2, expert_hidden=32, max_length=128),
        slot_decoder_layers=1,
        slot_decoder_heads=2,
        pretrain_epochs=1,
        posttrain_epochs=1,
        microbatch=2,
    )
    assert result["status"] == "cpu_smoke_candidate_only"
    assert result["data_contract"]["status"] == "passed"
    assert "slot_composition_exact_min" in result["worst_seed"]
    assert "per_slot" in result["candidates"][0]["post"]
    assert result["scientific_gate"]["trained_baseline_entropy_comparison"] is True
    assert all(value is False for value in result["promotion"].values())


def test_pg375_report_does_not_emit_raw_material() -> None:
    train, holdout, dataset = _fixture()
    result = run_candidate(train_rows=train, holdout_rows=holdout, dataset=dataset, seeds=(37501,), device="cpu", config=CausalMoEConfig(d_model=16, n_heads=2, n_layers=1, experts=2, expert_hidden=32, max_length=128), slot_decoder_layers=1, slot_decoder_heads=2)
    encoded = json.dumps(result, ensure_ascii=False)
    assert "raw_payload" not in encoded
    assert "response_body" not in encoded
    assert "http://" not in encoded
    assert "https://" not in encoded


def test_pg375_cuda_requires_explicit_remote_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLACKBOX_REMOTE_A800_TRAIN", raising=False)
    with pytest.raises(RuntimeError, match="BLACKBOX_REMOTE_A800_TRAIN"):
        _device_gate("cuda:0")
