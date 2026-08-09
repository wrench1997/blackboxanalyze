from __future__ import annotations

import copy
from pathlib import Path

import pytest

from app.pg295_causal_moe import CausalMoEConfig
from scripts.run_pg370_multitask_moe_candidate import (
    PROMOTION_KEYS,
    SLOTS,
    SharedCausalMoEMultiTask,
    _pad_context,
    _pad_lm,
    _safe_abstract_row,
    _slot_classes,
    _vocabulary_gaps,
    build_plan_report,
    build_vocabulary,
    load_locked_rows,
    run_candidate,
)


@pytest.fixture(scope="module")
def locked_rows():
    return load_locked_rows()


def test_pg370_locked_loader_preserves_full_abstract_context_only(locked_rows) -> None:
    train, holdout, locks = locked_rows
    assert len(train) == 1475
    assert len(holdout) == 1477
    assert set(locks["datasets"]) == {"pg362", "pg367"}
    row = train[0]
    assert set(row) == {"context_tokens", "target_tokens", "split", "_source", "_target_values"}
    assert row["_source"] in {"pg362", "pg367"}
    assert len(row["context_tokens"]) > 0
    assert len(row["target_tokens"]) == len(SLOTS) + 2
    assert "evaluator_projection" not in row and "source_meta" not in row
    assert all(value is not True for value in locks["row_audits"]["pg362"].get("failures", []))


def test_pg370_shared_backbone_has_independent_slot_and_auxiliary_heads(locked_rows) -> None:
    train, holdout, _ = locked_rows
    rows = [train[0], holdout[0]]
    vocabulary = build_vocabulary(rows)
    classes = _slot_classes(rows)
    config = CausalMoEConfig(d_model=16, n_heads=4, n_layers=1, experts=2, expert_hidden=32, max_length=768)
    model = SharedCausalMoEMultiTask(vocab_size=len(vocabulary), config=config, slot_classes=classes)
    assert set(model.slot_heads) == set(SLOTS)
    assert model.ask_head.out_features == 2
    assert model.repair_head.out_features == 4
    assert model.negative_head.out_features == 2
    context_ids, context_mask, _ = _pad_context(rows, vocabulary, __import__("torch").device("cpu"))
    lm_ids, lm_mask, target_mask = _pad_lm(rows, vocabulary, __import__("torch").device("cpu"))
    outputs = model(context_ids, context_mask, lm_ids=lm_ids[:, :-1], lm_mask=lm_mask[:, :-1])
    assert set(outputs["slot"]) == set(SLOTS)
    assert outputs["ask"].shape[0] == 2
    assert outputs["repair"].shape[0] == 2
    assert outputs["negative"].shape[0] == 2
    assert bool(target_mask.any())


def test_pg370_target_mask_covers_first_and_last_target_labels(locked_rows) -> None:
    train, _, _ = locked_rows
    row = train[0]
    vocabulary = build_vocabulary([row])
    _, _, target_mask = _pad_lm([row], vocabulary, __import__("torch").device("cpu"))
    positions = __import__("torch").nonzero(target_mask[0], as_tuple=False).flatten().tolist()
    # The first target token is predicted from the last context token, so its
    # logit index is context_len - 1.  The target EOS is included as well.
    assert positions[0] == len(row["context_tokens"]) - 1
    assert len(positions) == len(row["target_tokens"])


def test_pg370_train_only_vocab_exposes_holdout_unknowns_instead_of_leaking_them(locked_rows) -> None:
    train, holdout, _ = locked_rows
    train_vocab = build_vocabulary(train)
    unknown_tokens = {
        token
        for row in holdout
        for token in [*row["context_tokens"], *row["target_tokens"]]
        if token not in train_vocab
    }
    train_classes = _slot_classes(train)
    unknown_slot_values = {
        key: {
            (row.get("_target_values") or {})[key]
            for row in holdout
            if (row.get("_target_values") or {})[key] not in train_classes[key]
        }
        for key in SLOTS
    }
    assert len(unknown_tokens) == 23
    assert len(unknown_slot_values["encoding_ref"]) == 1
    assert len(unknown_slot_values["syntax_category_ref"]) == 1
    gaps = _vocabulary_gaps(train, holdout, train_vocab, train_classes)
    assert gaps["blocked"] is True
    assert gaps["unknown_token_count"] == 23
    assert gaps["unknown_slot_value_count"] == 2


def test_pg370_cpu_smoke_reports_all_required_metrics_without_action_replication(locked_rows) -> None:
    train, holdout, _ = locked_rows
    config = CausalMoEConfig(d_model=16, n_heads=4, n_layers=1, experts=2, expert_hidden=32, max_length=768)
    result = run_candidate(train_rows=train[:2], holdout_rows=holdout[:2], seeds=(37001,), device="cpu", epochs=1, microbatch=1, config=config)
    assert result["status"] == "cpu_smoke_candidate_only"
    assert result["training"]["action_balance_replication"] is False
    metrics = result["candidates"][0]["post"]
    assert {"predictive_entropy", "sequence_exact", "slot_accuracy", "ask_recall", "repair_recall", "positive_recall", "negative_false_allow"}.issubset(metrics)
    assert set(result["promotion"]) == set(PROMOTION_KEYS)
    assert all(value is False for value in result["promotion"].values())
    assert result["scientific_gate"]["claim_allowed"] is False


def test_pg370_plan_report_is_read_only_and_closed(locked_rows) -> None:
    train, holdout, locks = locked_rows
    result = build_plan_report(train_rows=train, holdout_rows=holdout, locks=locks, runner_path=Path("scripts/run_pg370_multitask_moe_candidate.py"))
    assert result["status"] == "plan_only"
    assert result["execution"] == {
        "trainer_invoked": False,
        "gpu_touched": False,
        "docker_started": False,
        "network_used": False,
        "checkpoint_written": False,
    }
    assert all(value is False for value in result["promotion"].values())


def test_pg370_raw_sidecar_is_rejected_before_batch_construction(locked_rows) -> None:
    train, _, _ = locked_rows
    raw = copy.deepcopy(train[0])
    raw["context_tokens"] = [*raw["context_tokens"], "payload=literal"]
    with pytest.raises(ValueError, match="raw/evaluator"):
        _safe_abstract_row(raw, source="synthetic")


def test_pg370_remote_lane_is_explicitly_gated_and_has_checkpoint_option() -> None:
    source = Path("scripts/run_pg370_multitask_moe_candidate.py").read_text(encoding="utf-8")
    assert "--remote-candidate" in source
    assert "--checkpoint-dir" in source
    assert "BLACKBOX_REMOTE_A800_TRAIN" in source
    assert "CUDA_VISIBLE_DEVICES" in source


def test_pg370_non_cpu_execution_requires_explicit_remote_flag(locked_rows, monkeypatch) -> None:
    train, holdout, _ = locked_rows
    monkeypatch.delenv("BLACKBOX_REMOTE_A800_TRAIN", raising=False)
    with pytest.raises(RuntimeError, match="explicit remote training flag"):
        run_candidate(train_rows=train[:1], holdout_rows=holdout[:1], seeds=(37001,), device="cuda:0", epochs=1, microbatch=1)
