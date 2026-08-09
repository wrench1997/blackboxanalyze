from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pg295_causal_moe import CausalMoEConfig
from scripts.run_pg377_entropy_preserved_context_candidate import (
    _build_vocabulary,
    _load_teacher,
    _safe_context_rows,
    run_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg375_strict_filtered_rule_ir_dataset_v1.json"
AUDIT = ROOT / "research" / "pg375_strict_filtered_rule_ir_audit_v1.json"
RULES = ROOT / "research" / "improvement_rules.json"
TEACHER = ROOT / "artifacts" / "pg375-context-representation-a800" / "pg375_context_seed_37521.pt"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_pg377_context_loader_never_reads_target_or_raw() -> None:
    dataset = _load(DATASET)
    rows, failures, count = _safe_context_rows(dataset, split="train")
    assert failures == []
    assert count == 1208
    assert rows and set(rows[0]) == {"context_tokens"}
    encoded = json.dumps(rows[:2], ensure_ascii=False)
    for marker in ("http://", "https://", "payload=", "response_body=", "wire="):
        assert marker not in encoded


def test_pg377_teacher_is_trained_context_only_and_vocab_locked() -> None:
    dataset = _load(DATASET)
    train, _, _ = _safe_context_rows(dataset, split="train")
    vocabulary = _build_vocabulary(dataset, train)
    teacher, metadata = _load_teacher(TEACHER, vocabulary=vocabulary, device=__import__("torch").device("cpu"))
    assert teacher.config.d_model == 384
    assert metadata["seed"] == 37521
    assert metadata["checkpoint_sha256"]


def test_pg377_cpu_candidate_uses_trained_teacher_and_keeps_promotion_closed() -> None:
    result = run_candidate(
        dataset=_load(DATASET),
        audit=_load(AUDIT),
        dataset_path=DATASET,
        audit_path=AUDIT,
        rules_path=RULES,
        teacher_checkpoint=TEACHER,
        device="cpu",
        seeds=(37701,),
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        temperature=2.0,
        kl_weight=1.0,
        entropy_weight=0.25,
        config=CausalMoEConfig(d_model=32, n_heads=4, n_layers=1, experts=2, expert_hidden=64, max_length=768),
        train_limit=2,
        holdout_limit=2,
    )
    assert result["training"]["target_tokens_read"] is False
    assert result["training"]["holdout_used_for_optimization"] is False
    assert result["training"]["teacher"]["seed"] == 37521
    assert result["candidates"][0]["teacher_holdout"]["mean_predictive_entropy_nats"] is not None
    assert result["promotion"] == {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    assert result["status"] in {"entropy_preserved_candidate_only", "blocked_entropy_preservation"}


def test_pg377_wrong_teacher_contract_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not-a-checkpoint")
    dataset = _load(DATASET)
    train, _, _ = _safe_context_rows(dataset, split="train")
    vocabulary = _build_vocabulary(dataset, train)
    with pytest.raises(Exception):
        _load_teacher(bad, vocabulary=vocabulary, device=__import__("torch").device("cpu"))


def test_pg377_report_contract_is_not_capability_or_payload() -> None:
    source = (ROOT / "scripts" / "run_pg377_entropy_preserved_context_candidate.py").read_text(encoding="utf-8")
    assert "target_tokens_read" in source
    assert "capability_training" in source
    assert "payload_catalog_promotion_allowed" in source
    assert "torch.cuda.device_count() != 1" in source
