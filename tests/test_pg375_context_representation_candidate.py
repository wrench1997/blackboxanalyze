from __future__ import annotations

import json
from pathlib import Path

from app.pg293_failure_next_action import PAD, UNK
from app.pg295_causal_moe import CausalMoEConfig
from scripts.run_pg375_context_representation_candidate import _safe_context_rows, run_candidate


def _dataset() -> dict[str, object]:
    return {
        "status": "candidate_only",
        "representation_pretrain_candidate_allowed": True,
        "capability_training_allowed": False,
        "source_contract": {"operator_reviewed": False, "typed_evaluator_complete": False, "fresh_reset_role_attested": False, "capability_training_eligible": False},
        "vocabulary": {"context_tokens": ["document_presence=observed", "request_method=get", "request_method=post", "surface=alpha", "surface=beta"]},
        "records": [
            {"split": "train", "context_tokens": ["document_presence=observed", "request_method=get", "surface=alpha"], "target_tokens": ["forbidden_target_must_not_be_read"], "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True}, "raw_payload_stored": False, "raw_response_body_stored": False, "oracle_answer_in_context": False},
            {"split": "train", "context_tokens": ["document_presence=observed", "request_method=post", "surface=beta"], "target_tokens": ["forbidden_second_target"], "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True}, "raw_payload_stored": False, "raw_response_body_stored": False, "oracle_answer_in_context": False},
            {"split": "implementation_holdout", "context_tokens": ["document_presence=observed", "request_method=get", "surface=beta"], "target_tokens": ["forbidden_holdout_target"], "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True}, "raw_payload_stored": False, "raw_response_body_stored": False, "oracle_answer_in_context": False},
        ],
    }


def test_context_loader_does_not_include_target_tokens() -> None:
    rows, failures, count = _safe_context_rows(_dataset(), split="train")
    assert not failures
    assert count == 2
    assert rows[0] == {"context_tokens": ["document_presence=observed", "request_method=get", "surface=alpha"]}
    assert rows[1]["context_tokens"] == ["document_presence=observed", "request_method=post", "surface=beta"]
    assert "forbidden_target_must_not_be_read" not in json.dumps(rows)


def test_context_representation_cpu_smoke_keeps_capability_closed(tmp_path: Path) -> None:
    root = tmp_path
    dataset_path = root / "dataset.json"; audit_path = root / "audit.json"; rules_path = root / "rules.json"
    dataset_path.write_text(json.dumps(_dataset()), encoding="utf-8")
    audit_path.write_text(json.dumps({"status": "passed_candidate_audit", "counts": {"active_cross_split_context_overlap": 0, "active_cross_split_exact_overlap": 0}}), encoding="utf-8")
    rules_path.write_text("{}", encoding="utf-8")
    result = run_candidate(dataset=_dataset(), audit=json.loads(audit_path.read_text()), dataset_path=dataset_path, audit_path=audit_path, rules_path=rules_path, device="cpu", seeds=(37521,), epochs=1, batch_size=1, config=CausalMoEConfig(d_model=8, n_heads=2, n_layers=1, experts=2, expert_hidden=16, max_length=16))
    assert result["status"] == "representation_pretrain_candidate_only"
    assert result["execution"]["gpu_touched"] is False
    assert result["training"]["target_tokens_read"] is False
    assert all(value is False for value in result["promotion"].values())


def test_context_candidate_rejects_unknown_holdout_token(tmp_path: Path) -> None:
    dataset = _dataset(); dataset["records"] = list(dataset["records"])  # type: ignore[index]
    dataset["records"][2]["context_tokens"] = ["unknown=holdout"]  # type: ignore[index]
    dataset_path = tmp_path / "dataset.json"; audit_path = tmp_path / "audit.json"; rules_path = tmp_path / "rules.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8"); audit_path.write_text(json.dumps({"status": "passed_candidate_audit", "counts": {"active_cross_split_context_overlap": 0, "active_cross_split_exact_overlap": 0}}), encoding="utf-8"); rules_path.write_text("{}", encoding="utf-8")
    result = run_candidate(dataset=dataset, audit=json.loads(audit_path.read_text()), dataset_path=dataset_path, audit_path=audit_path, rules_path=rules_path, device="cpu", seeds=(37521,), config=CausalMoEConfig(d_model=8, n_heads=2, n_layers=1, experts=2, expert_hidden=16, max_length=16))
    assert result["status"] == "blocked_representation_contract"
    assert result["gate"]["checks"]["holdout_vocabulary_closed"] is False
    assert result["execution"]["optimizer_started"] is False
