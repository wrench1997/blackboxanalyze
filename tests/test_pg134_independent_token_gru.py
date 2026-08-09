from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from app.pg134_independent_policy import (
    HASH_BUCKETS,
    MAX_STEPS,
    MAX_TOKENS_PER_STEP,
    SCALAR_DIM,
    TOKEN_MODES,
    IndependentTokenHashGRUPolicy,
    encode_prefix,
)


def _step() -> dict[str, object]:
    return {
        "source_token_layers": [
            {"modality": "html", "tokens": [{"kind": "tag", "value": "form"}, {"kind": "attribute", "value": "name", "value_hash": "a" * 64}]},
            {"modality": "javascript", "tokens": [{"kind": "api", "value": "fetch", "count_bucket": "1-4"}]},
            {"modality": "transport", "tokens": [{"kind": "method", "value": "POST"}, {"kind": "placement", "value": "json"}]},
        ],
        "ir_layer": {"tokens": [{"slot_id": "failure.kind", "value": "candidate_without_typed_effect", "weight": 2.0}, {"slot_id": "oracle.availability", "value": "typed", "weight": 1.25}]},
    }


def test_independent_encoder_has_bounded_modes_and_weight_only_channel() -> None:
    step = _step()
    for mode in TOKEN_MODES:
        ids, scalars = encode_prefix([step, step], mode=mode)
        assert len(ids) == len(scalars) == 2
        assert len(ids[0]) == MAX_TOKENS_PER_STEP
        assert len(scalars[0]) == MAX_TOKENS_PER_STEP
        assert len(scalars[0][0]) == SCALAR_DIM
    ids, scalars = encode_prefix([step], mode="weight_only")
    assert len({value for value in ids[0] if value}) == 1
    assert any(value != 0.0 for row in scalars[0] for value in row)


def test_independent_gru_forward_shape_and_provenance() -> None:
    model = IndependentTokenHashGRUPolicy(seed=13402)
    ids, scalars = encode_prefix([_step(), _step()])
    ids = ids + [[0] * MAX_TOKENS_PER_STEP for _ in range(MAX_STEPS - len(ids))]
    scalars = scalars + [[[0.0] * SCALAR_DIM for _ in range(MAX_TOKENS_PER_STEP)] for _ in range(MAX_STEPS - len(scalars))]
    logits = model(torch.tensor([ids]), torch.tensor([scalars]), torch.tensor([[True, True] + [False] * (MAX_STEPS - 2)]))
    assert logits.shape == (1, 7)
    assert model.embedding_provenance["representation"] == "blake2b_fixed_bucket"
    assert model.embedding_provenance["hash_buckets"] == HASH_BUCKETS
    assert model.embedding_provenance["pretrained"] is False


def test_pg134_report_passes_hard_gates_but_is_not_promoted() -> None:
    report = json.loads(Path("research/pg134_independent_token_gru_report_v1.json").read_text(encoding="utf-8"))
    assert report["hard_gates_passed"] is True
    assert report["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["holdout"]["pg133_history_holdout"]["history_pair"]["prediction_separation_rate"] == 1.0
    assert all(report["holdout"][name]["unknown_abstain_rate"] == 1.0 for name in ("pg133_history_holdout", "pg127_seed_holdout", "pg125_family_ood", "pg122_family_ood"))
    assert report["holdout"]["weight_sensitivity"]["mean_abs_logit_delta"] > 0.001
    assert report["training"]["safety_label_correction_count"] == 81
    assert report["diagnosis"]["safety_label_policy"] == "当 typed oracle 不可用时，模型标签强制为 abstain_unknown_oracle；原始 trace_next_action 与 label_correction 保留用于审计。"
    text = json.dumps(report, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "union select" not in text


def test_pg134_report_manifest_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg134_independent_token_gru_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
