from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pg378_residual", ROOT / "scripts" / "run_pg378_teacher_residual_context_candidate.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FixedModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def eval(self):
        return self

    def __call__(self, input_ids: torch.Tensor, *, valid_mask=None):
        batch, length = input_ids.shape
        return self.logits[:batch, :length].to(input_ids.device), torch.zeros((), device=input_ids.device)


def _rows_and_vocab():
    vocab = {MODULE.PAD: 0, MODULE.UNK: 1, "a": 2, "b": 3, "c": 4}
    rows = [{"context_tokens": ["a", "b", "c"]}]
    return rows, vocab


def test_residual_scale_zero_is_exact_teacher_distribution():
    rows, vocab = _rows_and_vocab()
    teacher_logits = torch.tensor([[[3.0, 0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0]]])
    student_logits = torch.tensor([[[0.0, 0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0, 0.0]]])
    metrics = MODULE._residual_metrics(_FixedModel(student_logits), _FixedModel(teacher_logits), rows, vocab, torch.device("cpu"), max_length=8, batch_size=1, residual_scale=0.0)
    teacher_entropy = float(MODULE._entropy(teacher_logits.reshape(-1, 5)).mean())
    assert abs(float(metrics["mean_predictive_entropy_nats"]) - teacher_entropy) < 1e-6
    assert float(metrics["entropy_relative_delta"]) == 0.0


def test_residual_scale_changes_only_bounded_mixed_logits():
    rows, vocab = _rows_and_vocab()
    teacher_logits = torch.zeros((1, 2, 5))
    student_logits = torch.ones((1, 2, 5))
    zero = MODULE._residual_metrics(_FixedModel(student_logits), _FixedModel(teacher_logits), rows, vocab, torch.device("cpu"), max_length=8, batch_size=1, residual_scale=0.0)
    bounded = MODULE._residual_metrics(_FixedModel(student_logits), _FixedModel(teacher_logits), rows, vocab, torch.device("cpu"), max_length=8, batch_size=1, residual_scale=0.1)
    assert zero["teacher_kl"] == 0.0
    assert bounded["teacher_kl"] == 0.0
    assert bounded["mean_predictive_entropy_nats"] == zero["mean_predictive_entropy_nats"]


def test_runner_source_has_context_only_and_no_raw_wire_literals():
    source = (ROOT / "scripts" / "run_pg378_teacher_residual_context_candidate.py").read_text(encoding="utf-8")
    assert "target_tokens" in source
    assert "target_tokens_read" in source
    assert '"docker_started": False' in source
    assert '"network_used": False' in source


def test_remote_lane_is_explicit_gpu0_weekend_gate():
    source = (ROOT / "scripts" / "run_pg378_teacher_residual_context_candidate.py").read_text(encoding="utf-8")
    assert "--remote-candidate" in source
    assert "cuda:0" in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "BLACKBOX_REMOTE_A800_TRAIN" in source
