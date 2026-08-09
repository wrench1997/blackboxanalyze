"""Permutation-aware neural decoder for PG-102 active probe signatures.

The model consumes a set of nine canonical probe observations.  Mean/max
pooling makes the forward pass invariant to the order in which observations
arrive; canonical probe IDs remain explicit action tokens.  Family labels are
targets only.  A separate, calibrated fail-closed guard handles unseen probe
slots and no-effect signatures, so the raw neural proposal and the guarded
decision can be scored separately.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .active_probe_signature import PROBE_IDS, model_input_has_forbidden_field


SCHEMA_VERSION = "neural-active-probe-set-decoder-v1"
_SIGN_TYPES = 11
_TOKEN_DIM = len(PROBE_IDS) + 1 + _SIGN_TYPES + 2 + 4  # probe id, delta bit, geometry signs, method, phase


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def signature_to_tokens(signature: Mapping[str, Any]) -> torch.Tensor:
    """Encode a label-free signature into nine permutationable token rows."""

    if model_input_has_forbidden_field(signature):
        raise ValueError("neural active decoder received evaluator/raw fields")
    order = list(signature.get("probe_order") or [])
    if order != list(PROBE_IDS):
        raise ValueError("neural active decoder requires the fixed canonical probe bank")
    pattern = list(signature.get("delta_pattern") or [])
    signs = list(signature.get("geometry_sign_pattern") or [])
    if len(pattern) != len(PROBE_IDS) or len(signs) != len(PROBE_IDS):
        raise ValueError("neural active decoder signature length is invalid")
    method = str(signature.get("method", ""))
    phase = str(signature.get("phase", ""))
    method_index = {"GET": 0, "POST": 1}.get(method, -1)
    phase_index = {"screen": 0, "confirm": 1, "error": 2, "timeout": 3}.get(phase, -1)
    rows: list[list[float]] = []
    for index, probe_id in enumerate(PROBE_IDS):
        probe = _one_hot(index, len(PROBE_IDS))
        raw_signs = list(signs[index]) if isinstance(signs[index], list) else []
        if len(raw_signs) != _SIGN_TYPES:
            raise ValueError("neural active decoder geometry sign width is invalid")
        sign_values = [1.0 if int(value) > 0 else -1.0 if int(value) < 0 else 0.0 for value in raw_signs]
        row = probe + [float(bool(pattern[index]))] + sign_values + _one_hot(method_index, 2) + _one_hot(phase_index, 4)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


class NeuralActiveProbeSetDecoder(nn.Module):
    """Small DeepSets-style classifier with explicit OOD safety calibration."""

    def __init__(self, class_names: Sequence[str], *, hidden_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        names = tuple(str(name) for name in class_names)
        if not names:
            raise ValueError("neural active decoder requires known class names")
        self.class_names = names
        self.token_encoder = nn.Sequential(
            nn.Linear(_TOKEN_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.projector = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.classifier = nn.Linear(hidden_dim, len(names))
        self.calibration: dict[str, Any] = {}

    @property
    def token_dim(self) -> int:
        return _TOKEN_DIM

    def forward_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != _TOKEN_DIM:
            raise ValueError("neural active decoder token tensor has an invalid shape")
        encoded = self.token_encoder(tokens)
        pooled = torch.cat((encoded.mean(dim=1), encoded.max(dim=1).values), dim=-1)
        embedding = self.projector(pooled)
        return self.classifier(embedding), embedding

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(tokens)[0]

    def fit(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        epochs: int = 240,
        learning_rate: float = 3e-3,
        seed: int = 102,
    ) -> "NeuralActiveProbeSetDecoder":
        if not rows:
            raise ValueError("neural active decoder requires training rows")
        torch.manual_seed(int(seed))
        tokens = torch.stack([signature_to_tokens(row["model_input"] if "model_input" in row else row["signature"]) for row in rows])
        labels = torch.tensor([self.class_names.index(str(row["family"])) for row in rows], dtype=torch.long)
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(learning_rate), weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        self.train()
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            logits = self(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
        self.eval()
        with torch.inference_mode():
            _, embeddings = self.forward_tokens(tokens)
        self.calibration["train_embedding_support"] = embeddings.detach().cpu()
        self.calibration["known_probe_ids"] = sorted({
            PROBE_IDS[index]
            for row in rows
            for index, value in enumerate(list((row.get("model_input") or row.get("signature") or {}).get("delta_pattern") or []))
            if bool(value)
        })
        self.calibration["train_row_count"] = len(rows)
        return self

    def calibrate(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        margin_epsilon: float = 0.02,
        distance_epsilon: float = 0.05,
        confidence_scale: float = 0.5,
        margin_scale: float = 0.5,
        enforce_distance_guard: bool = False,
    ) -> dict[str, Any]:
        if not rows:
            raise ValueError("neural active decoder calibration requires development rows")
        self.eval()
        with torch.inference_mode():
            tokens = torch.stack([signature_to_tokens(row["model_input"] if "model_input" in row else row["signature"]) for row in rows])
            logits, embeddings = self.forward_tokens(tokens)
            probabilities = torch.softmax(logits, dim=-1)
            ordered = torch.sort(probabilities, dim=-1, descending=True).values
            confidence = ordered[:, 0]
            margin = ordered[:, 0] - ordered[:, 1] if ordered.shape[1] > 1 else ordered[:, 0]
            support = self.calibration.get("train_embedding_support")
            if not isinstance(support, torch.Tensor) or support.numel() == 0:
                raise ValueError("neural active decoder has no train embedding support")
            distances = torch.cdist(embeddings, support).min(dim=1).values
        # Cross-implementation geometry is deliberately allowed to move away
        # from the train embedding support.  Dev data calibrates a bounded
        # confidence/margin floor; raw embedding distance remains diagnostic
        # unless a future protocol explicitly enables that guard.
        self.calibration["confidence_floor"] = max(0.0, float(confidence.min().item()) * float(confidence_scale) - float(margin_epsilon))
        self.calibration["margin_floor"] = max(0.0, float(margin.min().item()) * float(margin_scale) - float(margin_epsilon))
        self.calibration["distance_ceiling"] = float(distances.max().item()) + float(distance_epsilon)
        self.calibration["distance_guard_enabled"] = bool(enforce_distance_guard)
        self.calibration["confidence_scale"] = float(confidence_scale)
        self.calibration["margin_scale"] = float(margin_scale)
        self.calibration["dev_confidence_min"] = float(confidence.min().item())
        self.calibration["dev_margin_min"] = float(margin.min().item())
        self.calibration["dev_distance_max"] = float(distances.max().item())
        return dict(self.calibration)

    def _raw_predict(self, signature: Mapping[str, Any]) -> dict[str, Any]:
        self.eval()
        with torch.inference_mode():
            tokens = signature_to_tokens(signature).unsqueeze(0)
            logits, embedding = self.forward_tokens(tokens)
            probabilities = torch.softmax(logits, dim=-1)[0]
            ordered = torch.sort(probabilities, descending=True).values
            order = torch.argsort(probabilities, descending=True)
            support = self.calibration.get("train_embedding_support")
            distance = float(torch.cdist(embedding, support).min().item()) if isinstance(support, torch.Tensor) and support.numel() else float("inf")
        confidence = float(ordered[0].item())
        margin = float((ordered[0] - ordered[1]).item()) if len(ordered) > 1 else confidence
        candidate = self.class_names[int(order[0].item())]
        return {
            "candidate_family": candidate,
            "confidence": round(confidence, 6),
            "margin": round(margin, 6),
            "embedding_distance": round(distance, 6),
            "raw_decision": "candidate",
            "raw_abstain": False,
        }

    def predict(self, signature: Mapping[str, Any], *, guarded: bool = True) -> dict[str, Any]:
        raw = self._raw_predict(signature)
        if not guarded:
            return {**raw, "decision": raw["raw_decision"], "abstain": False, "guarded": False}
        pattern = list(signature.get("delta_pattern") or [])
        positive_ids = {PROBE_IDS[index] for index, value in enumerate(pattern) if bool(value)}
        known_ids = set(self.calibration.get("known_probe_ids") or [])
        confidence_ok = float(raw["confidence"]) >= float(self.calibration.get("confidence_floor", 1.0))
        margin_ok = float(raw["margin"]) >= float(self.calibration.get("margin_floor", 1.0))
        distance_ok = (
            not bool(self.calibration.get("distance_guard_enabled", False))
            or float(raw["embedding_distance"]) <= float(self.calibration.get("distance_ceiling", -1.0))
        )
        if not positive_ids:
            return {**raw, "decision": "abstain", "abstain": True, "guarded": True, "reason": "no_observable_active_effect"}
        if len(positive_ids) != 1:
            return {**raw, "decision": "abstain", "abstain": True, "guarded": True, "reason": "ambiguous_active_effect"}
        if not positive_ids.issubset(known_ids):
            return {**raw, "decision": "abstain", "abstain": True, "guarded": True, "reason": "unseen_probe_slot"}
        if not (confidence_ok and margin_ok and distance_ok):
            return {**raw, "decision": "abstain", "abstain": True, "guarded": True, "reason": "calibrated_ood_or_low_confidence"}
        return {**raw, "decision": "candidate", "abstain": False, "guarded": True, "reason": "calibrated_known_signature"}


def model_input_vector(signature: Mapping[str, Any]) -> list[float]:
    return signature_to_tokens(signature).reshape(-1).tolist()


__all__ = ["NeuralActiveProbeSetDecoder", "SCHEMA_VERSION", "model_input_vector", "sha256_json", "signature_to_tokens"]
