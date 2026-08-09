"""Fail-closed bridge from the shared family router to active probing.

The bridge is diagnostic only.  It can provide a weak family prior to the
belief controller when the observation is in the shared head's training
feature neighbourhood; an OOD or low-confidence observation becomes a uniform
prior.  It never emits a positive Rule IR finding and never bypasses a
family-specific oracle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .belief_state import DECODER_FAMILIES
from .ood_gate import nearest_reference_distances, ood_flags
from .shared_family_representation import SHARED_FAMILY_CLASSES, SharedFamilyRouter, shared_model_input


SHARED_ROUTER_BRIDGE_SCHEMA = "sift-shared-router-active-bridge-v1"


def _uniform_prior() -> dict[str, float]:
    value = 1.0 / len(DECODER_FAMILIES)
    return {family: value for family in DECODER_FAMILIES}


class SharedRouterBridge:
    def __init__(self, checkpoint_path: Path, *, strict_ood: bool = True, abstain_threshold: float | None = None, margin_threshold: float | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model = SharedFamilyRouter().eval()
        self.model.load_state_dict(checkpoint["model_state"])
        self.mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
        self.std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
        reference = checkpoint.get("ood_reference_features")
        self.reference = torch.tensor(reference, dtype=torch.float32) if reference else None
        mask = checkpoint.get("ood_feature_mask")
        self.ood_feature_mask = torch.tensor(mask, dtype=torch.bool) if mask else None
        self.ood_fit = dict(checkpoint.get("ood_fit") or {})
        self.ood_clip = float(checkpoint.get("ood_clip", 3.0))
        self.strict_ood = bool(strict_ood)
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.abstain_threshold = float(checkpoint.get("abstain_threshold", 0.75) if abstain_threshold is None else abstain_threshold)
        self.margin_threshold = float(checkpoint.get("margin_threshold", 0.10) if margin_threshold is None else margin_threshold)

    def inspect(self, record: dict[str, Any]) -> dict[str, Any]:
        raw = torch.tensor([shared_model_input(record)], dtype=torch.float32)
        features = (raw - self.mean) / self.std
        distance = None
        is_ood = False
        if self.reference is not None and self.ood_fit:
            ood_features = features[:, self.ood_feature_mask] if self.ood_feature_mask is not None else features
            ood_features = ood_features.clamp(-self.ood_clip, self.ood_clip)
            distance = nearest_reference_distances(ood_features, self.reference)[0]
            is_ood = ood_flags([distance], self.ood_fit)[0]
        if self.strict_ood and is_ood:
            return {
                "schema_version": SHARED_ROUTER_BRIDGE_SCHEMA,
                "candidate_family": None,
                "confidence": 0.0,
                "margin": 0.0,
                "abstained": True,
                "ood": True,
                "ood_distance": round(float(distance or 0.0), 6),
                "temperature": self.temperature,
                "reason": "strict_ood",
                "route_only": True,
                "positive_authority": False,
                "shared_probabilities": {name: 0.0 for name in SHARED_FAMILY_CLASSES},
                "belief_prior": _uniform_prior(),
            }
        decoded = self.model.decode(
            features,
            abstain_threshold=self.abstain_threshold,
            margin_threshold=self.margin_threshold,
            temperature=self.temperature,
        )[0]
        probabilities = decoded.get("probabilities") or {}
        prior = {family: float(probabilities.get(family, 0.0)) for family in DECODER_FAMILIES}
        # A shared head has no URL-redirect/authentication/observability logits;
        # keep those branches alive instead of assigning zero probability.
        for family in DECODER_FAMILIES:
            if prior[family] <= 0.0:
                prior[family] = 1.0
        total = sum(prior.values())
        prior = {family: value / total for family, value in prior.items()}
        if decoded.get("abstained"):
            prior = _uniform_prior()
        return {
            "schema_version": SHARED_ROUTER_BRIDGE_SCHEMA,
            "candidate_family": decoded.get("family"),
            "candidate_route": decoded.get("candidate_family"),
            "shared_probabilities": {name: float(probabilities.get(name, 0.0)) for name in SHARED_FAMILY_CLASSES},
            "confidence": float(decoded.get("confidence", 0.0)),
            "margin": float(decoded.get("margin", 0.0)),
            "abstained": bool(decoded.get("abstained", True)),
            "ood": bool(is_ood),
            "ood_distance": round(float(distance), 6) if distance is not None else None,
            "temperature": self.temperature,
            "reason": "model_route" if not decoded.get("abstained") else "router_abstain",
            "route_only": True,
            "positive_authority": False,
            "belief_prior": prior,
        }


__all__ = ["SHARED_ROUTER_BRIDGE_SCHEMA", "SharedRouterBridge"]
