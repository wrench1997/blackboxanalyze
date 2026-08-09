"""Neural, oracle-blind goal/label proposal decoder for PG-97.

The decoder learns an unsupervised representation of bounded observation
difference tokens.  It reconstructs token presence, clusters the resulting
latent vectors, and names the two clusters by a generic visible property
(effect-change density).  It never receives a vulnerability family, typed
oracle result, target id, or raw request/response.  The resulting proposal is
still only a hypothesis: the caller must run an independent typed oracle and
the normal promotion gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import torch
from torch import nn


SCHEMA_VERSION = "neural-auto-goal-label-decoder-v1"
_FORBIDDEN_FIELDS = {
    "family",
    "hypothesis",
    "oracle",
    "oracle_projection",
    "decision",
    "belief_before",
    "belief_after",
    "next_action",
    "target_instance_id",
    "route_template_id",
    "probe_ref",
    "probe_sha256",
    "body_sha256",
    "semantic_body_sha256",
    "projection_sha256",
}


class _TokenPresenceAutoencoder(nn.Module):
    def __init__(self, vocabulary_size: int, *, hidden_dim: int = 48, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocabulary_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )
        self.decoder = nn.Linear(latent_dim, vocabulary_size)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(values))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _bounded_token(value: Any) -> str:
    text = str(value)
    if not re.fullmatch(r"[A-Z0-9_:-]{1,160}", text):
        raise ValueError("proposal decoder received an unbounded token")
    return text


def _validate_visible_row(row: Mapping[str, Any]) -> None:
    leaked = _FORBIDDEN_FIELDS.intersection(row)
    if leaked:
        raise ValueError(f"oracle or target fields leaked into neural proposal: {sorted(leaked)}")
    for token in row.get("delta_tokens", []):
        _bounded_token(token)


def _kmeans(values: torch.Tensor, *, seed: int, iterations: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[0] < 2:
        raise ValueError("at least two visible rows are required for clustering")
    # Farthest-point initialization is deterministic and does not use labels.
    first = int(seed % values.shape[0])
    distance = torch.sum((values - values[first]) ** 2, dim=1)
    second = int(torch.argmax(distance).item())
    centroids = values[[first, second]].clone()
    assignments = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    for _ in range(iterations):
        distances = torch.cdist(values, centroids)
        next_assignments = torch.argmin(distances, dim=1)
        next_centroids = []
        for cluster in range(2):
            members = values[next_assignments == cluster]
            next_centroids.append(members.mean(dim=0) if len(members) else centroids[cluster])
        next_centroids_tensor = torch.stack(next_centroids)
        if torch.equal(next_assignments, assignments):
            centroids = next_centroids_tensor
            assignments = next_assignments
            break
        centroids = next_centroids_tensor
        assignments = next_assignments
    return assignments, centroids


class NeuralGoalLabelDecoder:
    """Fit an unlabeled representation and emit a bounded proposal."""

    def __init__(self, *, seed: int = 20260803, epochs: int = 80, device: str | None = None) -> None:
        self.seed = int(seed)
        self.epochs = int(epochs)
        if self.epochs <= 0 or self.epochs > 400:
            raise ValueError("epochs must be in [1, 400]")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.vocabulary: dict[str, int] = {}
        self.model: _TokenPresenceAutoencoder | None = None
        self.centroids: torch.Tensor | None = None
        self.high_effect_cluster: int | None = None
        self.degenerate: bool = False
        self.train_loss: float | None = None
        self.cluster_stats: dict[str, Any] = {}

    def _matrix(self, rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        values = torch.zeros((len(rows), len(self.vocabulary)), dtype=torch.float32, device=self.device)
        for index, row in enumerate(rows):
            tokens = set(_bounded_token(token) for token in row.get("delta_tokens", []))
            if not tokens:
                tokens = {"NO_DELTA"}
            for token in tokens:
                vocabulary_index = self.vocabulary.get(token)
                if vocabulary_index is not None:
                    values[index, vocabulary_index] = 1.0
        return values

    def fit(self, rows: Sequence[Mapping[str, Any]], *, extra_tokens: Sequence[str] = ()) -> "NeuralGoalLabelDecoder":
        if not rows:
            raise ValueError("cannot fit a goal/label decoder on an empty set")
        for row in rows:
            _validate_visible_row(row)
        tokens = {"NO_DELTA"}
        for token in extra_tokens:
            tokens.add(_bounded_token(token))
        for row in rows:
            tokens.update(_bounded_token(token) for token in row.get("delta_tokens", []))
        self.vocabulary = {token: index for index, token in enumerate(sorted(tokens))}
        matrix = self._matrix(rows)
        self.model = _TokenPresenceAutoencoder(len(self.vocabulary)).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            # A small masked reconstruction objective supplies a learning signal
            # without any typed label.  One input feature is hidden each epoch
            # when possible, while the target remains the complete visible set.
            input_matrix = matrix.clone()
            if input_matrix.shape[1] > 2:
                mask_index = int(torch.randint(input_matrix.shape[1], (1,), generator=generator).item())
                input_matrix[:, mask_index] = 0.0
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(self.model(input_matrix), matrix)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        self.model.eval()
        with torch.inference_mode():
            latent = self.model.encode(matrix)
            _, self.centroids = _kmeans(latent, seed=self.seed)
        # Cluster names are assigned from a visible statistic only.  This is
        # not an oracle label: a dense change cluster is merely a candidate
        # "effect present" state and must still be typed-oracle validated.
        assignments = self._cluster_latent(latent)
        delta_counts = torch.tensor([len(set(row.get("delta_tokens", []))) for row in rows], device=self.device, dtype=torch.float32)
        means = [float(delta_counts[assignments == cluster].mean().cpu()) if bool(torch.any(assignments == cluster)) else -math.inf for cluster in range(2)]
        self.degenerate = bool(float(delta_counts.max().cpu()) == float(delta_counts.min().cpu()))
        self.high_effect_cluster = int(max(range(2), key=lambda cluster: means[cluster])) if not self.degenerate else -1
        counts = [int(torch.sum(assignments == cluster).cpu()) for cluster in range(2)]
        self.cluster_stats = {
            "cluster_count": 2,
            "cluster_row_counts": counts,
            "cluster_mean_delta_count": [round(value, 6) if math.isfinite(value) else None for value in means],
            "high_effect_cluster": self.high_effect_cluster,
            "degenerate_visible_signal": self.degenerate,
        }
        with torch.inference_mode():
            self.train_loss = float(criterion(self.model(matrix), matrix).cpu())
        return self

    def _require_fitted(self) -> tuple[_TokenPresenceAutoencoder, torch.Tensor, int]:
        if self.model is None or self.centroids is None or self.high_effect_cluster is None:
            raise RuntimeError("goal/label decoder is not fitted")
        return self.model, self.centroids, int(self.high_effect_cluster)

    def _cluster_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise RuntimeError("goal/label decoder centroids are unavailable")
        return torch.argmin(torch.cdist(latent, self.centroids), dim=1)

    def predict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        model, centroids, high_effect_cluster = self._require_fitted()
        _validate_visible_row(row)
        if self.degenerate:
            return {"label_id": "AUTO_MODEL_UNSUPPORTED_OR_AMBIGUOUS", "decision": "abstain", "unknown_tokens": [], "reason": "degenerate_visible_signal"}
        tokens = set(_bounded_token(token) for token in row.get("delta_tokens", []))
        if not tokens:
            tokens = {"NO_DELTA"}
        unknown = sorted(token for token in tokens if token not in self.vocabulary)
        if unknown:
            return {"label_id": "AUTO_MODEL_UNSUPPORTED_OR_AMBIGUOUS", "decision": "abstain", "unknown_tokens": unknown}
        matrix = self._matrix([row])
        with torch.inference_mode():
            latent = model.encode(matrix)
            distances = torch.cdist(latent, centroids)[0]
        cluster = int(torch.argmin(distances).cpu())
        if cluster == high_effect_cluster:
            return {
                "label_id": "AUTO_MODEL_EFFECT_CHANGE",
                "decision": "confirm_candidate",
                "cluster": cluster,
                "cluster_distance": round(float(distances[cluster].cpu()), 6),
                "unknown_tokens": [],
            }
        return {
            "label_id": "AUTO_MODEL_NO_OBSERVED_CHANGE",
            "decision": "reject",
            "cluster": cluster,
            "cluster_distance": round(float(distances[cluster].cpu()), 6),
            "unknown_tokens": [],
        }

    def proposal(self, *, design_row_count: int) -> dict[str, Any]:
        _, centroids, high_effect_cluster = self._require_fitted()
        if self.train_loss is None:
            raise RuntimeError("goal/label decoder has no training diagnostics")
        centroid_summary = [[round(float(value), 6) for value in row[:8].detach().cpu()] for row in centroids]
        proposal = {
            "schema_version": SCHEMA_VERSION,
            "proposal_id": "pg97-neural-auto-goal-label-v1",
            "proposal_inputs": {
                "design_row_count": int(design_row_count),
                "oracle_visible": False,
                "family_visible": False,
                "raw_probe_visible": False,
                "raw_response_visible": False,
                "training_signal": "masked_bounded_delta_token_presence_reconstruction",
            },
            "model": {
                "architecture": "token_presence_autoencoder_plus_two_means_kmeans",
                "device": str(self.device),
                "epochs": self.epochs,
                "vocabulary_size": len(self.vocabulary),
                "train_reconstruction_loss": round(float(self.train_loss), 6),
                "cluster_stats": self.cluster_stats,
                "centroid_prefix_preview": centroid_summary,
            },
            "goal": {
                "goal_id": "neural_auto_goal_stable_effect_discovery_v1",
                "success_condition": [
                    "model assigns a supported observation pair to the high-change cluster",
                    "the high-change state repeats on a compatible second probe",
                    "a matched negative control does not receive the success decision",
                    "fresh reset, loopback, and evidence hash checks remain valid",
                ],
                "failure_condition": ["low-change cluster", "unseen token or conflicting state", "negative-control confirmation"],
                "abstain_condition": ["unseen token vocabulary", "ambiguous cluster distance", "unknown-family review gate"],
                "budget": {"max_steps": 2, "requires_fresh_reset": True, "requires_get_post_pair": True},
            },
            "labels": [
                {"label_id": "AUTO_MODEL_NO_OBSERVED_CHANGE", "decision": "reject"},
                {"label_id": "AUTO_MODEL_EFFECT_CHANGE", "decision": "confirm_candidate"},
                {"label_id": "AUTO_MODEL_UNSUPPORTED_OR_AMBIGUOUS", "decision": "abstain"},
            ],
            "selected_cluster": {"high_effect_cluster": high_effect_cluster},
            "degenerate_visible_signal": self.degenerate,
            "audit": {
                "oracle_is_evaluator_only": True,
                "family_labels_are_not_model_features": True,
                "training_promotion_allowed": False,
                "memory_promotion_allowed": False,
            },
        }
        proposal["proposal_sha256"] = _digest(proposal)
        return proposal

    def checkpoint(self) -> dict[str, Any]:
        model, centroids, high_effect_cluster = self._require_fitted()
        return {
            "schema_version": "pg97-neural-auto-goal-label-checkpoint-v1",
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "vocabulary": self.vocabulary,
            "centroids": centroids.detach().cpu(),
            "high_effect_cluster": high_effect_cluster,
            "degenerate_visible_signal": self.degenerate,
            "seed": self.seed,
            "epochs": self.epochs,
            "oracle_visible": False,
            "family_visible": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "training_promotion_allowed": False,
            "memory_promotion_allowed": False,
        }


__all__ = ["NeuralGoalLabelDecoder", "SCHEMA_VERSION"]
