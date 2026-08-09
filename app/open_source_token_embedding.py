"""Open-source tokenizer/embedding bridge for canonical Rule IR tokens.

This module deliberately separates three things that are often conflated in
experiments:

* the *tokenizer* (the open-source Hugging Face ``tokenizers`` package),
* the *embedding matrix* (fresh and trainable by default, or an explicitly
  attested local checkpoint), and
* the numeric evidence/weight scalars used by the action policy.

Only the bounded Rule IR layer is accepted.  Unknown slot/value strings are
mapped to ``unknown`` before tokenization, so target names, source text,
request bodies, response bodies and oracle authority cannot become accidental
vocabulary entries.  No network access or code execution is performed here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:  # ``tokenizers`` is an optional research dependency.
    import tokenizers as _hf_tokenizers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
    _hf_tokenizers = None
    Tokenizer = None  # type: ignore[assignment]
    WordLevel = None  # type: ignore[assignment]
    WhitespaceSplit = None  # type: ignore[assignment]
    _TOKENIZERS_IMPORT_ERROR = exc
else:
    _TOKENIZERS_IMPORT_ERROR = None

from .pg131_layered_ir_policy import (
    IR_MODES,
    IR_SLOT_IDS,
    IR_VALUE_VOCAB,
    MAX_IR_TOKENS,
    MAX_STEPS,
    TOKENS_PER_STEP,
)


SCHEMA_VERSION = "pg132-open-source-token-embedding-v1"
TOKENIZER_BACKEND = "huggingface-tokenizers-wordlevel"
SPECIAL_PAD = "[PAD]"
SPECIAL_UNK = "[UNK]"
SCALAR_DIM = 4  # slot weight, normalized token weight, current flag, position
DEFAULT_EMBEDDING_DIM = 48
TOKEN_ABLATION_MODES = tuple(IR_MODES) + ("tokens_zeroed",)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _bounded_value(value: Any) -> str:
    """Return an allow-listed Rule IR value without retaining arbitrary text."""

    text = str(value)
    return text if text in IR_VALUE_VOCAB else "unknown"


def _bounded_slot(value: Any) -> str:
    text = str(value)
    return text if text in IR_SLOT_IDS else "unknown"


def canonical_ir_pair_token(slot_id: Any, value: Any) -> str:
    """Map one IR slot/value pair to a bounded vocabulary token."""

    return f"ir.{_bounded_slot(slot_id)}={_bounded_value(value)}"


def _pair_vocabulary() -> tuple[str, ...]:
    # Include every legal pair so WordLevel never falls through for a valid
    # Rule IR value.  The vocabulary is small (8 * 41 plus two specials) and
    # is deterministic across machines and data splits.
    return tuple(canonical_ir_pair_token(slot, value) for slot in IR_SLOT_IDS for value in IR_VALUE_VOCAB)


PAIR_VOCABULARY = _pair_vocabulary()
VOCABULARY = (SPECIAL_PAD, SPECIAL_UNK) + PAIR_VOCABULARY
PAD_ID = 0
UNK_ID = 1


def build_rule_ir_tokenizer() -> Any:
    """Build the local deterministic tokenizer from the fixed Rule IR vocab."""

    if _hf_tokenizers is None:  # pragma: no cover - minimal install path
        raise RuntimeError("the optional 'tokenizers' package is required for PG-132") from _TOKENIZERS_IMPORT_ERROR
    vocab = {token: index for index, token in enumerate(VOCABULARY)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=SPECIAL_UNK))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer


def canonical_ir_layer_tokens(layer: Mapping[str, Any]) -> list[str]:
    """Return one bounded pair token for each Rule IR slot in stable order."""

    by_slot = {
        str(token.get("slot_id")): token
        for token in list(layer.get("tokens") or [])
        if isinstance(token, Mapping)
    }
    return [canonical_ir_pair_token(slot, (by_slot.get(slot) or {}).get("value", "unknown")) for slot in IR_SLOT_IDS]


def canonical_ir_token_strings(prefix_layers: Sequence[Mapping[str, Any]]) -> list[str]:
    """Flatten a Rule IR prefix without exposing any unbounded source field."""

    if len(prefix_layers) > MAX_STEPS:
        raise ValueError("Rule IR prefix exceeds the bounded token window")
    result: list[str] = []
    for layer in prefix_layers:
        result.extend(canonical_ir_layer_tokens(layer))
    return result


@dataclass(frozen=True)
class EmbeddingProvenance:
    """Auditable description of tokenizer and embedding weights."""

    schema_version: str
    tokenizer_backend: str
    tokenizer_version: str
    tokenizer_vocab_size: int
    embedding_dim: int
    weights_source: str
    pretrained: bool
    source_id: str | None
    license: str | None
    weights_sha256: str
    initialization_seed: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tokenizer_backend": self.tokenizer_backend,
            "tokenizer_version": self.tokenizer_version,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "embedding_dim": self.embedding_dim,
            "weights_source": self.weights_source,
            "pretrained": self.pretrained,
            "source_id": self.source_id,
            "license": self.license,
            "weights_sha256": self.weights_sha256,
            "initialization_seed": self.initialization_seed,
        }


def _tensor_sha256(weight: torch.Tensor) -> str:
    contiguous = weight.detach().cpu().contiguous()
    return _sha256_bytes(contiguous.numpy().tobytes())


def _load_weight_tensor(path: Path) -> torch.Tensor:
    """Load a local matrix only; this function never downloads a model."""

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch releases do not expose weights_only
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        weight = payload
    elif isinstance(payload, Mapping):
        candidate = payload.get("weight")
        if candidate is None:
            candidate = payload.get("embedding.weight")
        if not isinstance(candidate, torch.Tensor):
            raise ValueError("local embedding checkpoint must contain a tensor under 'weight'")
        weight = candidate
    else:
        raise ValueError("local embedding checkpoint has an unsupported format")
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("local embedding checkpoint must be a finite rank-2 tensor")
    return weight.detach().cpu().float()


class RuleIRTokenEmbedding(nn.Module):
    """Token embedding over the bounded Rule IR pair vocabulary.

    By default this is a fresh, seeded matrix and is trainable.  A checkpoint
    becomes a *pretrained* source only when the caller supplies its expected
    SHA-256, source identifier and license.  This prevents an arbitrary local
    file from being mislabeled as an open-source pretrained model.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        seed: int = 13201,
        pretrained_weights_path: str | Path | None = None,
        expected_sha256: str | None = None,
        source_id: str | None = None,
        license: str | None = None,
        freeze_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.tokenizer = build_rule_ir_tokenizer()
        self.embedding = nn.Embedding(len(VOCABULARY), embedding_dim, padding_idx=PAD_ID)
        pretrained = pretrained_weights_path is not None
        initialization_seed: int | None = seed
        weights_source = "fresh_seeded_torch_embedding"
        provenance_source_id: str | None = None
        provenance_license: str | None = None
        if pretrained:
            if not expected_sha256 or not source_id or not license:
                raise ValueError("pretrained weights require expected_sha256, source_id and license")
            path = Path(pretrained_weights_path)
            raw_sha = _sha256_bytes(path.read_bytes())
            if raw_sha != expected_sha256:
                raise ValueError("pretrained embedding checkpoint SHA-256 mismatch")
            weight = _load_weight_tensor(path)
            if tuple(weight.shape) != (len(VOCABULARY), embedding_dim):
                raise ValueError("pretrained embedding shape does not match the Rule IR vocabulary")
            with torch.no_grad():
                self.embedding.weight.copy_(weight)
            weights_source = "attested_local_pretrained_checkpoint"
            initialization_seed = None
            provenance_source_id = str(source_id)
            provenance_license = str(license)
            if freeze_pretrained:
                self.embedding.weight.requires_grad_(False)
        else:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            with torch.no_grad():
                self.embedding.weight.normal_(mean=0.0, std=0.02, generator=generator)
                self.embedding.weight[PAD_ID].zero_()
        self._provenance = EmbeddingProvenance(
            schema_version=SCHEMA_VERSION,
            tokenizer_backend=TOKENIZER_BACKEND,
            tokenizer_version=str(getattr(_hf_tokenizers, "__version__", "unknown")),
            tokenizer_vocab_size=len(VOCABULARY),
            embedding_dim=embedding_dim,
            weights_source=weights_source,
            pretrained=pretrained,
            source_id=provenance_source_id,
            license=provenance_license,
            weights_sha256=_tensor_sha256(self.embedding.weight),
            initialization_seed=initialization_seed,
        )

    @property
    def provenance(self) -> EmbeddingProvenance:
        return self._provenance

    def token_ids(self, prefix_layers: Sequence[Mapping[str, Any]], *, max_tokens: int = MAX_IR_TOKENS) -> list[int]:
        token_strings = canonical_ir_token_strings(prefix_layers)
        if len(token_strings) > max_tokens:
            raise ValueError("Rule IR token sequence exceeds max_tokens")
        encoding = self.tokenizer.encode(token_strings, is_pretokenized=True)
        ids = [int(value) for value in encoding.ids]
        ids.extend([PAD_ID] * (max_tokens - len(ids)))
        if len(ids) != max_tokens:
            raise AssertionError("token id shape drift")
        return ids

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[-1] != MAX_IR_TOKENS:
            raise ValueError("Rule IR token ids must have shape [batch, max_tokens]")
        return self.embedding(token_ids)


def _slot_weights(prefix_layers: Sequence[Mapping[str, Any]], *, mode: str) -> list[list[float]]:
    if mode not in TOKEN_ABLATION_MODES:
        raise ValueError(f"unknown PG-132 IR mode: {mode}")
    if len(prefix_layers) > MAX_STEPS:
        raise ValueError("Rule IR prefix exceeds the bounded token window")
    scalars: list[list[float]] = []
    for step_index, layer in enumerate(prefix_layers):
        by_slot = {
            str(token.get("slot_id")): token
            for token in list(layer.get("tokens") or [])
            if isinstance(token, Mapping)
        }
        raw_weights: list[float] = []
        for slot in IR_SLOT_IDS:
            token = by_slot.get(slot) or {}
            weight = max(0.0, min(float(token.get("weight", 1.0)), 2.0))
            if mode == "uniform":
                weight = 1.0
            elif mode == "no_failure_slots" and slot.startswith("failure."):
                weight = 0.0
            elif mode == "zero":
                weight = 0.0
            raw_weights.append(weight)
        denominator = sum(raw_weights) or 1.0
        current = 1.0 if step_index == len(prefix_layers) - 1 else 0.0
        position = float(step_index) / float(max(MAX_STEPS - 1, 1))
        if mode == "zero":
            scalars.extend([[0.0] * SCALAR_DIM for _ in raw_weights])
        else:
            scalars.extend([[weight / 2.0, weight / denominator, current, position] for weight in raw_weights])
    scalars.extend([[0.0] * SCALAR_DIM for _ in range(MAX_IR_TOKENS - len(scalars))])
    if len(scalars) != MAX_IR_TOKENS:
        raise AssertionError("Rule IR scalar shape drift")
    return scalars


def open_source_ir_token_inputs(
    embedding: RuleIRTokenEmbedding,
    prefix_layers: Sequence[Mapping[str, Any]],
    *,
    mode: str = "weighted",
) -> tuple[list[int], list[list[float]]]:
    """Return token IDs and auditable scalar side channels for a Rule IR prefix."""

    scalars = _slot_weights(prefix_layers, mode=mode)
    if mode in {"zero", "tokens_zeroed"}:
        # A true information ablation must remove the categorical pair IDs as
        # well as their weight side channel for ``zero``.  ``tokens_zeroed``
        # intentionally retains the scalar side channel so embedding reliance
        # can be measured independently.
        return [PAD_ID] * MAX_IR_TOKENS, scalars
    return embedding.token_ids(prefix_layers), scalars


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "EmbeddingProvenance",
    "IR_SLOT_IDS",
    "IR_VALUE_VOCAB",
    "MAX_IR_TOKENS",
    "PAIR_VOCABULARY",
    "PAD_ID",
    "RuleIRTokenEmbedding",
    "SCALAR_DIM",
    "SCHEMA_VERSION",
    "TOKEN_ABLATION_MODES",
    "TOKENIZER_BACKEND",
    "VOCABULARY",
    "build_rule_ir_tokenizer",
    "canonical_ir_layer_tokens",
    "canonical_ir_pair_token",
    "canonical_ir_token_strings",
    "open_source_ir_token_inputs",
]
