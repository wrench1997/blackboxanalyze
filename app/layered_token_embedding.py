"""Dual source-token and Rule-IR token embedding for PG-133.

The page is never represented as raw text here.  ``layered_ir_tokenizer``
already reduces HTML, JavaScript and GET/POST manifests to bounded categorical
source tokens.  This module turns those atoms and the resulting Rule-IR slot
tokens into one auditable vocabulary, adds explicit layer boundary tokens, and
keeps a separate weight/scalar channel for every atom.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:
    import tokenizers as _hf_tokenizers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
except ImportError as exc:  # pragma: no cover - minimal installs
    _hf_tokenizers = None
    Tokenizer = None  # type: ignore[assignment]
    WordLevel = None  # type: ignore[assignment]
    WhitespaceSplit = None  # type: ignore[assignment]
    _TOKENIZERS_IMPORT_ERROR = exc
else:
    _TOKENIZERS_IMPORT_ERROR = None

from .open_source_token_embedding import (
    EmbeddingProvenance,
    IR_SLOT_IDS,
    IR_VALUE_VOCAB,
    PAD_ID,
    canonical_ir_pair_token,
)
from .pg131_layered_ir_policy import IR_MODES, MAX_STEPS, TOKENS_PER_STEP


SCHEMA_VERSION = "pg133-layered-token-embedding-v1"
TOKENIZER_BACKEND = "huggingface-tokenizers-layered-wordlevel"
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[SRC_HTML]", "[SRC_JAVASCRIPT]", "[SRC_TRANSPORT]", "[IR]", "[STEP]")
SCALAR_DIM = 5  # bounded atom weight, normalized weight, current, position, layer flag
DEFAULT_EMBEDDING_DIM = 48
MAX_SOURCE_ATOMS_PER_STEP = 64
MAX_LAYERED_TOKENS = 512
TOKEN_ABLATION_MODES = tuple(IR_MODES) + ("tokens_zeroed",)

# PG-133 needs one additional *evidence* slot.  It is deliberately kept
# local to the layered tokenizer instead of changing PG-131/132's fixed IR
# contract: those older experiments have frozen manifests and token counts.
# Availability is a typed-validation fact, not the evaluator's action label
# or positive authority, so it may be exposed to the policy as a bounded
# ``typed``/``unknown`` token.
PG133_EXTRA_IR_SLOT_IDS = ("oracle.availability",)
PG133_IR_VALUE_VOCAB = ("typed", "unknown")

_SOURCE_VALUES = {
    "html": {
        "tag": {"form", "input", "script", "main", "div", "a", "body", "html", "button", "select", "textarea", "unknown"},
        "attribute": {"method", "name", "id", "class", "action", "href", "src", "unknown"},
        "form_method": {"GET", "POST", "OTHER", "unknown"},
        "text_length_bucket": {"0", "1-4", "5-16", "17+", "unknown"},
        "script_count": {"0", "1-4", "5-16", "17+", "unknown"},
    },
    "javascript": {
        "api": {"document", "fetch", "xmlhttprequest", "location", "innerhtml", "eval", "settimeout", "postmessage", "unknown"},
        "keyword": {"if", "else", "for", "while", "function", "return", "const", "let", "var", "try", "catch", "throw", "new", "unknown"},
        "event_handler": {"present", "unknown"},
        "length_bucket": {"0", "1-4", "5-16", "17+", "unknown"},
    },
    "transport": {
        "method": {"GET", "POST", "unknown"},
        "placement": {"query", "json", "form", "body", "path", "unknown"},
        "route_template": {"hash_present", "unknown"},
        "form_field_count": {"0", "1-4", "5-16", "17+", "unknown"},
        "form_field": {"hash_present", "unknown"},
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha256(weight: torch.Tensor) -> str:
    return _sha256_bytes(weight.detach().cpu().contiguous().numpy().tobytes())


def _source_value(modality: str, kind: str, value: Any, token: Mapping[str, Any]) -> str:
    if "value_hash" in token:
        return "hash_present"
    allowed = _SOURCE_VALUES.get(modality, {}).get(kind, {"unknown"})
    text = str(value)
    return text if text in allowed else "unknown"


def canonical_source_atom(modality: Any, token: Mapping[str, Any]) -> str:
    modality_text = str(modality) if str(modality) in _SOURCE_VALUES else "unknown"
    kind = str(token.get("kind", "unknown"))
    value = _source_value(modality_text, kind, token.get("value", "unknown"), token)
    return f"src.{modality_text}.{kind if kind in _SOURCE_VALUES.get(modality_text, {}) else 'unknown'}={value}"


def _source_boundary(modality: str) -> str:
    return f"[SRC_{modality.upper()}]" if modality in {"html", "javascript", "transport"} else "[UNK]"


def canonical_layered_ir_pair_token(slot_id: Any, value: Any) -> str:
    """Canonicalize a bounded Rule-IR pair, including PG-133 evidence facts."""

    slot = str(slot_id)
    if slot in PG133_EXTRA_IR_SLOT_IDS:
        bounded_value = str(value) if str(value) in PG133_IR_VALUE_VOCAB else "unknown"
        return f"ir.{slot}={bounded_value}"
    return canonical_ir_pair_token(slot, value)


def _bucket_weight(value: Any) -> float:
    bucket = str(value)
    return {"0": 0.5, "1-4": 1.0, "5-16": 1.25, "17+": 1.5}.get(bucket, 1.0)


def source_layer_atoms(source_layers: Sequence[Mapping[str, Any]]) -> list[tuple[str, float, int]]:
    """Return ``(atom, weight, layer_flag)`` without retaining raw fields."""

    atoms: list[tuple[str, float, int]] = []
    for layer in source_layers:
        modality = str(layer.get("modality", "unknown"))
        atoms.append((_source_boundary(modality), 0.5, 0))
        tokens = list(layer.get("tokens") or [])
        if len(tokens) > MAX_SOURCE_ATOMS_PER_STEP:
            raise ValueError("source token layer exceeds the bounded PG-133 atom limit")
        for token in tokens:
            if not isinstance(token, Mapping):
                continue
            atom = canonical_source_atom(modality, token)
            count_weight = _bucket_weight(token.get("count_bucket", token.get("value") if token.get("kind") in {"text_length_bucket", "script_count", "length_bucket", "form_field_count"} else "1-4"))
            atoms.append((atom, count_weight, 0))
    return atoms


def ir_layer_atoms(ir_layer: Mapping[str, Any]) -> list[tuple[str, float, int]]:
    by_slot = {
        str(token.get("slot_id")): token
        for token in list(ir_layer.get("tokens") or [])
        if isinstance(token, Mapping)
    }
    slots = (*IR_SLOT_IDS, *PG133_EXTRA_IR_SLOT_IDS)
    return [(canonical_layered_ir_pair_token(slot, (by_slot.get(slot) or {}).get("value", "unknown")), max(0.0, min(float((by_slot.get(slot) or {}).get("weight", 1.0)), 2.0)), 1) for slot in slots]


def _vocabulary() -> tuple[str, ...]:
    source_atoms: list[str] = []
    for modality, kinds in _SOURCE_VALUES.items():
        for kind, values in kinds.items():
            for value in sorted(values):
                source_atoms.append(f"src.{modality}.{kind}={value}")
    ir_atoms = [canonical_layered_ir_pair_token(slot, value) for slot in (*IR_SLOT_IDS, *PG133_EXTRA_IR_SLOT_IDS) for value in (*IR_VALUE_VOCAB, *PG133_IR_VALUE_VOCAB)]
    return tuple(dict.fromkeys((*SPECIAL_TOKENS, *source_atoms, *ir_atoms)))


VOCABULARY = _vocabulary()
VOCAB_IDS = {token: index for index, token in enumerate(VOCABULARY)}
PAD_ID = VOCAB_IDS["[PAD]"]
UNK_ID = VOCAB_IDS["[UNK]"]


def build_layered_tokenizer() -> Any:
    if _hf_tokenizers is None:  # pragma: no cover
        raise RuntimeError("the optional 'tokenizers' package is required for PG-133") from _TOKENIZERS_IMPORT_ERROR
    tokenizer = Tokenizer(WordLevel(vocab=VOCAB_IDS, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer


class LayeredTokenEmbedding(nn.Module):
    """Embedding over source atoms, Rule-IR pairs and boundary tokens."""

    def __init__(self, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM, seed: int = 13301) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.tokenizer = build_layered_tokenizer()
        self.embedding = nn.Embedding(len(VOCABULARY), embedding_dim, padding_idx=PAD_ID)
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
            weights_source="fresh_seeded_torch_embedding",
            pretrained=False,
            source_id=None,
            license=None,
            weights_sha256=_tensor_sha256(self.embedding.weight),
            initialization_seed=seed,
        )

    @property
    def provenance(self) -> EmbeddingProvenance:
        return self._provenance

    def ids(self, atoms: Sequence[str]) -> list[int]:
        encoding = self.tokenizer.encode(list(atoms), is_pretokenized=True)
        return [int(value) for value in encoding.ids]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[-1] != MAX_LAYERED_TOKENS:
            raise ValueError("layered token ids have the wrong shape")
        return self.embedding(token_ids)


def layered_token_inputs(
    embedding: LayeredTokenEmbedding,
    prefix_steps: Sequence[Mapping[str, Any]],
    *,
    mode: str = "weighted",
) -> tuple[list[int], list[list[float]]]:
    if mode not in TOKEN_ABLATION_MODES:
        raise ValueError(f"unknown PG-133 token mode: {mode}")
    if len(prefix_steps) > MAX_STEPS:
        raise ValueError("layered token prefix exceeds MAX_STEPS")
    all_atoms: list[str] = []
    all_meta: list[tuple[float, int]] = []
    for step_index, step in enumerate(prefix_steps):
        step_atoms: list[tuple[str, float, int]] = [("[STEP]", 0.5, 0)]
        step_atoms.extend(source_layer_atoms(list(step.get("source_token_layers") or [])))
        step_atoms.append(("[IR]", 0.5, 1))
        step_atoms.extend(ir_layer_atoms(step.get("ir_layer") or {}))
        if mode in {"uniform", "tokens_zeroed"}:
            step_atoms = [(atom, 1.0, layer_flag) for atom, _, layer_flag in step_atoms]
        elif mode == "no_failure_slots":
            step_atoms = [(atom, (0.0 if (layer_flag == 1 and ("failure.kind=" in atom or "failure.failed_gate=" in atom or "failure.recovery_phase=" in atom)) else weight), layer_flag) for atom, weight, layer_flag in step_atoms]
        elif mode == "zero":
            step_atoms = [(atom, 0.0, layer_flag) for atom, _, layer_flag in step_atoms]
        denominator = sum(max(weight, 0.0) for _, weight, _ in step_atoms) or 1.0
        current = 1.0 if step_index == len(prefix_steps) - 1 else 0.0
        position = float(step_index) / float(max(MAX_STEPS - 1, 1))
        all_atoms.extend(atom for atom, _, _ in step_atoms)
        all_meta.extend((weight, layer_flag) for _, weight, layer_flag in step_atoms)
        # Store normalized weights per step; this avoids sequence length being
        # a hidden global weight and keeps source/IR contribution auditable.
        start = len(all_meta) - len(step_atoms)
        all_meta[start:] = [(weight / 2.0, weight / denominator, layer_flag) for _, weight, layer_flag in step_atoms]
    if len(all_atoms) > MAX_LAYERED_TOKENS:
        raise ValueError("layered token sequence exceeds MAX_LAYERED_TOKENS")
    ids = embedding.ids(all_atoms)
    scalars: list[list[float]] = []
    cursor = 0
    for step_index, step in enumerate(prefix_steps):
        # Keep this derived from the actual Rule-IR atom list: PG-133 adds an
        # evidence availability slot without changing the older IR policy.
        step_atoms_count = 1 + len(source_layer_atoms(list(step.get("source_token_layers") or []))) + 1 + len(ir_layer_atoms(step.get("ir_layer") or {}))
        current = 1.0 if step_index == len(prefix_steps) - 1 else 0.0
        position = float(step_index) / float(max(MAX_STEPS - 1, 1))
        for _, _, layer_flag in all_meta[cursor : cursor + step_atoms_count]:
            # The normalized weight is read from the second metadata field.
            raw_weight, normalized_weight, flag = all_meta[cursor]
            scalars.append([raw_weight, normalized_weight, current, position, float(flag)])
            cursor += 1
    if mode == "zero":
        ids = [PAD_ID] * len(ids)
        scalars = [[0.0] * SCALAR_DIM for _ in scalars]
    elif mode == "tokens_zeroed":
        ids = [PAD_ID] * len(ids)
    ids.extend([PAD_ID] * (MAX_LAYERED_TOKENS - len(ids)))
    scalars.extend([[0.0] * SCALAR_DIM for _ in range(MAX_LAYERED_TOKENS - len(scalars))])
    if len(ids) != MAX_LAYERED_TOKENS or len(scalars) != MAX_LAYERED_TOKENS:
        raise AssertionError("layered token input shape drift")
    return ids, scalars


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "IR_SLOT_IDS",
    "IR_VALUE_VOCAB",
    "LayeredTokenEmbedding",
    "MAX_LAYERED_TOKENS",
    "MAX_SOURCE_ATOMS_PER_STEP",
    "PG133_EXTRA_IR_SLOT_IDS",
    "PG133_IR_VALUE_VOCAB",
    "PAD_ID",
    "SCALAR_DIM",
    "SCHEMA_VERSION",
    "SPECIAL_TOKENS",
    "TOKEN_ABLATION_MODES",
    "TOKENIZER_BACKEND",
    "VOCABULARY",
    "build_layered_tokenizer",
    "canonical_source_atom",
    "canonical_layered_ir_pair_token",
    "ir_layer_atoms",
    "layered_token_inputs",
    "source_layer_atoms",
]
