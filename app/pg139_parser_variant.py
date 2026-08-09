"""Independent bounded parser used only for PG-139 parser-OOD evaluation."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from .pg136_causal_token_lm import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN


SCHEMA_VERSION = "pg139-independent-parser-variant-v1"


def _safe(value: object) -> str:
    text = str(value or "unknown")
    if len(text) > 32:
        return "unknown"
    text = re.sub(r"[^A-Za-z0-9_+.-]", "_", text)
    return text or "unknown"


def _weight(value: object) -> str:
    try:
        number = max(0.0, min(float(value), 2.0))
    except (TypeError, ValueError):
        number = 1.0
    if number <= 0.5:
        return "0.5"
    if number <= 1.0:
        return "1.0"
    if number <= 1.25:
        return "1.25"
    if number <= 1.5:
        return "1.5"
    return "2.0"


def alternate_tokens(layered_steps: Sequence[Mapping[str, object]]) -> list[str]:
    """Encode the same bounded facts with an intentionally unseen grammar.

    The parser sorts Rule-IR slots and source modalities, separates slot/value
    atoms, and uses ``[ALT_*]`` boundaries.  It never reads raw response or
    probe fields; its purpose is to test whether a learned model transfers
    beyond the canonical encoder rather than memorizing token spelling.
    """

    if not layered_steps:
        raise ValueError("PG-139 alternate parser requires at least one step")
    tokens = [BOS_TOKEN]
    for step in layered_steps:
        tokens.append("[ALT_STEP]")
        layers = [layer for layer in list(step.get("source_token_layers") or []) if isinstance(layer, Mapping)]
        for layer in sorted(layers, key=lambda item: _safe(item.get("modality", "unknown")), reverse=True):
            modality = _safe(layer.get("modality", "unknown")).lower()
            tokens.append(f"[ALT_SRC_{modality.upper()}]")
            items = [item for item in list(layer.get("tokens") or []) if isinstance(item, Mapping)]
            for item in sorted(items, key=lambda value: (_safe(value.get("kind", "unknown")), _safe(value.get("value", "unknown")))):
                kind = _safe(item.get("kind", "unknown"))
                value = "hash_present" if "value_hash" in item else _safe(item.get("value", "unknown"))
                tokens.append(f"alt.src.{kind}={value}")
                if item.get("count_bucket") is not None:
                    tokens.append(f"alt.src.count={_safe(item.get('count_bucket'))}")
        tokens.append("[ALT_IR]")
        ir_items = [item for item in list((step.get("ir_layer") or {}).get("tokens") or []) if isinstance(item, Mapping)]
        for item in sorted(ir_items, key=lambda value: _safe(value.get("slot_id", "unknown"))):
            tokens.extend(["alt.ir." + _safe(item.get("slot_id", "unknown")) + "=" + _safe(item.get("value", "unknown")), "alt.ir.weight=" + _weight(item.get("weight", 1.0))])
    tokens.append(EOS_TOKEN)
    if len(tokens) > 384:
        raise ValueError("PG-139 alternate sequence exceeds bound")
    return tokens


__all__ = ["SCHEMA_VERSION", "alternate_tokens"]
