"""Language-neutral Rule IR decoding primitives for the Loop 12 experiment.

The decoder deliberately operates on a projection of visible observations.  It
does not consume challenge keys, family labels, evaluator state, source code, or
the synthetic oracle fields (``intended_output``/``is_counterexample``).  The
neural component predicts an abstract policy family; the emitted Rule IR is a
grammar-checked, language-independent template whose policy slots can later be
bound by a separate evidence/reconciliation stage.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from html import unescape
from typing import Any, Iterable, Iterator
from urllib.parse import unquote, urlsplit

from .juice_shop_adapter import RULE_FAMILY_TEMPLATES
from .rule_ir import canonical

import torch
from torch import nn


DECODER_FAMILIES = (
    "access_control",
    "authentication",
    "input_validation",
    "injection",
    "observability",
    "url_redirect",
    "xss",
)

# Fixed feature size is part of the checkpoint contract.  Features are ratios
# or bounded indicators; the trainer may additionally standardise them.
FEATURE_DIM = 256
_TYPE_NAMES = ("bool", "number", "string", "null", "other")
_IGNORE_KEYS = {"intended_output", "is_counterexample", "family", "record_id", "episode_id", "step"}
_TAG_RE = re.compile(r"<\s*/?\s*[a-z][^>]*>", re.IGNORECASE)


def abstract_rule_ir(family: str) -> dict[str, Any]:
    """Return a defensive copy of the family-level abstract Rule IR template."""

    if family not in DECODER_FAMILIES:
        raise KeyError(f"unsupported Rule IR decoder family: {family}")
    return copy.deepcopy(RULE_FAMILY_TEMPLATES[family])


def abstract_rule_ir_canonical(family: str) -> str:
    return canonical(abstract_rule_ir(family))


def _stable_bucket(value: str, size: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % size


def _iter_leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            if str(key).casefold() in _IGNORE_KEYS:
                continue
            child = f"{path}.{key}" if path else str(key)
            yield from _iter_leaves(value[key], child)
        return
    if isinstance(value, (list, tuple)):
        for index, child_value in enumerate(value):
            yield from _iter_leaves(child_value, f"{path}[{index}]")
        return
    yield path, value


def _safe_url(value: str) -> tuple[str, str, str] | None:
    if "://" not in value:
        return None
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return None
        return parsed.scheme.casefold(), parsed.hostname.casefold(), parsed.path or "/"
    except (TypeError, ValueError):
        return None


def _string_flags(value: str) -> list[int]:
    lowered = value.casefold()
    url = _safe_url(value)
    flags = [0] * 32
    flags[0] = int(bool(value))
    flags[1] = int(value != value.strip())
    flags[2] = int(value != lowered)
    flags[3] = int(bool(re.search(r"[\"']", value)))
    flags[4] = int("%" in value or "\\u" in lowered)
    flags[5] = int("&lt;" in lowered or "&#60;" in lowered or "&amp;" in lowered)
    flags[6] = int("<" in value)
    flags[7] = int(">" in value)
    flags[8] = int(bool(_TAG_RE.search(value)))
    flags[9] = int("onerror" in lowered or "javascript:" in lowered)
    flags[10] = int(any(token in lowered for token in ("select ", " union ", " or 1=1", "--", "drop ", "insert ")))
    flags[11] = int("=" in value or "&" in value)
    flags[12] = int(value.strip().casefold() in {"null", "undefined", "nan", "infinity"})
    flags[13] = int(bool(re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value.strip())))
    flags[14] = int(bool(re.fullmatch(r"[+-]?0[xX][0-9a-fA-F]+", value.strip())))
    flags[15] = int(len(value) == 0)
    if url is not None:
        scheme, host, path = url
        flags[16] = 1
        flags[17] = int(scheme == "https")
        flags[18] = int(scheme in {"http", "https"})
        flags[19] = int(scheme in {"ws", "wss"})
        flags[20] = int(scheme == "ftp")
        flags[21] = int(host.count(".") >= 1)
        flags[22] = int(host.count(".") >= 2)
        flags[23] = int(path not in {"", "/"})
    flags[24] = int(value.startswith("/") or value.startswith("\\"))
    flags[25] = int("." in value)
    flags[26] = int("/" in value)
    flags[27] = int(":" in value)
    flags[28] = int("{" in value or "}" in value or "[" in value or "]" in value)
    flags[29] = int("=" in value and "&" in value)
    flags[30] = int(len(value) >= 32)
    flags[31] = int(len(value) >= 128)
    return flags


def _encoding_depth_flags(value: str) -> list[int]:
    """Expose bounded decode-depth signals without retaining source text.

    The first four slots indicate whether markup becomes parser-visible after
    0..3 HTML-entity passes.  The URL-encoded and nested-marker slots cover
    percent-encoded payloads and mixed ``&amp;lt;``/``%253C`` forms.  This is
    deliberately a shape feature, not an executable decoder.
    """

    flags = [0] * 24
    html_value = value
    url_value = value
    changed = 0
    for depth in range(4):
        flags[depth] = int(bool(_TAG_RE.search(html_value)))
        flags[4 + depth] = int("<" in html_value or ">" in html_value)
        flags[8 + depth] = int("&" in html_value)
        next_html = unescape(html_value)
        next_url = unquote(url_value)
        changed += int(next_html != html_value or next_url != url_value)
        html_value = next_html
        url_value = next_url
    lowered = value.casefold()
    flags[16] = int("&amp;lt" in lowered or "&#38;lt" in lowered or "&amp;#" in lowered)
    flags[17] = int("&amp;amp;" in lowered or "&amp;#38;" in lowered)
    flags[18] = int("%3c" in lowered or "%3e" in lowered)
    flags[19] = int("%253c" in lowered or "%253e" in lowered)
    flags[20] = int("%25253c" in lowered or "%25253e" in lowered)
    flags[21] = min(changed, 4)
    flags[22] = int(not bool(_TAG_RE.search(value)) and bool(_TAG_RE.search(unescape(value))))
    flags[23] = int(not bool(_TAG_RE.search(value)) and not bool(_TAG_RE.search(unescape(value))) and bool(_TAG_RE.search(unescape(unescape(value)))))
    return flags


def _surface_flags(value: str) -> list[int]:
    lowered = value.casefold()
    tokens = (
        "metrics", "logs", "debug", "ftp", "redirect", "basket", "admin",
        "users", "login", "reset-password", "products/search", "products",
        "robots", "script", "img", "security",
    )
    return [int(token in lowered) for token in tokens]


def trace_feature_vector(traces: Iterable[dict[str, Any]]) -> list[float]:
    """Project visible black-box traces into a fixed, source-language-neutral vector.

    The projection intentionally ignores oracle-only fields.  Field names only
    contribute to stable anonymous hash buckets; the classifier therefore sees
    shapes and semantics rather than a memorisable JavaScript identifier.
    """

    features = [0.0] * FEATURE_DIM
    rows = list(traces)
    if not rows:
        return features
    n = float(len(rows))

    def add(index: int, value: float = 1.0) -> None:
        if 0 <= index < FEATURE_DIM:
            features[index] += float(value)

    true_count = 0
    history_count = 0
    prev_equal_count = 0
    leaf_count = 0
    for row in rows:
        if bool(row.get("output", False)):
            true_count += 1
        history = row.get("history") or []
        if history:
            history_count += 1
            current_leaves = dict(_iter_leaves(row.get("input", {}), "input"))
            previous_leaves = dict(_iter_leaves(history[-1].get("input", {}), "input")) if isinstance(history[-1], dict) else {}
            if current_leaves and previous_leaves and any(current_leaves.get(key) == value for key, value in previous_leaves.items()):
                prev_equal_count += 1
        for path, value in _iter_leaves({
            "input": row.get("input", {}),
            "context": row.get("context", {}),
            "state": row.get("state", {}),
        }):
            leaf_count += 1
            add(8 + _stable_bucket(path.casefold(), 32))
            if isinstance(value, bool):
                add(40)
                add(41 + int(value))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                add(43)
                add(44 + (0 if value < 0 else 1 if value == 0 else 2))
            elif isinstance(value, str):
                add(46)
                length_bucket = min(len(value) // 16, 7)
                add(47 + length_bucket)
                for flag, present in enumerate(_string_flags(value)):
                    add(55 + flag, present)
                for flag, present in enumerate(_encoding_depth_flags(value)):
                    add(120 + flag, present)
                for flag, present in enumerate(_surface_flags(value)):
                    add(145 + flag, present)
                if "action.path" in path.casefold():
                    add(161, int(value.startswith("/")))
                    add(162, int("?" in value))
                    add(163, int("%" in value))
            if "response.status_code" in path.casefold() and isinstance(value, (int, float)):
                status = int(value)
                add(165, int(200 <= status < 300))
                add(166, int(300 <= status < 400))
                add(167, int(400 <= status < 500))
                add(168, int(status >= 500))
                add(169, int(status in {401, 403}))
                add(170, int(status in {406, 422}))
                add(171, int(status == 500))
            if "response.body_shape" in path.casefold() and isinstance(value, str):
                for flag, present in enumerate(_surface_flags(value)):
                    add(173 + flag, present)
            elif value is None:
                add(87)
            else:
                add(88)
        for history_row in history:
            for path, value in _iter_leaves(history_row if isinstance(history_row, dict) else {}, "history"):
                add(89 + _stable_bucket(path.casefold(), 16))

    # Global behavior counters and ratios.
    add(0, min(len(rows), 32))
    add(1, true_count / n)
    add(2, (len(rows) - true_count) / n)
    add(3, history_count / n)
    add(4, prev_equal_count / n)
    add(5, leaf_count / n)
    add(6, len({str(row.get("output", False)) for row in rows}) / 2.0)
    add(7, sum(1 for row in rows if row.get("state")) / n)

    # Turn count-like blocks into bounded per-row rates.  Feature 0 remains a
    # trace-count indicator and is deliberately left as a small count.
    for index in range(8, FEATURE_DIM):
        features[index] = min(features[index] / n, 8.0) / 8.0
    features[0] = min(features[0] / 32.0, 1.0)
    return features


def validate_abstract_rule_ir(expr: dict[str, Any]) -> None:
    """Validate the restricted abstract-template grammar before persistence."""

    if not isinstance(expr, dict) or not isinstance(expr.get("op"), str):
        raise ValueError("Rule IR node must be an object with a string op")
    op = expr["op"]
    if op == "policy_slot":
        if not isinstance(expr.get("name"), str) or not expr["name"]:
            raise ValueError("policy_slot requires a non-empty name")
        return
    if op in {"and", "or", "xor"}:
        args = expr.get("args")
        if not isinstance(args, list) or not args:
            raise ValueError(f"{op} requires a non-empty args list")
        for arg in args:
            validate_abstract_rule_ir(arg)
        return
    if op == "not":
        validate_abstract_rule_ir(expr.get("arg"))
        return
    if op == "origin_eq":
        validate_abstract_rule_ir(expr.get("left"))
        validate_abstract_rule_ir(expr.get("right"))
        return
    if op == "html_creates_nodes":
        validate_abstract_rule_ir(expr.get("arg"))
        return
    raise ValueError(f"unsupported abstract Rule IR op: {op}")


for _family in DECODER_FAMILIES:
    validate_abstract_rule_ir(abstract_rule_ir(_family))


def confidence_margin(probabilities: Iterable[float]) -> tuple[float, float]:
    values = sorted((float(value) for value in probabilities), reverse=True)
    if not values:
        return 0.0, 0.0
    top = values[0]
    second = values[1] if len(values) > 1 else 0.0
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in values if value > 0)
    return top, top - second


def calibrate_abstention_threshold(
    probabilities: Iterable[Iterable[float]],
    labels: Iterable[int],
    *,
    minimum_precision: float = 0.95,
) -> dict[str, float | int | None]:
    """Choose the lowest confidence threshold meeting a precision gate.

    Calibration is performed only on a labelled validation split.  The chosen
    threshold is persisted with the checkpoint so runtime binding does not
    depend on a hand-tuned constant.  If the gate is impossible, the function
    returns a fully-abstaining threshold of 1.0.
    """

    rows = []
    for probability, label in zip(probabilities, labels):
        values = [float(value) for value in probability]
        if not values:
            continue
        prediction = max(range(len(values)), key=values.__getitem__)
        rows.append((max(values), int(prediction == int(label))))
    if not rows:
        return {"threshold": 1.0, "precision": 0.0, "coverage": 0.0, "accepted": 0, "total": 0}
    candidates = sorted({confidence for confidence, _ in rows}, reverse=True)
    candidates.append(1.0)
    best: tuple[float, float, float, int] | None = None
    for threshold in candidates:
        accepted = [correct for confidence, correct in rows if confidence >= threshold]
        if not accepted:
            continue
        precision = sum(accepted) / len(accepted)
        coverage = len(accepted) / len(rows)
        if precision + 1e-12 < minimum_precision:
            continue
        candidate = (coverage, precision, -threshold, len(accepted))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return {"threshold": 1.0, "precision": 0.0, "coverage": 0.0, "accepted": 0, "total": len(rows)}
    coverage, precision, negative_threshold, accepted = best
    return {
        "threshold": round(-negative_threshold, 6),
        "precision": round(precision, 6),
        "coverage": round(coverage, 6),
        "accepted": accepted,
        "total": len(rows),
    }


class RuleIRDecoder(nn.Module):
    """Small GPU-friendly template decoder with a grammar-constrained emission."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 192, dropout: float = 0.10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(DECODER_FAMILIES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))

    @torch.inference_mode()
    def decode(self, features: torch.Tensor, abstain_threshold: float = 0.55) -> list[dict[str, Any]]:
        logits = self(features)
        probabilities = torch.softmax(logits, dim=-1)
        values, indices = probabilities.max(dim=-1)
        decoded: list[dict[str, Any]] = []
        for confidence, index, row in zip(values.detach().cpu(), indices.detach().cpu(), probabilities.detach().cpu()):
            family = DECODER_FAMILIES[int(index)]
            confidence_value = float(confidence)
            decoded.append({
                "family": None if confidence_value < abstain_threshold else family,
                "candidate_family": family,
                "confidence": round(confidence_value, 6),
                "rule_ir": None if confidence_value < abstain_threshold else abstract_rule_ir(family),
                "probabilities": {name: round(float(probability), 6) for name, probability in zip(DECODER_FAMILIES, row)},
            })
        return decoded
