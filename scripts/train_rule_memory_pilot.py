#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from app.synthetic_curriculum import generate_curriculum  # noqa: E402
from app.research_events import emit_event  # noqa: E402


PAD = 0
EOS = 257
VOCAB_SIZE = 258


def flatten(value: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows = []
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else key
        item = value[key]
        if isinstance(item, dict):
            rows.extend(flatten(item, path))
        else:
            rows.append((path, item))
    return rows


def _semantic_projection(visible: dict[str, Any]) -> list[str]:
    features = []
    for key, value in flatten(visible):
        if isinstance(value, bool):
            features.append(f"{key}#type=bool")
        elif isinstance(value, (int, float)):
            sign = "zero" if value == 0 else "negative" if value < 0 else "positive"
            features.extend([f"{key}#type=number", f"{key}#sign={sign}"])
        elif isinstance(value, str):
            features.append(f"{key}#type=string")
            if "://" in value:
                try:
                    parsed = urlsplit(value)
                    if parsed.scheme and parsed.hostname:
                        features.extend([
                            f"{key}#url_scheme={parsed.scheme.lower()}",
                            f"{key}#url_host={parsed.hostname.lower()}",
                            f"{key}#url_host_labels={len(parsed.hostname.split('.'))}",
                            f"{key}#url_has_path={int(bool(parsed.path and parsed.path != '/'))}",
                        ])
                except ValueError:
                    features.append(f"{key}#url_parse=invalid")
            lowered = value.lower()
            tag_lexeme = int(re.search(r"<\s*/?\s*[a-z][^>]*>", value, re.IGNORECASE) is not None)
            features.extend([
                f"{key}#has_lt={int('<' in value)}",
                f"{key}#has_gt={int('>' in value)}",
                f"{key}#tag_lexeme={tag_lexeme}",
                f"{key}#encoded_angle={int('&lt;' in lowered or '&#60;' in lowered)}",
            ])
    return features


def _compact_semantic_projection(visible: dict[str, Any]) -> list[str]:
    features = []
    for key, value in flatten(visible):
        if isinstance(value, bool):
            features.append(f"{key}=b{int(value)}")
        elif isinstance(value, (int, float)):
            features.append(f"{key}=n{value}")
        elif isinstance(value, str):
            if "://" in value:
                try:
                    parsed = urlsplit(value)
                    if parsed.scheme and parsed.hostname:
                        tail = parsed.path or "/"
                        if parsed.query:
                            tail += f"?{parsed.query}"
                        features.append(f"{key}=u:{parsed.scheme.lower()}|{parsed.hostname.lower()}|{tail}")
                        continue
                except ValueError:
                    pass
            if key.endswith(("payload", "message")):
                lowered = value.lower()
                tag_lexeme = int(re.search(r"<\s*/?\s*[a-z][^>]*>", value, re.IGNORECASE) is not None)
                length_bucket = min(len(value) // 8, 9)
                features.append(
                    f"{key}=m:{int('<' in value)}{int('>' in value)}{tag_lexeme}"
                    f"{int('&lt;' in lowered or '&#60;' in lowered)}l{length_bucket}"
                )
            else:
                features.append(f"{key}=s:{value}")
    return features


def _routed_semantic_projection(visible: dict[str, Any], canonical_url_slots: bool = False) -> list[str]:
    features = []
    for key, value in flatten(visible):
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        if isinstance(value, str) and "://" in value:
            try:
                parsed = urlsplit(value)
                if parsed.scheme and parsed.hostname:
                    tail = parsed.path or "/"
                    if parsed.query:
                        tail += f"?{parsed.query}"
                    if canonical_url_slots:
                        default_ports = {"http": 80, "https": 443, "ws": 80, "wss": 443, "ftp": 21}
                        effective_port = parsed.port if parsed.port is not None else default_ports.get(parsed.scheme.lower())
                        features.append(f"url.value=u:{parsed.scheme.lower()}|{parsed.hostname.lower()}|p{effective_port or 0}|{tail}")
                    else:
                        features.append(f"{key}=u:{parsed.scheme.lower()}|{parsed.hostname.lower()}|{tail}")
                    continue
            except ValueError:
                pass
        if isinstance(value, str) and key.endswith(("payload", "message", "encoded")):
            lowered = value.lower()
            tag_lexeme = int(re.search(r"<\s*/?\s*[a-z][^>]*>", value, re.IGNORECASE) is not None)
            decoded_once = unescape(value)
            decoded_twice = unescape(decoded_once)
            decoded_once_tag = int(re.search(r"<\s*/?\s*[a-z][^>]*>", decoded_once, re.IGNORECASE) is not None)
            decoded_twice_tag = int(re.search(r"<\s*/?\s*[a-z][^>]*>", decoded_twice, re.IGNORECASE) is not None)
            length_bucket = min(len(value) // 8, 9)
            features.append(
                f"{key}=m:{int('<' in value)}{int('>' in value)}{tag_lexeme}"
                f"{int('&lt;' in lowered or '&#60;' in lowered)}d{decoded_once_tag}{decoded_twice_tag}l{length_bucket}"
            )
        elif isinstance(value, str) and key.endswith(("role", "token")):
            features.append(f"{key}=s:{value}|cf:{value.casefold()}")
        elif isinstance(value, str) and key.endswith(("amount", "total", "balance")):
            canonical = int(re.fullmatch(r"0|[1-9][0-9]*", value) is not None)
            features.append(
                f"{key}=s:{value}|canon:{canonical}|ws:{int(value != value.strip())}"
                f"|hex:{int(value.strip().lower().startswith(('0x', '+0x', '-0x')))}|exp:{int('e' in value.lower())}"
            )
        else:
            features.append(f"{key}={rendered}")
    return features


def _common_suffix(values: list[str]) -> str:
    if not values:
        return ""
    reversed_values = [value[::-1] for value in values]
    limit = min(len(value) for value in reversed_values)
    length = 0
    while length < limit and len({value[length] for value in reversed_values}) == 1:
        length += 1
    return reversed_values[0][:length][::-1]


def _parsed_url_parts(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or "://" not in value:
        return None
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return None
        default_ports = {"http": 80, "https": 443, "ws": 80, "wss": 443, "ftp": 21}
        return {
            "scheme": parsed.scheme.lower(),
            "hostname": parsed.hostname.lower(),
            "port": parsed.port if parsed.port is not None else default_ports.get(parsed.scheme.lower()),
        }
    except ValueError:
        return None


def _structured_url_episode_rule(context: list[dict[str, Any]], query: dict[str, Any]) -> str | None:
    """Induce URL-component predicates from observations without family metadata."""
    if len(context) < 4:
        return None
    query_visible = {"input": query.get("input", {}), "context": query.get("context", {}), "state": query.get("state", {})}
    query_values = dict(flatten(query_visible))
    labels = [bool(row.get("output")) for row in context]
    if len(set(labels)) < 2:
        return None
    candidates: list[tuple[int, str, list[bool], bool]] = []
    for path, query_value in query_values.items():
        query_parts = _parsed_url_parts(query_value)
        row_parts = []
        for row in context:
            visible = {"input": row.get("input", {}), "context": row.get("context", {}), "state": row.get("state", {})}
            row_parts.append(_parsed_url_parts(dict(flatten(visible)).get(path)))
        valid_parts = [parts for parts in row_parts if parts is not None]
        if len(valid_parts) < max(3, len(context) // 2):
            continue
        for component, priority in (("hostname", 300), ("scheme", 220), ("port", 180)):
            expected_values = sorted({parts[component] for parts in valid_parts if parts.get(component) is not None}, key=str)
            for expected in expected_values:
                predictions = [bool(parts is not None and parts.get(component) == expected) for parts in row_parts]
                query_prediction = bool(query_parts is not None and query_parts.get(component) == expected)
                text = f"url.{component}({path})=={expected}|q={int(query_prediction)}|fit={len(labels)}/{len(labels)}|kind=url_{component}"
                candidates.append((priority, text, predictions, query_prediction))
    exact = [candidate for candidate in candidates if candidate[2] == labels]
    if not exact:
        return None
    return max(exact, key=lambda candidate: (candidate[0], -len(candidate[1]), candidate[1]))[1]


def _generic_episode_rule(context: list[dict[str, Any]], query: dict[str, Any]) -> str | None:
    """Induce a small executable rule from observations only.

    This deliberately covers language-agnostic primitives rather than curriculum
    family names: numeric cuts and a categorical guard combined with a non-zero
    numeric flag.  A rule is exposed only when it explains every visible row, so
    the neural decoder remains responsible when the episode is ambiguous.
    """
    if len(context) < 6:
        return None
    query_visible = {"input": query.get("input", {}), "context": query.get("context", {}), "state": query.get("state", {})}
    query_values = dict(flatten(query_visible))
    row_values = []
    labels = []
    for row in context:
        visible = {"input": row.get("input", {}), "context": row.get("context", {}), "state": row.get("state", {})}
        row_values.append(dict(flatten(visible)))
        labels.append(bool(row.get("output")))

    atoms: list[dict[str, Any]] = []
    for path, query_value in query_values.items():
        values = [row.get(path) for row in row_values]
        if any(value is None for value in values):
            continue
        if isinstance(query_value, bool) and all(isinstance(value, bool) for value in values):
            for expected in (False, True):
                atoms.append({
                    "text": f"{path}=={str(expected).lower()}",
                    "pred": [value is expected for value in values],
                    "q": query_value is expected,
                    "path": path,
                    "kind": "bool_eq",
                })
        elif isinstance(query_value, (int, float)) and not isinstance(query_value, bool) and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        ):
            atoms.append({
                "text": f"{path}!=0",
                "pred": [value != 0 for value in values],
                "q": query_value != 0,
                "path": path,
                "kind": "numeric_nonzero",
            })
            for cut in sorted(set(values)):
                for symbol, predicate in (
                    ("lt", lambda value, boundary=cut: value < boundary),
                    ("le", lambda value, boundary=cut: value <= boundary),
                    ("gt", lambda value, boundary=cut: value > boundary),
                    ("ge", lambda value, boundary=cut: value >= boundary),
                ):
                    atoms.append({
                        "text": f"{path}#{symbol}={cut}",
                        "pred": [predicate(value) for value in values],
                        "q": predicate(query_value),
                        "path": path,
                        "kind": "numeric_boundary",
                    })
        elif isinstance(query_value, str) and all(isinstance(value, str) for value in values):
            for expected in sorted(set(values)):
                atoms.append({
                    "text": f"{path}=={expected}",
                    "pred": [value == expected for value in values],
                    "q": query_value == expected,
                    "path": path,
                    "kind": "string_eq",
                })

    query_history = query.get("history") or []
    if query_history:
        previous_query_values = dict(flatten(query_history[-1]))
        for path, query_value in query_values.items():
            if not path.startswith("input.") or path not in previous_query_values:
                continue
            predictions_eq = []
            usable = True
            for row, values in zip(context, row_values):
                history = row.get("history") or []
                previous_values = dict(flatten(history[-1])) if history else {}
                if path not in values or path not in previous_values:
                    usable = False
                    break
                predictions_eq.append(values[path] == previous_values[path])
            if usable and len(predictions_eq) == len(context):
                query_eq = query_value == previous_query_values[path]
                atoms.extend([
                    {"text": f"{path}#eq_prev1", "pred": predictions_eq, "q": query_eq, "path": path, "kind": "history_relation"},
                    {"text": f"{path}#ne_prev1", "pred": [not value for value in predictions_eq], "q": not query_eq, "path": path, "kind": "history_relation"},
                ])

    candidates: list[tuple[int, str, list[bool], bool]] = []
    for atom in atoms:
        priority = 110 if atom["kind"] == "history_relation" else 80 if atom["kind"] == "numeric_boundary" else 30
        candidates.append((priority, atom["text"], atom["pred"], atom["q"]))
    for left in atoms:
        for right in atoms:
            if left["path"] >= right["path"]:
                continue
            kinds = {left["kind"], right["kind"]}
            if kinds == {"string_eq", "numeric_nonzero"}:
                predictions = [a or b for a, b in zip(left["pred"], right["pred"])]
                candidates.append((100, f"({left['text']})or({right['text']})", predictions, bool(left["q"] or right["q"])))
            elif kinds == {"bool_eq"}:
                for operator in ("and", "or"):
                    predictions = [a and b if operator == "and" else a or b for a, b in zip(left["pred"], right["pred"])]
                    query_prediction = bool(left["q"] and right["q"] if operator == "and" else left["q"] or right["q"])
                    candidates.append((90, f"({left['text']}){operator}({right['text']})", predictions, query_prediction))

    exact = [candidate for candidate in candidates if candidate[2] == labels]
    if not exact:
        return None
    _, text, _, query_prediction = max(exact, key=lambda candidate: (candidate[0], -len(candidate[1]), candidate[1]))
    return f"generic={text}|q={int(query_prediction)}|fit={len(labels)}/{len(labels)}"


def _episode_rule_projection(
    context: list[dict[str, Any]],
    query: dict[str, Any],
    suppress_url_suffix: bool = False,
    structured_url_rule: bool = False,
) -> list[str]:
    query_visible = {"input": query.get("input", {}), "context": query.get("context", {}), "state": query.get("state", {})}
    features = []
    if structured_url_rule:
        url_rule = _structured_url_episode_rule(context, query)
        if url_rule:
            features.append(url_rule)
    for path, query_value in flatten(query_visible):
        if not isinstance(query_value, str):
            continue
        positive = []
        negative = []
        for row in context:
            visible = {"input": row.get("input", {}), "context": row.get("context", {}), "state": row.get("state", {})}
            values = dict(flatten(visible))
            value = values.get(path)
            if not isinstance(value, str):
                continue
            (positive if bool(row.get("output")) else negative).append(value)
        url_like = sum(_parsed_url_parts(value) is not None for value in positive + negative)
        if suppress_url_suffix and url_like >= max(2, (len(positive) + len(negative)) // 2):
            continue
        suffix = _common_suffix(positive)
        suffix = suffix.lstrip(":/")
        if len(suffix) < 4 or any(character.isspace() for character in suffix):
            continue
        correct = sum(value.endswith(suffix) for value in positive) + sum(not value.endswith(suffix) for value in negative)
        total = len(positive) + len(negative)
        if total and correct / total >= 0.75:
            features.append(f"{path}#suffix={suffix}|q={int(query_value.endswith(suffix))}|fit={correct}/{total}")
    generic = _generic_episode_rule(context, query)
    if generic:
        features.append(generic)
    return features


def compact_input(
    trace: dict[str, Any],
    semantic_features: bool = False,
    compact_semantic_features: bool = False,
    routed_semantic_features: bool = False,
    canonical_url_slots: bool = False,
) -> str:
    visible = {
        "input": trace.get("input", {}),
        "context": trace.get("context", {}),
        "state": trace.get("state", {}),
    }
    if compact_semantic_features:
        return ",".join(_compact_semantic_projection(visible))
    if routed_semantic_features:
        features = _routed_semantic_projection(visible, canonical_url_slots)
        history = trace.get("history") or []
        for offset, previous in enumerate(reversed(history[-2:]), start=1):
            previous_visible = {
                "input": previous.get("input", {}), "context": previous.get("context", {}), "state": previous.get("state", {})
            }
            features.extend(f"prev{offset}.{feature}" for feature in _routed_semantic_projection(previous_visible, canonical_url_slots))
        return ",".join(features)
    parts = []
    for key, value in flatten(visible):
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        parts.append(f"{key}={rendered}")
    if semantic_features:
        parts.extend(_semantic_projection(visible))
    return ",".join(parts)


def make_prompt(
    context: list[dict[str, Any]],
    query: dict[str, Any],
    include_memory: bool = True,
    semantic_features: bool = False,
    compact_semantic_features: bool = False,
    routed_semantic_features: bool = False,
    episode_rule_features: bool = False,
    suppress_url_suffix: bool = False,
    structured_url_rule: bool = False,
    canonical_url_slots: bool = False,
) -> str:
    if include_memory:
        trace_text = "|".join(
            f"{compact_input(row, semantic_features, compact_semantic_features, routed_semantic_features, canonical_url_slots)}:{int(bool(row['output']))}"
            for row in context
        )
    else:
        trace_text = ""
    mode = "<RSEM>" if routed_semantic_features else "<CSEM>" if compact_semantic_features else "<SEM>" if semantic_features else "<RAW>"
    rule_memory = ";".join(_episode_rule_projection(context, query, suppress_url_suffix, structured_url_rule)) if episode_rule_features else ""
    return f"{mode}<TRACE>{trace_text}<RULEMEM>{rule_memory}<QUERY>{compact_input(query, semantic_features, compact_semantic_features, routed_semantic_features, canonical_url_slots)}<ANSWER>"


def balanced_queries(traces: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    positive = [row for row in traces if bool(row["output"])]
    negative = [row for row in traces if not bool(row["output"])]
    if not positive or not negative:
        return [rng.choice(traces) for _ in range(count)]
    rows = []
    for index in range(count):
        pool = positive if index % 2 == 0 else negative
        rows.append(rng.choice(pool))
    rng.shuffle(rows)
    return rows


def context_for(traces: list[dict[str, Any]], query: dict[str, Any], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates = [row for row in traces if row is not query]
    positive = [row for row in candidates if bool(row["output"])]
    negative = [row for row in candidates if not bool(row["output"])]
    selected = []
    while len(selected) < min(limit, len(candidates)):
        pool = positive if len(selected) % 2 == 0 else negative
        pool = [row for row in pool if row not in selected]
        if not pool:
            pool = [row for row in candidates if row not in selected]
        if not pool:
            break
        selected.append(rng.choice(pool))
    rng.shuffle(selected)
    return selected


@dataclass
class Example:
    prompt: str
    label: int
    family: str
    record_id: str
    intended_label: int | None = None


def records_to_examples(
    records: list[dict[str, Any]],
    rng: random.Random,
    examples_per_program: int,
    memory_items: int,
    semantic_features: bool = False,
    compact_semantic_features: bool = False,
    routed_semantic_features: bool = False,
    episode_rule_features: bool = False,
    suppress_url_suffix: bool = False,
    structured_url_rule: bool = False,
    canonical_url_slots: bool = False,
    meta_label_permutation: bool = False,
    permutation_seed: int = 0,
) -> list[Example]:
    examples = []
    for record in records:
        traces = record["modalities"]["trace"]
        digest = hashlib.sha256(f"{record['record_id']}:{permutation_seed}".encode("utf-8")).digest()
        flip_labels = meta_label_permutation and bool(digest[0] & 1)
        for query in balanced_queries(traces, examples_per_program, rng):
            context = context_for(traces, query, memory_items, rng)
            visible_context = (
                [{**row, "output": not bool(row["output"])} for row in context]
                if flip_labels
                else context
            )
            examples.append(Example(
                prompt=make_prompt(
                    visible_context,
                    query,
                    include_memory=True,
                    semantic_features=semantic_features,
                    compact_semantic_features=compact_semantic_features,
                    routed_semantic_features=routed_semantic_features,
                    episode_rule_features=episode_rule_features,
                    suppress_url_suffix=suppress_url_suffix,
                    structured_url_rule=structured_url_rule,
                    canonical_url_slots=canonical_url_slots,
                ),
                label=int(not bool(query["output"]) if flip_labels else bool(query["output"])),
                family=record["family"],
                record_id=record["record_id"],
                intended_label=int(bool(query.get("intended_output", query["output"]))),
            ))
    rng.shuffle(examples)
    return examples


def stratified_iid_split(
    records: list[dict[str, Any]],
    validation_stride: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split within each family so record ordering cannot create family leakage."""
    if validation_stride < 2:
        raise ValueError("validation_stride must be at least 2")
    offsets: dict[str, int] = {}
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for record in records:
        family = str(record["family"])
        family_index = offsets.get(family, 0)
        offsets[family] = family_index + 1
        (validation if family_index % validation_stride == 0 else train).append(record)
    return train, validation


def records_to_counterexample_examples(
    records: list[dict[str, Any]],
    rng: random.Random,
    memory_items: int,
    semantic_features: bool,
    compact_semantic_features: bool = False,
    routed_semantic_features: bool = False,
    episode_rule_features: bool = False,
    canonical_url_slots: bool = False,
) -> list[Example]:
    examples = []
    for record in records:
        traces = record["modalities"]["trace"]
        for query in traces:
            context = context_for(traces, query, memory_items, rng)
            examples.append(Example(
                prompt=make_prompt(
                    context,
                    query,
                    include_memory=True,
                    semantic_features=semantic_features,
                    compact_semantic_features=compact_semantic_features,
                    routed_semantic_features=routed_semantic_features,
                    episode_rule_features=episode_rule_features,
                    canonical_url_slots=canonical_url_slots,
                ),
                label=int(bool(query["output"])),
                intended_label=int(bool(query["intended_output"])),
                family=record["family"],
                record_id=record["record_id"],
            ))
    return examples


def encode(text: str, max_length: int) -> list[int]:
    tokens = [byte + 1 for byte in text.encode("utf-8", errors="replace")]
    tokens = tokens[-(max_length - 1):] + [EOS]
    return tokens


class PromptDataset(Dataset):
    def __init__(self, examples: list[Example], max_length: int):
        self.examples = examples
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        tokens = encode(example.prompt, self.max_length)
        return {"tokens": tokens, "label": example.label, "example": example}


def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(len(row["tokens"]) for row in rows)
    tokens = torch.full((len(rows), width), PAD, dtype=torch.long)
    lengths = torch.zeros(len(rows), dtype=torch.long)
    for index, row in enumerate(rows):
        values = torch.tensor(row["tokens"], dtype=torch.long)
        tokens[index, : len(values)] = values
        lengths[index] = len(values)
    return {
        "tokens": tokens,
        "lengths": lengths,
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
        "examples": [row["example"] for row in rows],
    }


class TinyRuleGPT(nn.Module):
    def __init__(self, max_length: int, hidden: int, layers: int, heads: int):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, hidden, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_length, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, width = tokens.shape
        positions = torch.arange(width, device=tokens.device).unsqueeze(0).expand(batch, width)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal = torch.triu(torch.ones(width, width, device=tokens.device, dtype=torch.bool), diagonal=1)
        padding = tokens.eq(PAD)
        hidden = self.blocks(hidden, mask=causal, src_key_padding_mask=padding)
        last = hidden[torch.arange(batch, device=tokens.device), lengths - 1]
        return self.classifier(self.norm(last))


def _rule_memory_prediction(
    prompt: str,
    confidence_arbitration: bool = False,
    abstain_on_rule_conflict: bool = False,
) -> int | None:
    if confidence_arbitration:
        memory = prompt.split("<RULEMEM>", 1)[1].split("<QUERY>", 1)[0] if "<RULEMEM>" in prompt else ""
        candidates = []
        for index, segment in enumerate(memory.split(";")):
            match = re.search(r"\|q=([01])\|fit=(\d+)/(\d+)", segment)
            if not match:
                continue
            correct, total = int(match.group(2)), int(match.group(3))
            if not total or correct / total < 0.75:
                continue
            if "|kind=url_hostname" in segment:
                priority = 400
            elif "|kind=url_scheme" in segment:
                priority = 320
            elif "|kind=url_port" in segment:
                priority = 300
            elif segment.startswith("generic="):
                priority = 220
            elif "#suffix=" in segment:
                priority = 100
            else:
                priority = 150
            candidates.append((priority, correct / total, -index, int(match.group(1))))
        if not candidates:
            return None
        if abstain_on_rule_conflict:
            best_fit = max(candidate[1] for candidate in candidates)
            empirically_best = [candidate for candidate in candidates if abs(candidate[1] - best_fit) < 1e-12]
            if len({candidate[3] for candidate in empirically_best}) > 1:
                return None
            return max(empirically_best)[3]
        return max(candidates)[3]
    match = re.search(r"<RULEMEM>[^<]*\|q=([01])\|fit=(\d+)/(\d+)", prompt)
    if not match:
        return None
    correct, total = int(match.group(2)), int(match.group(3))
    return int(match.group(1)) if total and correct / total >= 0.75 else None


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    drop_memory: bool = False,
    max_length: int = 384,
    rule_memory_gate: bool = False,
    confidence_arbitration: bool = False,
    abstain_on_rule_conflict: bool = False,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    failures = []
    family_totals: dict[str, list[int]] = {}
    for batch in loader:
        if drop_memory:
            examples = batch["examples"]
            token_rows = []
            for item in examples:
                mode = "<RSEM>" if item.prompt.startswith("<RSEM>") else "<CSEM>" if item.prompt.startswith("<CSEM>") else "<SEM>" if item.prompt.startswith("<SEM>") else "<RAW>"
                query_text = item.prompt.split("<QUERY>", 1)[1].split("<ANSWER>", 1)[0]
                token_rows.append(encode(f"{mode}<TRACE><RULEMEM><QUERY>{query_text}<ANSWER>", max_length))
            width = max(len(row) for row in token_rows)
            tokens = torch.full((len(token_rows), width), PAD, dtype=torch.long)
            lengths = torch.tensor([len(row) for row in token_rows], dtype=torch.long)
            for index, row in enumerate(token_rows):
                tokens[index, :len(row)] = torch.tensor(row, dtype=torch.long)
        else:
            tokens = batch["tokens"]
            lengths = batch["lengths"]
        labels = batch["labels"].to(device)
        logits = model(tokens.to(device), lengths.to(device))
        predictions = logits.argmax(dim=-1)
        if rule_memory_gate and not drop_memory:
            for index, item in enumerate(batch["examples"]):
                memory_prediction = _rule_memory_prediction(item.prompt, confidence_arbitration, abstain_on_rule_conflict)
                if memory_prediction is not None:
                    predictions[index] = memory_prediction
        matches = predictions.eq(labels)
        correct += int(matches.sum())
        total += len(labels)
        for item, predicted, expected, matched in zip(batch["examples"], predictions.cpu(), labels.cpu(), matches.cpu()):
            stats = family_totals.setdefault(item.family, [0, 0])
            stats[1] += 1
            stats[0] += int(matched)
            if not matched and len(failures) < 12:
                failures.append({
                    "family": item.family,
                    "record_id": item.record_id,
                    "expected": int(expected),
                    "predicted": int(predicted),
                    "prompt": item.prompt[:600],
                })
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "correct": correct,
        "total": total,
        "by_family": {family: round(good / count, 6) for family, (good, count) in family_totals.items()},
        "failures": failures,
    }


@torch.inference_mode()
def evaluate_counterexample_at_k(
    model: nn.Module,
    records: list[dict[str, Any]],
    device: torch.device,
    max_length: int,
    memory_items: int,
    semantic_features: bool,
    compact_semantic_features: bool,
    routed_semantic_features: bool,
    episode_rule_features: bool,
    rule_memory_gate: bool,
    seed: int,
    k: int = 10,
    batch_size: int = 128,
    canonical_url_slots: bool = False,
) -> dict[str, Any]:
    examples = records_to_counterexample_examples(
        records,
        random.Random(seed + 971),
        memory_items,
        semantic_features,
        compact_semantic_features,
        routed_semantic_features,
        episode_rule_features,
        canonical_url_slots,
    )
    loader = DataLoader(PromptDataset(examples, max_length), batch_size=batch_size, shuffle=False, collate_fn=collate)
    grouped: dict[str, list[dict[str, Any]]] = {}
    model.eval()
    for batch in loader:
        logits = model(batch["tokens"].to(device), batch["lengths"].to(device))
        probabilities = torch.softmax(logits, dim=-1).cpu()
        if rule_memory_gate:
            for index, item in enumerate(batch["examples"]):
                memory_prediction = _rule_memory_prediction(item.prompt)
                if memory_prediction is not None:
                    probabilities[index, :] = 0.0
                    probabilities[index, memory_prediction] = 1.0
        for item, probability in zip(batch["examples"], probabilities):
            intended = int(item.intended_label or 0)
            grouped.setdefault(item.record_id, []).append({
                "family": item.family,
                "score": float(probability[1 - intended]),
                "actual_counterexample": item.label != intended,
            })

    family_totals: dict[str, list[int]] = {}
    successes = 0
    top1_successes = 0
    precision_sum = 0.0
    random_precision_sum = 0.0
    random_success_sum = 0.0
    for rows in grouped.values():
        ranked = sorted(rows, key=lambda row: row["score"], reverse=True)[:k]
        success = any(row["actual_counterexample"] for row in ranked)
        counterexample_count = sum(int(row["actual_counterexample"]) for row in rows)
        draw_count = min(k, len(rows))
        precision_sum += sum(int(row["actual_counterexample"]) for row in ranked) / max(1, draw_count)
        random_precision_sum += counterexample_count / max(1, len(rows))
        if draw_count >= len(rows) - counterexample_count + 1:
            random_success_sum += 1.0
        else:
            random_success_sum += 1.0 - math.comb(len(rows) - counterexample_count, draw_count) / math.comb(len(rows), draw_count)
        top1_successes += int(bool(ranked and ranked[0]["actual_counterexample"]))
        family = rows[0]["family"]
        stats = family_totals.setdefault(family, [0, 0])
        stats[0] += int(success)
        stats[1] += 1
        successes += int(success)
    total = len(grouped)
    top1 = top1_successes / total if total else 0.0
    random_top1 = random_precision_sum / total if total else 0.0
    return {
        "k": k,
        "score": round(successes / total, 6) if total else 0.0,
        "successful_programs": successes,
        "total_programs": total,
        "by_family": {family: round(good / count, 6) for family, (good, count) in family_totals.items()},
        "top1": round(top1, 6),
        "random_top1": round(random_top1, 6),
        "top1_lift": round(top1 - random_top1, 6),
        "precision_at_k": round(precision_sum / total, 6) if total else 0.0,
        "random_precision_at_k": round(random_precision_sum / total, 6) if total else 0.0,
        "random_success_at_k": round(random_success_sum / total, 6) if total else 0.0,
        "metric_warning": "Counterexample@K is non-discriminative when the random success baseline is near one.",
        "policy_visibility": "declared intended output is visible only to the verifier, not to the behavior model",
    }


def _query_from_prompt(prompt: str) -> dict[str, Any]:
    query = prompt.split("<QUERY>", 1)[1].split("<ANSWER>", 1)[0]
    values: dict[str, Any] = {}
    for part in query.split(","):
        if not part or "=" not in part:
            continue
        path, raw = part.split("=", 1)
        if raw == "true":
            value: Any = True
        elif raw == "false":
            value = False
        else:
            try:
                value = int(raw)
            except ValueError:
                value = raw
        set_nested(values, path, value)
    return values


def set_nested(root: dict[str, Any], path: str, value: Any) -> None:
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny causal Transformer on black-box rule traces.")
    parser.add_argument("--programs", type=int, default=1200)
    parser.add_argument("--examples-per-program", type=int, default=4)
    parser.add_argument("--memory-items", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--traces-per-program", type=int, default=12)
    parser.add_argument("--semantic-features", action="store_true")
    parser.add_argument("--compact-semantic-features", action="store_true")
    parser.add_argument("--routed-semantic-features", action="store_true")
    parser.add_argument("--canonical-url-slots", action="store_true")
    parser.add_argument("--episode-rule-features", action="store_true")
    parser.add_argument("--rule-memory-gate", action="store_true")
    parser.add_argument("--meta-label-permutation", action="store_true")
    parser.add_argument("--train-families", default="")
    parser.add_argument("--test-families", default="")
    parser.add_argument("--counterexample-k", type=int, default=10)
    parser.add_argument("--experiment-name", default="tiny-rule-gpt-family-holdout-v1")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rule-memory-pilot"))
    parser.add_argument("--report-name", default="tiny_rule_gpt_experiment.json")
    args = parser.parse_args()
    if sum(int(value) for value in (args.semantic_features, args.compact_semantic_features, args.routed_semantic_features)) > 1:
        parser.error("semantic feature serialization modes are mutually exclusive")
    if args.rule_memory_gate and not args.episode_rule_features:
        parser.error("--rule-memory-gate requires --episode-rule-features")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = random.Random(args.seed)
    records = generate_curriculum(args.programs, traces_per_program=args.traces_per_program, seed=args.seed)
    requested_train_families = {item.strip() for item in args.train_families.split(",") if item.strip()}
    requested_test_families = {item.strip() for item in args.test_families.split(",") if item.strip()}
    if requested_train_families:
        all_train_records = [row for row in records if row["family"] in requested_train_families]
        train_records, iid_records = stratified_iid_split(all_train_records)
        validation_records = iid_records
    else:
        source_train_records = [row for row in records if row["split"] == "train"]
        validation_records = [row for row in records if row["split"] == "validation"]
        train_records, iid_records = stratified_iid_split(source_train_records)
    test_records = (
        [row for row in records if row["family"] in requested_test_families]
        if requested_test_families
        else [row for row in records if row["split"] == "test"]
    )

    train_examples = records_to_examples(
        train_records,
        rng,
        args.examples_per_program,
        args.memory_items,
        semantic_features=args.semantic_features,
        compact_semantic_features=args.compact_semantic_features,
        routed_semantic_features=args.routed_semantic_features,
        episode_rule_features=args.episode_rule_features,
        canonical_url_slots=args.canonical_url_slots,
        meta_label_permutation=args.meta_label_permutation,
        permutation_seed=args.seed,
    )
    example_options = {
        "semantic_features": args.semantic_features,
        "compact_semantic_features": args.compact_semantic_features,
        "routed_semantic_features": args.routed_semantic_features,
        "episode_rule_features": args.episode_rule_features,
        "canonical_url_slots": args.canonical_url_slots,
    }
    iid_examples = records_to_examples(iid_records, rng, args.examples_per_program, args.memory_items, **example_options)
    validation_examples = records_to_examples(validation_records, rng, args.examples_per_program, args.memory_items, **example_options)
    test_examples = records_to_examples(test_records, rng, args.examples_per_program, args.memory_items, **example_options)

    train_loader = DataLoader(PromptDataset(train_examples, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    iid_loader = DataLoader(PromptDataset(iid_examples, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    validation_loader = DataLoader(PromptDataset(validation_examples, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(PromptDataset(test_examples, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRuleGPT(args.max_length, args.hidden, args.layers, args.heads).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    started = time.perf_counter()
    history = []
    best_score = -math.inf
    best_state = None
    emit_event(
        actor="trainer",
        tool="tiny-rule-gpt.train",
        phase="training",
        status="running",
        message="启动 Tiny Rule GPT 训练",
        payload={"programs": args.programs, "epochs": args.epochs, "memory_items": args.memory_items, "parameters": parameter_count, "semantic_features": args.semantic_features, "compact_semantic_features": args.compact_semantic_features, "routed_semantic_features": args.routed_semantic_features, "episode_rule_features": args.episode_rule_features, "meta_label_permutation": args.meta_label_permutation, "seed": args.seed},
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tokens, lengths)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        iid = evaluate(model, iid_loader, device, max_length=args.max_length, rule_memory_gate=args.rule_memory_gate)
        family_validation = evaluate(model, validation_loader, device, max_length=args.max_length, rule_memory_gate=args.rule_memory_gate)
        score = iid["accuracy"] + family_validation["accuracy"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        row = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, seen), 6),
            "iid_accuracy": iid["accuracy"],
            "validation_family_accuracy": family_validation["accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        emit_event(
            actor="trainer",
            tool="tiny-rule-gpt.epoch",
            phase="training",
            status="running" if epoch < args.epochs else "complete",
            message=f"Epoch {epoch}/{args.epochs}",
            payload=row,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    iid = evaluate(model, iid_loader, device, max_length=args.max_length, rule_memory_gate=args.rule_memory_gate)
    validation = evaluate(model, validation_loader, device, max_length=args.max_length, rule_memory_gate=args.rule_memory_gate)
    test_with_memory = evaluate(model, test_loader, device, max_length=args.max_length, rule_memory_gate=args.rule_memory_gate)
    test_without_memory = evaluate(model, test_loader, device, drop_memory=True, max_length=args.max_length, rule_memory_gate=args.rule_memory_gate)
    counterexample_at_k = evaluate_counterexample_at_k(
        model,
        test_records,
        device,
        args.max_length,
        args.memory_items,
        args.semantic_features,
        args.compact_semantic_features,
        args.routed_semantic_features,
        args.episode_rule_features,
        args.rule_memory_gate,
        args.seed,
        k=args.counterexample_k,
        batch_size=args.batch_size,
        canonical_url_slots=args.canonical_url_slots,
    )
    train_positive_rate = sum(item.label for item in train_examples) / len(train_examples)
    majority_accuracy = max(train_positive_rate, 1.0 - train_positive_rate)

    elapsed = time.perf_counter() - started
    report = {
        "experiment": args.experiment_name,
        "status": "completed",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "model": {
            "type": "causal_transformer_classifier",
            "parameters": parameter_count,
            "hidden": args.hidden,
            "layers": args.layers,
            "heads": args.heads,
            "max_length": args.max_length,
            "semantic_features": args.semantic_features,
            "compact_semantic_features": args.compact_semantic_features,
            "routed_semantic_features": args.routed_semantic_features,
            "canonical_url_slots": args.canonical_url_slots,
            "episode_rule_features": args.episode_rule_features,
            "rule_memory_gate": args.rule_memory_gate,
        },
        "data": {
            "programs": args.programs,
            "train_examples": len(train_examples),
            "iid_examples": len(iid_examples),
            "validation_family_examples": len(validation_examples),
            "test_family_examples": len(test_examples),
            "train_families": sorted({item.family for item in train_examples}),
            "validation_families": sorted({item.family for item in validation_examples}),
            "test_families": sorted({item.family for item in test_examples}),
            "source_or_rule_ir_visible_to_model": False,
            "feature_mode": "routed_semantic_projection" if args.routed_semantic_features else "compact_semantic_projection" if args.compact_semantic_features else "verbose_semantic_projection" if args.semantic_features else "raw_trace_text",
            "meta_label_permutation_in_training_only": args.meta_label_permutation,
        },
        "baselines": {"balanced_majority_accuracy": round(majority_accuracy, 6)},
        "results": {
            "iid": iid,
            "validation_family_holdout": validation,
            "test_family_with_memory": test_with_memory,
            "test_family_without_memory": test_without_memory,
            "counterexample_at_k": counterexample_at_k,
            "memory_delta": round(test_with_memory["accuracy"] - test_without_memory["accuracy"], 6),
        },
        "training": {"epochs": args.epochs, "seconds": round(elapsed, 3), "history": history},
        "research_guardrails": {
            "family_holdout": True,
            "black_box_only_input": True,
            "failure_examples_preserved": True,
            "claim_scope": "pilot only; no claim of production generalization",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": vars(args)}, args.output_dir / "tiny_rule_gpt.pt")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    research_report = PROJECT_ROOT / "research" / args.report_name
    research_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit_event(
        actor="trainer",
        tool="tiny-rule-gpt.evaluate",
        phase="family-holdout",
        status="complete",
        message=f"族外准确率 {test_with_memory['accuracy']:.2%}",
        payload={"test_with_memory": test_with_memory["accuracy"], "test_without_memory": test_without_memory["accuracy"], "memory_delta": report["results"]["memory_delta"], "counterexample_at_k": counterexample_at_k["score"], "semantic_features": args.semantic_features, "compact_semantic_features": args.compact_semantic_features, "routed_semantic_features": args.routed_semantic_features, "episode_rule_features": args.episode_rule_features, "seed": args.seed},
        artifact=str(research_report.relative_to(PROJECT_ROOT)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
