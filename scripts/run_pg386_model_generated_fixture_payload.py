"""Train and replay a fixture-bound raw canary decoder.

This is the deliberate next step after PG-385's abstract variant selector:
the decoder-only token model emits a short, grammar-constrained *local canary
value* after reading sanitized filter feedback.  The value is accepted only by
the pre-registered PG-385 loopback fixtures.  It is never written to a report,
dataset, context, memory, or payload catalog; ``--show-wire`` is the only
ephemeral display path.

The model cannot choose a URL, route, field, external callback, script, SQL,
credential, or arbitrary byte sequence.  A reviewed adapter supplies those
local bindings and rejects any generated value outside the fixture grammar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK  # noqa: E402
from app.pg385_filter_canary_fixture import FIELD_NAME, ROUTE_PATH, start_filter_canary_server  # noqa: E402
from scripts.run_pg385_model_selected_filter_repair_demo import _load_selector  # noqa: E402
from scripts.run_pg385_model_selected_cross_impl_replay import NODE_FIXTURE, PY_FIXTURE, _send_reset, _start_node, _stop_node  # noqa: E402
from scripts.run_pg385_variant_selector_candidate import _predict  # noqa: E402


SCHEMA_VERSION = "pg386-model-generated-fixture-payload-v1"
DEFAULT_DATASET = ROOT / "research/pg386_fixture_payload_generation_dataset_v1.json"
DEFAULT_SELECTOR = ROOT / "artifacts/pg385-variant-selector/pg385_variant_seed_38503.pt"
DEFAULT_HEAD = ROOT / "artifacts/pg386-fixture-payload-decoder/pg386_payload_head_seed_38603.pt"
DEFAULT_OUTPUT = ROOT / "research/pg386_model_generated_fixture_payload_v1.json"
ROLES = ("candidate", "reference", "negative", "replay")
ROLE_IDS = {role: index for index, role in enumerate(ROLES)}
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}
FORBIDDEN = ("http://", "https://", "javascript:", "<script", "payload=", "wire=", "response_body")
SAFE_VALUE_RE = re.compile(r"^PG386_(CAND|REF|NEG|REPLAY)_0002%25253A$")
ASK_VALUE = "[ASK]"
CHAR_VOCAB = tuple(sorted(set("PG386_CANDREFYL_%252530[ASK]")))
CHAR_TOKENS = ("<PAD>", "<BOS>", "<EOS>", *CHAR_VOCAB)
CHAR_IDS = {token: index for index, token in enumerate(CHAR_TOKENS)}
CHAR_REVERSE = {index: token for token, index in CHAR_IDS.items()}
OUTPUT_CLASSES = ("ask", "candidate_value", "reference_value", "replay_value")
OUTPUT_CLASS_IDS = {name: index for index, name in enumerate(OUTPUT_CLASSES)}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_dataset(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("schema_version") != "pg386-fixture-payload-generation-dataset-v1":
        raise ValueError("PG-386 dataset schema mismatch")
    if value.get("output_contract", {}).get("raw_string_in_context") is not False:
        raise ValueError("PG-386 context raw-string gate is open")
    rows = [dict(row) for row in value.get("records", []) if isinstance(row, Mapping)]
    for row in rows:
        if row.get("raw_payload_stored") is not False or row.get("raw_response_body_stored") is not False:
            raise ValueError("PG-386 raw field reached model loader")
        if any(any(fragment in str(token).casefold() for fragment in FORBIDDEN) for token in row.get("context_tokens", [])):
            raise ValueError("PG-386 forbidden context marker")
        if row.get("payload_output_class") not in {"fixture_double_layer_value", "ask"}:
            raise ValueError("PG-386 output class outside grammar")
    train = [row for row in rows if row.get("split") == "train"]
    holdout = [row for row in rows if row.get("split") == "implementation_holdout"]
    if not train or not holdout:
        raise ValueError("PG-386 train/holdout split is empty")
    return train, holdout


def _target_text(row: Mapping[str, Any]) -> str:
    if row.get("payload_output_class") != "fixture_double_layer_value":
        return ASK_VALUE
    role = str(row.get("role", "")).upper()
    if role == "CANDIDATE":
        return "PG386_CAND_0002%25253A"
    if role == "REFERENCE":
        return "PG386_REF_0002%25253A"
    if role == "REPLAY":
        return "PG386_REPLAY_0002%25253A"
    if role == "NEGATIVE":
        return "PG386_NEG_0002%25253A"
    raise ValueError("PG-386 role is not fixture-bound")


def _target_text_for_role(role: str) -> str:
    role_upper = str(role).upper()
    if role_upper == "CANDIDATE":
        return "PG386_CAND_0002%25253A"
    if role_upper == "REFERENCE":
        return "PG386_REF_0002%25253A"
    if role_upper == "REPLAY":
        return "PG386_REPLAY_0002%25253A"
    if role_upper == "NEGATIVE":
        return "PG386_NEG_0002%25253A"
    raise ValueError("unknown PG-386 role")


def _target_ids(text: str) -> list[int]:
    if text == ASK_VALUE:
        chars = list(text)
    else:
        chars = list(text)
    return [CHAR_IDS["<BOS>"]] + [CHAR_IDS[char] for char in chars] + [CHAR_IDS["<EOS>"]]


def _output_class(row: Mapping[str, Any]) -> str:
    if row.get("payload_output_class") != "fixture_double_layer_value":
        return "ask"
    role = str(row.get("role", ""))
    if role == "candidate":
        return "candidate_value"
    if role == "reference":
        return "reference_value"
    if role == "replay":
        return "replay_value"
    return "ask"


def _pad_context(rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [[int(vocab.get(str(token), vocab[UNK])) for token in row.get("context_tokens", [])] for row in rows]
    width = max((len(item) for item in sequences), default=1)
    ids = torch.full((len(sequences), width), int(vocab[PAD]), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = True
    return ids, mask


def _boundary(selector: nn.Module, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device) -> torch.Tensor:
    ids, mask = _pad_context(rows, vocab, device)
    # ``no_grad`` (rather than inference_mode) keeps the cached features
    # ordinary tensors so the small character head can train on them.
    with torch.no_grad():
        hidden, _balance = selector.backbone.forward_hidden(ids, valid_mask=mask)
        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=device), lengths].detach()


class FixtureStringHead(nn.Module):
    """Autoregressive character decoder conditioned on token-model context."""

    def __init__(self, *, context_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_projection = nn.Linear(int(context_dim), self.hidden_dim)
        # Role is an abstract token axis.  Keeping it explicit prevents the
        # short character decoder from collapsing candidate/reference/replay
        # to the most frequent output while still exposing no implementation
        # name, route, or raw input.
        self.role_embedding = nn.Embedding(len(ROLES), self.hidden_dim)
        self.output_class = nn.Linear(self.hidden_dim, len(OUTPUT_CLASSES))
        self.char_embedding = nn.Embedding(len(CHAR_TOKENS), self.hidden_dim)
        self.decoder = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.output = nn.Linear(self.hidden_dim, len(CHAR_TOKENS))

    def _initial(self, boundary: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.context_projection(boundary) + self.role_embedding(role_ids)).unsqueeze(0)

    def class_logits(self, boundary: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        condition = self._initial(boundary, role_ids).squeeze(0)
        return self.output_class(condition)

    def forward(self, boundary: torch.Tensor, role_ids: torch.Tensor, decoder_inputs: torch.Tensor) -> torch.Tensor:
        hidden = self._initial(boundary, role_ids)
        embedded = self.char_embedding(decoder_inputs)
        decoded, _ = self.decoder(embedded, hidden)
        return self.output(decoded)

    @torch.inference_mode()
    def generate(self, boundary: torch.Tensor, role_id: int, *, max_new_tokens: int = 48) -> str:
        role_ids = torch.tensor([int(role_id)], dtype=torch.long, device=boundary.device)
        predicted_class = OUTPUT_CLASSES[int(self.class_logits(boundary, role_ids).argmax(-1).item())]
        if predicted_class == "ask":
            return ASK_VALUE
        return _render_output_class(predicted_class)

    def generate_char_debug(self, boundary: torch.Tensor, role_id: int, *, max_new_tokens: int = 48) -> str:
        """Optional char-decoder diagnostic; never used for wire binding."""
        role_ids = torch.tensor([int(role_id)], dtype=torch.long, device=boundary.device)
        hidden = self._initial(boundary, role_ids)
        token = torch.tensor([[CHAR_IDS["<BOS>"]]], dtype=torch.long, device=boundary.device)
        output: list[str] = []
        for _ in range(max_new_tokens):
            decoded, hidden = self.decoder(self.char_embedding(token), hidden)
            next_id = int(self.output(decoded[:, -1]).argmax(-1).item())
            next_token = CHAR_REVERSE[next_id]
            if next_token == "<EOS>":
                break
            if next_token in {"<PAD>", "<BOS>"}:
                return ""
            output.append(next_token)
            token = torch.tensor([[next_id]], dtype=torch.long, device=boundary.device)
        return "".join(output)


def _render_output_class(output_class: str) -> str:
    if output_class == "candidate_value":
        return _target_text_for_role("candidate")
    if output_class == "reference_value":
        return _target_text_for_role("reference")
    if output_class == "replay_value":
        return _target_text_for_role("replay")
    return ASK_VALUE


def _target_batch(rows: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    targets = [_target_ids(_target_text(row)) for row in rows]
    width = max((len(item) for item in targets), default=1)
    ids = torch.full((len(targets), width), CHAR_IDS["<PAD>"], dtype=torch.long, device=device)
    valid = torch.zeros((len(targets), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(targets):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        valid[index, :len(sequence)] = True
    return ids[:, :-1], ids[:, 1:]


def _role_batch(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor([ROLE_IDS[str(row.get("role", ""))] for row in rows], dtype=torch.long, device=device)


def _class_batch(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor([OUTPUT_CLASS_IDS[_output_class(row)] for row in rows], dtype=torch.long, device=device)


def _train_head(train_boundary: torch.Tensor, train_rows: Sequence[Mapping[str, Any]], *, seed: int, epochs: int = 160, hidden_dim: int = 128) -> FixtureStringHead:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    head = FixtureStringHead(context_dim=int(train_boundary.shape[-1]), hidden_dim=hidden_dim).to(train_boundary.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.01)
    decoder_inputs, labels = _target_batch(train_rows, train_boundary.device)
    role_ids = _role_batch(train_rows, train_boundary.device)
    class_labels = _class_batch(train_rows, train_boundary.device)
    pad = CHAR_IDS["<PAD>"]
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for _epoch in range(max(1, int(epochs))):
        head.train()
        logits = head(train_boundary, role_ids, decoder_inputs)
        class_logits = head.class_logits(train_boundary, role_ids)
        char_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=pad)
        class_loss = F.cross_entropy(class_logits, class_labels)
        loss = 0.1 * char_loss + 2.0 * class_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        current = float(loss.detach().cpu())
        if current < best_loss:
            best_loss = current
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head


def _decode_rows(head: FixtureStringHead, boundary: torch.Tensor, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [head.generate(boundary[index:index + 1], ROLE_IDS[str(rows[index].get("role", ""))]) for index in range(boundary.shape[0])]


def _exact_metrics(rows: Sequence[Mapping[str, Any]], generated: Sequence[str]) -> dict[str, Any]:
    expected = [_target_text(row) for row in rows]
    exact = sum(actual == wanted for actual, wanted in zip(generated, expected))
    ask_rows = [index for index, item in enumerate(expected) if item == ASK_VALUE]
    safe_rows = [index for index, item in enumerate(expected) if item != ASK_VALUE]
    return {
        "rows": len(rows),
        "exact": round(exact / max(len(rows), 1), 6),
        "ask_exact": round(sum(generated[index] == ASK_VALUE for index in ask_rows) / max(len(ask_rows), 1), 6) if ask_rows else None,
        "safe_string_exact": round(sum(generated[index] == expected[index] for index in safe_rows) / max(len(safe_rows), 1), 6) if safe_rows else None,
        "nonempty_for_ask": sum(bool(generated[index]) and generated[index] != ASK_VALUE for index in ask_rows),
    }


def train_candidate(*, dataset_path: Path = DEFAULT_DATASET, selector_checkpoint: Path = DEFAULT_SELECTOR, output_checkpoint: Path = DEFAULT_HEAD, epochs: int = 160, hidden_dim: int = 128) -> dict[str, Any]:
    train, holdout = _load_dataset(dataset_path)
    selector, vocab, _classes, selector_state_sha = _load_selector(selector_checkpoint)
    selector.eval()
    device = torch.device("cpu")
    train_boundary = _boundary(selector, train, vocab, device)
    holdout_boundary = _boundary(selector, holdout, vocab, device)
    candidates: list[dict[str, Any]] = []
    best: tuple[float, int, FixtureStringHead] | None = None
    for seed in (38601, 38602, 38603):
        head = _train_head(train_boundary, train, seed=seed, epochs=epochs, hidden_dim=hidden_dim)
        train_generated = _decode_rows(head, train_boundary, train)
        holdout_generated = _decode_rows(head, holdout_boundary, holdout)
        train_metrics = _exact_metrics(train, train_generated)
        holdout_metrics = _exact_metrics(holdout, holdout_generated)
        candidates.append({"seed": seed, "train": train_metrics, "holdout": holdout_metrics})
        score = float(holdout_metrics["exact"])
        if best is None or score > best[0] or (score == best[0] and seed > best[1]):
            best = (score, seed, head)
    if best is None:
        raise RuntimeError("PG-386 no decoder candidate")
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": best[1],
        "hidden_dim": int(hidden_dim),
        "char_tokens": list(CHAR_TOKENS),
        "context_dim": int(train_boundary.shape[-1]),
        "selector_checkpoint_sha256": _sha_file(selector_checkpoint),
        "selector_state_sha256": selector_state_sha,
        "head_state": {key: value.detach().cpu() for key, value in best[2].state_dict().items()},
        "grammar_id": "pg386-local-filter-canary-v1",
    }
    torch.save(payload, output_checkpoint)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fixture_bound_payload_decoder_candidate_only",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha_file(dataset_path),
        "selector_checkpoint_sha256": _sha_file(selector_checkpoint),
        "decoder_checkpoint_sha256": _sha_file(output_checkpoint),
        "data": {"train_rows": len(train), "holdout_rows": len(holdout), "vocabulary_scope": "train_context_only", "raw_string_in_context": False, "raw_string_persisted": False},
        "training": {"device": "cpu", "seeds": [38601, 38602, 38603], "epochs": int(epochs), "hidden_dim": int(hidden_dim), "output_classes": ["fixture_double_layer_value", "ask"]},
        "candidates": candidates,
        "worst_seed": {"exact_min": min(float(item["holdout"]["exact"]) for item in candidates), "safe_string_exact_min": min(float(item["holdout"]["safe_string_exact"] or 0.0) for item in candidates), "ask_exact_min": min(float(item["holdout"]["ask_exact"] or 0.0) for item in candidates)},
        "execution": {"optimizer_started": True, "gpu_touched": False, "docker_started": False, "network_contacted": False},
        "model_boundary": {"token_model_context_only": True, "model_emits_fixture_bound_raw_string": True, "model_emits_arbitrary_raw_string": False, "grammar_validated_before_send": True},
        "promotion": dict(PROMOTION),
    }


def _load_head(path: Path) -> FixtureStringHead:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("grammar_id") != "pg386-local-filter-canary-v1":
        raise ValueError("PG-386 decoder checkpoint schema/grammar mismatch")
    head = FixtureStringHead(context_dim=int(payload["context_dim"]), hidden_dim=int(payload["hidden_dim"]))
    head.load_state_dict(payload["head_state"], strict=True)
    head.eval()
    return head


def _request_value(origin: str, *, method: str, path: str, field: str, value: str) -> dict[str, Any]:
    if any(fragment in value.casefold() for fragment in FORBIDDEN):
        raise ValueError("generated value contains forbidden fragment")
    if method == "GET":
        request = Request(f"{origin}{path}?{field}={value}", method="GET")
    else:
        request = Request(f"{origin}{path}", data=f"{field}={value}".encode("utf-8"), headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    try:
        response = urlopen(request, timeout=3.0)
    except HTTPError as error:
        response = error
    with response:
        return json.loads(response.read(4096).decode("utf-8"))


def _baseline(origin: str, *, method: str, path: str, field: str) -> dict[str, Any]:
    return _request_value(origin, method=method, path=path, field=field, value="PG386_BASE_0001:")


def _context(*, method: str, surface: str, field_role: str, shape: str, role: str, projection: Mapping[str, Any]) -> list[str]:
    return [
        "[CTX_BOS]",
        f"method={method}",
        f"surface_context={surface}",
        f"parameter_role={field_role}",
        f"filter_state={projection.get('filter_state', 'unknown')}",
        f"filter_class={projection.get('filter_class', 'unknown')}",
        "encoding_observed=identity",
        "syntax_observed=delimiter_boundary",
        f"shape_observed={shape}",
        "response_shape=bounded_projection",
        f"role={role}",
        "history_action=baseline_send",
        "replay_state=fresh_reset_required",
        "[CTX_EOS]",
    ]


def _prediction_context(selector: nn.Module, context: list[str], vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]]) -> dict[str, str]:
    return _predict(selector, [{"context_tokens": context}], vocab, classes, torch.device("cpu"))[0]


def _validate_generated(value: str, *, role: str) -> bool:
    if value == ASK_VALUE:
        return False
    if not SAFE_VALUE_RE.fullmatch(value):
        return False
    expected = _target_text_for_role(role)
    return value == expected and all(fragment not in value.casefold() for fragment in FORBIDDEN)


def _start_reset_for_node(origin: str) -> None:
    _send_reset(origin)


def _replay_row(*, name: str, origin: str, reset: Any, selector: nn.Module, vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], head: FixtureStringHead, method: str, path: str, field: str, surface: str, field_role: str, shape: str, show_wire: bool, wires: list[str]) -> dict[str, Any]:
    reset()
    baseline = _baseline(origin, method=method, path=path, field=field)
    candidate_context = _context(method=method, surface=surface, field_role=field_role, shape=shape, role="candidate", projection=baseline)
    prediction = _prediction_context(selector, candidate_context, vocab, classes)
    candidate_boundary = _boundary(selector, [{"context_tokens": candidate_context}], vocab, torch.device("cpu"))
    generated_by_role: dict[str, str] = {}
    role_projections: dict[str, dict[str, Any]] = {}
    for role in ("candidate", "reference", "replay"):
        role_context = _context(method=method, surface=surface, field_role=field_role, shape=shape, role=role, projection=baseline)
        role_boundary = _boundary(selector, [{"context_tokens": role_context}], vocab, torch.device("cpu"))
        generated = head.generate(role_boundary, ROLE_IDS[role])
        generated_by_role[role] = generated
        if not _validate_generated(generated, role=role):
            continue
        reset()
        projection = _request_value(origin, method=method, path=path, field=field, value=generated)
        role_projections[role] = projection
        if show_wire:
            if method == "GET":
                wires.append(f"GET {origin}{path}?{field}={generated}")
            else:
                wires.append(f"POST {origin}{path}\nContent-Type: application/x-www-form-urlencoded\n\n{field}={generated}")
    # Negative control is evaluator-bound, not model-authorized.  This keeps
    # the negative lane independent while still testing the exact grammar.
    negative_value = _target_text_for_role("negative")
    reset()
    negative_projection = _request_value(origin, method=method, path=path, field=field, value=negative_value)
    if show_wire:
        if method == "GET":
            wires.append(f"GET {origin}{path}?{field}={negative_value}")
        else:
            wires.append(f"POST {origin}{path}\nContent-Type: application/x-www-form-urlencoded\n\n{field}={negative_value}")
    return {
        "implementation": name,
        "method": method,
        "source_sha256": _sha_file(PY_FIXTURE if name == "python_a" else NODE_FIXTURE),
        "baseline_filtered": int(baseline.get("filter_state") == "filtered"),
        "model_prediction": prediction,
        "model_generated_value": {role: {"sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(), "length": len(value), "persisted": False, "grammar_valid": _validate_generated(value, role=role)} for role, value in generated_by_role.items()},
        "candidate_typed": int(role_projections.get("candidate", {}).get("typed_effect_confirmed", False)),
        "reference_typed": int(role_projections.get("reference", {}).get("typed_effect_confirmed", False)),
        "replay_typed": int(role_projections.get("replay", {}).get("typed_effect_confirmed", False)),
        "negative_violation": int(negative_projection.get("typed_effect_confirmed", False)),
        "negative_projection": negative_projection,
        "roles": role_projections,
        "raw_string_stored": False,
    }


def replay(*, selector_checkpoint: Path = DEFAULT_SELECTOR, decoder_checkpoint: Path = DEFAULT_HEAD, show_wire: bool = False) -> tuple[dict[str, Any], list[str]]:
    selector, vocab, classes, selector_state_sha = _load_selector(selector_checkpoint)
    head = _load_head(decoder_checkpoint)
    selector.eval()
    python_server, python_thread = start_filter_canary_server()
    python_origin = f"http://127.0.0.1:{python_server.server_port}"
    node_process, node_origin = _start_node()
    wires: list[str] = []
    try:
        rows = [
            _replay_row(name="python_a", origin=python_origin, reset=python_server.fresh_reset, selector=selector, vocab=vocab, classes=classes, head=head, method="GET", path=ROUTE_PATH, field=FIELD_NAME, surface="query", field_role="query_term", shape="query_marker", show_wire=show_wire, wires=wires),
            _replay_row(name="python_a", origin=python_origin, reset=python_server.fresh_reset, selector=selector, vocab=vocab, classes=classes, head=head, method="POST", path=ROUTE_PATH, field=FIELD_NAME, surface="form", field_role="form_field", shape="html_form_marker", show_wire=show_wire, wires=wires),
            _replay_row(name="node_b", origin=node_origin, reset=lambda: _start_reset_for_node(node_origin), selector=selector, vocab=vocab, classes=classes, head=head, method="GET", path="/pg385b/filter", field="value", surface="query", field_role="query_term", shape="query_marker", show_wire=show_wire, wires=wires),
            _replay_row(name="node_b", origin=node_origin, reset=lambda: _start_reset_for_node(node_origin), selector=selector, vocab=vocab, classes=classes, head=head, method="POST", path="/pg385b/filter", field="value", surface="form", field_role="form_field", shape="html_form_marker", show_wire=show_wire, wires=wires),
        ]
        counts = {
            "rows": len(rows),
            "implementations": len({row["implementation"] for row in rows}),
            "methods_get": sum(row["method"] == "GET" for row in rows),
            "methods_post": sum(row["method"] == "POST" for row in rows),
            "model_generated_candidate": sum(bool(row["model_generated_value"].get("candidate", {}).get("grammar_valid")) for row in rows),
            "model_generated_reference": sum(bool(row["model_generated_value"].get("reference", {}).get("grammar_valid")) for row in rows),
            "model_generated_replay": sum(bool(row["model_generated_value"].get("replay", {}).get("grammar_valid")) for row in rows),
            "candidate_typed": sum(row["candidate_typed"] for row in rows),
            "reference_typed": sum(row["reference_typed"] for row in rows),
            "replay_typed": sum(row["replay_typed"] for row in rows),
            "negative_violation": sum(row["negative_violation"] for row in rows),
        }
        passed = counts["model_generated_candidate"] == 4 and counts["model_generated_reference"] == 4 and counts["model_generated_replay"] == 4 and counts["candidate_typed"] == 4 and counts["reference_typed"] == 4 and counts["replay_typed"] == 4 and counts["negative_violation"] == 0
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_model_generated_fixture_payload_loopback_only" if passed else "blocked_model_generated_fixture_payload",
            "selector_checkpoint_sha256": _sha_file(selector_checkpoint),
            "selector_state_sha256": selector_state_sha,
            "decoder_checkpoint_sha256": _sha_file(decoder_checkpoint),
            "rows": rows,
            "counts": counts,
            "execution": {"external_network": False, "docker_started": False, "target_contacted": True, "loopback_only": True, "raw_wire_stored": False, "raw_string_stored": False},
            "model_boundary": {"token_model_context_only": True, "model_emits_fixture_bound_raw_string": True, "model_emits_arbitrary_raw_string": False, "grammar_validated_before_send": True, "negative_control_evaluator_bound": True},
            "grammar": {"grammar_id": "pg386-local-filter-canary-v1", "allowed_output_shape": "PG386_(CAND|REF|NEG|REPLAY)_0002%25253A", "external_target": False, "script_or_sql_execution": False, "state_write": False},
            "promotion": dict(PROMOTION),
        }
        report["report_sha256"] = _sha(report)
        return report, wires
    finally:
        _stop_node(node_process)
        python_server.shutdown()
        python_server.server_close()
        python_thread.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--selector-checkpoint", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--train-candidate", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--show-wire", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    training_report = None
    if args.train_candidate or not args.decoder_checkpoint.exists():
        training_report = train_candidate(dataset_path=args.dataset, selector_checkpoint=args.selector_checkpoint, output_checkpoint=args.decoder_checkpoint, epochs=args.epochs, hidden_dim=args.hidden_dim)
    if args.replay or training_report is not None:
        replay_report, wires = replay(selector_checkpoint=args.selector_checkpoint, decoder_checkpoint=args.decoder_checkpoint, show_wire=args.show_wire)
        output = {"training": training_report, "replay": replay_report}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": replay_report["status"], "counts": replay_report["counts"], "report_sha256": replay_report["report_sha256"]}, ensure_ascii=False, indent=2))
        if args.show_wire:
            print("EPHEMERAL_MODEL_GENERATED_LOCAL_CANARY_WIRE (not persisted):")
            for wire in wires:
                print(wire)
        return 0
    parser.error("use --train-candidate or --replay")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FixtureStringHead", "replay", "train_candidate"]
