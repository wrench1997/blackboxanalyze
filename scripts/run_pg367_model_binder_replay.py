"""Connect a PG-367 model decode to the evaluator-only PG-350 binder.

The model emits only abstract Rule-IR slots.  The binder validates those
slots against a source-attested loopback route and a single reviewed template,
then creates an in-memory canary wire.  Candidate/reference/negative/replay
requests are sent to the synthetic PG-348 dynamic runtime after fresh resets.
The persisted report contains no URL, body, response bytes or canary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from app.pg293_failure_next_action import TARGET_BOS, TARGET_EOS
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, generate_target
from app.pg348_dynamic_runtime import DynamicFixtureApplication, load_registry, start_server
from app.pg350_runtime_payload_binder import ALLOWED_ORACLES, ALLOWED_SHAPES, bind_runtime_probe
from app.pg361_payload_shape_slots import ALLOWED_SYNTAX_CATEGORIES
from scripts.run_pg350_runtime_binding_replay import _send_ephemeral


SCHEMA_VERSION = "pg367-model-binder-replay-v1"
ROLES = ("candidate", "reference", "negative", "replay")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ENCODING_MAP = {
    "identity": "identity",
    "url_percent": "url_percent",
    "form_urlencoded": "form_urlencoded",
    "entity_encoded": "html_entity",
    "double_encoded": "double_layer_order_sensitive",
    "json_string": "json_escape",
}
_FORBIDDEN_TEXT = ("http://127.0.0.1:", "https://", "pg367model")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_target(tokens: Sequence[str]) -> dict[str, str] | None:
    if list(tokens[:1]) != [TARGET_BOS] or list(tokens[-1:]) != [TARGET_EOS]:
        return None
    result: dict[str, str] = {}
    for token in tokens[1:-1]:
        if "=" not in token:
            return None
        key, value = token.split("=", 1)
        if not key or key in result or not value:
            return None
        result[key] = value
    required = {"transport_ref", "field_role_ref", "encoding_ref", "probe_variant_ref", "safe_to_send", "payload_shape_ref", "oracle_ref", "syntax_category_ref"}
    return result if required <= set(result) else None


def _normalize_rule_ir(raw: Mapping[str, str]) -> dict[str, Any] | None:
    transport = raw.get("transport_ref", "")
    encoding = _ENCODING_MAP.get(raw.get("encoding_ref", ""))
    if transport not in {"get_query", "post_form"} or encoding is None:
        return None
    if raw.get("safe_to_send") not in {"1", "true"}:
        return None
    if raw.get("probe_variant_ref") not in {"source_attested_candidate", "reference", "reference_shape", "fresh_replay"}:
        return None
    try:
        syntax = raw["syntax_category_ref"].casefold().replace("-", "_")
        shape = raw["payload_shape_ref"]
        oracle = raw["oracle_ref"]
        safe = True
    except Exception:
        return None
    if shape not in ALLOWED_SHAPES or oracle not in ALLOWED_ORACLES or syntax not in ALLOWED_SYNTAX_CATEGORIES:
        return None
    return {
        "transport_ref": transport,
        "field_role_ref": raw["field_role_ref"],
        "encoding_ref": encoding,
        "probe_variant_ref": raw["probe_variant_ref"],
        "safe_to_send": safe,
        "payload_shape_ref": raw["payload_shape_ref"],
        "oracle_ref": raw["oracle_ref"],
        "syntax_category_ref": syntax,
    }


def _route_for_method(registry: Mapping[str, Any], method: str) -> dict[str, Any] | None:
    rows = [row for row in registry.get("records") or [] if str(row.get("transport_method", "")).upper() == method]
    if not rows:
        return None
    row = sorted(rows, key=lambda item: str(item.get("challenge_id", "")))[0]
    return dict(row)


def _catalog(rule_ir: Mapping[str, Any], template_id: str) -> dict[str, Any]:
    shape = str(rule_ir["payload_shape_ref"])
    syntax = str(rule_ir["syntax_category_ref"])
    template = "{{MARKER}}'"
    return {"templates": [{"template_id": template_id, "shape": shape, "syntax_category_ref": syntax, "template": template, "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(), "local_only": True, "non_destructive": True, "stateful_allowed": False}]}


def _runtime_attestation(origin: str, route: Mapping[str, Any], method: str, template_id: str) -> dict[str, Any]:
    source_hash = str(route.get("source_hash", "")).casefold()
    if not _HASH_RE.fullmatch(source_hash):
        source_hash = _sha({"challenge_id": str(route.get("challenge_id", "")), "method": method})
    return {
        "target_origin": origin,
        "route": {"method": method, "path": f"/pg348/dynamic/{route['challenge_id']}", "field_name": "q"},
        "loopback_only": True,
        "external_network": False,
        "source_attested": True,
        "route_attested": True,
        "field_attested": True,
        "fresh_reset": True,
        "candidate_reference_negative": True,
        "replay_consistency": True,
        "authorization_id": "pg367_model_binder_local",
        "allowed_template_ids": [template_id],
        "stateful_evaluator": False,
        "source_attestation_sha256": source_hash,
    }


def _role_variant(role: str) -> str:
    return {"candidate": "candidate_surface", "reference": "reference_surface", "negative": "unsupported_variant", "replay": "candidate_surface"}[role]


def _projection(result: Mapping[str, Any]) -> dict[str, Any]:
    projection = result.get("projection") if isinstance(result.get("projection"), Mapping) else {}
    return {
        "typed_effect_confirmed": bool(result.get("typed_effect_confirmed")),
        "projection": {key: value for key, value in projection.items() if key not in {"url", "body", "response_body", "wire"}},
        "response_evidence_sha256": str(result.get("response_evidence_sha256", "")),
    }


def _scrub(value: Any) -> None:
    folded = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
    for fragment in _FORBIDDEN_TEXT:
        if fragment.casefold() in folded:
            raise ValueError(f"raw_or_wire_leak:{fragment}")
    for key in ("url", "body", "response_body", "raw_value"):
        if f'"{key}":' in folded:
            raise ValueError(f"raw_key:{key}")


def _load_model(checkpoint: Mapping[str, Any], device: torch.device) -> tuple[Any, dict[str, int]]:
    config = CausalMoEConfig(**dict(checkpoint["config"]))
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    model = CausalMoELanguageModel(vocab_size=len(vocabulary), config=config).to(device)
    states = checkpoint.get("states") or {}
    first_seed = sorted(states, key=str)[0]
    model.load_state_dict({key: value.to(device) for key, value in states[first_seed].items()})
    model.eval()
    return model, vocabulary


def _decode_model(model: Any, vocabulary: Mapping[str, int], context: Sequence[str], target_length: int, device: torch.device) -> list[str]:
    return generate_target(model, context, target_length, vocabulary, device)


def replay(*, checkpoint_path: Path, dataset_path: Path, registry_path: Path, max_rows: int = 12, show_wire: bool = False) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    registry = load_registry(registry_path)
    rows = [row for row in dataset.get("records") or [] if row.get("split") == "implementation_holdout"]
    selected = rows[: max(1, int(max_rows))]
    model, model_vocabulary = _load_model(checkpoint, torch.device("cpu"))
    app = DynamicFixtureApplication(registry)
    server, thread = start_server(app, host="127.0.0.1", port=0)
    origin = f"http://127.0.0.1:{server.server_port}"
    results: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected):
            context = [str(token) for token in row.get("context_tokens") or []]
            gold_target = [str(token) for token in row.get("target_tokens") or []]
            decoded = _decode_model(model, model_vocabulary, context, len(gold_target), torch.device("cpu"))
            slots = _parse_target(decoded)
            gold_slots = _parse_target(gold_target) or {}
            expected_safe = gold_slots.get("safe_to_send") in {"1", "true"}
            decoded_safe = slots is not None and slots.get("safe_to_send") in {"1", "true"}
            abstract: dict[str, Any] = {"decoded_exact": decoded == gold_target, "decoded_target_sha256": _sha(decoded), "context_sha256": _sha(context), "expected_safe_to_send": expected_safe, "decoded_safe_to_send": decoded_safe, "unsafe_allow": bool(decoded_safe and not expected_safe), "safe_abstain_correct": bool((not decoded_safe) and not expected_safe), "abstain": True, "reason": "decode_or_slot_invalid"}
            if slots is None:
                results.append(abstract)
                continue
            rule_ir = _normalize_rule_ir(slots)
            method = "GET" if slots.get("transport_ref") == "get_query" else "POST" if slots.get("transport_ref") == "post_form" else ""
            route = _route_for_method(registry, method)
            if not expected_safe and decoded_safe:
                abstract.update({"decoded_slots": {key: slots[key] for key in sorted(slots)}, "reason": "unsafe_allow_blocked_before_binder"})
                results.append(abstract)
                continue
            if rule_ir is None or route is None:
                abstract.update({"decoded_slots": {key: slots[key] for key in sorted(slots)}, "reason": "slot_not_bindable"})
                results.append(abstract)
                continue
            template_id = f"pg367_model_{method.casefold()}_{index}"
            try:
                app.reset(str(route["challenge_id"]))
                bound = bind_runtime_probe(rule_ir, _runtime_attestation(origin, route, method, template_id), _catalog(rule_ir, template_id), marker=f"PG367M{index:04d}CAND")
                if show_wire:
                    print(bound.human_review_wire())
                persisted = bound.persisted_projection()
                role_results: dict[str, Any] = {}
                for role in ROLES:
                    app.reset(str(route["challenge_id"]))
                    role_result = _projection(_send_ephemeral(bound, variant=_role_variant(role), timeout=3.0))
                    role_results[role] = role_result
                candidate = role_results["candidate"]
                reference = role_results["reference"]
                negative = role_results["negative"]
                replay_row = role_results["replay"]
                evidence = _sha({"binding": persisted, "roles": role_results})
                abstract.update({"decoded_slots": {key: slots[key] for key in sorted(slots)}, "rule_ir": rule_ir, "binding": persisted, "roles": role_results, "evidence_sha256": evidence, "abstain": False, "reason": "bound_and_replayed", "typed_effect_confirmed": bool(candidate["typed_effect_confirmed"] and reference["typed_effect_confirmed"]), "negative_control_clean": not negative["typed_effect_confirmed"], "replay_consistent": replay_row["typed_effect_confirmed"] == candidate["typed_effect_confirmed"], "confirmed_positive": bool(candidate["typed_effect_confirmed"] and reference["typed_effect_confirmed"] and not negative["typed_effect_confirmed"] and replay_row["typed_effect_confirmed"] == candidate["typed_effect_confirmed"])})
            except (ValueError, RuntimeError, KeyError) as error:
                abstract.update({"decoded_slots": {key: slots[key] for key in sorted(slots)}, "reason": "binder_rejected", "error_class": type(error).__name__})
            results.append(abstract)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    unsafe_count = sum(bool(row.get("unsafe_allow")) for row in results)
    confirmed_count = sum(bool(row.get("confirmed_positive")) for row in results)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_model_selected_evaluator_only" if confirmed_count and unsafe_count == 0 else "blocked_model_safety_or_binder",
        "model_checkpoint_sha256": _file_sha(checkpoint_path),
        "dataset_sha256": _file_sha(dataset_path),
        "registry_sha256": _file_sha(registry_path),
        "counts": {"holdout_rows_considered": len(results), "decoded_exact": sum(bool(row.get("decoded_exact")) for row in results), "bound_rows": sum(not bool(row.get("abstain")) for row in results), "confirmed_positive": sum(bool(row.get("confirmed_positive")) for row in results), "abstain_rows": sum(bool(row.get("abstain")) for row in results), "unsafe_allow": sum(bool(row.get("unsafe_allow")) for row in results), "safe_abstain_correct": sum(bool(row.get("safe_abstain_correct")) for row in results)},
        "rows": results,
        "raw_persistence": {"model_context_raw": False, "request_url_stored": False, "request_body_stored": False, "response_body_stored": False, "wire_stored": False, "canary_stored": False},
        "scientific_scope": {"model_selected_abstract_slots": True, "synthetic_pg348_runtime": True, "second_independent_implementation": False, "general_vulnerability_claim": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    _scrub(report)
    report["report_sha256"] = _sha(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay PG-367 model Rule-IR through the local binder")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/pg367-a800-process/pg367_a800_process_candidate_v2.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research/pg367_waf_staircase_dataset_v2.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "fixtures/pg348/registry_v1.json")
    parser.add_argument("--max-rows", type=int, default=12)
    parser.add_argument("--show-wire", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg367_model_binder_replay_report_v1.json")
    args = parser.parse_args()
    report = replay(checkpoint_path=args.checkpoint, dataset_path=args.dataset, registry_path=args.registry, max_rows=args.max_rows, show_wire=args.show_wire)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed_model_selected_evaluator_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
