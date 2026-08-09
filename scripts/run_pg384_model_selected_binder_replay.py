"""PG-384 model-selected abstract Rule-IR -> local evaluator binder replay.

The model emits only 13 abstract slots.  A reviewed, loopback-only evaluator
may expand one bounded ``{{MARKER}}`` template in memory, send candidate /
reference / negative / replay variants, and retain only projections and
hashes.  No arbitrary target, raw payload, response body, or wire is written
to the report.  The default is dry-run; ``--live`` additionally requires
``PG384_LOCAL_EVAL=1`` and uses the synthetic PG-348 loopback runtime only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg348_dynamic_runtime import DynamicFixtureApplication, load_registry, start_server  # noqa: E402
from app.pg350_runtime_payload_binder import ALLOWED_ORACLES, ALLOWED_SHAPES, bind_runtime_probe  # noqa: E402
from scripts.run_pg350_runtime_binding_replay import _send_ephemeral  # noqa: E402
from scripts.run_pg375_composed_rule_ir_candidate import (  # noqa: E402
    ComposedRuleIRModel,
    SLOTS,
    _pad_context,
)
from scripts.run_pg370_multitask_moe_candidate import _target_values  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig  # noqa: E402


SCHEMA_VERSION = "pg384-model-selected-binder-replay-v1"
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
_ROLE_VARIANT = {
    "candidate": "candidate_surface",
    "reference": "reference_surface",
    "negative": "unsupported_variant",
    "replay": "candidate_surface",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_slots(tokens: Sequence[str]) -> dict[str, str] | None:
    if not tokens or tokens[0] != "[TARGET_BOS]" or tokens[-1] != "[TARGET_EOS]":
        return None
    parsed: dict[str, str] = {}
    for token in tokens[1:-1]:
        text = str(token)
        if "=" not in text:
            return None
        key, value = text.split("=", 1)
        if key in parsed or not key or not value:
            return None
        parsed[key] = value
    return parsed if set(SLOTS) <= set(parsed) else None


def normalize_rule_ir(slots: Mapping[str, str]) -> dict[str, Any] | None:
    """Return only binder-safe abstract slots; reject unsafe/unknown output."""

    if slots.get("safe_to_send") not in {"1", "true"}:
        return None
    transport = str(slots.get("transport_ref", ""))
    if transport not in {"get_query", "post_form"}:
        return None
    encoding = _ENCODING_MAP.get(str(slots.get("encoding_ref", "")))
    shape = str(slots.get("payload_shape_ref", ""))
    oracle = str(slots.get("oracle_ref", ""))
    variant = str(slots.get("probe_variant_ref", ""))
    syntax = str(slots.get("syntax_category_ref", "")).casefold().replace("-", "_")
    if encoding is None or shape not in ALLOWED_SHAPES or oracle not in ALLOWED_ORACLES:
        return None
    if variant not in {"source_attested_candidate", "reference_shape", "fresh_replay"}:
        return None
    if not syntax or syntax in {"none", "unknown"}:
        return None
    return {
        "transport_ref": transport,
        "field_role_ref": str(slots.get("field_role_ref", "unknown")),
        "encoding_ref": encoding,
        "payload_shape_ref": shape,
        "oracle_ref": oracle,
        "probe_variant_ref": variant,
        "syntax_category_ref": syntax,
        "safe_to_send": True,
    }


def _route_for_method(registry: Mapping[str, Any], method: str) -> dict[str, Any] | None:
    rows = [row for row in registry.get("records") or [] if str(row.get("transport_method", "")).upper() == method]
    return dict(sorted(rows, key=lambda row: str(row.get("challenge_id", "")))[0]) if rows else None


def _attestation(route: Mapping[str, Any], method: str, origin: str, template_id: str) -> dict[str, Any]:
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
        "authorization_id": "pg384_local_synthetic_evaluator",
        "allowed_template_ids": [template_id],
        "stateful_evaluator": False,
        "source_attestation_sha256": source_hash,
    }


def _catalog(rule_ir: Mapping[str, Any], template_id: str) -> dict[str, Any]:
    # This is a reviewed non-destructive canary placeholder, not an attack
    # string.  It is expanded only in evaluator memory and never persisted.
    template = "{{MARKER}}"
    return {
        "templates": [
            {
                "template_id": template_id,
                "shape": str(rule_ir["payload_shape_ref"]),
                "syntax_category_ref": str(rule_ir["syntax_category_ref"]),
                "template": template,
                "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                "local_only": True,
                "non_destructive": True,
                "stateful_allowed": False,
            }
        ]
    }


def _projection(result: Mapping[str, Any]) -> dict[str, Any]:
    projection = result.get("projection") if isinstance(result.get("projection"), Mapping) else {}
    return {
        "typed_effect_confirmed": bool(result.get("typed_effect_confirmed")),
        "projection": {key: value for key, value in projection.items() if key not in {"url", "body", "response_body", "wire"}},
        "response_evidence_sha256": str(result.get("response_evidence_sha256", "")),
    }


def _scrub(value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
    for marker in ("http://", "https://", "wire=", "payload=", "response_body=", "raw_value="):
        if marker in text:
            raise ValueError("raw_or_wire_leak")
    for key in ('"url":', '"body":', '"response_body":', '"raw_value":', '"wire":'):
        if key in text:
            raise ValueError("raw_key_leak")


def _load_model(checkpoint_path: Path) -> tuple[ComposedRuleIRModel, dict[str, int], dict[str, dict[str, int]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    slot_classes = {str(slot): {str(value): int(index) for value, index in values.items()} for slot, values in checkpoint["slot_classes"].items()}
    config = CausalMoEConfig(**dict(checkpoint["config"]))
    model = ComposedRuleIRModel(
        vocab_size=len(vocabulary),
        config=config,
        slot_classes=slot_classes,
        slot_decoder_layers=int(checkpoint.get("slot_decoder_layers", 2)),
        slot_decoder_heads=int(checkpoint.get("slot_decoder_heads", config.n_heads)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocabulary, slot_classes


def _decode(model: ComposedRuleIRModel, vocabulary: Mapping[str, int], slot_classes: Mapping[str, Mapping[str, int]], row: Mapping[str, Any]) -> dict[str, str]:
    reverse = {slot: {int(index): str(value) for value, index in classes.items()} for slot, classes in slot_classes.items()}
    with torch.inference_mode():
        context_ids, context_mask = _pad_context([row], vocabulary, torch.device("cpu"))
        output = model(context_ids, context_mask, decode_composition=True)
    return {slot: reverse[slot][int(output["composition"][slot].argmax(-1).item())] for slot in SLOTS}


def replay(
    *,
    checkpoint_path: Path,
    dataset_path: Path,
    registry_path: Path,
    max_rows: int = 48,
    live: bool = False,
    ephemeral_wire_sink: list[str] | None = None,
) -> dict[str, Any]:
    if live and os.environ.get("PG384_LOCAL_EVAL") != "1":
        raise RuntimeError("PG384 live evaluator requires PG384_LOCAL_EVAL=1")
    if ephemeral_wire_sink is not None and not live:
        raise ValueError("ephemeral wire preview requires live local evaluator")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    rows = [dict(row) for row in dataset.get("records") or [] if row.get("split") == "implementation_holdout"]
    selected = rows[: max(1, int(max_rows))]
    model, vocabulary, slot_classes = _load_model(checkpoint_path)
    registry = load_registry(registry_path)
    server = thread = None
    app = None
    origin = None
    if live:
        app = DynamicFixtureApplication(registry)
        server, thread = start_server(app, host="127.0.0.1", port=0)
        origin = f"http://127.0.0.1:{server.server_port}"
    results: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected):
            gold = _target_values(row["target_tokens"])
            decoded = _decode(model, vocabulary, slot_classes, row)
            expected_safe = str(gold.get("safe_to_send")) == "1"
            decoded_safe = str(decoded.get("safe_to_send")) == "1"
            item: dict[str, Any] = {
                "record_ref_sha256": _sha({"record_id": row.get("record_id"), "split": row.get("split")}),
                "context_sha256": _sha(row.get("context_tokens", [])),
                "decoded_slots": {slot: decoded.get(slot) for slot in SLOTS},
                "decoded_exact": decoded == gold,
                "expected_safe_to_send": expected_safe,
                "decoded_safe_to_send": decoded_safe,
                "unsafe_allow": bool(decoded_safe and not expected_safe),
                "safe_abstain_correct": bool(not decoded_safe and not expected_safe),
                "abstain": True,
                "reason": "model_abstain_or_invalid_slots",
            }
            if not decoded_safe:
                results.append(item)
                continue
            if not expected_safe:
                item["reason"] = "unsafe_allow_blocked_before_binder"
                results.append(item)
                continue
            rule_ir = normalize_rule_ir(decoded)
            method = "GET" if decoded.get("transport_ref") == "get_query" else "POST" if decoded.get("transport_ref") == "post_form" else ""
            route = _route_for_method(registry, method)
            if rule_ir is None or route is None:
                item["reason"] = "binder_preflight_rejected"
                results.append(item)
                continue
            if not live:
                item["reason"] = "dry_run_model_selected_abstract_slots"
                item["abstain"] = False
                item["model_selected"] = True
                results.append(item)
                continue
            template_id = f"pg384_model_{method.casefold()}_{index}"
            try:
                bound = bind_runtime_probe(
                    rule_ir,
                    _attestation(route, method, str(origin), template_id),
                    _catalog(rule_ir, template_id),
                    marker=f"PG384M{index:04d}CAND",
                )
                # The concrete canary wire is deliberately available only to
                # an explicit operator preview sink.  It is never inserted
                # into ``item`` or the persisted report.
                if ephemeral_wire_sink is not None:
                    ephemeral_wire_sink.append(bound.human_review_wire())
                role_results: dict[str, Any] = {}
                for role in ROLES:
                    assert app is not None
                    app.reset(str(route["challenge_id"]))
                    role_results[role] = _projection(_send_ephemeral(bound, variant=_ROLE_VARIANT[role], timeout=3.0))
                candidate = role_results["candidate"]
                reference = role_results["reference"]
                negative = role_results["negative"]
                replay_row = role_results["replay"]
                item.update(
                    {
                        "abstain": False,
                        "model_selected": True,
                        "reason": "bound_and_replayed_loopback_only",
                        "binding": bound.persisted_projection(),
                        "roles": role_results,
                        "evidence_sha256": _sha({"binding": bound.persisted_projection(), "roles": role_results}),
                        "typed_effect_confirmed": bool(candidate["typed_effect_confirmed"] and reference["typed_effect_confirmed"]),
                        "negative_control_clean": not negative["typed_effect_confirmed"],
                        "replay_consistent": replay_row["typed_effect_confirmed"] == candidate["typed_effect_confirmed"],
                    }
                )
                item["confirmed_positive"] = bool(item["typed_effect_confirmed"] and item["negative_control_clean"] and item["replay_consistent"])
            except (KeyError, RuntimeError, ValueError):
                item["reason"] = "binder_rejected"
            results.append(item)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_model_selected_loopback_evaluator_only" if live else "dry_run_model_selected_abstract_only",
        "checkpoint_sha256": _file_sha(checkpoint_path),
        "dataset_sha256": _file_sha(dataset_path),
        "registry_sha256": _file_sha(registry_path),
        "counts": {
            "holdout_rows_considered": len(results),
            "decoded_exact": sum(bool(item.get("decoded_exact")) for item in results),
            "model_selected": sum(bool(item.get("model_selected")) for item in results),
            "confirmed_positive": sum(bool(item.get("confirmed_positive")) for item in results),
            "abstain_rows": sum(bool(item.get("abstain")) for item in results),
            "unsafe_allow": sum(bool(item.get("unsafe_allow")) for item in results),
            "binder_rejected": sum(item.get("reason") == "binder_rejected" for item in results),
        },
        "rows": results,
        "execution": {"live": bool(live), "target_contacted": bool(live), "docker_started": False, "external_network": False, "loopback_runtime": bool(live)},
        "raw_persistence": {"model_context_raw": False, "request_url_stored": False, "request_body_stored": False, "response_body_stored": False, "wire_stored": False, "canary_stored": False},
        "scientific_scope": {"model_selected_abstract_slots": True, "synthetic_pg348_runtime": bool(live), "second_independent_implementation": False, "general_vulnerability_claim": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    _scrub(report)
    report["report_sha256"] = _sha(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-384 model-selected abstract binder replay")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/pg384-binding-composition-a800/pg375_seed_38101.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research/pg384_binding_abstract_adversarial_dataset_v1.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "fixtures/pg348/registry_v1.json")
    parser.add_argument("--max-rows", type=int, default=48)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--show-wire",
        action="store_true",
        help="print one-shot local canary wires to stdout; requires --live and PG384_LOCAL_EVAL=1; never persists them",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg384_model_selected_binder_replay_v1.json")
    args = parser.parse_args()
    if args.show_wire and not args.live:
        parser.error("--show-wire requires --live")
    ephemeral_wires: list[str] = []
    report = replay(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        registry_path=args.registry,
        max_rows=args.max_rows,
        live=args.live,
        ephemeral_wire_sink=ephemeral_wires if args.show_wire else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    if args.show_wire:
        print("EPHEMERAL_LOCAL_CANARY_WIRE_PREVIEW (not persisted):")
        for wire in ephemeral_wires[:3]:
            print(wire)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
