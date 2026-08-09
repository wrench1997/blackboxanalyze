"""Run the PG-385 local filter demo with a trained abstract selector.

The checkpoint is loaded only as an abstract token/slot coordinate system.  A
prediction that passes the local allowlist is handed to the existing reviewed
runtime binder; the binder, not the model, materializes the ephemeral canary
string.  Reports contain projections and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg295_causal_moe import CausalMoEConfig  # noqa: E402
from app.pg350_runtime_payload_binder import bind_runtime_probe  # noqa: E402
from app.pg385_filter_canary_fixture import FIELD_NAME, ROUTE_PATH, start_filter_canary_server  # noqa: E402
from scripts.run_pg385_filter_repair_demo import _catalog, _runtime, _scrub, _send_and_project  # noqa: E402
from scripts.run_pg385_variant_selector_candidate import VariantSelector, _predict  # noqa: E402


SCHEMA_VERSION = "pg385-model-selected-filter-repair-demo-v1"
DEFAULT_CHECKPOINT = ROOT / "artifacts/pg385-variant-selector/pg385_variant_seed_38503.pt"
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha_state(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(str(key).encode("utf-8"))
        value = state[key]
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _load_selector(path: Path) -> tuple[VariantSelector, dict[str, int], dict[str, dict[str, int]], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != "pg385-abstract-variant-selector-candidate-v1":
        raise ValueError("checkpoint is not a PG-385 abstract selector")
    vocabulary = {str(key): int(value) for key, value in dict(payload.get("vocabulary") or {}).items()}
    classes = {str(key): {str(item): int(index) for item, index in dict(values).items()} for key, values in dict(payload.get("classes") or {}).items()}
    if set(classes) != {"encoding_ref", "probe_variant_ref", "repair_action", "next_action", "question", "safe_to_send"}:
        raise ValueError("checkpoint head inventory is incomplete")
    config = CausalMoEConfig(**{str(key): value for key, value in dict(payload.get("config") or {}).items()})
    model = VariantSelector(vocab_size=len(vocabulary), config=config, classes=classes)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, vocabulary, classes, _sha_state(payload.get("model_state") or {})


def _context_from_projection(projection: dict[str, Any]) -> list[str]:
    """Build the exact abstract context shape used by the PG-385 dataset."""

    return [
        "[CTX_BOS]",
        "method=GET",
        "surface_context=query",
        "parameter_role=query_term",
        f"filter_state={projection.get('filter_state', 'unknown')}",
        f"filter_class={projection.get('filter_class', 'unknown')}",
        "encoding_observed=identity",
        "syntax_observed=delimiter_boundary",
        "shape_observed=query_marker",
        "response_shape=bounded_projection",
        "role=candidate",
        "history_action=baseline_send",
        "replay_state=fresh_reset_required",
        "[CTX_EOS]",
    ]


def _rule_from_prediction(
    prediction: dict[str, str],
    *,
    transport_ref: str = "get_query",
    field_role_ref: str = "query_term",
    payload_shape_ref: str = "query_marker",
) -> dict[str, str]:
    """Expand abstract heads to the full reviewed local Rule-IR shape."""

    return {
        "question": prediction["question"],
        "ask_reason": "none" if prediction["question"] == "none" else "missing_observation",
        "next_action": prediction["next_action"],
        "repair_action": prediction["repair_action"],
        "transport_ref": transport_ref,
        "field_role_ref": field_role_ref,
        "encoding_ref": prediction["encoding_ref"],
        "syntax_category_ref": "structured_value",
        "probe_variant_ref": prediction["probe_variant_ref"],
        "safe_to_send": prediction["safe_to_send"],
        "payload_shape_ref": payload_shape_ref,
        "oracle_ref": "response_shape",
        "negative_control_presence_ref": "matched_triplet",
    }


def run_demo(*, checkpoint: Path = DEFAULT_CHECKPOINT, show_wire: bool = False) -> tuple[dict[str, Any], list[str]]:
    model, vocabulary, classes, model_state_sha = _load_selector(checkpoint)
    server, thread = start_filter_canary_server()
    origin = f"http://127.0.0.1:{server.server_port}"
    wires: list[str] = []
    try:
        server.fresh_reset()
        baseline = {
            "transport_ref": "get_query",
            "field_role_ref": "query_term",
            "encoding_ref": "identity",
            "payload_shape_ref": "query_marker",
            "syntax_category_ref": "delimiter_boundary",
            "probe_variant_ref": "source_attested_candidate",
            "oracle_ref": "response_shape",
            "safe_to_send": "1",
        }
        baseline_bound = bind_runtime_probe(baseline, _runtime(origin), _catalog(baseline), marker="PG385_BASE_0001")
        if show_wire:
            wires.append(baseline_bound.human_review_wire())
        baseline_projection = _send_and_project(baseline_bound)
        context = _context_from_projection(baseline_projection)
        prediction = _predict(model, [{"context_tokens": context}], vocabulary, classes, torch.device("cpu"))[0]
        rule = _rule_from_prediction(prediction)
        selected = prediction["probe_variant_ref"] == "one_variable_repair" and prediction["encoding_ref"] == "double_layer_order_sensitive" and prediction["next_action"] == "repair" and prediction["safe_to_send"] == "1"
        role_projections: dict[str, Any] = {}
        if selected:
            # The model's abstract ``one_variable_repair`` is translated to
            # the binder's source-attested runtime-canary alias only at the
            # evaluator boundary.  The model checkpoint never contains that
            # concrete template or a wire string.
            evaluator_rule = dict(rule)
            evaluator_rule["probe_variant_ref"] = "runtime_canary"
            for role, marker in (("candidate", "PG385_CAND_0002"), ("reference", "PG385_REF_0002"), ("negative", "PG385_NEG_0002"), ("replay", "PG385_REPLAY_0002")):
                server.fresh_reset()
                bound = bind_runtime_probe(evaluator_rule, _runtime(origin), _catalog(evaluator_rule), marker=marker)
                if show_wire:
                    wires.append(bound.human_review_wire())
                role_projections[role] = _send_and_project(bound)
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_model_selected_filter_repair_loopback_only" if selected else "blocked_model_variant_not_selected",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model_state_sha256": model_state_sha,
            "model_prediction": prediction,
            "model_boundary": {"abstract_context_only": True, "model_emits_raw_string": False, "model_emits_variant_reference": True, "evaluator_last_hop_canary_binding": True, "raw_response_in_context": False},
            "baseline_projection": baseline_projection,
            "selected": selected,
            "roles": role_projections,
            "counts": {
                "baseline_filtered": int(baseline_projection.get("filter_state") == "filtered"),
                "model_variant_selected": int(selected),
                "candidate_typed": int(role_projections.get("candidate", {}).get("typed_effect_confirmed", False)),
                "reference_typed": int(role_projections.get("reference", {}).get("typed_effect_confirmed", False)),
                "negative_violation": int(role_projections.get("negative", {}).get("typed_effect_confirmed", False)),
                "replay_typed": int(role_projections.get("replay", {}).get("typed_effect_confirmed", False)),
            },
            "promotion": dict(PROMOTION),
        }
        _scrub(report)
        report["report_sha256"] = _sha(report)
        return report, wires
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--show-wire", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg385_model_selected_filter_repair_demo_v1.json")
    args = parser.parse_args()
    report, wires = run_demo(checkpoint=args.checkpoint, show_wire=args.show_wire)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "model_prediction": report["model_prediction"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    if args.show_wire:
        print("EPHEMERAL_LOCAL_CANARY_WIRE_PREVIEW (not persisted):")
        for wire in wires:
            print(wire)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_demo"]
