"""Replay the PG-385 selector on two independent loopback implementations.

Implementation A is the Python fixture already used by the single-implementation
demo.  Implementation B is a dependency-free Node process with a different
source/runtime boundary and route/field names.  Both expose only the same
abstract filter projection.  The model sees no implementation name or route
literal; the evaluator owns the route binding and ephemeral canary wire.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg350_runtime_payload_binder import bind_runtime_probe  # noqa: E402
from app.pg385_filter_canary_fixture import start_filter_canary_server  # noqa: E402
from scripts.run_pg385_filter_repair_demo import _catalog, _runtime, _scrub, _send_and_project  # noqa: E402
from scripts.run_pg385_model_selected_filter_repair_demo import _load_selector, _rule_from_prediction, _context_from_projection  # noqa: E402
from scripts.run_pg385_variant_selector_candidate import _predict  # noqa: E402


SCHEMA_VERSION = "pg385-model-selected-cross-implementation-replay-v1"
DEFAULT_CHECKPOINT = ROOT / "artifacts/pg385-variant-selector/pg385_variant_seed_38503.pt"
NODE_FIXTURE = ROOT / "fixtures/pg385/impl_b/server.js"
PY_FIXTURE = ROOT / "app/pg385_filter_canary_fixture.py"
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _abstract_context(*, method: str, surface: str, field_role: str, shape: str, projection: Mapping[str, Any]) -> list[str]:
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
        "role=candidate",
        "history_action=baseline_send",
        "replay_state=fresh_reset_required",
        "[CTX_EOS]",
    ]


def _runtime_for(origin: str, *, method: str, path: str, field: str) -> dict[str, Any]:
    runtime = _runtime(origin)
    runtime["route"] = {"method": method, "path": path, "field_name": field}
    return runtime


def _send_reset(origin: str) -> None:
    request = Request(f"{origin}/__reset", data=b"", headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=3.0) as response:  # loopback origin is supplied by an attested local process
        response.read(2048)


def _start_node() -> tuple[subprocess.Popen[str], str]:
    if not NODE_FIXTURE.exists():
        raise FileNotFoundError(NODE_FIXTURE)
    env = dict(os.environ)
    env["PORT"] = "0"
    process = subprocess.Popen(["node", str(NODE_FIXTURE)], cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 5.0
    try:
        while time.time() < deadline:
            line = process.stdout.readline() if process.stdout is not None else ""
            if line:
                event = json.loads(line)
                port = int(event["port"])
                return process, f"http://127.0.0.1:{port}"
            if process.poll() is not None:
                raise RuntimeError("node fixture exited before ready")
            time.sleep(0.02)
        raise TimeoutError("node fixture readiness timeout")
    except Exception:
        process.terminate()
        process.wait(timeout=2)
        raise


def _stop_node(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _run_impl(*, name: str, origin: str, reset: Any, model: Any, vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], method: str, path: str, field: str, surface: str, field_role: str, shape: str, show_wire: bool, wires: list[str]) -> dict[str, Any]:
    transport = "get_query" if method == "GET" else "post_form"
    baseline_rule = {"transport_ref": transport, "field_role_ref": field_role, "encoding_ref": "identity", "payload_shape_ref": shape, "syntax_category_ref": "delimiter_boundary", "probe_variant_ref": "source_attested_candidate", "oracle_ref": "response_shape", "safe_to_send": "1"}
    reset()
    baseline_bound = bind_runtime_probe(baseline_rule, _runtime_for(origin, method=method, path=path, field=field), _catalog(baseline_rule), marker=f"PG385_{name.upper()}_{method}_BASE")
    if show_wire:
        wires.append(baseline_bound.human_review_wire())
    baseline = _send_and_project(baseline_bound)
    context = _abstract_context(method=method, surface=surface, field_role=field_role, shape=shape, projection=baseline)
    prediction = _predict(model, [{"context_tokens": context}], vocab, classes, torch.device("cpu"))[0]
    selected = prediction["probe_variant_ref"] == "one_variable_repair" and prediction["encoding_ref"] == "double_layer_order_sensitive" and prediction["next_action"] == "repair" and prediction["safe_to_send"] == "1"
    roles: dict[str, Any] = {}
    if selected:
        abstract_rule = _rule_from_prediction(prediction, transport_ref=transport, field_role_ref=field_role, payload_shape_ref=shape)
        evaluator_rule = dict(abstract_rule)
        evaluator_rule["probe_variant_ref"] = "runtime_canary"
        for role in ("candidate", "reference", "negative", "replay"):
            reset()
            role_marker = {"candidate": "CAND_0002", "reference": "REF_0002", "negative": "NEG_0002", "replay": "REPLAY_0002"}[role]
            bound = bind_runtime_probe(evaluator_rule, _runtime_for(origin, method=method, path=path, field=field), _catalog(evaluator_rule), marker=f"PG385_{name.upper()}_{method}_{role_marker}")
            if show_wire:
                wires.append(bound.human_review_wire())
            roles[role] = _send_and_project(bound)
    return {
        "implementation": name,
        "method": method,
        "source_sha256": _sha_file(PY_FIXTURE if name == "python_a" else NODE_FIXTURE),
        "baseline_filtered": int(baseline.get("filter_state") == "filtered"),
        "model_prediction": prediction,
        "model_variant_selected": int(selected),
        "candidate_typed": int(roles.get("candidate", {}).get("typed_effect_confirmed", False)),
        "reference_typed": int(roles.get("reference", {}).get("typed_effect_confirmed", False)),
        "negative_violation": int(roles.get("negative", {}).get("typed_effect_confirmed", False)),
        "replay_typed": int(roles.get("replay", {}).get("typed_effect_confirmed", False)),
        "baseline_projection": baseline,
        "roles": roles,
    }


def run_cross_impl(*, checkpoint: Path = DEFAULT_CHECKPOINT, show_wire: bool = False) -> tuple[dict[str, Any], list[str]]:
    model, vocab, classes, state_sha = _load_selector(checkpoint)
    python_server, python_thread = start_filter_canary_server()
    python_origin = f"http://127.0.0.1:{python_server.server_port}"
    node_process, node_origin = _start_node()
    wires: list[str] = []
    try:
        rows = []
        rows.append(_run_impl(name="python_a", origin=python_origin, reset=python_server.fresh_reset, model=model, vocab=vocab, classes=classes, method="GET", path="/pg385/filter", field="q", surface="query", field_role="query_term", shape="query_marker", show_wire=show_wire, wires=wires))
        rows.append(_run_impl(name="python_a", origin=python_origin, reset=python_server.fresh_reset, model=model, vocab=vocab, classes=classes, method="POST", path="/pg385/filter", field="q", surface="form", field_role="form_field", shape="html_form_marker", show_wire=show_wire, wires=wires))
        rows.append(_run_impl(name="node_b", origin=node_origin, reset=lambda: _send_reset(node_origin), model=model, vocab=vocab, classes=classes, method="GET", path="/pg385b/filter", field="value", surface="query", field_role="query_term", shape="query_marker", show_wire=show_wire, wires=wires))
        rows.append(_run_impl(name="node_b", origin=node_origin, reset=lambda: _send_reset(node_origin), model=model, vocab=vocab, classes=classes, method="POST", path="/pg385b/filter", field="value", surface="form", field_role="form_field", shape="html_form_marker", show_wire=show_wire, wires=wires))
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_model_selected_cross_implementation_loopback_only" if all(item["model_variant_selected"] and item["candidate_typed"] and item["reference_typed"] and item["replay_typed"] and not item["negative_violation"] for item in rows) else "blocked_cross_implementation_gate",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model_state_sha256": state_sha,
            "implementations": {"python_a": {"source_sha256": _sha_file(PY_FIXTURE), "external_network": False, "docker_started": False}, "node_b": {"source_sha256": _sha_file(NODE_FIXTURE), "external_network": False, "docker_started": False}},
            "rows": rows,
            "counts": {
                "rows": len(rows),
                "implementations": len({item["implementation"] for item in rows}),
                "methods_get": sum(item["method"] == "GET" for item in rows),
                "methods_post": sum(item["method"] == "POST" for item in rows),
                "model_variant_selected": sum(item["model_variant_selected"] for item in rows),
                "candidate_typed": sum(item["candidate_typed"] for item in rows),
                "reference_typed": sum(item["reference_typed"] for item in rows),
                "negative_violation": sum(item["negative_violation"] for item in rows),
                "replay_typed": sum(item["replay_typed"] for item in rows),
            },
            "execution": {"external_network": False, "docker_started": False, "target_contacted": True, "loopback_only": True, "raw_wire_stored": False},
            "model_boundary": {"abstract_context_only": True, "model_emits_raw_string": False, "model_emits_variant_reference": True, "evaluator_last_hop_canary_binding": True},
            "promotion": dict(PROMOTION),
        }
        _scrub(report)
        report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return report, wires
    finally:
        _stop_node(node_process)
        python_server.shutdown()
        python_server.server_close()
        python_thread.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--show-wire", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg385_model_selected_cross_impl_replay_v1.json")
    args = parser.parse_args()
    report, wires = run_cross_impl(checkpoint=args.checkpoint, show_wire=args.show_wire)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    if args.show_wire:
        print("EPHEMERAL_LOCAL_CANARY_WIRE_PREVIEW (not persisted):")
        for wire in wires:
            print(wire)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_cross_impl"]
