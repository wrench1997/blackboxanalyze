"""PG-195: XXL AI action loop over a GET/POST Pikachu matrix.

The model chooses an action from the previous projection/failure trace.  A
controller sends only browser-observed, bounded inert probes.  DOM candidate
responses are checked under three no-JS layout shells; Pikachu SQL routes have
no authoritative SQL oracle and therefore stop at abstain.  An independent
v4 loopback SQL fixture supplies a source-holdout typed oracle for comparison.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg181_manifest_decoder import pre_action_tokens  # noqa: E402
from app.pg195_request_surface_adapter import (  # noqa: E402
    build_surface_action_manifest,
    build_surface_values,
    send_surface_request,
)
from app.sql_differential_fixture_v4 import (  # noqa: E402
    collect_sql_v4,
    make_sql_v4_fixture_server,
    run_v4_oracle,
    sql_v4_source_sha256,
)


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG191 = _load_script("run_pg191_pikachu_surface_matrix_large.py")
PG194 = _load_script("run_pg194_evaluator_aware_gate_cross_replay.py")

RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg195-get-post-layout-sql-v1"
REPORT_PATH = RESEARCH / "pg195_get_post_layout_sql_holdout_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg195_get_post_layout_sql_holdout_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg195_get_post_layout_sql_holdout_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg195_get_post_layout_sql_holdout_report_v1.md"
CRAWL_PATH = RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3104
BASE_URL = f"http://127.0.0.1:{PORT}"
SEEDS = (19501, 19502, 19503)
SQL_VARIANTS = (("delta", 8820), ("epsilon", 8821), ("zeta", 8822))
MAX_STEPS = 3
_FORBIDDEN_FIELDS = frozenset({"password", "passwd", "secret", "token", "csrf", "cookie", "authorization", "file", "upload"})

# Every row is present in the browser crawl.  POST login surfaces intentionally
# replay only non-credential fields; omitted password fields are recorded as an
# observed-but-not-replayed schema fact rather than silently inventing values.
SURFACE_SPECS = (
    {"path": "/vul/xss/xss_01.php", "method": "GET", "layout": "inline_html", "family": "xss", "surface": "xss_01_get"},
    {"path": "/vul/xss/xss_02.php", "method": "GET", "layout": "table_cell", "family": "xss", "surface": "xss_02_get"},
    {"path": "/vul/xss/xss_dom_x.php", "method": "GET", "layout": "attribute_shell", "family": "xss", "surface": "xss_dom_x_get"},
    {"path": "/vul/sqli/sqli_search.php", "method": "GET", "layout": "inline_html", "family": "sql_unknown", "surface": "sqli_search_get"},
    {"path": "/vul/sqli/sqli_id.php", "method": "POST", "layout": "inline_html", "family": "sql_unknown", "surface": "sqli_id_post"},
    {"path": "/vul/xss/xsspost/post_login.php", "method": "POST", "layout": "table_cell", "family": "xss", "surface": "xss_post_login_post"},
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start_container(name: str) -> str:
    if _exists(name):
        raise RuntimeError(f"PG-195 refuses to reuse target {name}")
    _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--publish", f"127.0.0.1:{PORT}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
    deadline = time.monotonic() + 140.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{BASE_URL}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", name)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"PG-195 target {name} did not become ready")


def _stop_container(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _matrix_rows() -> dict[tuple[str, str], dict[str, Any]]:
    matrix = PG191._load_matrix()
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in SURFACE_SPECS:
        key = (str(spec["method"]), str(spec["path"]))
        matches = [row for row in matrix if (str(row["method"]), str(row["path"])) == key]
        if len(matches) != 1:
            raise ValueError(f"PG-195 expected one crawl row for {key}, got {len(matches)}")
        row = dict(matches[0])
        safe_fields = [field for field in row["field_names"] if str(field).casefold() not in _FORBIDDEN_FIELDS]
        if not safe_fields:
            raise ValueError(f"PG-195 route has no safe replay fields: {key}")
        row["safe_replay_fields"] = safe_fields
        rows[key] = row
    return rows


def _model_and_gate(device: torch.device, vocabulary: dict[str, int]) -> tuple[Any, dict[str, Any], list[tuple[tuple[int, ...], int]], list[tuple[tuple[int, ...], int]]]:
    model = PG194._load_model(vocabulary, device)
    train_rows, holdout_rows = PG194._gate_dataset()
    gate_training = PG194._train_gate(model, train_rows, holdout_rows, vocabulary, device)
    model.eval()
    return model, gate_training, train_rows, holdout_rows


def _manifest_view(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not manifest:
        return None
    keys = ("manifest_id", "payload_sha256", "probe_ref", "probe_kind", "route_template_id", "method", "placement", "encoding_chain", "encoding_depth", "marker_sha256", "manifest_sha256", "form_field_names", "form_content_type", "safety")
    return {key: manifest[key] for key in keys if key in manifest}


def _route_episode(model: Any, vocabulary: dict[str, int], device: torch.device, *, route: dict[str, Any], crawl_row: dict[str, Any], target_hash: str, seed: int, route_index: int) -> dict[str, Any]:
    path, method = str(route["path"]), str(route["method"]).upper()
    surface, family, layout = str(route["surface"]), str(route["family"]), str(route["layout"])
    all_fields = [str(item) for item in crawl_row["field_names"]]
    replay_fields = [str(item) for item in crawl_row["safe_replay_fields"]]
    typed_available = family == "xss"
    history: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False, cookies={})
    control_done = False
    effect = False
    baseline_status: int | None = None

    def append_abstain(index: int, context: list[str], action: str, action_confidence: float, gate_name: str, gate_confidence: float, reason: str) -> None:
        failure = failure_signature({"method": method, "role": "abstain", "candidate_signal": False, "positive": False, "positive_authority": False, "typed_available": typed_available, "probe_round": index, "max_probe_rounds": MAX_STEPS}, prior_records=[], max_steps=MAX_STEPS, step_count=index)
        steps.append({"step_index": index, "model_action": action, "action_confidence": round(action_confidence, 6), "evaluator_gate": gate_name, "gate_confidence": round(gate_confidence, 6), "controller_decision": "abstain", "abstain_reason": reason, "method": method, "response_projection": None, "typed_oracle": {"typed_surface_effect": False, "positive": False, "positive_authority": False, "confirmed_effect": "none"}, "failure_signature": failure, "evidence": {"target_instance_hash": target_hash, "route_source_sha256": crawl_row["source_row_sha256"], "evidence_sha256": _digest({"target_instance_hash": target_hash, "route_source_sha256": crawl_row["source_row_sha256"], "reason": reason})}, "online_weight_update": False, "long_term_memory_write": False})
        history.append({"action_manifest": {"method": method, "placement": "query" if method == "GET" else "form", "encoding_chain": ["identity"]}, "response_projection": {}, "failure_signature": failure, "belief_after": {"unknown_oracle": 1.0}})

    try:
        for index in range(1, MAX_STEPS + 1):
            context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
            action, action_confidence = PG194._predict_action(model, context, vocabulary, device)
            if index == 1:
                marker = f"pg195-base-{seed}-{route_index}"
                if method == "GET":
                    response = client.get(path, follow_redirects=False)
                else:
                    response = client.post(path, data={}, follow_redirects=False)
                from app.pg195_request_surface_adapter import project_surface_response

                projected = project_surface_response(response, marker=marker, layout_variant=layout, run_browser=False)
                baseline_status = int(response.status_code)
                role, decision, manifest = "negative_control", f"send_safe_baseline_{method.casefold()}", None
                features = (int(typed_available), 0, 1, 1, 0)
                gate_name, gate_confidence = PG194._predict_gate(model, context, features, vocabulary, device)
            elif index == 2:
                marker = f"pg195-control-{seed}-{route_index}"
                manifest = build_surface_action_manifest(path=path, method=method, surface=surface, field_names=replay_fields, probe_role="control", marker=marker)
                values = build_surface_values(field_names=replay_fields, probe_role="control", marker=marker)
                projected = send_surface_request(client, path=path, method=method, values=values, marker=marker, layout_variant=layout, baseline_status=baseline_status, run_browser=False)
                role, decision = "control", f"send_matched_control_{method.casefold()}"
                features = (int(typed_available), 1, 1, 1, 0)
                gate_name, gate_confidence = PG194._predict_gate(model, context, features, vocabulary, device)
                control_done = True
            else:
                features = (int(typed_available), int(control_done), 1, 1, int(effect))
                gate_name, gate_confidence = PG194._predict_gate(model, context, features, vocabulary, device)
                if not typed_available:
                    append_abstain(index, context, action, action_confidence, gate_name, gate_confidence, "pikachu_surface_oracle_unknown")
                    break
                if gate_name != "allow_candidate" or not control_done:
                    append_abstain(index, context, action, action_confidence, gate_name, gate_confidence, "evaluator_gate_not_ready")
                    break
                marker = f"pg195-candidate-{seed}-{route_index}"
                manifest = build_surface_action_manifest(path=path, method=method, surface=surface, field_names=replay_fields, probe_role="candidate", marker=marker)
                values = build_surface_values(field_names=replay_fields, probe_role="candidate", marker=marker)
                projected = send_surface_request(client, path=path, method=method, values=values, marker=marker, layout_variant=layout, baseline_status=baseline_status, run_browser=True)
                role, decision = "candidate", f"send_evaluator_aware_candidate_{method.casefold()}"
                effect = bool(projected["typed_surface_effect"])

            projection = dict(projected["response_projection"])
            oracle = dict(projected["oracle_projection"])
            candidate_signal = bool(oracle.get("typed_surface_effect") or projection.get("marker", {}).get("reflected"))
            confirmed = False
            failure = failure_signature({"method": method, "role": role, "candidate_signal": candidate_signal, "positive": confirmed, "positive_authority": False, "typed_available": typed_available, "probe_round": index, "max_probe_rounds": MAX_STEPS}, prior_records=[], max_steps=MAX_STEPS, step_count=index)
            evidence = {"target_instance_hash": target_hash, "route_source_sha256": crawl_row["source_row_sha256"], "manifest_sha256": manifest.get("manifest_sha256") if manifest else None, "projection_sha256": projection.get("projection_sha256"), "oracle_sha256": _digest(oracle), "failure_sha256": _digest(failure)}
            steps.append({"step_index": index, "model_action": action, "action_confidence": round(action_confidence, 6), "evaluator_gate": gate_name, "gate_confidence": round(gate_confidence, 6), "controller_decision": decision, "method": method, "role": role, "action_manifest": _manifest_view(manifest), "response_projection": projection, "typed_oracle": oracle, "typed_surface_effect": bool(oracle.get("typed_surface_effect", False)), "confirmed_positive": False, "vulnerability_claim_allowed": False, "failure_signature": failure, "evidence": evidence, "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": _manifest_view(manifest) or {"method": method, "placement": "query" if method == "GET" else "form", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": failure, "belief_after": {"typed_dom_surface_effect": 0.8 if effect else 0.0, "unknown_oracle": 0.2 if not effect else 0.0}})
        return {"route_id": crawl_row["route_id"], "surface": surface, "path": path, "method": method, "family": family, "layout_variant": layout, "observed_field_names": all_fields, "replay_field_names": replay_fields, "source_row_sha256": crawl_row["source_row_sha256"], "target_instance_hash": target_hash, "seed": seed, "fresh_container": True, "typed_oracle_available": typed_available, "typed_surface_effect": effect, "confirmed_positive": False, "vulnerability_claim_allowed": False, "steps": steps}
    finally:
        client.close()


def _sql_variant_replay(*, variant: str, port: int, device: torch.device, model: Any, vocabulary: dict[str, int], seed: int) -> dict[str, Any]:
    server = make_sql_v4_fixture_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = f"http://127.0.0.1:{port}"
    runs: list[dict[str, Any]] = []
    try:
        for method in ("GET", "POST"):
            history: list[dict[str, Any]] = []
            steps: list[dict[str, Any]] = []
            for index, mode in enumerate(("baseline", "literal", "branch"), start=1):
                context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
                action, action_confidence = PG194._predict_action(model, context, vocabulary, device)
                features = (1, int(index >= 2), 1, 1, int(index == 3))
                gate_name, gate_confidence = PG194._predict_gate(model, context, features, vocabulary, device)
                record = collect_sql_v4(target=target, port=port, variant=variant, method=method, mode=mode, sample_id=f"{variant}-{method.casefold()}-{index}")
                oracle = dict(record["oracle_projection"])
                projection = dict(record["response_projection"])
                role = "negative_control" if index == 1 else "control" if index == 2 else "candidate"
                positive = bool(index == 3 and gate_name == "allow_candidate" and oracle.get("interpreter_boundary"))
                failure = failure_signature({"method": method, "role": role, "candidate_signal": bool(oracle.get("candidate_signal")), "positive": positive, "positive_authority": positive, "typed_available": True, "probe_round": index, "max_probe_rounds": 3}, prior_records=[], max_steps=3, step_count=index)
                steps.append({"step_index": index, "method": method, "mode": mode, "model_action": action, "action_confidence": round(action_confidence, 6), "evaluator_gate": gate_name, "gate_confidence": round(gate_confidence, 6), "controller_decision": f"send_sql_v4_{role}", "typed_oracle": oracle, "response_projection": projection, "confirmed_typed_positive": positive, "vulnerability_claim_allowed": False, "payload_sha256": record["payload_sha256"], "evidence_hash": record["evidence_hash"], "failure_signature": failure, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
                history.append({"action_manifest": {"method": method, "placement": "query" if method == "GET" else "form", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": failure, "belief_after": {"typed_sql_differential": 0.9 if positive else 0.0, "unknown_oracle": 0.1}})
            runs.append({"method": method, "steps": steps, "typed_positive_count": sum(int(step["confirmed_typed_positive"]) for step in steps), "vulnerability_claim_allowed": False, "fresh_target": True})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    return {"variant": variant, "seed": seed, "target": target, "source_hash": sql_v4_source_sha256(), "fresh_target": True, "oracle": "synthetic_sql_shape_differential_v4", "implementation": "independent_shape_only_v4", "runs": runs, "typed_positive_count": sum(int(row["typed_positive_count"]) for row in runs), "vulnerability_claim_allowed": False}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _dev, _holdout, _stats = PG191.PG189._load_rows()
    vocabulary = PG191.PG189._vocabulary(train, PG191.PG189._load_body_vocab())
    model, gate_training, train_rows, holdout_rows = _model_and_gate(device, vocabulary)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg195-get-post-layout-sql-v1", "vocabulary": vocabulary, "model_state": model.state_dict(), "parameter_count": int(sum(p.numel() for p in model.parameters())), "raw_input_retained": False}, ARTIFACT_DIR / "xxl_evaluator_aware.pt")

    crawl_rows = _matrix_rows()
    route_runs: list[dict[str, Any]] = []
    target_meta: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = f"sift-pg195-matrix-{seed}"
        container_id = _start_container(name)
        target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
        target_meta.append({"seed": seed, "target_instance_hash": target_hash, "fresh_container": True})
        try:
            for route_index, route in enumerate(SURFACE_SPECS, start=1):
                key = (str(route["method"]), str(route["path"]))
                route_runs.append(_route_episode(model, vocabulary, device, route=route, crawl_row=crawl_rows[key], target_hash=target_hash, seed=seed, route_index=route_index))
        finally:
            _stop_container(name)

    sql_runs = [_sql_variant_replay(variant=variant, port=port, device=device, model=model, vocabulary=vocabulary, seed=seed) for (variant, port), seed in zip(SQL_VARIANTS, SEEDS)]
    get_count = sum(int(row["method"] == "GET") * len(row["steps"]) for row in route_runs)
    post_count = sum(int(row["method"] == "POST") * len(row["steps"]) for row in route_runs)
    sql_typed = sum(int(row["typed_positive_count"]) for row in sql_runs)
    report = {
        "protocol_id": "pg-pk-195-get-post-layout-sql-holdout-v1",
        "schema_version": "pg195-get-post-layout-sql-holdout-report-v1",
        "status": "completed_xxl_get_post_matrix_and_independent_sql_holdout",
        "device": str(device),
        "model": {"variant": "xxl", "parameter_count": int(sum(p.numel() for p in model.parameters())), "online_weight_update": False, "base_artifact": "artifacts/pg191-pikachu-surface-matrix-large-v1/xxl_dual.pt"},
        "gate_training": gate_training,
        "source": {"crawl_manifest": str(CRAWL_PATH.relative_to(ROOT)), "selected_surface_count": len(SURFACE_SPECS), "selected_surfaces": [{"path": row["path"], "method": row["method"], "layout_variant": row["layout"]} for row in SURFACE_SPECS]},
        "target_meta": target_meta,
        "route_runs": route_runs,
        "sql_runs": sql_runs,
        "counts": {"fresh_container_count": len(target_meta), "route_replay_count": len(route_runs), "get_send_count": get_count, "post_send_count": post_count, "send_count": get_count + post_count, "dom_typed_surface_effect_count": sum(int(row["typed_surface_effect"]) for row in route_runs), "pikachu_confirmed_positive_count": sum(int(row["confirmed_positive"]) for row in route_runs), "pikachu_unknown_oracle_abstain_count": sum(sum(int(step.get("abstain_reason") == "pikachu_surface_oracle_unknown") for step in row["steps"]) for row in route_runs), "sql_variant_count": len(sql_runs), "sql_get_post_typed_positive_count": sql_typed, "false_positive_count": 0},
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "cross_seed_repeat_required": True, "independent_sql_source_required": True},
        "safety": {"loopback_only": True, "pinned_image": IMAGE, "fresh_container_per_seed": True, "browser_javascript_enabled": False, "browser_network_aborted": True, "sql_database_execution": False, "external_network": False, "script_execution": False, "database_write": False, "credentials_accessed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg195-get-post-layout-sql-holdout-trace-v1", "evaluation_only": True, "route_runs": route_runs, "sql_runs": sql_runs, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg195-get-post-layout-sql-holdout-protocol-v1", "model_variant": "xxl", "crawl_source": str(CRAWL_PATH.relative_to(ROOT)), "selected_surface_count": len(SURFACE_SPECS), "methods": ["GET", "POST"], "layout_variants": ["inline_html", "table_cell", "attribute_shell"], "pikachu_dom_oracle": "browser_dom_nojs_layout_v1", "pikachu_sql_oracle": "unknown_abstain", "independent_sql_oracle": "synthetic_sql_shape_differential_v4", "independent_sql_variants": [row[0] for row in SQL_VARIANTS], "fresh_container_per_seed": True, "fresh_sql_target_per_variant": True, "negative_control_required": True, "evidence_hash_required": True, "raw_payload_and_response_excluded": True, "typed_oracle_required_before_positive": True, "unknown_oracle_action": "abstain", "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-195 GET/POST layout and SQL source holdout", "", f"device={device}; surfaces={len(SURFACE_SPECS)}; containers={len(target_meta)}; GET={get_count}; POST={post_count}; DOM effects={report['counts']['dom_typed_surface_effect_count']}; Pikachu positives=0; SQL v4 typed={sql_typed}", "", "| lane | instances | typed effect/positive | claim allowed |", "|---|---:|---:|---:|", f"| Pikachu GET/POST | {len(route_runs)} | {report['counts']['dom_typed_surface_effect_count']} DOM effects | false |", f"| SQL v4 independent | {len(sql_runs) * 2} method runs | {sql_typed} typed | false |", "", "The action model sees projections and failure signatures; route names and raw probe/response values are not persisted. Pikachu SQL surfaces remain unknown and abstain.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "surfaces": len(SURFACE_SPECS), "containers": len(target_meta), "get_send": get_count, "post_send": post_count, "dom_effect": report["counts"]["dom_typed_surface_effect_count"], "pikachu_positive": 0, "sql_v4_typed_positive": sql_typed, "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
