"""Collect PG-50's multi-implementation, multi-seed stability matrix."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import ReadOnlySafeCatalogCollector, build_catalog, sha256_json  # noqa: E402
from app.pg50_stability_matrix_fixture import IMPLEMENTATIONS, LAYOUTS, PHASES, PG50_SCHEMA, SURFACE_SPECS, VARIANTS, make_pg50_server  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json as trace_sha256_json, validate_trace_step  # noqa: E402

REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
CATALOG_OUTPUT = ROOT / "research" / "pg50_stability_matrix_catalog_v1.json"
TRACE_OUTPUT = ROOT / "research" / "pg50_stability_matrix_trace_v1.json"
SEEDS = (503, 509, 521, 523, 541)
TARGET_PORT = 32080
MAX_WORKERS = 16
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg50-stability-matrix-typed-oracle-v1").hexdigest()
EFFECTS = {surface["family"]: surface["effect"] for surface in SURFACE_SPECS.values()}


def _worker_port() -> int:
    local = getattr(_worker_port, "local", None)
    if local is None:
        local = threading.local()
        _worker_port.local = local
    if not hasattr(local, "port"):
        lock = getattr(_worker_port, "lock", None)
        if lock is None:
            lock = threading.Lock()
            _worker_port.lock = lock
        with lock:
            index = getattr(_worker_port, "counter", 0)
            _worker_port.counter = index + 1
        local.port = TARGET_PORT + index
    return int(local.port)


class _FreshTarget:
    def __init__(self, implementation: str) -> None:
        self.implementation = implementation
        self.port = _worker_port()
        self.server = make_pg50_server(self.port, implementation)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.005}, daemon=True)
        self.client: httpx.Client | None = None

    def __enter__(self) -> httpx.Client:
        self.thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.005)
        else:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2.0)
            raise RuntimeError("PG-50 target failed to start")
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=3.0, follow_redirects=False, headers={"accept": "application/json"})
        return self.client

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.client is not None:
            self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def _route(implementation: str, surface: str) -> str:
    return f"{LAYOUTS[implementation]['prefix']}/{surface}"


def _call(client: httpx.Client, implementation: str, surface: str, method: str, variant: str, phase: str, candidate: bool) -> httpx.Response:
    layout = LAYOUTS[implementation]
    spec = SURFACE_SPECS[surface]
    values = {layout["surface_key"]: surface, layout["probe_key"]: spec["probe"] if candidate else "ordinary-observation", layout["variant_key"]: variant, layout["phase_key"]: phase}
    if method == "GET":
        return client.get(_route(implementation, surface), params=values)
    headers = {"content-type": layout["post_content_type"]}
    if layout["post_content_type"] == "application/json":
        return client.post(_route(implementation, surface), content=json.dumps(values, separators=(",", ":")).encode(), headers=headers)
    body = "&".join(f"{key}={value}" for key, value in values.items())
    return client.post(_route(implementation, surface), content=body.encode(), headers=headers)


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        scalars = [item for item in value.values() if not isinstance(item, (dict, list))]
        return {"kind": "object", "key_count": len(value), "scalar_count": len(scalars), "array_count": sum(isinstance(item, list) for item in value.values()), "bool_count": sum(isinstance(item, bool) for item in scalars), "number_count": sum(isinstance(item, (int, float)) and not isinstance(item, bool) for item in scalars), "string_count": sum(isinstance(item, str) for item in scalars)}
    return {"kind": "other", "key_count": 0, "scalar_count": 1, "array_count": 0, "bool_count": 0, "number_count": 0, "string_count": 0}


def _projection(response: httpx.Response) -> tuple[dict[str, Any], dict[str, Any]]:
    body = bytes(response.content)
    parsed = response.json()
    shape = _shape(parsed)
    status = int(response.status_code)
    content_type = str(response.headers.get("content-type", "")).split(";", 1)[0].casefold()
    headers = sorted({str(key).casefold() for key in response.headers if str(key).casefold() in {"content-type", "x-pg50-surface"}})
    projection = {"status_code": status, "status_class": f"{status // 100}xx" if 100 <= status <= 599 else "other", "content_type_class": "json" if content_type == "application/json" else "other", "body_length_bucket": "0" if not body else "1-255" if len(body) <= 255 else "256-4095", "body_sha256": hashlib.sha256(body).hexdigest(), "semantic_body_sha256": sha256_json(shape), "shape": shape, "header_names": headers, "marker": {"reflected": False, "location": "none", "count": 0}, "frame_policy": "unknown", "transport_error": False, "status_changed": status >= 400, "state_changed": False, "location_origin_changed": False}
    projection["projection_sha256"] = sha256_json(projection)
    return projection, parsed


def _source(implementation: str, collector_hash: str, fixture_hash: str) -> dict[str, Any]:
    return {"target_id": "pg50_stability_matrix_fixture", "app_family": "standalone_python_http_stability_matrix_v1", "source_id": f"pg50-stability-source-{implementation}", "source_type": "in_repo_synthetic", "origin_ref": "app/pg50_stability_matrix_fixture.py:/stability/<implementation>/<surface>", "license": "in-repo-synthetic", "authorization": "workspace_local_only", "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": TARGET_PORT}, "fixture_source_sha256": hashlib.sha256(f"{fixture_hash}:{implementation}:pg50".encode()).hexdigest(), "collector_sha256": collector_hash, "reset_adapter_sha256": hashlib.sha256(f"pg50-reset-adapter:{implementation}".encode()).hexdigest(), "oracle_contract_sha256": ORACLE_CONTRACT_SHA256, "read_only": True, "external_network": False}


def _role(implementation: str, surface: str, variant: str, seed: int) -> str:
    if surface == "surface-10":
        return "negative_control"
    if surface in {"surface-08", "surface-09"}:
        return "family_holdout"
    if implementation == "quartz":
        return "implementation_holdout"
    if implementation == "frost":
        return "ood_source"
    if variant == "framed" or seed in {523, 541}:
        return "dev"
    return "train"


def _reset(source: dict[str, Any], implementation: str, surface: str, variant: str, seed: int, method: str, phase: str, pair_role: str, baseline: dict[str, Any]) -> dict[str, Any]:
    suffix = f"{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}-{phase}-{pair_role}"
    result = {"reset_id": f"pg50-reset-{suffix}", "kind": "fresh_pg50_stability_http_server", "target_instance_id": f"pg50-target-{suffix}", "state_epoch": f"pg50-epoch-{suffix}", "reset_adapter_sha256": source["reset_adapter_sha256"], "baseline_projection_sha256": baseline["projection_sha256"], "fresh_target": True, "completed": True, "evaluator_state_hidden": True, "state_change_allowed": False, "external_network": False, "transport": "httpx_loopback"}
    result["reset_sha256"] = sha256_json(result)
    return result


def _manifest(implementation: str, surface: str, variant: str, seed: int, method: str, phase: str, pair_role: str) -> dict[str, Any]:
    layout = LAYOUTS[implementation]
    spec = SURFACE_SPECS[surface]
    basis = {"adapter": "pg50", "implementation": implementation, "surface": surface, "semantic": spec["semantic"], "channel": spec["channel"], "variant": variant, "seed": seed, "method": method, "phase": phase, "pair_role": pair_role}
    return {"manifest_id": f"pg50-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}-{phase}-{pair_role}", "payload_sha256": sha256_json(basis), "probe_ref": f"pg50-semantic-{spec['semantic']}", "probe_kind": "inert_dom_markup" if spec["family"] == "xss" else "abstract_channel_class", "route_template_id": f"pg50-{implementation}-{surface}", "method": method, "placement": "query" if method == "GET" else "form", "encoding_chain": ["identity"], "encoding_depth": 0, "marker_sha256": hashlib.sha256(f"pg50-marker:{implementation}:{surface}:{variant}:{seed}".encode()).hexdigest(), "max_bytes": 256, "form_field_names": [layout["surface_key"], layout["probe_key"], layout["variant_key"], layout["phase_key"]] if method == "POST" else [], "form_content_type": layout["post_content_type"] if method == "POST" else "", "safety": {"does_not_execute": True, "no_external_network": True, "no_script_execution": True, "no_database_write": True, "no_credential_access": True}}


def _oracle(implementation: str, surface: str, variant: str, method: str, phase: str, parsed: dict[str, Any]) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    positive = bool(parsed.get("effect", {}).get("confirmed", False)) and phase == "confirm"
    allowed = {"candidate_signal", "ambiguous", "bounded_observation", "dom_structure_delta", "marker_hits", "ast_shape_delta", "operator_boundary", "ast_node_delta", "authentication_boundary", "authenticated", "authorization_boundary", "cross_subject_access", "business_invariant_boundary", "history_changed", "redirect_candidate", "same_origin", "external_redirect", "validation_boundary", "rejected", "command_boundary", "command_canary", "executed", "template_context_delta", "render_executed", "ordinary_response", "fault", "state_mutated", "database_touched", "credentials_accessed", "external_network", "script_execution", "effect", "proof"}
    safe = {key: parsed[key] for key in parsed if key in allowed}
    return {"oracle_id": f"pg50-stability-{spec['family']}-oracle-v1", "oracle_contract_sha256": ORACLE_CONTRACT_SHA256, "family": spec["family"], "modality": spec["modality"] if positive else "negative_control", "candidate_signal": bool(parsed.get("candidate_signal", False)), "positive": positive, "positive_authority": positive, "confirmed_effect": EFFECTS[spec["family"]] if positive else "none", "signals": {"implementation": implementation, "semantic_reference": spec["semantic"], "channel_reference": spec["channel"], "surface_variant": variant, "method": method, "phase": phase, **safe}, "safety": {"external_network": False, "script_execution": False, "database_write": False, "persistent_state_mutated": False, "credentials_accessed": False, "raw_body_stored": False}}


def _rule(surface: str, variant: str, method: str, phase: str) -> dict[str, Any]:
    family = SURFACE_SPECS[surface]["family"]
    return {"rule_key": f"{family}.pg50.{variant}.{method.casefold()}.{phase}", "grammar_version": "rule-ir-v1", "family_candidate": family, "operator_set": ["and", "eq", "present"], "required_slots": ["surface", "transport", "oracle"], "bound_slots": ["surface", "transport", "oracle"], "executable": False}


def _trace(row: dict[str, Any], control: dict[str, Any], episode_id: str, step_id: str, parent: str | None, next_action: str, belief: dict[str, float]) -> dict[str, Any]:
    oracle = dict(row["oracle_projection"])
    oracle["negative_control_pair_id"] = control["sample_id"]
    manifest = row["payload_manifest"]
    action = {"method": manifest["method"], "route_template_id": manifest["route_template_id"], "placement": manifest["placement"], "encoding_chain": manifest["encoding_chain"], "probe_ref": manifest["probe_ref"], "probe_sha256": manifest["payload_sha256"], "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if manifest["method"] == "POST":
        action["form_field_names"] = manifest["form_field_names"]
    positive = bool(oracle.get("positive", False))
    after = {"unknown": 0.05, row["family"]: 0.95} if positive else {"unknown": 1.0}
    decision = "confirmed_positive" if positive else "confirmed_negative" if next_action == "stop_episode" else "candidate"
    echo_body = {"action_manifest": action, "baseline_projection": control["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_before": belief, "belief_after": after, "decision": decision, "next_action": next_action}
    return validate_trace_step({"episode_id": episode_id, "step_id": step_id, "parent_step_id": parent, "sampling_seed": int(row["sampling_seed"]), "target_instance_id": row["target_instance_id"], "hypothesis": "unknown_surface", "belief_before": belief, "action_manifest": action, "baseline_projection": control["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_after": after, "decision": decision, "next_action": next_action, "fresh_reset": row["reset"], "evidence_sha256": row["evidence"]["evidence_hash"], "dataset_stage": "trace_only", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": trace_sha256_json(echo_body)}})


def _episode(task: tuple[str, str, int, str, dict[str, Any], str, str]) -> dict[str, Any]:
    implementation, surface, seed, variant, source, collector_hash, fixture_hash = task
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector = ReadOnlySafeCatalogCollector(source, registry=registry)
    spec = SURFACE_SPECS[surface]
    episode_id = f"pg50-episode-{implementation}-{surface}-{variant}-s{seed}"
    records: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    previous: str | None = None
    belief = {"unknown": 1.0}
    for method in ("GET", "POST"):
        for phase in PHASES:
            with _FreshTarget(implementation) as client:
                control_response = _call(client, implementation, surface, method, variant, phase, False)
            with _FreshTarget(implementation) as client:
                candidate_response = _call(client, implementation, surface, method, variant, phase, True)
            baseline, _ = _projection(control_response)
            control_projection, control_parsed = _projection(control_response)
            candidate_projection, candidate_parsed = _projection(candidate_response)
            control_oracle = _oracle(implementation, surface, variant, method, phase, control_parsed)
            candidate_oracle = _oracle(implementation, surface, variant, method, phase, candidate_parsed)
            control_id = f"pg50-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}-{phase}-control"
            candidate_id = f"pg50-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}-{phase}-candidate"
            role = _role(implementation, surface, variant, seed)
            control = collector.collect(sample_id=control_id, sample_role="negative_control", sampling_seed=seed, reset=_reset(collector.source, implementation, surface, variant, seed, method, phase, "control", baseline), payload_manifest=_manifest(implementation, surface, variant, seed, method, phase, "control"), response_projection=control_projection, oracle_projection=control_oracle, rule_ir=_rule(surface, variant, method, phase))
            positive = bool(candidate_oracle["positive"])
            candidate = collector.collect(sample_id=candidate_id, sample_role="candidate" if positive else "negative_control", sampling_seed=seed, reset=_reset(collector.source, implementation, surface, variant, seed, method, phase, "candidate", baseline), payload_manifest=_manifest(implementation, surface, variant, seed, method, phase, "candidate"), response_projection=candidate_projection, oracle_projection=candidate_oracle, rule_ir=_rule(surface, variant, method, phase), negative_control={"control_sample_id": control["sample_id"], "control_evidence_hash": control["evidence"]["evidence_hash"], "intervention": "stability-probe-vs-ordinary-control", "verdict": "confirmed_negative", "same_source": True, "same_surface": True} if positive else None)
            for row, sample_role, pair_role in ((control, "negative_control", "control"), (candidate, "candidate" if positive else "negative_control", "candidate")):
                row.update({"dataset_role": role, "implementation": implementation, "surface_id": surface, "surface_variant": variant, "semantic_reference": spec["semantic"], "channel_reference": spec["channel"], "required_channel": spec["channel"], "family": spec["family"], "method": method, "phase": phase, "sample_role": sample_role, "pair_role": pair_role})
                records.append(row)
            for row, action in ((control, "replay_candidate"), (candidate, "stop_episode" if positive else "next_probe")):
                step = _trace(row, control, episode_id, f"{episode_id}-{method.casefold()}-{phase}-{row['pair_role']}", previous, action, belief)
                previous = step["step_id"]
                belief = step["belief_after"]
                steps.append(step)
    return {"implementation": implementation, "records": records, "steps": steps, "episode": evaluate_episode(steps)}


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_path = ROOT / "app" / "pg50_stability_matrix_fixture.py"
    fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    raw_sources = {implementation: _source(implementation, collector_hash, fixture_hash) for implementation in IMPLEMENTATIONS}
    tasks = [(implementation, surface, seed, variant, raw_sources[implementation], collector_hash, fixture_hash) for implementation in IMPLEMENTATIONS for surface in SURFACE_SPECS for seed in SEEDS for variant in VARIANTS]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="pg50") as executor:
        futures = [executor.submit(_episode, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["implementation"], item["records"][0]["sample_id"]))
    all_records = [row for result in results for row in result["records"]]
    trace_steps = [step for result in results for step in result["steps"]]
    episodes = [result["episode"] for result in results]
    sources = [ReadOnlySafeCatalogCollector(raw_sources[implementation], registry=registry).source for implementation in IMPLEMENTATIONS]
    source_catalogs = []
    for source in sources:
        implementation = source["source_id"].rsplit("-", 1)[-1]
        rows = [row for row in all_records if row["implementation"] == implementation]
        variant_catalog = build_catalog(f"pg50-stability-{source['source_id']}-catalog", source, rows)
        source_catalogs.append({"source_id": source["source_id"], "source_sha256": source["source_sha256"], "catalog_sha256": variant_catalog["catalog_sha256"], "sample_count": len(rows), "training_eligible": variant_catalog["training_eligible"]})
    dataset_tests = []
    for role in ("train", "dev", "family_holdout", "ood_source", "implementation_holdout", "negative_control"):
        rows = [row for row in all_records if row["dataset_role"] == role]
        if not rows:
            continue
        ids = sorted(row["sample_id"] for row in rows)
        targets = sorted({row["target_instance_id"] for row in rows})
        source_hashes = sorted({row["source_sha256"] for row in rows})
        summary = {"sample_id": f"pg50-test-{role}", "dataset_id": f"pg50-{role}-v1", "source_id": f"pg50-source-{role}", "source_hash": sha256_json(source_hashes), "target_instance_ids": targets, "family_set": sorted({row["family"] for row in rows}), "sampling_seed": min(int(row["sampling_seed"]) for row in rows), "role": role, "sample_count": len(rows), "unique_sample_count": len(ids), "denominator": len(rows), "positive_count": sum(int(row["oracle_projection"]["positive"]) for row in rows), "negative_count": sum(int(not row["oracle_projection"]["positive"]) for row in rows), "abstain_count": 0, "method_set": ["GET", "POST"], "phase_set": list(PHASES), "semantic_reference_set": sorted({row["semantic_reference"] for row in rows}), "channel_reference_set": sorted({row["channel_reference"] for row in rows}), "source_count": len(source_hashes), "dataset_manifest_sha256": sha256_json(ids), "split_manifest_sha256": sha256_json({"role": role, "sources": source_hashes}), "probe_sha256": sha256_json([row["payload_manifest"]["payload_sha256"] for row in rows]), "oracle_contract_sha256": ORACLE_CONTRACT_SHA256, "checkpoint_sha256": sha256_json("pending-model-run"), "metrics_status": "pending_model_run", "metrics": {"typed_recall": 0.0, "precision": 0.0, "false_positive_rate": 0.0, "abstain_precision": 0.0, "ece": 0.0, "median_queries": 0.0}}
        summary["evidence_hash"] = sha256_json(summary)
        dataset_tests.append(summary)
    episodes.sort(key=lambda item: item["episode_id"])
    trace_steps.sort(key=lambda item: (item["episode_id"], item["step_id"]))
    catalog = {"schema_version": "pg-pk-50-stability-matrix-catalog-v1", "catalog_id": "pg50-stability-matrix-v1", "purpose": "multi-implementation multi-seed multi-surface GET/POST typed-oracle stability matrix", "runtime_replay": True, "independent_target_implementation": True, "evaluation_only": True, "training_eligible": True, "training_artifact_generated": False, "model_evaluation_completed": False, "methods": ["GET", "POST"], "phases": list(PHASES), "surface_variants": list(VARIANTS), "seeds": list(SEEDS), "implementations": list(IMPLEMENTATIONS), "families": sorted({row["family"] for row in all_records}), "semantic_references": sorted({row["semantic_reference"] for row in all_records}), "channel_references": sorted({row["channel_reference"] for row in all_records}), "sources": sources, "source_catalogs": source_catalogs, "samples": all_records, "dataset_tests": dataset_tests, "trace_dataset": str(TRACE_OUTPUT.relative_to(ROOT)), "trace_episode_count": len(episodes), "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episodes), "typed_positive_count": sum(int(row["oracle_projection"]["positive"]) for row in all_records), "negative_control_count": sum(int(not row["oracle_projection"]["positive"]) for row in all_records), "fresh_reset_count": len(all_records), "source_count": len({row["source_sha256"] for row in all_records}), "target_instance_count": len({row["target_instance_id"] for row in all_records}), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "external_network": False, "authorization": "workspace_local_only", "manifest_sha256": sha256_json({"samples": [row["evidence"]["evidence_hash"] for row in all_records], "dataset_tests": dataset_tests})}
    trace = {"schema_version": "pg-pk-50-stability-matrix-trace-v1", "purpose": "multi-implementation multi-seed multi-surface GET/POST replay trace", "evaluation_only": True, "training_eligible": False, "independent_target_implementation": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "methods": ["GET", "POST"], "phases": list(PHASES), "surface_variants": list(VARIANTS), "episodes": episodes, "episode_count": len(episodes), "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episodes), "steps": trace_steps, "catalog_manifest_sha256": catalog["manifest_sha256"], "trace_manifest_sha256": trace_sha256_json([step["trace_sha256"] for step in trace_steps])}
    CATALOG_OUTPUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_OUTPUT.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"catalog": str(CATALOG_OUTPUT.relative_to(ROOT)), "trace": str(TRACE_OUTPUT.relative_to(ROOT)), "sample_count": len(all_records), "typed_positive_count": catalog["typed_positive_count"], "negative_count": catalog["negative_control_count"], "pair_count": len(all_records) // 2, "source_count": catalog["source_count"], "episode_count": len(episodes), "accepted_evaluation_episodes": catalog["accepted_evaluation_episode_count"], "implementations": catalog["implementations"], "families": catalog["families"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
