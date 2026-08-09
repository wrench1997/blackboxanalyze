"""Collect PG-40 with language-neutral semantic probe references.

The underlying target is the safe PG-37 loopback fixture, but the adapter
attaches a bounded semantic ``probe_ref`` such as ``state-invariant``.  The
reference never contains a vulnerability-family name or a raw probe.
"""

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
from app.pg37_counterfactual_fixture import LAYOUTS, PHASES, SURFACE_SPECS, make_pg37_server  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json as trace_sha256_json, validate_trace_step  # noqa: E402


REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
CATALOG_OUTPUT = ROOT / "research" / "pg40_semantic_router_catalog_v1.json"
TRACE_OUTPUT = ROOT / "research" / "pg40_semantic_router_trace_v1.json"
SEEDS = (361, 367, 373)
TARGET_PORT = 31940
MAX_WORKERS = 8
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg40-semantic-router-typed-oracle-v1").hexdigest()
ROLE_FAMILIES = {"train": ["xss", "injection", "authentication", "access_control", "logic", "url_redirect", "input_validation", "command_injection", "ordinary_response", "unknown_surface"]}
SEMANTIC_REFERENCE = {
    "surface-01": "markup-context",
    "surface-02": "operator-context",
    "surface-03": "auth-boundary",
    "surface-04": "subject-boundary",
    "surface-05": "state-invariant",
    "surface-06": "url-target",
    "surface-07": "scalar-boundary",
    "surface-08": "local-canary",
    "surface-09": "ordinary-surface",
    "surface-10": "ordinary-surface",
}
SEMANTIC_BY_FAMILY = {spec["family"]: SEMANTIC_REFERENCE[surface] for surface, spec in SURFACE_SPECS.items()}


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
        self.server = make_pg37_server(self.port, implementation)
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
            self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2.0)
            raise RuntimeError("PG-40 target failed to start")
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=3.0, follow_redirects=False, headers={"accept": "application/json"})
        return self.client

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.client is not None:
            self.client.close()
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2.0)


def _route(implementation: str, surface: str) -> str:
    return f"{LAYOUTS[implementation]['prefix']}/{surface}"


def _call(client: httpx.Client, implementation: str, surface: str, method: str, phase: str, *, positive: bool) -> httpx.Response:
    layout = LAYOUTS[implementation]
    spec = SURFACE_SPECS[surface]
    values = {layout["slot_key"]: surface, layout["probe_key"]: spec["positive"] if positive else "normal", layout["variant_key"]: "compact", layout["phase_key"]: phase}
    if method == "GET":
        return client.get(_route(implementation, surface), params=values)
    headers = {"content-type": layout["post_content_type"]}
    if layout["post_content_type"] == "application/json":
        return client.post(_route(implementation, surface), content=json.dumps(values, separators=(",", ":")).encode(), headers=headers)
    return client.post(_route(implementation, surface), content="&".join(f"{k}={v}" for k, v in values.items()).encode(), headers=headers)


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
    projection = {"status_code": status, "status_class": f"{status // 100}xx" if 100 <= status <= 599 else "other", "content_type_class": "json", "body_length_bucket": "0" if not body else "1-255" if len(body) <= 255 else "256-4095", "body_sha256": hashlib.sha256(body).hexdigest(), "semantic_body_sha256": sha256_json(shape), "shape": shape, "header_names": ["content-type"], "marker": {"reflected": False, "location": "none", "count": 0}, "frame_policy": "unknown", "transport_error": False, "status_changed": status >= 400, "state_changed": False, "location_origin_changed": False}
    projection["projection_sha256"] = sha256_json(projection)
    return projection, parsed


def _source(implementation: str, collector_hash: str, fixture_hash: str) -> dict[str, Any]:
    return {"target_id": "pg40_semantic_router_fixture", "app_family": "standalone_python_http_semantic_router_v1", "source_id": f"pg40-semantic-source-{implementation}", "source_type": "in_repo_synthetic", "origin_ref": "app/pg37_counterfactual_fixture.py:/semantic/<implementation>/<surface>", "license": "in-repo-synthetic", "authorization": "workspace_local_only", "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": TARGET_PORT}, "fixture_source_sha256": hashlib.sha256(f"{fixture_hash}:{implementation}:pg40".encode()).hexdigest(), "collector_sha256": collector_hash, "reset_adapter_sha256": hashlib.sha256(f"pg40-semantic-reset:{implementation}".encode()).hexdigest(), "oracle_contract_sha256": ORACLE_CONTRACT_SHA256, "read_only": True, "external_network": False}


def _reset(source: dict[str, Any], implementation: str, surface: str, seed: int, method: str, phase: str, pair_role: str, baseline: dict[str, Any]) -> dict[str, Any]:
    suffix = f"{implementation}-{surface}-s{seed}-{method.casefold()}-{phase}-{pair_role}"
    result = {"reset_id": f"pg40-reset-{suffix}", "kind": "fresh_pg40_semantic_http_server", "target_instance_id": f"pg40-target-{suffix}", "state_epoch": f"pg40-epoch-{suffix}", "reset_adapter_sha256": source["reset_adapter_sha256"], "baseline_projection_sha256": baseline["projection_sha256"], "fresh_target": True, "completed": True, "evaluator_state_hidden": True, "state_change_allowed": False, "external_network": False, "transport": "httpx_loopback"}
    result["reset_sha256"] = sha256_json(result)
    return result


def _manifest(implementation: str, surface: str, seed: int, method: str, phase: str, pair_role: str) -> dict[str, Any]:
    layout = LAYOUTS[implementation]
    spec = SURFACE_SPECS[surface]
    probe_kind = "http_canary" if phase == "timeout" else "inert_dom_markup" if spec["family"] == "xss" else "abstract_channel_class"
    semantic = SEMANTIC_REFERENCE[surface]
    basis = {"adapter": "pg40", "implementation": implementation, "semantic": semantic, "seed": seed, "method": method, "phase": phase, "pair_role": pair_role}
    return {"manifest_id": f"pg40-{implementation}-{surface}-s{seed}-{method.casefold()}-{phase}-{pair_role}", "payload_sha256": sha256_json(basis), "probe_ref": f"pg40-semantic-{semantic}", "probe_kind": probe_kind, "route_template_id": f"pg40-{implementation}-{surface}", "method": method, "placement": "query" if method == "GET" else "form", "encoding_chain": ["identity"], "encoding_depth": 0, "marker_sha256": hashlib.sha256(f"pg40-marker:{implementation}:{surface}:{seed}".encode()).hexdigest(), "max_bytes": 256, "form_field_names": [layout["slot_key"], layout["probe_key"], layout["variant_key"], layout["phase_key"]] if method == "POST" else [], "form_content_type": layout["post_content_type"] if method == "POST" else "", "safety": {"does_not_execute": True, "no_external_network": True, "no_script_execution": True, "no_database_write": True, "no_credential_access": True}}


def _oracle(implementation: str, surface: str, method: str, phase: str, parsed: dict[str, Any]) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    positive = bool(parsed.get("typed_effect_ready", False)) and phase == "confirm" and spec["family"] not in {"ordinary_response", "unknown_surface"}
    allowed = {"candidate_signal", "ambiguous", "typed_effect_ready", "dom_change", "marker_hits", "ast_shape_diff", "interpreter_boundary", "authentication_boundary", "authorization_boundary", "cross_subject_access", "business_invariant_boundary", "redirect_candidate", "same_origin", "external_redirect", "validation_boundary", "rejected", "local_canary_observed", "command_executed", "ordinary_response", "error_class", "timeout_class", "state_mutated", "database_touched", "credentials_accessed", "external_network", "script_execution", "bounded_response_delta"}
    safe = {key: parsed[key] for key in parsed if key in allowed}
    return {"oracle_id": f"pg40-semantic-{spec['family']}-oracle-v1", "oracle_contract_sha256": ORACLE_CONTRACT_SHA256, "family": spec["family"], "modality": spec["modality"] if positive else "negative_control", "candidate_signal": bool(parsed.get("candidate_signal", False)), "positive": positive, "positive_authority": positive, "confirmed_effect": spec["effect"] if positive else "none", "signals": {"implementation": implementation, "semantic_reference": SEMANTIC_REFERENCE[surface], "method": method, "phase": phase, **safe}, "safety": {"external_network": False, "script_execution": False, "database_write": False, "persistent_state_mutated": False, "credentials_accessed": False, "raw_body_stored": False}}


def _rule(surface: str, method: str, phase: str) -> dict[str, Any]:
    family = SURFACE_SPECS[surface]["family"]
    return {"rule_key": f"{family}.pg40-semantic.{method.casefold()}.{phase}", "grammar_version": "rule-ir-v1", "family_candidate": family, "operator_set": ["and", "eq", "present"], "required_slots": ["surface", "transport", "oracle"], "bound_slots": ["surface", "transport", "oracle"], "executable": False}


def _trace(row: dict[str, Any], episode_id: str, step_id: str, parent: str | None, next_action: str, pair_id: str | None, belief: dict[str, float]) -> dict[str, Any]:
    oracle = dict(row["oracle_projection"])
    if pair_id:
        oracle["negative_control_pair_id"] = pair_id
    family = row["family"]
    positive = bool(oracle.get("positive", False))
    after = {"unknown": 0.05, family: 0.95} if positive else {"unknown": 0.97, family: 0.03}
    manifest = row["payload_manifest"]
    action = {"method": manifest["method"], "route_template_id": manifest["route_template_id"], "placement": manifest["placement"], "encoding_chain": manifest["encoding_chain"], "probe_ref": manifest["probe_ref"], "probe_sha256": manifest["payload_sha256"], "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if manifest["method"] == "POST":
        action["form_field_names"] = manifest["form_field_names"]
    decision = "confirmed_positive" if positive else "confirmed_negative"
    echo_body = {"action_manifest": action, "baseline_projection": row["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_before": belief, "belief_after": after, "decision": decision, "next_action": next_action}
    return validate_trace_step({"episode_id": episode_id, "step_id": step_id, "parent_step_id": parent, "sampling_seed": int(row["sampling_seed"]), "target_instance_id": row["target_instance_id"], "hypothesis": family, "belief_before": belief, "action_manifest": action, "baseline_projection": row["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_after": after, "decision": decision, "next_action": next_action, "fresh_reset": row["reset"], "evidence_sha256": row["evidence"]["evidence_hash"], "dataset_stage": "trace_only", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": trace_sha256_json(echo_body)}})


def _episode(task: tuple[str, str, int, int, dict[str, Any], str, str]) -> dict[str, Any]:
    implementation, surface, seed, _hint, source, collector_hash, fixture_hash = task
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector = ReadOnlySafeCatalogCollector(source, registry=registry)
    spec = SURFACE_SPECS[surface]
    role = "train"
    episode_id = f"pg40-episode-{implementation}-{surface}-s{seed}"
    records: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    previous: str | None = None
    belief = {"unknown": 1.0}
    for method in ("GET", "POST"):
        for phase in PHASES:
            with _FreshTarget(implementation) as client:
                control_response = _call(client, implementation, surface, method, phase, positive=False)
            with _FreshTarget(implementation) as client:
                candidate_response = _call(client, implementation, surface, method, phase, positive=spec["family"] not in {"ordinary_response", "unknown_surface"})
            baseline, _ = _projection(control_response)
            control_projection, control_parsed = _projection(control_response)
            candidate_projection, candidate_parsed = _projection(candidate_response)
            control_oracle = _oracle(implementation, surface, method, phase, control_parsed)
            candidate_oracle = _oracle(implementation, surface, method, phase, candidate_parsed)
            positive = bool(candidate_oracle["positive"])
            control_id = f"pg40-{implementation}-{surface}-s{seed}-{method.casefold()}-{phase}-control"
            candidate_id = f"pg40-{implementation}-{surface}-s{seed}-{method.casefold()}-{phase}-candidate"
            control = collector.collect(sample_id=control_id, sample_role="negative_control", sampling_seed=seed, reset=_reset(collector.source, implementation, surface, seed, method, phase, "control", baseline), payload_manifest=_manifest(implementation, surface, seed, method, phase, "control"), response_projection=control_projection, oracle_projection=control_oracle, rule_ir=_rule(surface, method, phase))
            candidate = collector.collect(sample_id=candidate_id, sample_role="candidate" if positive else "negative_control", sampling_seed=seed, reset=_reset(collector.source, implementation, surface, seed, method, phase, "candidate", baseline), payload_manifest=_manifest(implementation, surface, seed, method, phase, "candidate"), response_projection=candidate_projection, oracle_projection=candidate_oracle, rule_ir=_rule(surface, method, phase), negative_control=({"control_sample_id": control["sample_id"], "control_evidence_hash": control["evidence"]["evidence_hash"], "intervention": "semantic-class-vs-normal-control", "verdict": "confirmed_negative", "same_source": True, "same_surface": True} if positive else None))
            for row, sample_role, pair_role in ((control, "negative_control", "control"), (candidate, "candidate" if positive else "negative_control", "candidate")):
                row.update({"dataset_role": role, "implementation": implementation, "surface_id": surface, "surface_variant": "compact", "semantic_reference": SEMANTIC_REFERENCE[surface], "family": spec["family"], "method": method, "phase": phase, "sample_role": sample_role, "pair_role": pair_role})
                records.append(row)
            for row, pair_id, action in ((control, None, "replay_candidate"), (candidate, control["sample_id"] if positive else None, "confirm_same_surface" if phase == "screen" else "stop_episode" if positive else "next_probe")):
                step = _trace(row, episode_id, f"{episode_id}-{method.casefold()}-{phase}-{row['pair_role']}", previous, action, pair_id, belief)
                previous = step["step_id"]; belief = step["belief_after"]; steps.append(step)
    return {"implementation": implementation, "records": records, "steps": steps, "episode": evaluate_episode(steps)}


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_hash = hashlib.sha256((ROOT / "app" / "pg37_counterfactual_fixture.py").read_bytes()).hexdigest()
    raw_sources = {implementation: _source(implementation, collector_hash, fixture_hash) for implementation in LAYOUTS}
    tasks: list[tuple[str, str, int, int, dict[str, Any], str, str]] = []
    index = 0
    for implementation in LAYOUTS:
        for surface in SURFACE_SPECS:
            for seed in SEEDS:
                tasks.append((implementation, surface, seed, index, raw_sources[implementation], collector_hash, fixture_hash)); index += 1
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="pg40") as executor:
        futures = [executor.submit(_episode, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["implementation"], item["records"][0]["sample_id"]))
    all_records = [row for result in results for row in result["records"]]
    trace_steps = [step for result in results for step in result["steps"]]
    episodes = [result["episode"] for result in results]
    sources = [ReadOnlySafeCatalogCollector(raw_sources[implementation], registry=registry).source for implementation in LAYOUTS]
    source_catalogs = []
    for source in sources:
        implementation_records = [row for row in all_records if row["implementation"] == source["source_id"].rsplit("-", 1)[-1]]
        variant_catalog = build_catalog(f"pg40-semantic-{source['source_id']}-catalog", source, implementation_records)
        source_catalogs.append({"source_id": source["source_id"], "source_sha256": source["source_sha256"], "catalog_sha256": variant_catalog["catalog_sha256"], "sample_count": len(implementation_records), "training_eligible": variant_catalog["training_eligible"]})
    dataset_tests = []
    for seed in SEEDS:
        rows = [row for row in all_records if int(row["sampling_seed"]) == seed]
        ids = sorted(row["sample_id"] for row in rows)
        targets = sorted({row["target_instance_id"] for row in rows})
        source_hashes = sorted({row["source_sha256"] for row in rows})
        summary = {"sample_id": f"pg40-test-source-s{seed}", "dataset_id": f"pg40-source-s{seed}-v1", "source_id": f"pg40-source-s{seed}", "source_hash": sha256_json(source_hashes), "target_instance_ids": targets, "family_set": sorted({row["family"] for row in rows}), "sampling_seed": seed, "role": "train", "sample_count": len(rows), "unique_sample_count": len(ids), "denominator": len(rows), "positive_count": sum(int(row["oracle_projection"]["positive"]) for row in rows), "negative_count": sum(int(not row["oracle_projection"]["positive"]) for row in rows), "abstain_count": 0, "method_set": ["GET", "POST"], "phase_set": list(PHASES), "semantic_reference_set": sorted({row["semantic_reference"] for row in rows}), "source_count": len(source_hashes), "dataset_manifest_sha256": sha256_json(ids), "split_manifest_sha256": sha256_json({"seed": seed, "sources": source_hashes}), "probe_sha256": sha256_json([row["payload_manifest"]["payload_sha256"] for row in rows]), "oracle_contract_sha256": ORACLE_CONTRACT_SHA256, "checkpoint_sha256": sha256_json("pending-model-run"), "metrics_status": "pending_model_run", "metrics": {"typed_recall": 0.0, "precision": 0.0, "false_positive_rate": 0.0, "abstain_precision": 0.0, "ece": 0.0, "median_queries": 0.0}}
        summary["evidence_hash"] = sha256_json(summary); dataset_tests.append(summary)
    episodes.sort(key=lambda item: item["episode_id"]); trace_steps.sort(key=lambda item: (item["episode_id"], item["step_id"]))
    catalog = {"schema_version": "pg-pk-40-semantic-router-catalog-v1", "catalog_id": "pg40-semantic-router-v1", "purpose": "abstract semantic probe references with source-holdout typed replay", "runtime_replay": True, "independent_target_implementation": True, "evaluation_only": True, "training_eligible": True, "training_artifact_generated": False, "model_evaluation_completed": False, "methods": ["GET", "POST"], "phases": list(PHASES), "surface_variants": ["compact"], "seeds": list(SEEDS), "implementations": list(LAYOUTS), "semantic_references": sorted(set(SEMANTIC_REFERENCE.values())), "sources": sources, "source_catalogs": source_catalogs, "samples": all_records, "dataset_tests": dataset_tests, "trace_dataset": str(TRACE_OUTPUT.relative_to(ROOT)), "trace_episode_count": len(episodes), "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episodes), "typed_positive_count": sum(int(row["oracle_projection"]["positive"]) for row in all_records), "negative_control_count": sum(int(not row["oracle_projection"]["positive"]) for row in all_records), "fresh_reset_count": len(all_records), "source_count": len({row["source_sha256"] for row in all_records}), "target_instance_count": len({row["target_instance_id"] for row in all_records}), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "external_network": False, "authorization": "workspace_local_only", "manifest_sha256": sha256_json({"samples": [row["evidence"]["evidence_hash"] for row in all_records], "dataset_tests": dataset_tests})}
    trace = {"schema_version": "pg-pk-40-semantic-router-trace-v1", "purpose": "semantic probe route trace", "evaluation_only": True, "training_eligible": False, "independent_target_implementation": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "methods": ["GET", "POST"], "phases": list(PHASES), "episodes": episodes, "episode_count": len(episodes), "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episodes), "steps": trace_steps, "catalog_manifest_sha256": catalog["manifest_sha256"], "trace_manifest_sha256": trace_sha256_json([step["trace_sha256"] for step in trace_steps])}
    CATALOG_OUTPUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_OUTPUT.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"catalog": str(CATALOG_OUTPUT.relative_to(ROOT)), "trace": str(TRACE_OUTPUT.relative_to(ROOT)), "sample_count": len(all_records), "typed_positive_count": catalog["typed_positive_count"], "negative_count": catalog["negative_control_count"], "source_count": catalog["source_count"], "episode_count": len(episodes), "accepted_evaluation_episodes": catalog["accepted_evaluation_episode_count"], "semantic_references": catalog["semantic_references"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
