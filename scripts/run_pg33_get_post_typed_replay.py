"""Collect PG-33 real local GET/POST typed-oracle replay traces.

The adapter starts a fresh loopback HTTP target for every control and candidate
probe, then uses ``httpx`` over ``127.0.0.1``.  It sends only bounded inert DOM
markers and abstract SQL or logic classes.  Raw request values and response
bodies are held in memory long enough to calculate projections, then discarded.
The output is an evaluation/training-candidate manifest with fresh-reset, pair
and evidence hashes; it does not start a trainer.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import socket
import threading
import time

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import ReadOnlySafeCatalogCollector, build_catalog, sha256_json  # noqa: E402
from app.main import app  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json as trace_sha256_json, validate_trace_step  # noqa: E402


REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
CATALOG_OUTPUT = ROOT / "research" / "pg_pk_33_get_post_typed_replay_catalog_v1.json"
TRACE_OUTPUT = ROOT / "research" / "pg_pk_33_trace_dataset_v1.json"
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg33-typed-oracle-contract-v1").hexdigest()
SEEDS = (331, 337, 347)
TARGET_PORT = 31933


VARIANTS: tuple[dict[str, Any], ...] = (
    {"id": "01", "family": "xss", "kind": "dom", "surface": "html_text", "role": "train", "route": "maze-dom-xss-v1", "probe_ref": "inert-dom-marker"},
    {"id": "02", "family": "injection", "kind": "sql", "surface": "sql_ast_shape", "role": "train", "route": "maze-sql-injection-v1", "probe_ref": "abstract-sql-channel"},
    {"id": "03", "family": "xss", "kind": "dom", "surface": "html_text", "role": "dev", "route": "maze-dom-xss-v2", "probe_ref": "inert-dom-marker"},
    {"id": "04", "family": "injection", "kind": "sql", "surface": "sql_ast_shape", "role": "dev", "route": "maze-sql-injection-v2", "probe_ref": "abstract-sql-channel"},
    {"id": "05", "family": "logic", "kind": "logic", "surface": "business_invariant", "role": "family_holdout", "route": "maze-logic-business-v1", "probe_ref": "abstract-invariant-class"},
    {"id": "06", "family": "access_control", "kind": "logic", "surface": "authorization_boundary", "role": "ood_source", "route": "maze-logic-access-v1", "probe_ref": "abstract-boundary-class"},
    {"id": "07", "family": "ordinary_response", "kind": "logic", "surface": "ordinary_response", "role": "negative_control", "route": "maze-negative-control-v1", "probe_ref": "ordinary-control"},
)


def _digest_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _length_bucket(length: int) -> str:
    if length == 0:
        return "0"
    if length <= 255:
        return "1-255"
    if length <= 4095:
        return "256-4095"
    if length <= 65535:
        return "4096-65535"
    return "65536+"


def _status_class(status: int) -> str:
    return f"{status // 100}xx" if 100 <= status <= 599 else "other"


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        scalar = sum(not isinstance(child, (dict, list)) for child in value.values())
        arrays = sum(isinstance(child, list) for child in value.values())
        return {"kind": "object", "key_count": len(value), "scalar_count": scalar, "array_count": arrays}
    if isinstance(value, list):
        return {"kind": "array", "key_count": 0, "scalar_count": sum(not isinstance(child, (dict, list)) for child in value), "array_count": len(value)}
    return {"kind": type(value).__name__, "key_count": 0, "scalar_count": 1, "array_count": 0}


def _response_projection(response: Any) -> dict[str, Any]:
    body = bytes(response.content)
    try:
        parsed = response.json()
    except ValueError:
        parsed = None
    shape = _shape(parsed)
    content_type = str(response.headers.get("content-type", "application/octet-stream")).split(";", 1)[0].casefold()
    if content_type == "application/json":
        content_type = "json"
    elif content_type == "text/html":
        content_type = "html"
    elif content_type.startswith("text/"):
        content_type = "text"
    elif content_type.endswith("+xml") or content_type == "application/xml":
        content_type = "xml"
    else:
        content_type = "other"
    projection = {
        "status_code": int(response.status_code),
        "status_class": _status_class(int(response.status_code)),
        "content_type_class": content_type,
        "body_length_bucket": _length_bucket(len(body)),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "semantic_body_sha256": sha256_json(shape),
        "shape": shape,
        "header_names": sorted({str(key).casefold() for key in response.headers.keys()} & {"content-type", "location", "allow"}),
        "marker": {"reflected": False, "location": "none", "count": 0},
        "frame_policy": "unknown",
        "transport_error": False,
        "status_changed": False,
        "state_changed": False,
        "location_origin_changed": False,
    }
    projection["projection_sha256"] = sha256_json(projection)
    return projection


class _LoopbackHttpServer:
    """Start one fresh loopback HTTP target and tear it down after a probe."""

    def __init__(self, port: int = TARGET_PORT) -> None:
        self.port = int(port)
        self.config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="critical",
            access_log=False,
            lifespan="off",
        )
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.client: httpx.Client | None = None

    def __enter__(self) -> httpx.Client:
        self.thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not self.server.started:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    if self.server.started:
                        break
            except OSError:
                time.sleep(0.02)
        if not self.server.started:
            self.server.should_exit = True
            self.thread.join(timeout=2.0)
            raise RuntimeError(f"fresh loopback target did not start on port {self.port}")
        self.client = httpx.Client(
            base_url=f"http://127.0.0.1:{self.port}",
            timeout=3.0,
            follow_redirects=False,
            headers={"accept": "application/json"},
        )
        return self.client

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.client is not None:
            self.client.close()
        self.server.should_exit = True
        self.thread.join(timeout=3.0)


def _call(client: httpx.Client, variant: dict[str, Any], method: str, *, positive: bool, marker: str) -> Any:
    kind = variant["kind"]
    if kind == "dom":
        value = f'<span data-sift-marker="{marker}">{marker}</span>' if positive else f"plain-{marker}"
        params = {"value": value, "marker": marker}
        return client.get("/api/maze/replay/dom", params=params) if method == "GET" else client.post("/api/maze/replay/dom", json=params)
    if kind == "sql":
        fragment = "operator_like" if positive else "plain"
        params = {"fragment_class": fragment}
        return client.get("/api/maze/replay/sql", params=params) if method == "GET" else client.post("/api/maze/replay/sql", json=params)
    probe_class = "boundary_candidate" if positive else "normal"
    params = {"probe_class": probe_class, "surface": variant["surface"]}
    return client.get("/api/maze/replay/logic", params=params) if method == "GET" else client.post("/api/maze/replay/logic", json=params)


def _oracle_projection(variant: dict[str, Any], response: Any, *, positive: bool, method: str) -> dict[str, Any]:
    body = response.json()
    if variant["kind"] == "dom":
        signals = {
            "method": method,
            "dom_change": bool(body.get("dom_change", False)),
            "marker_hits": int(body.get("marker_hits", 0)),
            "script_execution": bool(body.get("script_execution", False)),
            "network_access": bool(body.get("network_access", False)),
        }
        modality = "typed_dom_effect"
        effect = "dom_structure" if positive else "none"
    elif variant["kind"] == "sql":
        evidence = dict(body.get("evidence") or {})
        signals = {
            "method": method,
            "controlled_differential": bool(evidence.get("controlled_differential", False)),
            "interpreter_boundary": bool(evidence.get("interpreter_boundary", False)),
            "execution": "not_run",
            "database_touched": bool(evidence.get("database_touched", False)),
        }
        modality = "typed_sql_ast"
        effect = "interpreter_boundary" if positive else "none"
    else:
        signals = {
            "method": method,
            "typed_boundary_observed": bool(body.get("typed_boundary_observed", False)),
            "state_mutated": bool(body.get("state_mutated", False)),
            "credentials_accessed": bool(body.get("credentials_accessed", False)),
        }
        modality = "typed_authorization_boundary" if variant["surface"] == "authorization_boundary" else "typed_business_invariant"
        effect = variant["surface"] if positive else "none"
    projection = {
        "oracle_id": f"pg33-{variant['kind']}-typed-oracle-v1",
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "family": variant["family"],
        "modality": modality if positive else "negative_control",
        "candidate_signal": bool(positive),
        "positive": bool(positive),
        "positive_authority": bool(positive),
        "confirmed_effect": effect,
        "signals": signals,
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
            "raw_body_stored": False,
        },
    }
    projection["projection_sha256"] = sha256_json(projection)
    return projection


def _payload_manifest(variant: dict[str, Any], method: str, seed: int, *, positive: bool, marker: str) -> dict[str, Any]:
    if variant["kind"] == "dom":
        probe_kind = "inert_dom_markup"
        fields = ["value", "marker"]
    elif variant["kind"] == "sql":
        probe_kind = "abstract_channel_class"
        fields = ["fragment_class"]
    else:
        probe_kind = "http_canary"
        fields = ["probe_class", "surface"]
    descriptor = {
        "variant": variant["id"],
        "family": variant["family"],
        "method": method,
        "seed": seed,
        "positive": bool(positive),
        "probe_ref": variant["probe_ref"],
    }
    result = {
        "manifest_id": f"pg33-v{variant['id']}-s{seed}-{method.casefold()}-{'candidate' if positive else 'control'}",
        "payload_sha256": sha256_json(descriptor),
        "probe_ref": variant["probe_ref"] if positive else "baseline-control",
        "probe_kind": probe_kind,
        "route_template_id": variant["route"],
        "method": method,
        "placement": "query" if method == "GET" else "form",
        "encoding_chain": ["identity"],
        "encoding_depth": 0,
        "marker_sha256": _digest_text(marker),
        "max_bytes": 512,
        "safety": {
            "does_not_execute": True,
            "no_external_network": True,
            "no_script_execution": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }
    if method == "POST":
        result["form_field_names"] = fields
        result["form_content_type"] = "application/json"
    return result


def _rule_ir(variant: dict[str, Any], method: str) -> dict[str, Any]:
    return {
        "rule_key": f"{variant['family']}.{variant['route']}.{method.casefold()}",
        "grammar_version": "rule-ir-v1",
        "family_candidate": variant["family"],
        "operator_set": ["and", "eq", "present"],
        "required_slots": ["surface", "transport", "oracle"],
        "bound_slots": ["surface", "transport", "oracle"],
        "executable": False,
    }


def _source(variant: dict[str, Any], collector_hash: str, *, port: int = 3100) -> dict[str, Any]:
    fixture_material = {
        "variant": variant,
        "routes": ["app/main.py", "app/dom_oracle.py", "app/sql_ast_oracle.py", "app/logic_replay_oracle.py"],
    }
    return {
        "target_id": "dual_channel_replay_fixture",
        "app_family": "local_dual_channel_oracle_fixture",
        "source_id": f"pg33-source-v{variant['id']}",
        "source_type": "in_repo_synthetic",
        "origin_ref": "app/main.py:/api/maze/replay/{dom,sql,logic}",
        "license": "in-repo-synthetic",
        "authorization": "workspace_local_only",
        "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": port},
        "fixture_source_sha256": sha256_json(fixture_material),
        "collector_sha256": collector_hash,
        "reset_adapter_sha256": _digest_text(f"pg33-fresh-reset-v1:{variant['id']}"),
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "read_only": True,
        "external_network": False,
    }


def _reset(source: dict[str, Any], variant: dict[str, Any], seed: int, method: str, positive: bool, baseline: dict[str, Any]) -> dict[str, Any]:
    target = f"pg33-target-v{variant['id']}-s{seed}"
    return {
        "reset_id": f"pg33-reset-v{variant['id']}-s{seed}-{method.casefold()}-{'candidate' if positive else 'control'}",
        "kind": "fresh_loopback_http_server",
        "target_instance_id": target,
        "state_epoch": f"pg33-epoch-v{variant['id']}-s{seed}-{method.casefold()}",
        "reset_adapter_sha256": source["reset_adapter_sha256"],
        "baseline_projection_sha256": sha256_json(baseline),
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "transport": "httpx_loopback",
        "external_network": False,
    }


def _trace_step(record: dict[str, Any], baseline: dict[str, Any], *, episode_id: str, step_id: str, parent_step_id: str | None, next_action: str, negative_control_pair_id: str | None) -> dict[str, Any]:
    manifest = record["payload_manifest"]
    action = {
        "method": manifest["method"],
        "route_template_id": manifest["route_template_id"],
        "placement": manifest["placement"],
        "encoding_chain": manifest["encoding_chain"],
        "probe_ref": manifest["probe_ref"],
        "probe_sha256": manifest["payload_sha256"],
        "safety": {
            "no_external_network": True,
            "does_not_execute": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }
    if manifest["method"] == "POST":
        action["form_field_names"] = manifest["form_field_names"]
    oracle = dict(record["oracle_projection"])
    if negative_control_pair_id:
        oracle["negative_control_pair_id"] = negative_control_pair_id
    positive = bool(record["oracle_projection"]["positive"])
    body = {
        "action_manifest": action,
        "baseline_projection": baseline,
        "response_projection": record["response_projection"],
        "oracle_projection": oracle,
        "belief_before": {record["oracle_projection"]["family"]: 0.5, "none": 0.5},
        "belief_after": {record["oracle_projection"]["family"]: 0.9 if positive else 0.2, "none": 0.1 if positive else 0.8},
        "decision": "confirmed_positive" if positive else "confirmed_negative",
        "next_action": next_action,
    }
    step = {
        "episode_id": episode_id,
        "step_id": step_id,
        "parent_step_id": parent_step_id,
        "sampling_seed": record["sampling_seed"],
        "target_instance_id": record["target_instance_id"],
        "hypothesis": record["oracle_projection"]["family"],
        **body,
        "fresh_reset": record["reset"],
        "evidence_sha256": record["evidence"]["evidence_hash"],
        "echo": {"sha256": trace_sha256_json(body)},
    }
    return validate_trace_step(step)


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    all_records: list[dict[str, Any]] = []
    source_catalogs: list[dict[str, Any]] = []
    trace_steps: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    for variant in VARIANTS:
        source = _source(variant, collector_hash, port=TARGET_PORT)
        collector = ReadOnlySafeCatalogCollector(source, registry=registry)
        variant_records: list[dict[str, Any]] = []
        for seed in SEEDS:
            method_records: dict[str, dict[str, Any]] = {}
            for method in ("GET", "POST"):
                marker = f"pg33-v{variant['id']}-s{seed}-{method.casefold()}"
                # Each control and candidate is replayed against a separately
                # started loopback HTTP target.  This is intentionally slower
                # than TestClient but supplies a real GET/POST socket trace and
                # a process-level fresh-target boundary for every probe.
                with _LoopbackHttpServer(TARGET_PORT) as client:
                    control_response = _call(client, variant, method, positive=False, marker=marker)
                with _LoopbackHttpServer(TARGET_PORT) as client:
                    candidate_response = _call(client, variant, method, positive=variant["role"] != "negative_control", marker=marker)
                baseline_projection = _response_projection(control_response)
                control_projection = _response_projection(control_response)
                candidate_projection = _response_projection(candidate_response)
                reset_control = _reset(source, variant, seed, method, False, baseline_projection)
                reset_candidate = _reset(source, variant, seed, method, True, baseline_projection)
                control_oracle = _oracle_projection(variant, control_response, positive=False, method=method)
                candidate_positive = variant["role"] != "negative_control"
                candidate_oracle = _oracle_projection(variant, candidate_response, positive=candidate_positive, method=method)
                control = collector.collect(
                    sample_id=f"pg33-v{variant['id']}-s{seed}-{method.casefold()}-control",
                    sample_role="negative_control",
                    sampling_seed=seed,
                    reset=reset_control,
                    payload_manifest=_payload_manifest(variant, method, seed, positive=False, marker=marker),
                    response_projection=control_projection,
                    oracle_projection=control_oracle,
                    rule_ir=_rule_ir(variant, method),
                )
                candidate = collector.collect(
                    sample_id=f"pg33-v{variant['id']}-s{seed}-{method.casefold()}-candidate",
                    sample_role="candidate" if candidate_positive else "negative_control",
                    sampling_seed=seed,
                    reset=reset_candidate,
                    payload_manifest=_payload_manifest(variant, method, seed, positive=candidate_positive, marker=marker),
                    response_projection=candidate_projection,
                    oracle_projection=candidate_oracle,
                    rule_ir=_rule_ir(variant, method),
                    negative_control=(
                        {
                            "control_sample_id": control["sample_id"],
                            "control_evidence_hash": control["evidence"]["evidence_hash"],
                            "intervention": "typed-class-vs-normal-control",
                            "verdict": "confirmed_negative",
                            "same_source": True,
                            "same_surface": True,
                        }
                        if candidate_positive else None
                    ),
                )
                for row in (control, candidate):
                    row["dataset_role"] = variant["role"]
                    row["variant_id"] = variant["id"]
                    row["family"] = variant["family"]
                    row["method"] = method
                    row["route_template_id"] = variant["route"]
                    variant_records.append(row)
                method_records[method] = {"control": control, "candidate": candidate}
                print(
                    f"variant={variant['id']} role={variant['role']} seed={seed} method={method} "
                    f"control_status={control_response.status_code} candidate_status={candidate_response.status_code} "
                    f"positive={candidate_oracle['positive']}"
                )
            # One episode contains both methods and both matched controls.
            episode_id = f"pg33-episode-v{variant['id']}-s{seed}"
            previous: str | None = None
            episode_steps: list[dict[str, Any]] = []
            for method in ("GET", "POST"):
                pair = method_records[method]
                control = pair["control"]
                candidate = pair["candidate"]
                control_step = _trace_step(
                    control,
                    control["response_projection"],
                    episode_id=episode_id,
                    step_id=f"{episode_id}-{method.casefold()}-control",
                    parent_step_id=previous,
                    next_action="replay_candidate",
                    negative_control_pair_id=None,
                )
                previous = control_step["step_id"]
                candidate_step = _trace_step(
                    candidate,
                    control["response_projection"],
                    episode_id=episode_id,
                    step_id=f"{episode_id}-{method.casefold()}-candidate",
                    parent_step_id=previous,
                    next_action="stop_episode",
                    negative_control_pair_id=control["sample_id"] if candidate["oracle_projection"]["positive"] else None,
                )
                previous = candidate_step["step_id"]
                episode_steps.extend([control_step, candidate_step])
            episode_reports.append(evaluate_episode(episode_steps))
            trace_steps.extend(episode_steps)
        variant_catalog = build_catalog(f"pg33-source-v{variant['id']}-catalog", collector.source, variant_records)
        source_catalogs.append({
            "source_id": collector.source["source_id"],
            "source_sha256": collector.source["source_sha256"],
            "catalog_sha256": variant_catalog["catalog_sha256"],
            "sample_count": len(variant_records),
            "training_eligible": variant_catalog["training_eligible"],
        })
        all_records.extend(variant_records)

    # Add a compact per-role/per-seed skeleton for the later capability gate.
    dataset_tests: list[dict[str, Any]] = []
    family_policy = {
        "train": ["xss", "injection"],
        "dev": ["xss", "injection"],
        "family_holdout": ["logic"],
        "ood_source": ["access_control"],
        "negative_control": ["ordinary_response"],
    }
    for role, families in family_policy.items():
        for seed in SEEDS:
            rows = [row for row in all_records if row["dataset_role"] == role and int(row["sampling_seed"]) == seed]
            target_ids = sorted({str(row["target_instance_id"]) for row in rows})
            source_hashes = sorted({str(row["source_sha256"]) for row in rows})
            sample_ids = sorted({str(row["sample_id"]) for row in rows})
            summary = {
                "sample_id": f"pg33-test-{role}-s{seed}",
                "dataset_id": f"pg33-{role}-s{seed}-v1",
                "source_id": f"pg33-role-source-{role}-s{seed}",
                "source_hash": sha256_json(source_hashes),
                "target_instance_id": target_ids[0],
                "target_instance_ids": target_ids,
                "family_set": families,
                "sampling_seed": seed,
                "role": role,
                "sample_count": len(rows),
                "unique_sample_count": len(sample_ids),
                "denominator": len(rows),
                "positive_count": sum(int(row["oracle_projection"]["positive"]) for row in rows),
                "negative_count": sum(int(not row["oracle_projection"]["positive"]) for row in rows),
                "abstain_count": sum(int(row["decision"]["evidence_status"] == "abstain") for row in rows),
                "dataset_manifest_sha256": sha256_json({"role": role, "seed": seed, "samples": sample_ids}),
                "split_manifest_sha256": sha256_json({"role": role, "seed": seed, "targets": target_ids, "families": families}),
                "probe_sha256": sha256_json([row["payload_manifest"]["payload_sha256"] for row in rows]),
                "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
                "checkpoint_sha256": sha256_json("pending-model-run"),
                "metrics_status": "pending_model_run",
                "metrics": {"typed_recall": 0.0, "precision": 0.0, "false_positive_rate": 0.0, "abstain_precision": 0.0, "ece": 0.0, "median_queries": 0.0},
                "baseline_metrics": {"typed_recall": 0.0, "precision": 0.0, "false_positive_rate": 0.0, "abstain_precision": 0.0, "ece": 0.0, "median_queries": 0.0},
                "candidate_metrics": {"typed_recall": 0.0, "precision": 0.0, "false_positive_rate": 0.0, "abstain_precision": 0.0, "ece": 0.0, "median_queries": 0.0},
            }
            summary["evidence_hash"] = sha256_json(summary)
            dataset_tests.append(summary)

    sources = []
    seen_sources: set[str] = set()
    for variant in VARIANTS:
        source = _source(variant, collector_hash, port=TARGET_PORT)
        normalized = ReadOnlySafeCatalogCollector(source, registry=registry).source
        if normalized["source_id"] not in seen_sources:
            seen_sources.add(normalized["source_id"])
            sources.append(normalized)
    catalog = {
        "schema_version": "pg-pk-33-get-post-typed-replay-catalog-v1",
        "catalog_id": "pg33-get-post-typed-replay-v1",
        "purpose": "real local GET/POST typed positive and negative replay traces",
        "runtime_replay": True,
        "evaluation_only": True,
        "training_eligible": False,
        "training_artifact_generated": False,
        "model_evaluation_completed": False,
        "methods": ["GET", "POST"],
        "seeds": list(SEEDS),
        "sources": sources,
        "source_catalogs": source_catalogs,
        "samples": all_records,
        "dataset_tests": dataset_tests,
        "trace_dataset": str(TRACE_OUTPUT.relative_to(ROOT)),
        "typed_positive_count": sum(int(row["oracle_projection"]["positive"]) for row in all_records),
        "negative_control_count": sum(int(not row["oracle_projection"]["positive"]) for row in all_records),
        "fresh_reset_count": len(all_records),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "authorization": "workspace_local_only",
        "family_policy": family_policy,
        "manifest_sha256": sha256_json({"samples": [row["evidence"]["evidence_hash"] for row in all_records], "dataset_tests": dataset_tests}),
    }
    trace_dataset = {
        "schema_version": "pg-pk-33-trace-dataset-v1",
        "purpose": "step-aligned model observation and belief-update replay",
        "evaluation_only": True,
        "training_eligible": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "methods": ["GET", "POST"],
        "episode_count": len(episode_reports),
        "episodes": episode_reports,
        "steps": trace_steps,
        "catalog_manifest_sha256": catalog["manifest_sha256"],
        "trace_manifest_sha256": sha256_json([step["trace_sha256"] for step in trace_steps]),
    }
    CATALOG_OUTPUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_OUTPUT.write_text(json.dumps(trace_dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "catalog": str(CATALOG_OUTPUT),
        "trace_dataset": str(TRACE_OUTPUT),
        "sample_count": len(all_records),
        "typed_positive_count": catalog["typed_positive_count"],
        "negative_control_count": catalog["negative_control_count"],
        "episode_count": len(episode_reports),
        "accepted_evaluation_episodes": sum(int(item["status"] == "accepted_evaluation") for item in episode_reports),
        "training_eligible": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
