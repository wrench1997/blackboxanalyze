"""Collect PG-34's independent HTTP fixture as a typed GET/POST catalog.

The fixture is a standalone ``http.server`` implementation, not the FastAPI
maze.  Every control and candidate is sent to a freshly started loopback
target.  Only bounded abstract probe classes are transmitted and only response
projections/oracle evidence hashes are persisted.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import ReadOnlySafeCatalogCollector, build_catalog, sha256_json  # noqa: E402
from app.pg53_cross_source_oracle import generic_effect_geometry, surface_observation  # noqa: E402
from app.pg34_independent_fixture import SURFACE_SPECS, make_independent_fixture_server  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json as trace_sha256_json, validate_trace_step  # noqa: E402


REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
CATALOG_OUTPUT = ROOT / "research" / "pg34_independent_fixture_catalog_v1.json"
TRACE_OUTPUT = ROOT / "research" / "pg34_independent_fixture_trace_v1.json"
SEEDS = (341, 347, 353)
TARGET_PORT = 31934
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg34-independent-typed-oracle-v1").hexdigest()

VARIANTS: tuple[dict[str, Any], ...] = (
    {"id": "01", "surface": "surface-01", "role": "train"},
    {"id": "02", "surface": "surface-02", "role": "train"},
    {"id": "03", "surface": "surface-03", "role": "dev"},
    {"id": "04", "surface": "surface-04", "role": "dev"},
    {"id": "05", "surface": "surface-05", "role": "family_holdout"},
    {"id": "06", "surface": "surface-06", "role": "family_holdout"},
    {"id": "07", "surface": "surface-07", "role": "ood_source"},
    {"id": "08", "surface": "surface-08", "role": "ood_source"},
    {"id": "09", "surface": "surface-09", "role": "negative_control"},
)


class _FreshTarget:
    def __init__(self, port: int = TARGET_PORT) -> None:
        self.port = int(port)
        self.server = make_independent_fixture_server(self.port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.client: httpx.Client | None = None

    def __enter__(self) -> httpx.Client:
        self.thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.02)
        else:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2.0)
            raise RuntimeError("independent fixture did not start")
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
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def _source(variant: dict[str, Any], collector_hash: str, fixture_hash: str) -> dict[str, Any]:
    spec = SURFACE_SPECS[variant["surface"]]
    return {
        "target_id": "pg34_independent_fixture",
        "app_family": "standalone_python_http_fixture",
        "source_id": f"pg34-independent-source-v{variant['id']}",
        "source_type": "in_repo_synthetic",
        "origin_ref": "app/pg34_independent_fixture.py:/pg34/surface/{surface-id}",
        "license": "in-repo-synthetic",
        "authorization": "workspace_local_only",
        "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": TARGET_PORT},
        "fixture_source_sha256": sha256_json({"fixture": fixture_hash, "surface": variant["surface"], "family": spec["family"]}),
        "collector_sha256": collector_hash,
        "reset_adapter_sha256": hashlib.sha256(f"pg34-independent-fresh-reset:{variant['id']}".encode()).hexdigest(),
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "read_only": True,
        "external_network": False,
    }


def _status_class(status: int) -> str:
    return f"{status // 100}xx" if 100 <= status <= 599 else "other"


def _length_bucket(length: int) -> str:
    if length == 0:
        return "0"
    if length <= 255:
        return "1-255"
    if length <= 4095:
        return "256-4095"
    return "4096-65535"


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "kind": "object",
            "key_count": len(value),
            "scalar_count": sum(not isinstance(child, (dict, list)) for child in value.values()),
            "array_count": sum(isinstance(child, list) for child in value.values()),
        }
    return {"kind": type(value).__name__, "key_count": 0, "scalar_count": 1, "array_count": 0}


def _response_projection(response: httpx.Response) -> dict[str, Any]:
    body = bytes(response.content)
    parsed = response.json()
    content_type = str(response.headers.get("content-type", "application/octet-stream")).split(";", 1)[0].casefold()
    content_type = "json" if content_type == "application/json" else "other"
    shape = _shape(parsed)
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
    # Preserve a bounded, label-free semantic surface signal while the parsed
    # body is still in local memory.  ``surface_observation`` excludes fixture
    # and evaluator keys, hashes only field-name buckets, and never persists
    # the body; geometry contains type counts only.  These fields are needed
    # because PG-34 positive/control responses intentionally share the same
    # coarse JSON shape.
    if isinstance(parsed, dict):
        projection["effect_surface"] = surface_observation(parsed)
        projection["effect_geometry"] = generic_effect_geometry(parsed)
        projection["projection_schema"] = "canonical_effect_projection_v3_surface_signal"
    projection["projection_sha256"] = sha256_json(projection)
    return projection


def _call(client: httpx.Client, variant: dict[str, Any], method: str, *, positive: bool) -> httpx.Response:
    spec = SURFACE_SPECS[variant["surface"]]
    probe = spec["positive"] if positive and variant["role"] != "negative_control" else "normal"
    params = {spec["field"]: probe}
    path = f"/pg34/surface/{variant['surface']}"
    if method == "GET":
        return client.get(path, params=params)
    return client.post(path, json=params)


def _oracle(variant: dict[str, Any], response: httpx.Response, *, positive: bool, method: str) -> dict[str, Any]:
    spec = SURFACE_SPECS[variant["surface"]]
    body = response.json()
    signals = {key: value for key, value in body.items() if key not in {"surface_slot"}}
    effect = spec["effect"] if positive and variant["role"] != "negative_control" else "none"
    modality = "typed_boundary" if spec["family"] not in {"xss", "injection"} else ("typed_dom_effect" if spec["family"] == "xss" else "typed_sql_ast")
    projection = {
        "oracle_id": f"pg34-independent-{spec['family']}-oracle-v1",
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "family": spec["family"],
        "modality": modality if effect != "none" else "negative_control",
        "candidate_signal": bool(positive and variant["role"] != "negative_control"),
        "positive": bool(positive and variant["role"] != "negative_control"),
        "positive_authority": bool(positive and variant["role"] != "negative_control"),
        "confirmed_effect": effect,
        "signals": {"method": method, **signals},
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


def _manifest(variant: dict[str, Any], method: str, seed: int, *, positive: bool) -> dict[str, Any]:
    spec = SURFACE_SPECS[variant["surface"]]
    probe_kind = "inert_dom_markup" if spec["family"] == "xss" else "abstract_channel_class" if spec["family"] == "injection" else "http_canary"
    fields = [spec["field"]]
    descriptor = {"surface": variant["surface"], "method": method, "seed": seed, "positive": positive}
    result = {
        "manifest_id": f"pg34-independent-v{variant['id']}-s{seed}-{method.casefold()}-{'candidate' if positive else 'control'}",
        "payload_sha256": sha256_json(descriptor),
        "probe_ref": "abstract-positive-class" if positive else "baseline-control",
        "probe_kind": probe_kind,
        "route_template_id": f"pg34-independent-{variant['surface']}",
        "method": method,
        "placement": "query" if method == "GET" else "form",
        "encoding_chain": ["identity"],
        "encoding_depth": 0,
        "marker_sha256": hashlib.sha256(f"pg34-{variant['id']}-{seed}-{method}".encode()).hexdigest(),
        "max_bytes": 256,
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


def _reset(source: dict[str, Any], variant: dict[str, Any], seed: int, method: str, positive: bool, baseline: dict[str, Any]) -> dict[str, Any]:
    role = "candidate" if positive else "control"
    return {
        "reset_id": f"pg34-independent-reset-v{variant['id']}-s{seed}-{method.casefold()}-{role}",
        "kind": "fresh_independent_http_server",
        # The target identity must distinguish every fresh control/candidate
        # server.  The old identity collapsed GET and POST (and control and
        # candidate) into one target per variant/seed, invalidating replay
        # independence despite the server process being restarted.
        "target_instance_id": f"pg34-independent-target-v{variant['id']}-s{seed}-{method.casefold()}-{role}",
        "state_epoch": f"pg34-independent-epoch-v{variant['id']}-s{seed}-{method.casefold()}-{role}",
        "reset_adapter_sha256": source["reset_adapter_sha256"],
        "baseline_projection_sha256": sha256_json(baseline),
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "transport": "httpx_loopback",
        "external_network": False,
    }


def _rule_ir(variant: dict[str, Any], method: str) -> dict[str, Any]:
    spec = SURFACE_SPECS[variant["surface"]]
    return {
        "rule_key": f"{spec['family']}.pg34-independent-{variant['surface']}.{method.casefold()}",
        "grammar_version": "rule-ir-v1",
        "family_candidate": spec["family"],
        "operator_set": ["and", "eq", "present"],
        "required_slots": ["surface", "transport", "oracle"],
        "bound_slots": ["surface", "transport", "oracle"],
        "executable": False,
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
        "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True},
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
    fixture_hash = hashlib.sha256((ROOT / "app" / "pg34_independent_fixture.py").read_bytes()).hexdigest()
    all_records: list[dict[str, Any]] = []
    source_catalogs: list[dict[str, Any]] = []
    trace_steps: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for variant in VARIANTS:
        source = _source(variant, collector_hash, fixture_hash)
        collector = ReadOnlySafeCatalogCollector(source, registry=registry)
        if source["source_id"] not in seen_source_ids:
            seen_source_ids.add(source["source_id"])
            sources.append(collector.source)
        variant_records: list[dict[str, Any]] = []
        for seed in SEEDS:
            method_records: dict[str, dict[str, Any]] = {}
            for method in ("GET", "POST"):
                with _FreshTarget(TARGET_PORT) as client:
                    control_response = _call(client, variant, method, positive=False)
                with _FreshTarget(TARGET_PORT) as client:
                    candidate_response = _call(client, variant, method, positive=variant["role"] != "negative_control")
                baseline = _response_projection(control_response)
                control_projection = _response_projection(control_response)
                candidate_projection = _response_projection(candidate_response)
                control_oracle = _oracle(variant, control_response, positive=False, method=method)
                candidate_positive = variant["role"] != "negative_control"
                candidate_oracle = _oracle(variant, candidate_response, positive=candidate_positive, method=method)
                control = collector.collect(
                    sample_id=f"pg34-independent-v{variant['id']}-s{seed}-{method.casefold()}-control",
                    sample_role="negative_control",
                    sampling_seed=seed,
                    reset=_reset(source, variant, seed, method, False, baseline),
                    payload_manifest=_manifest(variant, method, seed, positive=False),
                    response_projection=control_projection,
                    oracle_projection=control_oracle,
                    rule_ir=_rule_ir(variant, method),
                )
                candidate = collector.collect(
                    sample_id=f"pg34-independent-v{variant['id']}-s{seed}-{method.casefold()}-candidate",
                    sample_role="candidate" if candidate_positive else "negative_control",
                    sampling_seed=seed,
                    reset=_reset(source, variant, seed, method, True, baseline),
                    payload_manifest=_manifest(variant, method, seed, positive=candidate_positive),
                    response_projection=candidate_projection,
                    oracle_projection=candidate_oracle,
                    rule_ir=_rule_ir(variant, method),
                    negative_control=(
                        {"control_sample_id": control["sample_id"], "control_evidence_hash": control["evidence"]["evidence_hash"], "intervention": "typed-class-vs-normal-control", "verdict": "confirmed_negative", "same_source": True, "same_surface": True}
                        if candidate_positive else None
                    ),
                )
                for row in (control, candidate):
                    row.update({"dataset_role": variant["role"], "variant_id": variant["id"], "family": SURFACE_SPECS[variant["surface"]]["family"], "method": method, "route_template_id": f"pg34-independent-{variant['surface']}"})
                    variant_records.append(row)
                method_records[method] = {"control": control, "candidate": candidate}
            episode_id = f"pg34-independent-episode-v{variant['id']}-s{seed}"
            previous: str | None = None
            steps: list[dict[str, Any]] = []
            for method in ("GET", "POST"):
                pair = method_records[method]
                control = pair["control"]
                candidate = pair["candidate"]
                control_step = _trace_step(control, control["response_projection"], episode_id=episode_id, step_id=f"{episode_id}-{method.casefold()}-control", parent_step_id=previous, next_action="replay_candidate", negative_control_pair_id=None)
                previous = control_step["step_id"]
                candidate_step = _trace_step(candidate, control["response_projection"], episode_id=episode_id, step_id=f"{episode_id}-{method.casefold()}-candidate", parent_step_id=previous, next_action="stop_episode", negative_control_pair_id=candidate["sample_id"] if candidate["oracle_projection"]["positive"] else None)
                previous = candidate_step["step_id"]
                steps.extend([control_step, candidate_step])
            episode_reports.append(evaluate_episode(steps))
            trace_steps.extend(steps)
        variant_catalog = build_catalog(f"pg34-independent-v{variant['id']}-catalog", collector.source, variant_records)
        source_catalogs.append({"source_id": collector.source["source_id"], "source_sha256": collector.source["source_sha256"], "catalog_sha256": variant_catalog["catalog_sha256"], "sample_count": len(variant_records), "training_eligible": variant_catalog["training_eligible"]})
        all_records.extend(variant_records)

    role_families = {
        "train": ["xss", "injection"],
        "dev": ["authentication", "access_control"],
        "family_holdout": ["logic", "url_redirect"],
        "ood_source": ["input_validation", "command_injection"],
        "negative_control": ["ordinary_response"],
    }
    dataset_tests: list[dict[str, Any]] = []
    for role, families in role_families.items():
        for seed in SEEDS:
            rows = [row for row in all_records if row["dataset_role"] == role and int(row["sampling_seed"]) == seed]
            targets = sorted({row["target_instance_id"] for row in rows})
            source_hashes = sorted({row["source_sha256"] for row in rows})
            sample_ids = sorted(row["sample_id"] for row in rows)
            summary = {
                "sample_id": f"pg34-independent-test-{role}-s{seed}",
                "dataset_id": f"pg34-independent-{role}-s{seed}-v1",
                "source_id": f"pg34-independent-role-source-{role}-s{seed}",
                "source_hash": sha256_json(source_hashes),
                "target_instance_id": targets[0],
                "target_instance_ids": targets,
                "family_set": families,
                "sampling_seed": seed,
                "role": role,
                "sample_count": len(rows),
                "unique_sample_count": len(sample_ids),
                "denominator": len(rows),
                "positive_count": sum(int(row["oracle_projection"]["positive"]) for row in rows),
                "negative_count": sum(int(not row["oracle_projection"]["positive"]) for row in rows),
                "abstain_count": 0,
                "dataset_manifest_sha256": sha256_json({"role": role, "seed": seed, "samples": sample_ids}),
                "split_manifest_sha256": sha256_json({"role": role, "seed": seed, "targets": targets, "families": families}),
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

    catalog = {
        "schema_version": "pg-pk-34-independent-fixture-catalog-v1",
        "catalog_id": "pg34-independent-fixture-v1",
        "purpose": "independent local GET/POST typed replay across eight vulnerability families and one negative control",
        "runtime_replay": True,
        "independent_target_implementation": True,
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
        "target_instance_count": len({row["target_instance_id"] for row in all_records}),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "authorization": "workspace_local_only",
        "family_policy": role_families,
        "manifest_sha256": sha256_json({"samples": [row["evidence"]["evidence_hash"] for row in all_records], "dataset_tests": dataset_tests}),
    }
    trace_dataset = {
        "schema_version": "pg-pk-34-independent-fixture-trace-v1",
        "purpose": "independent target step-aligned GET/POST evidence replay",
        "evaluation_only": True,
        "training_eligible": False,
        "independent_target_implementation": True,
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
    print(json.dumps({"catalog": str(CATALOG_OUTPUT.relative_to(ROOT)), "trace_dataset": str(TRACE_OUTPUT.relative_to(ROOT)), "sample_count": len(all_records), "typed_positive_count": catalog["typed_positive_count"], "negative_control_count": catalog["negative_control_count"], "episode_count": len(episode_reports), "accepted_evaluation_episodes": sum(int(item["status"] == "accepted_evaluation") for item in episode_reports), "independent_target_implementation": True, "training_eligible": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
