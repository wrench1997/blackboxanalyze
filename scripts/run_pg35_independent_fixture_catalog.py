"""Collect PG-35's independent GET/POST, multi-encoding typed replay.

This adapter is the only component allowed to talk to the fixture.  It sends
bounded abstract class identifiers, starts a fresh loopback HTTP server for
each control/candidate observation, and persists only projections, hashes and
typed oracle evidence.  It never writes a raw request or response body.
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
from app.pg35_independent_fixture import PG35_VARIANTS, SURFACE_SPECS, make_pg35_server  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json as trace_sha256_json, validate_trace_step  # noqa: E402


REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
CATALOG_OUTPUT = ROOT / "research" / "pg35_independent_fixture_catalog_v1.json"
TRACE_OUTPUT = ROOT / "research" / "pg35_independent_fixture_trace_v1.json"
SEEDS = (351, 357, 367)
TARGET_PORT = 31935
ENCODINGS = ("identity", "url_percent")
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg35-independent-typed-oracle-v1").hexdigest()
ROLE_FAMILIES = {
    "train": ["xss", "injection"],
    "dev": ["authentication", "access_control"],
    "family_holdout": ["logic", "url_redirect"],
    "ood_source": ["input_validation", "command_injection"],
    "negative_control": ["ordinary_response"],
}
ROLE_BY_SURFACE = {
    surface: role
    for role, families in ROLE_FAMILIES.items()
    for surface, spec in SURFACE_SPECS.items()
    if spec["family"] in families
}


class _FreshTarget:
    def __init__(self, variant: str, port: int = TARGET_PORT) -> None:
        self.variant = str(variant)
        self.port = int(port)
        self.server = make_pg35_server(self.port, self.variant)
        # ``serve_forever`` defaults to a 0.5s shutdown poll interval.  PG-35
        # deliberately starts one fresh target for each control/candidate, so
        # that default would turn a bounded run into minutes of idle teardown.
        # A shorter poll interval changes no evidence semantics and keeps the
        # fresh-reset contract intact.
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
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
            raise RuntimeError("PG-35 fixture did not start")
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


def _encode(value: str, encoding: str) -> str:
    if encoding == "identity":
        return str(value)
    if encoding == "url_percent":
        return "".join(f"%{byte:02X}" for byte in str(value).encode("utf-8"))
    raise ValueError(f"unsupported PG-35 encoding: {encoding}")


def _route(variant: str, surface: str) -> str:
    return f"{PG35_VARIANTS[variant]['prefix']}/{surface}"


def _call(client: httpx.Client, variant: str, surface: str, method: str, encoding: str, *, positive: bool) -> httpx.Response:
    layout = PG35_VARIANTS[variant]
    spec = SURFACE_SPECS[surface]
    token = spec["positive"] if positive else "normal"
    wire = _encode(token, encoding)
    route = _route(variant, surface)
    if method == "GET":
        path = f"{route}?{layout['slot_key']}={surface}&{layout['probe_key']}={wire}"
        return client.get(path)
    if method != "POST":
        raise ValueError("PG-35 collector permits only GET and POST")
    content_type = layout["post_content_type"]
    headers = {"content-type": content_type}
    if content_type == "application/json":
        body = json.dumps({layout["slot_key"]: surface, layout["probe_key"]: wire}, separators=(",", ":"))
    else:
        # The value is already an inert identifier.  Percent encoding is
        # intentionally kept on the wire so parse_qs performs the same safe
        # one-layer decode as the GET path.
        body = f"{layout['slot_key']}={surface}&{layout['probe_key']}={wire}"
    return client.post(route, content=body.encode("utf-8"), headers=headers)


def _status_class(status: int) -> str:
    return f"{status // 100}xx" if 100 <= status <= 599 else "other"


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


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        scalars = [child for child in value.values() if not isinstance(child, (dict, list))]
        return {
            "kind": "object",
            "key_count": len(value),
            "scalar_count": len(scalars),
            "array_count": sum(isinstance(child, list) for child in value.values()),
            "bool_count": sum(isinstance(child, bool) for child in scalars),
            "number_count": sum(isinstance(child, (int, float)) and not isinstance(child, bool) for child in scalars),
            "string_count": sum(isinstance(child, str) for child in scalars),
        }
    return {"kind": "other", "key_count": 0, "scalar_count": 1, "array_count": 0, "bool_count": 0, "number_count": 0, "string_count": 0}


def _response_projection(response: httpx.Response) -> tuple[dict[str, Any], dict[str, Any]]:
    body = bytes(response.content)
    parsed = response.json()
    shape = _shape(parsed)
    content_type = str(response.headers.get("content-type", "application/octet-stream")).split(";", 1)[0].casefold()
    content_type = "json" if content_type == "application/json" else "other"
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
    return projection, parsed


def _oracle(variant: str, surface: str, method: str, parsed: dict[str, Any]) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    family = spec["family"]
    positive = bool(parsed.get("candidate_signal", False)) and family != "ordinary_response"
    safe_signal_keys = {
        key: parsed[key]
        for key in parsed
        if key in {
            "candidate_signal", "dom_change", "marker_hits", "ast_shape_diff", "interpreter_boundary",
            "authentication_boundary", "authorization_boundary", "cross_subject_access",
            "business_invariant_boundary", "redirect_candidate", "same_origin", "external_redirect",
            "validation_boundary", "rejected", "local_canary_observed", "command_executed", "ordinary_response",
            "state_mutated", "database_touched", "credentials_accessed", "external_network", "script_execution",
        }
    }
    return {
        "oracle_id": f"pg35-independent-{family}-oracle-v1",
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "family": family,
        "modality": spec["modality"] if positive else "negative_control",
        "candidate_signal": bool(parsed.get("candidate_signal", False)),
        "positive": positive,
        "positive_authority": positive,
        "confirmed_effect": spec["effect"] if positive else "none",
        "signals": {"variant": variant, "method": method, **safe_signal_keys},
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
            "raw_body_stored": False,
        },
    }


def _source(variant: str, collector_hash: str, fixture_hash: str) -> dict[str, Any]:
    return {
        "target_id": "pg35_independent_fixture",
        "app_family": "standalone_python_http_fixture_v3",
        "source_id": f"pg35-independent-source-{variant}",
        "source_type": "in_repo_synthetic",
        "origin_ref": "app/pg35_independent_fixture.py:/<variant>/<surface-id>",
        "license": "in-repo-synthetic",
        "authorization": "workspace_local_only",
        "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": TARGET_PORT},
        "fixture_source_sha256": sha256_json({"fixture": fixture_hash, "variant": variant}),
        "collector_sha256": collector_hash,
        "reset_adapter_sha256": hashlib.sha256(f"pg35-independent-fresh-reset:{variant}".encode()).hexdigest(),
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "read_only": True,
        "external_network": False,
    }


def _reset(source: dict[str, Any], variant: str, surface: str, seed: int, method: str, encoding: str, role: str, baseline: dict[str, Any]) -> dict[str, Any]:
    # The surface is part of the target identity.  Omitting it would make
    # independent fresh servers for different routes appear to be the same
    # target and would invalidate cross-target gates despite correct resets.
    suffix = f"{variant}-{surface}-s{seed}-{method.casefold()}-{encoding}-{role}"
    normalized = {
        "reset_id": f"pg35-reset-{suffix}",
        "kind": "fresh_pg35_http_server",
        "target_instance_id": f"pg35-target-{suffix}",
        "state_epoch": f"pg35-epoch-{suffix}",
        "reset_adapter_sha256": source["reset_adapter_sha256"],
        "baseline_projection_sha256": baseline["projection_sha256"],
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
        "transport": "httpx_loopback",
    }
    normalized["reset_sha256"] = sha256_json(normalized)
    return normalized


def _manifest(variant: str, surface: str, seed: int, method: str, encoding: str, role: str) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    layout = PG35_VARIANTS[variant]
    chain = [encoding]
    safe_basis = {"fixture": "pg35", "variant": variant, "surface": surface, "seed": seed, "method": method, "encoding": encoding, "role": role}
    return {
        "manifest_id": f"pg35-{variant}-{surface}-s{seed}-{method.casefold()}-{encoding}-{role}",
        "payload_sha256": sha256_json({"basis": safe_basis, "probe_kind": spec["probe_kind"]}),
        "probe_ref": f"pg35-safe-{spec['family']}-{encoding}",
        "probe_kind": spec["probe_kind"],
        "route_template_id": f"pg35-{variant}-{surface}",
        "method": method,
        "placement": "query" if method == "GET" else "form",
        "encoding_chain": chain,
        "encoding_depth": int(encoding != "identity"),
        "marker_sha256": hashlib.sha256(f"pg35-safe-marker:{variant}:{surface}:{seed}".encode()).hexdigest(),
        "max_bytes": 256,
        "form_field_names": [layout["slot_key"], layout["probe_key"]] if method == "POST" else [],
        "form_content_type": layout["post_content_type"] if method == "POST" else "",
        "safety": {
            "does_not_execute": True,
            "no_external_network": True,
            "no_script_execution": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }


def _rule_ir(variant: str, surface: str, method: str, encoding: str) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    return {
        "rule_key": f"{spec['family']}.pg35-{variant}-{surface}.{method.casefold()}.{encoding}",
        "grammar_version": "rule-ir-v1",
        "family_candidate": spec["family"],
        "operator_set": ["and", "eq", "present"],
        "required_slots": ["surface", "transport", "oracle"],
        "bound_slots": ["surface", "transport", "oracle"],
        "executable": False,
    }


def _trace_step(row: dict[str, Any], *, episode_id: str, step_id: str, parent_step_id: str | None, next_action: str, negative_control_pair_id: str | None, belief_before: dict[str, float]) -> dict[str, Any]:
    manifest = row["payload_manifest"]
    oracle = dict(row["oracle_projection"])
    if negative_control_pair_id:
        oracle["negative_control_pair_id"] = negative_control_pair_id
    family = row["family"]
    positive = bool(oracle.get("positive", False))
    belief_after = {"unknown": 0.05, family: 0.95} if positive else {"unknown": 0.97, family: 0.03}
    decision = "confirmed_positive" if positive else "confirmed_negative"
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
    echo_body = {
        "action_manifest": action,
        "baseline_projection": row["response_projection"],
        "response_projection": row["response_projection"],
        "oracle_projection": oracle,
        "belief_before": belief_before,
        "belief_after": belief_after,
        "decision": decision,
        "next_action": next_action,
    }
    step = {
        "schema_version": "sift-trace-aligned-step-v1",
        "episode_id": episode_id,
        "step_id": step_id,
        "parent_step_id": parent_step_id,
        "sampling_seed": int(row["sampling_seed"]),
        "target_instance_id": row["target_instance_id"],
        "hypothesis": family,
        "belief_before": belief_before,
        "action_manifest": action,
        "baseline_projection": row["response_projection"],
        "response_projection": row["response_projection"],
        "oracle_projection": oracle,
        "belief_after": belief_after,
        "decision": decision,
        "next_action": next_action,
        "fresh_reset": row["reset"],
        "evidence_sha256": row["evidence"]["evidence_hash"],
        "dataset_stage": "trace_only",
        "online_weight_update": False,
        "long_term_memory_write": False,
        "echo": {"sha256": trace_sha256_json(echo_body)},
    }
    return validate_trace_step(step)


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_hash = hashlib.sha256((ROOT / "app" / "pg35_independent_fixture.py").read_bytes()).hexdigest()
    all_records: list[dict[str, Any]] = []
    source_catalogs: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    trace_steps: list[dict[str, Any]] = []

    for variant in PG35_VARIANTS:
        source = _source(variant, collector_hash, fixture_hash)
        collector = ReadOnlySafeCatalogCollector(source, registry=registry)
        sources.append(collector.source)
        variant_records: list[dict[str, Any]] = []
        for surface, spec in SURFACE_SPECS.items():
            role = ROLE_BY_SURFACE[surface]
            for seed in SEEDS:
                episode_id = f"pg35-episode-{variant}-{surface}-s{seed}"
                previous: str | None = None
                belief: dict[str, float] = {"unknown": 1.0}
                pair_rows: dict[tuple[str, str], dict[str, Any]] = {}
                for method in ("GET", "POST"):
                    for encoding in ENCODINGS:
                        with _FreshTarget(variant) as client:
                            control_response = _call(client, variant, surface, method, encoding, positive=False)
                        with _FreshTarget(variant) as client:
                            candidate_response = _call(client, variant, surface, method, encoding, positive=role != "negative_control")
                        baseline, _ = _response_projection(control_response)
                        control_projection, control_parsed = _response_projection(control_response)
                        candidate_projection, candidate_parsed = _response_projection(candidate_response)
                        control_oracle = _oracle(variant, surface, method, control_parsed)
                        candidate_positive = role != "negative_control"
                        candidate_oracle = _oracle(variant, surface, method, candidate_parsed)
                        control_id = f"pg35-{variant}-{surface}-s{seed}-{method.casefold()}-{encoding}-control"
                        candidate_id = f"pg35-{variant}-{surface}-s{seed}-{method.casefold()}-{encoding}-candidate"
                        control = collector.collect(
                            sample_id=control_id,
                            sample_role="negative_control",
                            sampling_seed=seed,
                            reset=_reset(collector.source, variant, surface, seed, method, encoding, role + "-control", baseline),
                            payload_manifest=_manifest(variant, surface, seed, method, encoding, "control"),
                            response_projection=control_projection,
                            oracle_projection=control_oracle,
                            rule_ir=_rule_ir(variant, surface, method, encoding),
                        )
                        candidate = collector.collect(
                            sample_id=candidate_id,
                            sample_role="candidate" if candidate_positive else "negative_control",
                            sampling_seed=seed,
                            reset=_reset(collector.source, variant, surface, seed, method, encoding, role + "-candidate", baseline),
                            payload_manifest=_manifest(variant, surface, seed, method, encoding, "candidate"),
                            response_projection=candidate_projection,
                            oracle_projection=candidate_oracle,
                            rule_ir=_rule_ir(variant, surface, method, encoding),
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
                        for row, sample_role, sample_id, pair_role in (
                            (control, "negative_control", control_id, "control"),
                            (candidate, "candidate" if candidate_positive else "negative_control", candidate_id, "candidate"),
                        ):
                            row.update({
                                "dataset_role": role,
                                "variant": variant,
                                "surface_id": surface,
                                "family": spec["family"],
                                "method": method,
                                "encoding": encoding,
                                # Keep control/candidate distinct even for the
                                # ordinary-negative surface, whose candidate
                                # is intentionally stored as a negative role.
                                "encoding_pair_id": f"pg35-pair-{variant}-{surface}-s{seed}-{method.casefold()}-{pair_role}",
                                "pair_role": pair_role,
                                "sample_role": sample_role,
                            })
                            variant_records.append(row)
                            pair_rows[(method, encoding, sample_role)] = row
                        # Add two steps per encoding.  Keeping the pair control
                        # adjacent makes replay order and evidence provenance
                        # inspectable without retaining the request itself.
                        control_step = _trace_step(
                            control,
                            episode_id=episode_id,
                            step_id=f"{episode_id}-{method.casefold()}-{encoding}-control",
                            parent_step_id=previous,
                            next_action="replay_candidate",
                            negative_control_pair_id=None,
                            belief_before=belief,
                        )
                        previous = control_step["step_id"]
                        candidate_step = _trace_step(
                            candidate,
                            episode_id=episode_id,
                            step_id=f"{episode_id}-{method.casefold()}-{encoding}-candidate",
                            parent_step_id=previous,
                            next_action="continue_encoding" if encoding == "identity" else "stop_episode",
                            negative_control_pair_id=control["sample_id"] if candidate_positive else None,
                            belief_before=control_step["belief_after"],
                        )
                        previous = candidate_step["step_id"]
                        belief = candidate_step["belief_after"]
                        trace_steps.extend([control_step, candidate_step])
                episode_reports.append(evaluate_episode([step for step in trace_steps if step["episode_id"] == episode_id]))
        variant_catalog = build_catalog(f"pg35-independent-{variant}-catalog", collector.source, variant_records)
        source_catalogs.append({
            "source_id": collector.source["source_id"],
            "source_sha256": collector.source["source_sha256"],
            "catalog_sha256": variant_catalog["catalog_sha256"],
            "sample_count": len(variant_records),
            "training_eligible": variant_catalog["training_eligible"],
        })
        all_records.extend(variant_records)

    dataset_tests: list[dict[str, Any]] = []
    for role, families in ROLE_FAMILIES.items():
        for seed in SEEDS:
            rows = [row for row in all_records if row["dataset_role"] == role and int(row["sampling_seed"]) == seed]
            sample_ids = sorted(row["sample_id"] for row in rows)
            targets = sorted({row["target_instance_id"] for row in rows})
            source_hashes = sorted({row["source_sha256"] for row in rows})
            summary = {
                "sample_id": f"pg35-test-{role}-s{seed}",
                "dataset_id": f"pg35-{role}-s{seed}-v1",
                "source_id": f"pg35-role-source-{role}-s{seed}",
                "source_hash": sha256_json(source_hashes),
                "target_instance_ids": targets,
                "family_set": families,
                "sampling_seed": seed,
                "role": role,
                "sample_count": len(rows),
                "unique_sample_count": len(sample_ids),
                "denominator": len(rows),
                "positive_count": sum(int(row["oracle_projection"]["positive"]) for row in rows),
                "negative_count": sum(int(not row["oracle_projection"]["positive"]) for row in rows),
                "encoding_set": sorted({row["encoding"] for row in rows}),
                "method_set": sorted({row["method"] for row in rows}),
                "source_count": len(source_hashes),
                "target_instance_count": len(targets),
                "dataset_manifest_sha256": sha256_json({"role": role, "seed": seed, "samples": sample_ids}),
                "split_manifest_sha256": sha256_json({"role": role, "seed": seed, "targets": targets, "families": families}),
                "metrics_status": "pending_model_run",
                "metrics": {"typed_recall": 0.0, "precision": 0.0, "false_positive_rate": 0.0, "abstain_precision": 0.0, "ece": 0.0, "median_queries": 0.0},
            }
            summary["evidence_hash"] = sha256_json(summary)
            dataset_tests.append(summary)

    catalog = {
        "schema_version": "pg-pk-35-independent-fixture-catalog-v1",
        "catalog_id": "pg35-independent-fixture-v1",
        "purpose": "independent local GET/POST typed replay with same-family identity/url-percent pairs across three route implementations",
        "runtime_replay": True,
        "independent_target_implementation": True,
        "evaluation_only": True,
        "training_eligible": False,
        "training_artifact_generated": False,
        "model_evaluation_completed": False,
        "methods": ["GET", "POST"],
        "encodings": list(ENCODINGS),
        "seeds": list(SEEDS),
        "sources": sources,
        "source_catalogs": source_catalogs,
        "samples": all_records,
        "dataset_tests": dataset_tests,
        "trace_dataset": str(TRACE_OUTPUT.relative_to(ROOT)),
        "trace_episode_count": len(episode_reports),
        "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episode_reports),
        "typed_positive_count": sum(int(row["oracle_projection"]["positive"]) for row in all_records),
        "negative_control_count": sum(int(not row["oracle_projection"]["positive"]) for row in all_records),
        "fresh_reset_count": len(all_records),
        "source_count": len({row["source_sha256"] for row in all_records}),
        "target_instance_count": len({row["target_instance_id"] for row in all_records}),
        "encoding_pair_count": len({row["encoding_pair_id"] for row in all_records}),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "authorization": "workspace_local_only",
        "family_policy": ROLE_FAMILIES,
        "manifest_sha256": sha256_json({"samples": [row["evidence"]["evidence_hash"] for row in all_records], "dataset_tests": dataset_tests}),
    }
    trace_dataset = {
        "schema_version": "pg-pk-35-independent-fixture-trace-v1",
        "purpose": "independent target step-aligned GET/POST encoding-pair evidence replay",
        "evaluation_only": True,
        "training_eligible": False,
        "independent_target_implementation": True,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "methods": ["GET", "POST"],
        "encodings": list(ENCODINGS),
        "episode_count": len(episode_reports),
        "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episode_reports),
        "episodes": episode_reports,
        "steps": trace_steps,
        "catalog_manifest_sha256": catalog["manifest_sha256"],
        "trace_manifest_sha256": trace_sha256_json([step["trace_sha256"] for step in trace_steps]),
    }
    CATALOG_OUTPUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_OUTPUT.write_text(json.dumps(trace_dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "catalog": str(CATALOG_OUTPUT.relative_to(ROOT)),
        "trace_dataset": str(TRACE_OUTPUT.relative_to(ROOT)),
        "sample_count": len(all_records),
        "typed_positive_count": catalog["typed_positive_count"],
        "negative_control_count": catalog["negative_control_count"],
        "source_count": catalog["source_count"],
        "target_instance_count": catalog["target_instance_count"],
        "encoding_pair_count": catalog["encoding_pair_count"],
        "episode_count": len(episode_reports),
        "accepted_evaluation_episodes": catalog["accepted_evaluation_episode_count"],
        "independent_target_implementation": True,
        "training_eligible": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
