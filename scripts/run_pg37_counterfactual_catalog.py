"""Collect PG-37 same-family/multi-surface counterfactual traces.

Every control and candidate observation starts a fresh loopback HTTP server.
Only bounded projections, hashes, and typed-oracle metadata are retained;
request strings and response bodies are never written to the Catalog.
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
from app.pg37_counterfactual_fixture import LAYOUTS, PHASES, SURFACE_SPECS, VARIANTS, make_pg37_server  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json as trace_sha256_json, validate_trace_step  # noqa: E402


REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
CATALOG_OUTPUT = ROOT / "research" / "pg37_counterfactual_catalog_v1.json"
TRACE_OUTPUT = ROOT / "research" / "pg37_counterfactual_trace_v1.json"
SEEDS = (361, 367, 373)
TARGET_PORT = 31937
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg37-counterfactual-typed-oracle-v1").hexdigest()
ROLE_FAMILIES = {
    "train": ["xss", "injection"],
    "dev": ["authentication", "access_control"],
    "family_holdout": ["logic", "url_redirect"],
    "ood_source": ["input_validation", "command_injection"],
    "negative_control": ["ordinary_response", "unknown_surface"],
}
ROLE_BY_SURFACE = {
    surface: role
    for role, families in ROLE_FAMILIES.items()
    for surface, spec in SURFACE_SPECS.items()
    if spec["family"] in families
}


class _FreshTarget:
    def __init__(self, implementation: str, port: int = TARGET_PORT) -> None:
        self.implementation = str(implementation)
        self.port = int(port)
        self.server = make_pg37_server(self.port, self.implementation)
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
            raise RuntimeError("PG-37 fixture did not start")
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


def _call(client: httpx.Client, implementation: str, surface: str, variant: str, method: str, phase: str, *, positive: bool) -> httpx.Response:
    layout = LAYOUTS[implementation]
    spec = SURFACE_SPECS[surface]
    token = spec["positive"] if positive else "normal"
    values = {layout["slot_key"]: surface, layout["probe_key"]: token, layout["variant_key"]: variant, layout["phase_key"]: phase}
    if method == "GET":
        return client.get(_route(implementation, surface), params=values)
    if method != "POST":
        raise ValueError("PG-37 permits only GET and POST")
    headers = {"content-type": layout["post_content_type"]}
    if layout["post_content_type"] == "application/json":
        return client.post(_route(implementation, surface), content=json.dumps(values, separators=(",", ":")).encode("utf-8"), headers=headers)
    body = "&".join(f"{key}={value}" for key, value in values.items()).encode("utf-8")
    return client.post(_route(implementation, surface), content=body, headers=headers)


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
    projection = {
        "status_code": int(response.status_code),
        "status_class": _status_class(int(response.status_code)),
        "content_type_class": "json" if str(response.headers.get("content-type", "")).split(";", 1)[0].casefold() == "application/json" else "other",
        "body_length_bucket": _length_bucket(len(body)),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "semantic_body_sha256": sha256_json(shape),
        "shape": shape,
        "header_names": sorted({str(key).casefold() for key in response.headers.keys()} & {"content-type", "location", "allow"}),
        "marker": {"reflected": False, "location": "none", "count": 0},
        "frame_policy": "unknown",
        "transport_error": False,
        "status_changed": int(response.status_code) >= 400,
        "state_changed": False,
        "location_origin_changed": False,
    }
    projection["projection_sha256"] = sha256_json(projection)
    return projection, parsed


def _oracle(implementation: str, surface: str, variant: str, method: str, phase: str, parsed: dict[str, Any]) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    positive = bool(parsed.get("typed_effect_ready", False)) and phase == "confirm" and spec["family"] not in {"ordinary_response", "unknown_surface"}
    allowed = {
        "candidate_signal", "ambiguous", "typed_effect_ready", "dom_change", "marker_hits", "ast_shape_diff", "interpreter_boundary",
        "authentication_boundary", "authorization_boundary", "cross_subject_access", "business_invariant_boundary", "redirect_candidate",
        "same_origin", "external_redirect", "validation_boundary", "rejected", "local_canary_observed", "command_executed", "ordinary_response",
        "error_class", "timeout_class", "state_mutated", "database_touched", "credentials_accessed", "external_network", "script_execution",
        "bounded_response_delta",
    }
    safe = {key: parsed[key] for key in parsed if key in allowed}
    return {
        "oracle_id": f"pg37-counterfactual-{spec['family']}-oracle-v1",
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "family": spec["family"],
        "modality": spec["modality"] if positive else "negative_control",
        "candidate_signal": bool(parsed.get("candidate_signal", False)),
        "positive": positive,
        "positive_authority": positive,
        "confirmed_effect": spec["effect"] if positive else "none",
        "signals": {"implementation": implementation, "variant": variant, "method": method, "phase": phase, **safe},
        "safety": {"external_network": False, "script_execution": False, "database_write": False, "persistent_state_mutated": False, "credentials_accessed": False, "raw_body_stored": False},
    }


def _source(implementation: str, collector_hash: str, fixture_hash: str) -> dict[str, Any]:
    return {
        "target_id": "pg37_counterfactual_fixture",
        "app_family": "standalone_python_http_counterfactual_v1",
        "source_id": f"pg37-counterfactual-source-{implementation}",
        "source_type": "in_repo_synthetic",
        "origin_ref": "app/pg37_counterfactual_fixture.py:/<implementation>/<surface>/<variant>",
        "license": "in-repo-synthetic",
        "authorization": "workspace_local_only",
        "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": TARGET_PORT},
        "fixture_source_sha256": hashlib.sha256(f"{fixture_hash}:{implementation}".encode()).hexdigest(),
        "collector_sha256": collector_hash,
        "reset_adapter_sha256": hashlib.sha256(f"pg37-counterfactual-fresh-reset:{implementation}".encode()).hexdigest(),
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "read_only": True,
        "external_network": False,
    }


def _reset(source: dict[str, Any], implementation: str, surface: str, variant: str, seed: int, method: str, phase: str, role: str, pair_role: str, baseline: dict[str, Any]) -> dict[str, Any]:
    suffix = f"{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}-{phase}-{role}-{pair_role}"
    result = {
        "reset_id": f"pg37-reset-{suffix}",
        "kind": "fresh_pg37_http_server",
        "target_instance_id": f"pg37-target-{suffix}",
        "state_epoch": f"pg37-epoch-{suffix}",
        "reset_adapter_sha256": source["reset_adapter_sha256"],
        "baseline_projection_sha256": baseline["projection_sha256"],
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
        "transport": "httpx_loopback",
    }
    result["reset_sha256"] = sha256_json(result)
    return result


def _manifest(implementation: str, surface: str, variant: str, seed: int, method: str, phase: str, role: str, pair_role: str) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    layout = LAYOUTS[implementation]
    probe_kind = "http_canary" if phase == "timeout" else spec["probe_kind"]
    basis = {"fixture": "pg37", "implementation": implementation, "surface": surface, "variant": variant, "seed": seed, "method": method, "phase": phase, "role": role, "pair_role": pair_role, "probe_kind": probe_kind}
    return {
        "manifest_id": f"pg37-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}-{phase}-{role}-{pair_role}",
        "payload_sha256": sha256_json(basis),
        "probe_ref": f"pg37-safe-{spec['family']}-{phase}",
        "probe_kind": probe_kind,
        "route_template_id": f"pg37-{implementation}-{surface}-{variant}",
        "method": method,
        "placement": "query" if method == "GET" else "form",
        "encoding_chain": ["identity"],
        "encoding_depth": 0,
        "marker_sha256": hashlib.sha256(f"pg37-safe-marker:{implementation}:{surface}:{variant}:{seed}".encode()).hexdigest(),
        "max_bytes": 256,
        "form_field_names": [layout["slot_key"], layout["probe_key"], layout["variant_key"], layout["phase_key"]] if method == "POST" else [],
        "form_content_type": layout["post_content_type"] if method == "POST" else "",
        "safety": {"does_not_execute": True, "no_external_network": True, "no_script_execution": True, "no_database_write": True, "no_credential_access": True},
    }


def _rule_ir(surface: str, variant: str, method: str, phase: str) -> dict[str, Any]:
    family = SURFACE_SPECS[surface]["family"]
    return {"rule_key": f"{family}.pg37-{surface}-{variant}.{method.casefold()}.{phase}", "grammar_version": "rule-ir-v1", "family_candidate": family, "operator_set": ["and", "eq", "present"], "required_slots": ["surface", "transport", "oracle"], "bound_slots": ["surface", "transport", "oracle"], "executable": False}


def _trace_step(row: dict[str, Any], *, episode_id: str, step_id: str, parent_step_id: str | None, next_action: str, negative_control_pair_id: str | None, belief_before: dict[str, float]) -> dict[str, Any]:
    manifest = row["payload_manifest"]
    oracle = dict(row["oracle_projection"])
    if negative_control_pair_id:
        oracle["negative_control_pair_id"] = negative_control_pair_id
    family = row["family"]
    positive = bool(oracle.get("positive", False))
    belief_after = {"unknown": 0.05, family: 0.95} if positive else {"unknown": 0.97, family: 0.03}
    action = {"method": manifest["method"], "route_template_id": manifest["route_template_id"], "placement": manifest["placement"], "encoding_chain": manifest["encoding_chain"], "probe_ref": manifest["probe_ref"], "probe_sha256": manifest["payload_sha256"], "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if manifest["method"] == "POST":
        action["form_field_names"] = manifest["form_field_names"]
    decision = "confirmed_positive" if positive else "confirmed_negative"
    echo_body = {"action_manifest": action, "baseline_projection": row["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_before": belief_before, "belief_after": belief_after, "decision": decision, "next_action": next_action}
    return validate_trace_step({"schema_version": "sift-trace-aligned-step-v1", "episode_id": episode_id, "step_id": step_id, "parent_step_id": parent_step_id, "sampling_seed": int(row["sampling_seed"]), "target_instance_id": row["target_instance_id"], "hypothesis": family, "belief_before": belief_before, "action_manifest": action, "baseline_projection": row["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_after": belief_after, "decision": decision, "next_action": next_action, "fresh_reset": row["reset"], "evidence_sha256": row["evidence"]["evidence_hash"], "dataset_stage": "trace_only", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": trace_sha256_json(echo_body)}})


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_hash = hashlib.sha256((ROOT / "app" / "pg37_counterfactual_fixture.py").read_bytes()).hexdigest()
    all_records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    source_catalogs: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    trace_steps: list[dict[str, Any]] = []
    for implementation in LAYOUTS:
        source = _source(implementation, collector_hash, fixture_hash)
        collector = ReadOnlySafeCatalogCollector(source, registry=registry)
        sources.append(collector.source)
        implementation_records: list[dict[str, Any]] = []
        for surface, spec in SURFACE_SPECS.items():
            role = ROLE_BY_SURFACE[surface]
            for variant in VARIANTS:
                for seed in SEEDS:
                    episode_id = f"pg37-episode-{implementation}-{surface}-{variant}-s{seed}"
                    previous: str | None = None
                    belief = {"unknown": 1.0}
                    for method in ("GET", "POST"):
                        for phase in PHASES:
                            with _FreshTarget(implementation) as client:
                                control_response = _call(client, implementation, surface, variant, method, phase, positive=False)
                            with _FreshTarget(implementation) as client:
                                candidate_response = _call(client, implementation, surface, variant, method, phase, positive=role != "negative_control")
                            baseline, _ = _response_projection(control_response)
                            control_projection, control_parsed = _response_projection(control_response)
                            candidate_projection, candidate_parsed = _response_projection(candidate_response)
                            control_oracle = _oracle(implementation, surface, variant, method, phase, control_parsed)
                            candidate_oracle = _oracle(implementation, surface, variant, method, phase, candidate_parsed)
                            candidate_positive = bool(candidate_oracle["positive"])
                            control_id = f"pg37-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}-{phase}-control"
                            candidate_id = f"pg37-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}-{phase}-candidate"
                            control = collector.collect(sample_id=control_id, sample_role="negative_control", sampling_seed=seed, reset=_reset(collector.source, implementation, surface, variant, seed, method, phase, role, "control", baseline), payload_manifest=_manifest(implementation, surface, variant, seed, method, phase, role, "control"), response_projection=control_projection, oracle_projection=control_oracle, rule_ir=_rule_ir(surface, variant, method, phase))
                            candidate = collector.collect(sample_id=candidate_id, sample_role="candidate" if candidate_positive else "negative_control", sampling_seed=seed, reset=_reset(collector.source, implementation, surface, variant, seed, method, phase, role, "candidate", baseline), payload_manifest=_manifest(implementation, surface, variant, seed, method, phase, role, "candidate"), response_projection=candidate_projection, oracle_projection=candidate_oracle, rule_ir=_rule_ir(surface, variant, method, phase), negative_control=({"control_sample_id": control["sample_id"], "control_evidence_hash": control["evidence"]["evidence_hash"], "intervention": "typed-class-vs-normal-control", "verdict": "confirmed_negative", "same_source": True, "same_surface": True, "same_variant": True} if candidate_positive else None))
                            for row, sample_role, pair_role in ((control, "negative_control", "control"), (candidate, "candidate" if candidate_positive else "negative_control", "candidate")):
                                row.update({"dataset_role": role, "implementation": implementation, "surface_id": surface, "surface_variant": variant, "family": spec["family"], "method": method, "phase": phase, "sample_role": sample_role, "pair_role": pair_role})
                                implementation_records.append(row)
                            for row, pair_id, action in ((control, None, "replay_candidate"), (candidate, control["sample_id"] if candidate_positive else None, "confirm_same_surface" if phase == "screen" else "stop_episode" if candidate_positive else "next_probe")):
                                step = _trace_step(row, episode_id=episode_id, step_id=f"{episode_id}-{method.casefold()}-{phase}-{row['pair_role']}", parent_step_id=previous, next_action=action, negative_control_pair_id=pair_id, belief_before=belief)
                                previous = step["step_id"]
                                belief = step["belief_after"]
                                trace_steps.append(step)
                    episode_steps = [step for step in trace_steps if step["episode_id"] == episode_id]
                    episode_reports.append(evaluate_episode(episode_steps))
        variant_catalog = build_catalog(f"pg37-counterfactual-{implementation}-catalog", collector.source, implementation_records)
        source_catalogs.append({"source_id": collector.source["source_id"], "source_sha256": collector.source["source_sha256"], "catalog_sha256": variant_catalog["catalog_sha256"], "sample_count": len(implementation_records), "training_eligible": variant_catalog["training_eligible"]})
        all_records.extend(implementation_records)

    dataset_tests: list[dict[str, Any]] = []
    for role, families in ROLE_FAMILIES.items():
        for seed in SEEDS:
            rows = [row for row in all_records if row["dataset_role"] == role and int(row["sampling_seed"]) == seed]
            sample_ids = sorted(row["sample_id"] for row in rows)
            targets = sorted({row["target_instance_id"] for row in rows})
            source_hashes = sorted({row["source_sha256"] for row in rows})
            summary = {"sample_id": f"pg37-test-{role}-s{seed}", "dataset_id": f"pg37-{role}-s{seed}-v1", "source_id": f"pg37-role-source-{role}-s{seed}", "source_hash": sha256_json(source_hashes), "target_instance_ids": targets, "family_set": families, "sampling_seed": seed, "role": role, "sample_count": len(rows), "unique_sample_count": len(sample_ids), "denominator": len(rows), "positive_count": sum(int(row["oracle_projection"]["positive"]) for row in rows), "negative_count": sum(int(not row["oracle_projection"]["positive"]) for row in rows), "abstain_count": 0, "method_set": sorted({row["method"] for row in rows}), "phase_set": sorted({row["phase"] for row in rows}), "surface_variant_set": sorted({row["surface_variant"] for row in rows}), "source_count": len(source_hashes), "dataset_manifest_sha256": sha256_json({"role": role, "seed": seed, "samples": sample_ids}), "split_manifest_sha256": sha256_json({"role": role, "seed": seed, "targets": targets, "families": families}), "probe_sha256": sha256_json([row["payload_manifest"]["payload_sha256"] for row in rows]), "oracle_contract_sha256": ORACLE_CONTRACT_SHA256, "checkpoint_sha256": sha256_json("pending-model-run"), "metrics_status": "pending_model_run", "metrics": {"typed_recall": 0.0, "precision": 0.0, "false_positive_rate": 0.0, "abstain_precision": 0.0, "ece": 0.0, "median_queries": 0.0}}
            summary["evidence_hash"] = sha256_json(summary)
            dataset_tests.append(summary)

    catalog = {"schema_version": "pg-pk-37-counterfactual-catalog-v1", "catalog_id": "pg37-counterfactual-v1", "purpose": "same-family multi-surface counterfactual GET/POST typed replay", "runtime_replay": True, "independent_target_implementation": True, "evaluation_only": True, "training_eligible": True, "training_artifact_generated": False, "model_evaluation_completed": False, "methods": ["GET", "POST"], "phases": list(PHASES), "surface_variants": list(VARIANTS), "seeds": list(SEEDS), "implementations": list(LAYOUTS), "sources": sources, "source_catalogs": source_catalogs, "samples": all_records, "dataset_tests": dataset_tests, "trace_dataset": str(TRACE_OUTPUT.relative_to(ROOT)), "trace_episode_count": len(episode_reports), "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episode_reports), "typed_positive_count": sum(int(row["oracle_projection"]["positive"]) for row in all_records), "negative_control_count": sum(int(not row["oracle_projection"]["positive"]) for row in all_records), "counterfactual_pair_count": len({(row["implementation"], row["surface_id"], row["sampling_seed"], row["method"], row["phase"]) for row in all_records}), "fresh_reset_count": len(all_records), "source_count": len({row["source_sha256"] for row in all_records}), "target_instance_count": len({row["target_instance_id"] for row in all_records}), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "external_network": False, "authorization": "workspace_local_only", "family_policy": ROLE_FAMILIES, "manifest_sha256": sha256_json({"samples": [row["evidence"]["evidence_hash"] for row in all_records], "dataset_tests": dataset_tests})}
    trace_dataset = {"schema_version": "pg-pk-37-counterfactual-trace-v1", "purpose": "counterfactual surface trace with typed oracle after probe", "evaluation_only": True, "training_eligible": False, "independent_target_implementation": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "methods": ["GET", "POST"], "phases": list(PHASES), "surface_variants": list(VARIANTS), "episode_count": len(episode_reports), "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episode_reports), "episodes": episode_reports, "steps": trace_steps, "catalog_manifest_sha256": catalog["manifest_sha256"], "trace_manifest_sha256": trace_sha256_json([step["trace_sha256"] for step in trace_steps])}
    CATALOG_OUTPUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_OUTPUT.write_text(json.dumps(trace_dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"catalog": str(CATALOG_OUTPUT.relative_to(ROOT)), "trace_dataset": str(TRACE_OUTPUT.relative_to(ROOT)), "sample_count": len(all_records), "typed_positive_count": catalog["typed_positive_count"], "negative_control_count": catalog["negative_control_count"], "counterfactual_pair_count": catalog["counterfactual_pair_count"], "source_count": catalog["source_count"], "episode_count": len(episode_reports), "accepted_evaluation_episodes": catalog["accepted_evaluation_episode_count"], "training_eligible": catalog["training_eligible"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
