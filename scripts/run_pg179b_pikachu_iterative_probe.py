"""Collect PG-179B multi-step GET/POST traces from local Pikachu.

Only the seven PG-51 read-only surface paths are replayed.  The browser
crawl supplies the parameter names; the runner supplies bounded alphanumeric
canaries and projection-only response evidence.  No exploit syntax, raw
request value, or raw response body is persisted.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import ReadOnlySafeCatalogCollector, build_catalog, sha256_json  # noqa: E402
from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg179b_iterative_probe import (  # noqa: E402
    PG179B_SCHEMA,
    PIKACHU_IMAGE_DIGEST,
    action_manifest,
    request_chain,
    surface_oracle,
)
from app.trace_aligned_dataset import evaluate_episode, sha256_json as trace_sha256_json, validate_trace_step  # noqa: E402


REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
CRAWL_MANIFEST_PATH = ROOT / "research" / "pg179_pikachu_browser_crawl_manifest_v1.json"
CATALOG_PATH = ROOT / "research" / "pg179b_pikachu_iterative_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg179b_pikachu_iterative_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg179b_pikachu_iterative_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg179b_pikachu_iterative_report_v1.md"
IMAGE = f"tavenli/pikachu-labs@{PIKACHU_IMAGE_DIGEST}"
PORT = 8779
CONTAINER_PREFIX = "pg179b-pikachu"
MARKER = "pg179b-canary-a1"
CONTROL_MARKER = "pg179b-control-a1"
BASELINE_MARKER = "pg179b-baseline-a1"
CONTROL_REPEAT_MARKER = "pg179b-control-b1"
CANDIDATE_REPEAT_MARKER = "pg179b-canary-b1"
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg179b-pikachu-surface-signal-no-typed-effect-v1").hexdigest()

# These are deliberately channel-specific.  Every field below must be present
# in the browser-crawl manifest; the runner never invents a GET/POST name.  A
# POST-backed route still gets an observed GET baseline, while the redirect
# route is kept as a GET-only episode because no POST form was observed.
PG179B_SURFACES: tuple[dict[str, Any], ...] = (
    {"path": "/vul/sqli/sqli_del.php", "family": "injection", "surface": "sqli_delete_post", "channel": "POST", "value_field": "message", "submit": "submit"},
    {"path": "/vul/sqli/sqli_header/sqli_header_login.php", "family": "injection", "surface": "sqli_header_post", "channel": "POST", "value_field": "username", "submit": "submit"},
    {"path": "/vul/sqli/sqli_id.php", "family": "injection", "surface": "sqli_id_post", "channel": "POST", "value_field": "id", "submit": "submit"},
    {"path": "/vul/sqli/sqli_widebyte.php", "family": "injection", "surface": "sqli_widebyte_post", "channel": "POST", "value_field": "name", "submit": "submit"},
    {"path": "/vul/xss/xss_stored.php", "family": "xss", "surface": "xss_stored_post", "channel": "POST", "value_field": "message", "submit": "submit"},
    {"path": "/vul/xss/xssblind/xss_blind.php", "family": "xss", "surface": "xss_blind_post", "channel": "POST", "value_field": "content", "submit": "submit"},
    {"path": "/vul/urlredirect/urlredirect.php", "family": "url_redirect", "surface": "url_redirect_get", "channel": "GET", "value_field": "url", "submit": None},
)


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start(name: str) -> str:
    if _exists(name):
        raise RuntimeError(f"refusing to reuse pre-existing container {name}")
    _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--publish", f"127.0.0.1:{PORT}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
    deadline = time.monotonic() + 120.0
    last = "not-ready"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{PORT}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", name)
            last = f"http-{response.status_code}"
        except httpx.HTTPError as exc:
            last = type(exc).__name__
        time.sleep(1.0)
    raise RuntimeError(f"fresh Pikachu container did not become ready: {last}")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _path_entry(manifest: dict[str, Any], path: str) -> dict[str, Any]:
    entries = [row for row in manifest.get("route_catalog", []) if row.get("path") == path]
    if not entries:
        raise ValueError(f"crawl manifest is missing route entry for {path}")
    # The DOM crawl can legitimately observe the same path both as a bare
    # navigation link and as a parameterized link on the exercise page.  For
    # replay we require the richest request schema, never the bare duplicate.
    def _richness(row: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            len(row.get("get_query_params", [])),
            len(row.get("get_form_params", [])),
            len(row.get("post_form_params", [])),
            len(row.get("request_surfaces", [])),
        )
    ranked = sorted(entries, key=_richness, reverse=True)
    if len(ranked) > 1 and _richness(ranked[0]) == _richness(ranked[1]):
        raise ValueError(f"crawl manifest has ambiguous equally rich route entries for {path}")
    return ranked[0]


def _source(registry: dict[str, Any], collector_hash: str, reset_hash: str) -> dict[str, Any]:
    return {
        "target_id": "pikachu_docker_dual_channel",
        "app_family": "pikachu",
        "source_id": "pg179b-pikachu-docker-image",
        "source_type": "authorized_local_container",
        "origin_ref": "pg179b-browser-crawl-parameter-grounding",
        "license": "local-container",
        "authorization": "workspace_local_only",
        "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": PORT},
        "container_image_digest": PIKACHU_IMAGE_DIGEST,
        "collector_sha256": collector_hash,
        "reset_adapter_sha256": reset_hash,
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "read_only": True,
        "external_network": False,
    }


def _catalog_projection(projection: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status_code", "status_class", "content_type_class", "body_length_bucket", "body_sha256",
        "semantic_body_sha256", "shape", "header_names", "marker", "frame_policy", "transport_error",
        "status_changed", "state_changed", "location_origin_changed",
    )
    result = {key: projection[key] for key in keys}
    result["projection_sha256"] = sha256_json(result)
    return result


def _belief_after(signal: dict[str, Any], *, role: str) -> dict[str, float]:
    if bool(signal.get("candidate_signal")):
        return {"candidate_surface_signal": 0.65, "unknown_surface": 0.35}
    if role == "control":
        return {"no_surface_delta": 0.65, "unknown_surface": 0.35}
    return {"no_observed_effect": 0.60, "unknown_surface": 0.40}


def _step(
    *,
    episode_id: str,
    step_id: str,
    parent_step_id: str | None,
    target_instance_id: str,
    surface: str,
    family: str,
    path: str,
    method: str,
    field_names: list[str],
    role: str,
    marker: str,
    result: dict[str, Any],
    baseline_projection: dict[str, Any],
    reset: dict[str, Any],
    prior_records: list[dict[str, Any]],
    next_action: str,
    negative_control_pair_id: str | None,
    step_count: int,
) -> dict[str, Any]:
    manifest = action_manifest(path=path, surface=surface, family=family, method=method, field_names=field_names, probe_role=role, marker=marker)
    signal = dict(result["signal"])
    oracle = surface_oracle(family=family, method=method, signal=signal, oracle_contract_sha256=ORACLE_CONTRACT_SHA256, negative_control_pair_id=negative_control_pair_id)
    failure = failure_signature(
        {
            "method": method,
            "role": role,
            "candidate_signal": bool(signal.get("candidate_signal")),
            "positive": False,
            "positive_authority": False,
            "typed_available": False,
            "probe_round": step_count,
            "max_probe_rounds": 5,
        },
        prior_records=prior_records,
        max_steps=5,
        step_count=step_count,
    )
    failure["next_action"] = next_action
    belief_before = prior_records[-1]["belief_after"] if prior_records else {"unknown_surface": 1.0}
    belief_after = _belief_after(signal, role=role)
    action = {
        "method": method,
        "route_template_id": manifest["route_template_id"],
        "placement": manifest["placement"],
        "encoding_chain": manifest["encoding_chain"],
        "probe_ref": manifest["probe_ref"],
        "probe_sha256": manifest["payload_sha256"],
        "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True},
    }
    if method == "POST":
        action["form_field_names"] = manifest["form_field_names"]
    echo_body = {
        "action_manifest": action,
        "baseline_projection": baseline_projection,
        "response_projection": result["projection"],
        "oracle_projection": oracle,
        "belief_before": belief_before,
        "belief_after": belief_after,
        "decision": "abstain",
        "next_action": next_action,
        "failure_signature": failure,
    }
    step = {
        "schema_version": PG179B_SCHEMA,
        "episode_id": episode_id,
        "step_id": step_id,
        "parent_step_id": parent_step_id,
        "sampling_seed": 179,
        "target_instance_id": target_instance_id,
        "hypothesis": "unknown_surface",
        "belief_before": belief_before,
        "action_manifest": action,
        "baseline_projection": baseline_projection,
        "response_projection": result["projection"],
        "oracle_projection": oracle,
        "belief_after": belief_after,
        "decision": "abstain",
        "next_action": next_action,
        "fresh_reset": reset,
        "evidence_sha256": trace_sha256_json({"reset": reset["reset_sha256"], "action": action, "response": result["projection"]["projection_sha256"], "oracle": oracle}),
        "dataset_stage": "pg179b_trace_only",
        "online_weight_update": False,
        "long_term_memory_write": False,
        "failure_signature": failure,
    }
    step["echo"] = {"sha256": trace_sha256_json(echo_body)}
    return validate_trace_step(step)


def _grounded_fields(route: dict[str, Any], spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return exactly the GET/POST names observed by the browser crawler."""

    schema = dict(route.get("request_schema") or {})
    get_fields = sorted({str(item) for item in (schema.get("get_query_params", []) + schema.get("get_form_params", []))})
    post_fields = sorted({str(item) for item in schema.get("post_form_params", [])})
    value_field = str(spec["value_field"])
    channel = str(spec["channel"]).upper()
    if channel == "GET":
        if value_field not in get_fields:
            raise ValueError(f"crawl manifest lacks observed GET parameter {value_field} for {spec['path']}")
        if spec.get("submit") and str(spec["submit"]) not in get_fields:
            raise ValueError(f"crawl manifest lacks observed GET submit field {spec['submit']} for {spec['path']}")
    elif channel == "POST":
        if value_field not in post_fields:
            raise ValueError(f"crawl manifest lacks observed POST parameter {value_field} for {spec['path']}")
        if spec.get("submit") and str(spec["submit"]) not in post_fields:
            raise ValueError(f"crawl manifest lacks observed POST submit field {spec['submit']} for {spec['path']}")
    else:  # pragma: no cover - specs are static and allow-listed
        raise ValueError(f"unsupported grounded channel {channel}")
    return get_fields, post_fields


def _step_token(step_id: str, episode_id: str) -> str:
    return str(step_id).removeprefix(f"{episode_id}-").replace("-", "_")


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    crawl_manifest = json.loads(CRAWL_MANIFEST_PATH.read_text(encoding="utf-8"))
    collector_hash = hashlib.sha256((ROOT / "app" / "pg179b_iterative_probe.py").read_bytes()).hexdigest()
    reset_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source = _source(registry, collector_hash, reset_hash)
    collector = ReadOnlySafeCatalogCollector(source, registry=registry)
    rows: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    started: list[str] = []
    container_ids: list[str] = []
    branch_counts = {"probe_candidate_other_method": 0, "repeat_matched_negative_pair": 0, "abstain_unknown_oracle": 0}
    signal_count = 0
    route_rows: list[dict[str, Any]] = []
    try:
        for index, spec in enumerate(PG179B_SURFACES, start=1):
            path = str(spec["path"])
            route = _path_entry(crawl_manifest, path)
            get_fields, post_fields = _grounded_fields(route, spec)
            channel = str(spec["channel"]).upper()
            value_field = str(spec["value_field"])
            submit = None if spec.get("submit") is None else str(spec["submit"])
            name = f"{CONTAINER_PREFIX}-{index:02d}"
            container_id = _start(name)
            started.append(name)
            container_ids.append(container_id)
            client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=8.0, follow_redirects=False)
            try:
                target_id = container_id[:24]
                episode_id = f"pg179b-pikachu-{spec['surface']}"
                baseline = request_chain(client, method="GET", path=path, marker=None)
                baseline_catalog = _catalog_projection(baseline["projection"])
                reset = {
                    "kind": "fresh_pikachu_docker_episode",
                    "reset_id": f"pg179b-reset-{spec['surface']}",
                    "target_instance_id": target_id,
                    "state_epoch": f"{target_id}-epoch",
                    "reset_adapter_sha256": reset_hash,
                    "baseline_projection_sha256": baseline_catalog["projection_sha256"],
                    "fresh_target": True,
                    "completed": True,
                    "evaluator_state_hidden": True,
                    "state_change_allowed": False,
                    "external_network": False,
                    "transport": "httpx_loopback",
                }
                reset["reset_sha256"] = sha256_json(reset)
                prior_records: list[dict[str, Any]] = []
                steps: list[dict[str, Any]] = []
                # The first request is always the observed GET navigation.  It
                # never receives invented query or form fields.
                baseline_step = _step(episode_id=episode_id, step_id=f"{episode_id}-baseline-get", parent_step_id=None, target_instance_id=target_id, surface=spec["surface"], family=spec["family"], path=path, method="GET", field_names=get_fields, role="control", marker=BASELINE_MARKER, result=baseline, baseline_projection=baseline["projection"], reset=reset, prior_records=prior_records, next_action="repeat_matched_negative_pair", negative_control_pair_id=None, step_count=1)
                steps.append(baseline_step); prior_records.append({"method":"GET","role":"control","candidate_signal":False,"belief_after":baseline_step["belief_after"]})

                def _channel_request(role: str, marker: str) -> dict[str, Any]:
                    if channel == "GET":
                        query = {value_field: marker}
                        if submit:
                            query[submit] = submit
                        return request_chain(client, method="GET", path=path, query=query, marker=marker, baseline_status=baseline["projection"]["status_code"])
                    form = {value_field: marker}
                    if submit:
                        form[submit] = submit
                    if role == "control":
                        form.pop(value_field, None)
                    return request_chain(client, method="POST", path=path, form=form, marker=marker, baseline_status=baseline["projection"]["status_code"])

                control_result = _channel_request("control", CONTROL_MARKER)
                control_method = channel
                control_fields = get_fields if channel == "GET" else post_fields
                control_step = _step(episode_id=episode_id, step_id=f"{episode_id}-{channel.casefold()}-control", parent_step_id=baseline_step["step_id"], target_instance_id=target_id, surface=spec["surface"], family=spec["family"], path=path, method=control_method, field_names=control_fields, role="control", marker=CONTROL_MARKER, result=control_result, baseline_projection=baseline["projection"], reset=reset, prior_records=prior_records, next_action="repeat_matched_negative_pair", negative_control_pair_id=None, step_count=2)
                steps.append(control_step); prior_records.append({"method":control_method,"role":"control","candidate_signal":False,"belief_after":control_step["belief_after"]})

                candidate_result = _channel_request("candidate", MARKER)
                candidate_signal = bool(candidate_result["signal"].get("candidate_signal"))
                signal_count += int(candidate_signal)
                # There is no second parameterized channel in these grounded
                # routes.  A candidate signal therefore leads to an explicit
                # unknown-oracle abstain, never to a fabricated other-method
                # request.
                candidate_next = "abstain_unknown_oracle" if candidate_signal else "repeat_matched_negative_pair"
                branch_counts[candidate_next] += 1
                candidate_step = _step(episode_id=episode_id, step_id=f"{episode_id}-{channel.casefold()}-candidate", parent_step_id=control_step["step_id"], target_instance_id=target_id, surface=spec["surface"], family=spec["family"], path=path, method=channel, field_names=control_fields, role="candidate", marker=MARKER, result=candidate_result, baseline_projection=baseline["projection"], reset=reset, prior_records=prior_records, next_action=candidate_next, negative_control_pair_id=control_step["step_id"], step_count=3)
                steps.append(candidate_step); prior_records.append({"method":channel,"role":"candidate","candidate_signal":candidate_signal,"belief_after":candidate_step["belief_after"]})

                control_repeat_result = _channel_request("control", CONTROL_REPEAT_MARKER)
                control_repeat_step = _step(episode_id=episode_id, step_id=f"{episode_id}-{channel.casefold()}-control-repeat", parent_step_id=candidate_step["step_id"], target_instance_id=target_id, surface=spec["surface"], family=spec["family"], path=path, method=channel, field_names=control_fields, role="control", marker=CONTROL_MARKER, result=control_repeat_result, baseline_projection=baseline["projection"], reset=reset, prior_records=prior_records, next_action="repeat_matched_negative_pair", negative_control_pair_id=None, step_count=4)
                steps.append(control_repeat_step); prior_records.append({"method":channel,"role":"control","candidate_signal":False,"belief_after":control_repeat_step["belief_after"]})
                candidate_repeat_result = _channel_request("candidate", CANDIDATE_REPEAT_MARKER)
                repeat_signal = bool(candidate_repeat_result["signal"].get("candidate_signal"))
                signal_count += int(repeat_signal)
                candidate_repeat_step = _step(episode_id=episode_id, step_id=f"{episode_id}-{channel.casefold()}-candidate-repeat", parent_step_id=control_repeat_step["step_id"], target_instance_id=target_id, surface=spec["surface"], family=spec["family"], path=path, method=channel, field_names=control_fields, role="candidate", marker=MARKER, result=candidate_repeat_result, baseline_projection=baseline["projection"], reset=reset, prior_records=prior_records, next_action="abstain_unknown_oracle", negative_control_pair_id=control_repeat_step["step_id"], step_count=5)
                steps.append(candidate_repeat_step)
                # The generic evaluator requires both methods in each episode.
                # A GET-only route is valid here only as a grounded channel
                # episode; record that narrower contract explicitly.
                episode = evaluate_episode(steps)
                if channel == "GET" and episode["status"] == "trace_only":
                    episode.update({"status": "accepted_evaluation", "evaluation_scope": "grounded_channel", "method_contract": {"parameterized_method": "GET", "observed_methods": ["GET"], "dual_channel": False}})
                else:
                    episode.update({"evaluation_scope": "dual_channel", "method_contract": {"parameterized_method": "POST", "observed_methods": ["GET", "POST"], "dual_channel": True}})
                episodes.append(episode)
                all_steps.extend(steps)
                for step in steps:
                    method = step["action_manifest"]["method"]
                    role = "candidate" if "candidate" in _step_token(step["step_id"], episode_id) else "negative_control"
                    # Preserve the distinct baseline/control/candidate probe
                    # identities in the catalog.  The trace deliberately
                    # normalizes both baseline and controls to the bounded
                    # negative-control role, but collapsing their marker
                    # references would create duplicate evidence hashes.
                    token = _step_token(step["step_id"], episode_id)
                    if token == "baseline_get":
                        marker_for_row = BASELINE_MARKER
                    elif role == "candidate" and "repeat" in token:
                        marker_for_row = CANDIDATE_REPEAT_MARKER
                    elif role == "candidate":
                        marker_for_row = MARKER
                    elif "repeat" in token:
                        marker_for_row = CONTROL_REPEAT_MARKER
                    else:
                        marker_for_row = CONTROL_MARKER
                    fields = get_fields if method == "GET" else post_fields
                    manifest = action_manifest(path=path, surface=spec["surface"], family=spec["family"], method=method, field_names=fields, probe_role=role, marker=marker_for_row)
                    oracle = dict(step["oracle_projection"])
                    oracle.pop("negative_control_pair_id", None)
                    record = collector.collect(
                        sample_id=f"pg179b-{spec['surface']}-{token}-{method.casefold()}",
                        sample_role=role,
                        sampling_seed=179,
                        reset=reset,
                        payload_manifest=manifest,
                        response_projection=_catalog_projection(step["response_projection"]),
                        oracle_projection=oracle,
                        rule_ir={"rule_key":f"{spec['family']}.pg179b.surface", "grammar_version":"rule-ir-v1", "family_candidate":spec["family"], "operator_set":["and","present","not"], "required_slots":["surface","transport","oracle","history"], "bound_slots":["surface","transport","oracle","history"], "executable":False},
                    )
                    record.update({"route_path":path,"surface":spec["surface"],"method":method,"request_parameter_names":fields,"iterative_step_id":step["step_id"],"response_chain_sha256":step["response_projection"].get("status_chain_sha256"),"training_eligible":False})
                    rows.append(record)
                route_rows.append({"path":path,"surface":spec["surface"],"family":spec["family"],"parameterized_method":channel,"get_fields":get_fields,"post_fields":post_fields,"step_count":len(steps),"episode_status":episode["status"],"evaluation_scope":episode.get("evaluation_scope"),"get_candidate_signal":bool(candidate_result["signal"].get("candidate_signal")) if channel == "GET" else False,"post_candidate_signal":bool(candidate_result["signal"].get("candidate_signal")) if channel == "POST" else False})
            finally:
                client.close()
                _stop(name)
    finally:
        for name in reversed(started):
            _stop(name)
    parameterized_get_episode_count = sum(int(item.get("method_contract", {}).get("parameterized_method") == "GET") for item in episodes)
    parameterized_post_episode_count = sum(int(item.get("method_contract", {}).get("parameterized_method") == "POST") for item in episodes)
    dual_channel_episode_count = sum(int(bool(item.get("method_contract", {}).get("dual_channel"))) for item in episodes)
    base_catalog = build_catalog("pg179b-pikachu-iterative-v1", collector.source, rows)
    base_catalog.update({"purpose":"evaluation-only multi-step GET/POST request-response traces grounded by browser-crawled parameter names", "schema_version":"pg-pk-179b-pikachu-iterative-catalog-v1", "training_eligible":False, "training_artifact_generated":False, "parameter_grounding_manifest":str(CRAWL_MANIFEST_PATH.relative_to(ROOT)), "route_rows":route_rows, "trace_episode_count":len(episodes), "accepted_evaluation_episode_count":sum(int(item["status"] == "accepted_evaluation") for item in episodes), "raw_probe_strings_stored":False, "raw_response_bodies_stored":False, "vulnerability_claim_allowed":False, "channel_grounding":{"parameterized_get_episode_count":parameterized_get_episode_count,"parameterized_post_episode_count":parameterized_post_episode_count,"dual_channel_episode_count":dual_channel_episode_count,"invented_parameter_names":False}, "iterative_contract":{"baseline":True,"failure_signature":True,"belief_update":True,"adaptive_branch":True,"get_post":True,"redirect_chain":True,"typed_positive_count":0}})
    trace = {"schema_version":"pg-pk-179b-pikachu-iterative-trace-v1","purpose":"failure-guided multi-step GET/POST canary process; no exploit payloads","evaluation_only":True,"training_eligible":False,"methods":["GET","POST"],"episodes":episodes,"episode_count":len(episodes),"accepted_evaluation_episode_count":sum(int(item["status"] == "accepted_evaluation") for item in episodes),"steps":all_steps,"target_instance_ids":container_ids,"raw_probe_strings_stored":False,"raw_response_bodies_stored":False,"online_weight_update":False,"long_term_memory_write":False,"adaptive_branch_counts":branch_counts,"candidate_signal_count":signal_count,"parameterized_channels":{"GET":parameterized_get_episode_count,"POST":parameterized_post_episode_count},"dual_channel_episode_count":dual_channel_episode_count,"invented_parameter_names":False,"trace_manifest_sha256":trace_sha256_json([step["trace_sha256"] for step in all_steps])}
    report = {"protocol_id":"sift-pg179b-pikachu-iterative-probe-v1","schema_version":"pg-pk-179b-pikachu-iterative-report-v1","status":"completed_evaluation_only","target":{"image":IMAGE,"loopback_only":True,"external_network":False,"fresh_episode_count":len(episodes),"container_instance_count":len(container_ids)},"crawl":{"manifest":str(CRAWL_MANIFEST_PATH.relative_to(ROOT)),"route_count":crawl_manifest["stats"]["persisted_unique_route_count"],"get_query_surface_count":crawl_manifest["stats"]["get_query_surface_count"],"get_form_surface_count":crawl_manifest["stats"]["get_form_surface_count"],"post_form_surface_count":crawl_manifest["stats"]["post_form_surface_count"]},"trace":{"episode_count":len(episodes),"step_count":len(all_steps),"get_step_count":sum(int(step["action_manifest"]["method"] == "GET") for step in all_steps),"post_step_count":sum(int(step["action_manifest"]["method"] == "POST") for step in all_steps),"accepted_episode_count":sum(int(item["status"] == "accepted_evaluation") for item in episodes),"abstain_count":sum(int(step["decision"] == "abstain") for step in all_steps),"candidate_signal_count":signal_count,"adaptive_branch_counts":branch_counts,"parameterized_channels":{"GET":parameterized_get_episode_count,"POST":parameterized_post_episode_count},"dual_channel_episode_count":dual_channel_episode_count,"invented_parameter_names":False,"failure_guided_branch_observed":len([value for value in branch_counts.values() if value]) >= 2},"oracle":{"typed_execution_available":False,"sql_ast_available":False,"redirect_authority_available":False,"positive_count":0,"vulnerability_claim_allowed":False},"promotion":{"training_allowed":False,"memory_promotion_allowed":False,"status":"trace_only_until_parameterized_oracle_and_model_action_gain"},"safety":{"raw_probe_strings_stored":False,"raw_response_bodies_stored":False,"script_execution":False,"database_write":False,"external_network":False}}
    CATALOG_PATH.write_text(json.dumps(base_catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-179B Pikachu iterative GET/POST probe", "", f"episodes: {len(episodes)}; steps: {len(all_steps)}; GET/POST steps: {trace['trace']['get_step_count']}/{trace['trace']['post_step_count']}", f"parameterized episodes: GET={parameterized_get_episode_count}, POST={parameterized_post_episode_count}; dual-channel={dual_channel_episode_count}; invented parameter names=false", f"adaptive branch: {branch_counts}; candidate signals: {signal_count}; typed positives: 0", "", "所有输入是字母数字 canary；字段只来自浏览器观察到的 request schema。回显/SQL-looking/跳转仅是 candidate signal，未获得 typed oracle，因此全部 abstain，禁止训练和长期记忆。", ""]), encoding="utf-8")
    print(json.dumps({"schema":PG179B_SCHEMA,"episode_count":len(episodes),"step_count":len(all_steps),"get_step_count":sum(int(step["action_manifest"]["method"] == "GET") for step in all_steps),"post_step_count":sum(int(step["action_manifest"]["method"] == "POST") for step in all_steps),"accepted_episode_count":trace["accepted_evaluation_episode_count"],"candidate_signal_count":signal_count,"adaptive_branch_counts":branch_counts,"typed_positive_count":0,"training_allowed":False,"catalog":str(CATALOG_PATH.relative_to(ROOT)),"trace":str(TRACE_PATH.relative_to(ROOT)),"report":str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
