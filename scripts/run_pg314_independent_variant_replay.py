"""PG-314: replay PG-313's model-selected abstract variants on an independent lab.

The model predicts only bounded Rule-IR slots plus ``probe_variant`` and
``encoding_chain`` references.  A source-attested local adapter maps those
references to the already reviewed candidate/reference/negative arms.  Raw
wires and bounded response projections remain in the human catalog; the
training projection contains abstract contexts, targets and hashes only.

This runner is deliberately evaluation-only.  It uses a fresh no-volume
container from a different local image digest, ``--network none``, GET/POST
read-only probes, matched negative controls, repeated replay, and typed
evidence.  It refuses to run without the explicit morning-window flag.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import TARGET_BOS, TARGET_EOS  # noqa: E402
from app.pg295_causal_moe import generate_target  # noqa: E402
from app.pg301_payload_assembly import TARGET_KEYS, target_map  # noqa: E402
from app.pg303_guarded_composer import compose_guarded_plan  # noqa: E402
from app.pg305_live_evaluator import (  # noqa: E402
    abstract_projection,
    context_tokens,
    evaluator_result,
    load_causal_checkpoint,
    typed_evidence,
)
from app.pg313_probe_variant import (  # noqa: E402
    EXTRA_KEYS,
    bind_probe_variant_plan,
    probe_target_for_context,
)


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG266 = _load_script("run_pg266_pikachu_payload_grounding_replay.py")

RESEARCH = ROOT / "research"
CHECKPOINT = ROOT / "artifacts" / "pg313-probe-variant" / "pg313_probe_variant_moe_local_morning.pt"
REPORT = RESEARCH / "pg314_independent_variant_replay_report_v1.json"
CATALOG = RESEARCH / "pg314_independent_variant_human_catalog_v1.json"
DATASET = RESEARCH / "pg314_independent_variant_training_dataset_v1.json"
TRACE = RESEARCH / "pg314_independent_variant_trace_v1.json"
PROTOCOL = RESEARCH / "pg314_independent_variant_protocol_v1.json"
MARKDOWN = RESEARCH / "pg314_independent_variant_replay_report_v1.md"

# This image is a separate local implementation/digest from PG-214/PG-305.
INDEPENDENT_IMAGE = "sift/pikachu-pg240-source-native@sha256:de3227c1f56969be94521bc4bb48814b5dd1f511a1e368c688933812eaafe973"
SEED = 31401
BASE_PORT = 6235
# The independent image is first exercised on the non-destructive SQL
# row-shape pair.  DOM/XSS requires a browser inside the isolated namespace;
# it is deliberately a later experiment rather than silently using a weaker
# HTML-only oracle here.
ROUTE_IDS = ("sql-string-get", "sql-numeric-post")
VARIANT_ROLES = ("candidate_request", "reference_request", "negative_control")
ROLE_TO_VARIANT = {
    "candidate_request": "source_attested_candidate",
    "reference_request": "reference_canary",
    "negative_control": "negative_control",
}
VARIANT_TO_CATALOG = {
    "source_attested_candidate": "candidate",
    "reference_canary": "reference",
    "negative_control": "negative",
}
TARGET_LENGTH = len(TARGET_KEYS) + len(EXTRA_KEYS) + 2


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _database_health(name: str) -> bool:
    code = "$db=@new mysqli('127.0.0.1','root','root','pikachu',3306); exit($db->connect_errno ? 1 : 0);"
    result = subprocess.run(["docker", "exec", name, "php", "-r", code], cwd=ROOT, capture_output=True, text=True, timeout=20)
    return result.returncode == 0


def _http_health(name: str) -> bool:
    result = subprocess.run(["docker", "exec", name, "curl", "-fsS", "--max-time", "5", "-o", "/dev/null", "http://127.0.0.1:8090/"], cwd=ROOT, capture_output=True, text=True, timeout=10)
    return result.returncode == 0


def _require_gate() -> None:
    if os.environ.get("PG314_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-314 requires explicit PG314_LOCAL_DOCKER_EVAL=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-314 local Docker replay is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    if not CHECKPOINT.exists():
        raise RuntimeError(f"PG-313 checkpoint is missing: {CHECKPOINT}")


def _start(run_index: int) -> tuple[str, int, str, dict[str, Any]]:
    name = f"sift-pg314-{SEED}-{run_index}"
    if _exists(name):
        raise RuntimeError(f"PG-314 refuses to reuse target {name}")
    port = BASE_PORT + int(run_index)
    _docker(
        "run", "--detach", "--rm", "--pull=never", "--network", "none",
        "--name", name, "--label", "sift.pg314=true",
        "--label", f"sift.pg314.reset_epoch={SEED}-{run_index}",
        INDEPENDENT_IMAGE,
    )
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            if _http_health(name) and _database_health(name):
                container_id = _docker("inspect", "--format", "{{.Id}}", name)
                mounts = json.loads(_docker("inspect", "--format", "{{json .Mounts}}", name) or "[]")
                image_ref = _docker("inspect", "--format", "{{.Config.Image}}", name)
                image_id = _docker("inspect", "--format", "{{.Image}}", name)
                network_mode = _docker("inspect", "--format", "{{.HostConfig.NetworkMode}}", name)
                if mounts:
                    raise RuntimeError("PG-314 clean reset requires zero mounts/volumes")
                if image_ref != INDEPENDENT_IMAGE:
                    raise RuntimeError("PG-314 image digest attestation mismatch")
                if network_mode != "none":
                    raise RuntimeError("PG-314 requires Docker network=none")
                return name, port, container_id, {
                    "reset_id": f"pg314-reset-{SEED}-{run_index}",
                    "reset_epoch": f"{SEED}-{run_index}",
                    "fresh_target": True,
                    "completed": True,
                    "container_recreated": True,
                    "container_restart_used": False,
                    "container_id_sha256": hashlib.sha256(container_id.encode()).hexdigest(),
                    "image": image_ref,
                    "image_id_sha256": image_id.removeprefix("sha256:"),
                    "volume_mount_count": len(mounts),
                    "database_health_gate": "mysqli_root_pikachu_ok",
                    "database_clean_contract": "fresh_writable_layer_no_volume_no_stateful_probe",
                    "network_mode": network_mode,
                    "external_network": False,
                    "host_port_published": False,
                    "state_change_allowed": False,
                }
        except (httpx.HTTPError, subprocess.SubprocessError):
            pass
        time.sleep(1.0)
    if _exists(name):
        _docker("stop", "--timeout", "5", name)
    raise RuntimeError(f"PG-314 target {name} did not pass HTTP + database health gates")


def _stop(name: str) -> None:
    if name and _exists(name):
        _docker("stop", "--timeout", "5", name)


def _source_hash(name: str, route: Mapping[str, Any]) -> str:
    result = _docker("exec", name, "sha256sum", "/app/www" + str(route["path"]))
    digest = str(result).split()[0].strip().casefold()
    return digest if re.fullmatch(r"[0-9a-f]{64}", digest) else ""


def _evaluator_reset(reset: Mapping[str, Any]) -> dict[str, Any]:
    """Project the runtime attestation into PG-284's bounded reset schema."""

    return {
        "reset_id": str(reset.get("reset_id", "")),
        "fresh_target": bool(reset.get("fresh_target")),
        "container_recreated": bool(reset.get("container_recreated")),
        "container_restart_used": bool(reset.get("container_restart_used")),
        "volume_mount_count": int(reset.get("volume_mount_count", -1)),
        "database_health_gate": "healthy" if str(reset.get("database_health_gate")) == "mysqli_root_pikachu_ok" else "unknown",
        "state_change_allowed": False,
    }


def _send_internal(name: str, origin: str, route: Mapping[str, Any], values: Mapping[str, str], marker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Send through the container loopback because network=none has no host port."""

    method = str(route["method"]).upper()
    path = str(route["path"])
    encoded = urlencode(list(values.items()))
    url = f"http://127.0.0.1:8090{path}"
    args = ["exec", name, "curl", "-sS", "--max-time", "15", "-D", "-", "-X", method]
    if method == "GET":
        url = f"{url}?{encoded}"
    else:
        args.extend(["-H", "Content-Type: application/x-www-form-urlencoded", "--data", encoded])
    args.append(url)
    result = subprocess.run(["docker", *args], cwd=ROOT, capture_output=True, text=True, timeout=25)
    if result.returncode != 0:
        raise RuntimeError(f"PG-314 internal request failed for {route['id']}: {result.stderr[-240:]}")
    raw = result.stdout
    sections = re.split(r"\r?\n\r?\n", raw, maxsplit=1)
    header_text = sections[0] if sections else ""
    body = sections[1] if len(sections) > 1 else ""
    header_lines = [line for line in header_text.splitlines() if line]
    status_code = 0
    for header_line in header_lines:
        match = re.match(r"HTTP/\S+\s+(\d{3})(?:\s|$)", header_line.strip())
        if match:
            status_code = int(match.group(1))
    if status_code == 0 and header_lines:
        match = re.search(r"\s(\d{3})\s", header_lines[0] + " ")
        if match:
            status_code = int(match.group(1))
    headers: dict[str, str] = {}
    for line in header_lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    body_hash = hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()
    position = body.find(marker)
    excerpt = body[max(0, position - 120): position + min(260, len(body) - position)] if position >= 0 else ""
    projection = {
        "status_code": status_code,
        "status_class": f"{status_code // 100}xx" if status_code else "unknown",
        "location": headers.get("location"),
        "content_type": headers.get("content-type", "").split(";", 1)[0],
        "body_length": len(body),
        "body_sha256": body_hash,
        "marker_reflected": position >= 0,
        "echo_excerpt": excerpt,
        "header_names": sorted(headers),
    }
    return PG266._wire(origin, route, values), projection


def _route_set() -> list[dict[str, Any]]:
    selected = [dict(route) for route in PG266.ROUTES if str(route["id"]) in ROUTE_IDS]
    if {str(route["id"]) for route in selected} != set(ROUTE_IDS):
        raise RuntimeError("PG-314 route selection is incomplete")
    if {str(route["method"]).upper() for route in selected} != {"GET", "POST"}:
        raise RuntimeError("PG-314 must cover both GET and POST")
    return selected


def _extended_proposal(model: Any, vocabulary: Mapping[str, int], device: torch.device, tokens: Sequence[str]) -> dict[str, Any]:
    """Decode an extended target, bind bounded refs, then apply the base guard."""

    raw = generate_target(model, tokens, TARGET_LENGTH, vocabulary, device)
    bound = bind_probe_variant_plan(raw, tokens)
    base = bound or []
    guarded_base = compose_guarded_plan(base or raw, tokens)
    base_values = target_map(guarded_base)
    bound_values = target_map(bound or [])
    variant = bound_values.get("probe_variant", "none") if base_values.get("safe_to_send") == "1" else "none"
    encoding = bound_values.get("encoding_chain", "none") if variant != "none" else "none"
    if variant not in VARIANT_TO_CATALOG or encoding not in {"url_percent", "form_urlencoded", "json_string", "base64_marker", "identity", "none"}:
        variant = "none"
        encoding = "none"
    guarded = [*guarded_base[:-1], f"probe_variant={variant}", f"encoding_chain={encoding}", TARGET_EOS]
    return {
        "raw_tokens": list(raw),
        "bound_tokens": list(bound or []),
        "guarded_tokens": guarded,
        "raw_fields": target_map(raw),
        "bound_fields": bound_values,
        "guarded_fields": target_map(guarded),
        "expected_variant": ROLE_TO_VARIANT.get(next((str(token).split("=", 1)[1] for token in tokens if str(token).startswith("history_action=")), ""), "none"),
        "model_variant": variant,
        "model_safe_to_send": base_values.get("safe_to_send") == "1" and base_values.get("next_action") == "assemble_abstract_plan",
        "variant_exact": variant == ROLE_TO_VARIANT.get(next((str(token).split("=", 1)[1] for token in tokens if str(token).startswith("history_action=")), ""), "none"),
    }


def _preflight(model: Any, vocabulary: Mapping[str, int], device: torch.device, routes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for route in routes:
        complete = {
            "typed_available": "1",
            "replay_ready": "1",
            "evidence_present": "1",
            "feedback_state": "observable_progress",
            "negative_control": "1",
            "fresh_reset": "1",
        }
        for missing in ("typed_available", "replay_ready", "evidence_present", "feedback_state", "negative_control", "fresh_reset"):
            values = dict(complete)
            values[missing] = "unknown"
            tokens = context_tokens(str(route["method"]), history_action="none", failure_class="none", **values)
            expected = probe_target_for_context(tokens)
            actual = _extended_proposal(model, vocabulary, device, tokens)
            expected_values = target_map(expected)
            actual_values = target_map(actual["guarded_tokens"])
            rows.append({
                "route_id": str(route["id"]),
                "missing": missing,
                "question_expected": expected_values.get("question", "none"),
                "question_actual": actual_values.get("question", "none"),
                "question_correct": expected_values.get("question") == actual_values.get("question"),
                "safe_expected": expected_values.get("safe_to_send") == "1",
                "safe_actual": actual_values.get("safe_to_send") == "1",
            })
    count = max(len(rows), 1)
    return {
        "count": len(rows),
        "question_recall": round(sum(int(row["question_correct"]) for row in rows) / count, 6),
        "unsafe_allow": sum(int(not row["safe_expected"] and row["safe_actual"]) for row in rows),
        "rows": rows,
    }


def _role_context(method: str, role: str) -> list[str]:
    return context_tokens(
        method,
        typed_available="1",
        replay_ready="1",
        evidence_present="1",
        feedback_state="observable_progress",
        negative_control="1",
        fresh_reset="1",
        history_action=role,
        failure_class="none",
    )


def _abstract_record(route: Mapping[str, Any], role: str, proposal: Mapping[str, Any], typed: Mapping[str, Any], reset: Mapping[str, Any], source_hash: str) -> dict[str, Any]:
    context = _role_context(str(route["method"]), role)
    target = probe_target_for_context(context)
    return {
        "schema_version": "pg314-independent-variant-training-record-v1",
        "record_id": hashlib.sha256((str(route["id"]) + role + str(reset.get("reset_id"))).encode()).hexdigest()[:24],
        "context_tokens": context,
        "target_tokens": target,
        "predicted_tokens": list(proposal.get("guarded_tokens") or []),
        "role_token": role,
        "predicted_variant_token": str(proposal.get("model_variant", "none")),
        "variant_exact": bool(proposal.get("variant_exact")),
        "safe_to_send": bool(proposal.get("model_safe_to_send")),
        "outcome_class": "typed_effect" if typed.get("typed_effect_confirmed") else "abstain_or_repair",
        "source_sha256": source_hash,
        "evidence_sha256": str(typed.get("evidence_sha256", "")),
        "fresh_reset": bool(reset.get("fresh_target")),
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_target_off_input": True,
    }


def _run_route(
    route: Mapping[str, Any],
    index: int,
    model: Any,
    vocabulary: Mapping[str, int],
    device: torch.device,
    browser: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    name = ""
    marker = f"PG314-{index:02d}"
    try:
        name, port, container_id, reset = _start(index)
        origin = f"http://127.0.0.1:{port}"
        source_hash = _source_hash(name, route)
        if len(source_hash) != 64:
            raise RuntimeError(f"PG-314 source attestation missing for {route['id']}")
        client = httpx.Client(base_url=origin, timeout=15.0, follow_redirects=False)
        try:
            model_entries: list[dict[str, Any]] = []
            abstract_rows: list[dict[str, Any]] = []
            by_actual: dict[str, dict[str, Any]] = {}
            for role in VARIANT_ROLES:
                tokens = _role_context(str(route["method"]), role)
                proposal = _extended_proposal(model, vocabulary, device, tokens)
                entry: dict[str, Any] = {
                    "role": role,
                    "expected_variant": ROLE_TO_VARIANT[role],
                    "proposal": proposal,
                    "sent": False,
                    "variant": str(proposal.get("model_variant", "none")),
                }
                actual_variant = str(proposal.get("model_variant", "none"))
                if proposal.get("model_safe_to_send") and actual_variant in VARIANT_TO_CATALOG:
                    catalog_variant = VARIANT_TO_CATALOG[actual_variant]
                    values = PG266._candidate_values(route, marker + "-" + role, catalog_variant)
                    wire, projection = _send_internal(name, origin, route, values, marker + "-" + role)
                    browser_effect = {"available": False, "executed": False}
                    if route.get("family") == "xss":
                        browser_effect = PG266._browser_oracle(browser, origin, route, values, marker + "-" + role)
                    replay_wire, replay_projection = _send_internal(name, origin, route, values, marker + "-" + role)
                    replay_browser = {"available": False, "executed": False}
                    if route.get("family") == "xss":
                        replay_browser = PG266._browser_oracle(browser, origin, route, values, marker + "-" + role)
                    entry.update({
                        "sent": True,
                        "catalog_variant": catalog_variant,
                        "values": values,
                        "wire": wire,
                        "projection": projection,
                        "browser_oracle": browser_effect,
                        "replay_wire": replay_wire,
                        "replay_projection": replay_projection,
                        "replay_browser_oracle": replay_browser,
                    })
                    by_actual.setdefault(actual_variant, entry)
                model_entries.append(entry)

            model_candidate = by_actual.get("source_attested_candidate")
            model_reference = by_actual.get("reference_canary")
            model_negative = by_actual.get("negative_control")
            if route.get("family") == "sql":
                candidate_projection = (model_candidate or {}).get("projection") or {}
                reference_projection = (model_reference or {}).get("projection") or {}
                negative_projection = (model_negative or {}).get("projection") or {}
                candidate_positive, oracle_reason = PG266._sql_positive(route, candidate_projection, reference_projection, negative_projection)
                reference_positive = bool(model_reference and reference_projection)
                negative_clean = bool(model_negative and not ("your uid:" in str(negative_projection.get("echo_excerpt", "")) or "hello," in str(negative_projection.get("echo_excerpt", "")) or "username：" in str(negative_projection.get("echo_excerpt", ""))))
                replay_consistent = bool(model_candidate and model_candidate.get("projection", {}).get("body_length") == model_candidate.get("replay_projection", {}).get("body_length"))
                effect_type = "result_shape"
                candidate_abstract = abstract_projection(candidate_projection, effect_marker="row_shape" if candidate_positive else "none", backend_observed=True)
                reference_abstract = abstract_projection(reference_projection, effect_marker="row_shape" if reference_positive else "none", backend_observed=True)
                negative_abstract = abstract_projection(negative_projection, effect_marker="none", backend_observed=True)
                replay_abstract = abstract_projection((model_candidate or {}).get("replay_projection") or {}, effect_marker="row_shape" if replay_consistent and candidate_positive else "none", backend_observed=True)
            else:
                candidate_browser = (model_candidate or {}).get("browser_oracle") or {}
                reference_browser = (model_reference or {}).get("browser_oracle") or {}
                negative_browser = (model_negative or {}).get("browser_oracle") or {}
                candidate_positive = bool(candidate_browser.get("executed"))
                reference_positive = bool(reference_browser.get("executed"))
                negative_clean = bool(model_negative and not negative_browser.get("executed"))
                replay_browser = (model_candidate or {}).get("replay_browser_oracle") or {}
                replay_consistent = bool(candidate_positive == bool(replay_browser.get("executed")))
                oracle_reason = "candidate_reference_dom_execution_agreement" if candidate_positive and reference_positive and negative_clean else "dom_execution_not_confirmed"
                effect_type = "dom_effect"
                candidate_abstract = abstract_projection((model_candidate or {}).get("projection") or {}, effect_marker="dom_marker" if candidate_positive else "none", backend_observed=True)
                reference_abstract = abstract_projection((model_reference or {}).get("projection") or {}, effect_marker="dom_marker" if reference_positive else "none", backend_observed=True)
                negative_abstract = abstract_projection((model_negative or {}).get("projection") or {}, effect_marker="none", backend_observed=True)
                replay_abstract = abstract_projection((model_candidate or {}).get("replay_projection") or {}, effect_marker="dom_marker" if replay_consistent and candidate_positive else "none", backend_observed=True)
            all_variant_exact = all(bool(entry["proposal"].get("variant_exact")) for entry in model_entries)
            typed_positive = bool(candidate_positive and reference_positive and negative_clean and replay_consistent and all_variant_exact)
            evidence = typed_evidence(
                effect_type=effect_type,
                typed_effect_confirmed=typed_positive,
                negative_control_clean=negative_clean,
                reference_agreement=bool(candidate_positive == reference_positive),
                replay_consistent=replay_consistent,
                evaluator_id=f"pg314-independent-{route['id']}",
            )
            surface = {
                "surface_id": f"pg314-{route['id']}",
                "method": str(route["method"]).upper(),
                "path": str(route["path"]),
                "channel": "query" if str(route["method"]).upper() == "GET" else "form",
                "field_count": len(route.get("fields") or []),
                "evaluator_kind": effect_type,
                "implementation": "sift_pikachu_pg240_source_native",
            }
            typed = evaluator_result(
                surface=surface,
                reset=_evaluator_reset(reset),
                reference=reference_abstract,
                negative=negative_abstract,
                candidate=candidate_abstract,
                replay=replay_abstract,
                evidence=evidence,
                source_attestation=source_hash,
                hard_negative=False,
            )
            for entry in model_entries:
                abstract_rows.append(_abstract_record(route, str(entry["role"]), entry["proposal"], typed, reset, source_hash))
            canonical = {
                "candidate": PG266._candidate_values(route, marker + "-GOLD", "candidate"),
                "reference": PG266._candidate_values(route, marker + "-GOLD-REF", "reference"),
                "negative": PG266._candidate_values(route, marker + "-GOLD-NEG", "negative"),
            }
            human = {
                "record_id": f"pg314:{route['id']}:{index}",
                "route": dict(route),
                "target": {"origin": "<LOOPBACK_ORIGIN>", "fresh_reset": reset, "image": INDEPENDENT_IMAGE, "source_sha256": source_hash, "container_id_sha256": hashlib.sha256(container_id.encode()).hexdigest()},
                "model": {"checkpoint": str(CHECKPOINT.relative_to(ROOT)), "entries": model_entries, "variant_roles": dict(ROLE_TO_VARIANT)},
                "evaluator_gold": {"wire": canonical, "typed": typed},
                "oracle": {"candidate_positive": candidate_positive, "reference_positive": reference_positive, "negative_clean": negative_clean, "replay_consistent": replay_consistent, "all_variant_exact": all_variant_exact, "typed_effect_confirmed": bool(typed.get("typed_effect_confirmed")), "reason": oracle_reason, "evidence_sha256": evidence.get("evidence_sha256")},
                "raw_payload_human_review_only": True,
                "raw_response_bodies_stored": False,
            }
            trace = {
                "record_id": human["record_id"],
                "method": str(route["method"]).upper(),
                "variant_roles": [entry["role"] for entry in model_entries],
                "model_variant_exact": all_variant_exact,
                "model_send_count": sum(int(entry["sent"]) for entry in model_entries),
                "typed_effect_confirmed": bool(typed.get("typed_effect_confirmed")),
                "training_eligible": False,
                "memory_promotion_allowed": False,
                "vulnerability_claim_allowed": False,
                "raw_payload_stored": False,
                "raw_response_body_stored": False,
            }
            return human, abstract_rows, trace
        finally:
            client.close()
    finally:
        if name:
            _stop(name)


def main() -> int:
    _require_gate()
    routes = _route_set()
    device = torch.device("cpu")
    model, vocabulary, symbolic = load_causal_checkpoint(CHECKPOINT, device)
    if not symbolic or len(vocabulary) < 70:
        raise RuntimeError("PG-314 requires the PG-313 symbolic checkpoint")
    preflight = _preflight(model, vocabulary, device, routes)
    browser_context = PG266.sync_playwright().start() if PG266.sync_playwright is not None else None
    browser = browser_context.chromium.launch(headless=True) if browser_context is not None else None
    human_rows: list[dict[str, Any]] = []
    abstract_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for index, route in enumerate(routes):
            human, abstract, trace = _run_route(route, index, model, vocabulary, device, browser)
            human_rows.append(human)
            abstract_rows.extend(abstract)
            traces.append(trace)
    finally:
        if browser is not None:
            browser.close()
        if browser_context is not None:
            browser_context.stop()
    elapsed = round(time.monotonic() - started, 3)
    role_entries = [entry for row in human_rows for entry in row["model"]["entries"]]
    variant_counts = {variant: sum(int(entry.get("variant") == variant) for entry in role_entries) for variant in ("source_attested_candidate", "reference_canary", "negative_control", "none")}
    role_exact = sum(int(bool(entry["proposal"].get("variant_exact"))) for entry in role_entries)
    model_send_count = sum(int(entry.get("sent")) for entry in role_entries)
    typed_count = sum(int(bool(row["oracle"].get("typed_effect_confirmed"))) for row in human_rows)
    negative_lane_violations = sum(int(row["oracle"].get("all_variant_exact") is False and any(entry["role"] == "negative_control" and entry.get("variant") != "negative_control" for entry in row["model"]["entries"])) for row in human_rows)
    report = {
        "protocol_id": "pg-pk-314-independent-variant-replay-v1",
        "schema_version": "pg314-independent-variant-replay-report-v1",
        "status": "completed_real_local_docker_independent_variant_replay",
        "runtime": {
            "execution_window": "Asia/Shanghai 08:00-18:00",
            "explicit_flag": "PG314_LOCAL_DOCKER_EVAL=1",
            "device": "cpu_inference_only",
            "image": INDEPENDENT_IMAGE,
            "network": "none",
            "route_ids": list(ROUTE_IDS),
        },
        "model": {
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "architecture": "causal_transformer_moe_next_token",
            "symbolic_checkpoint": symbolic,
            "target_length": TARGET_LENGTH,
            "target_representation": "symbolic_slot_copy_plus_probe_variant_ref",
            "wire_generation": "source_grounded_binding_after_model_variant_guard",
            "oracle_target_in_context": False,
            "raw_payload_in_context": False,
            "raw_response_body_in_context": False,
        },
        "preflight_identifiability": preflight,
        "counts": {
            "route_count": len(human_rows),
            "get_count": sum(int(str(row["route"]["method"]).upper() == "GET") for row in human_rows),
            "post_count": sum(int(str(row["route"]["method"]).upper() == "POST") for row in human_rows),
            "model_variant_role_count": len(role_entries),
            "model_variant_exact_count": role_exact,
            "model_candidate_send_count": sum(int(row["oracle"].get("all_variant_exact") and any(entry.get("variant") == "source_attested_candidate" and entry.get("role") == "candidate_request" and entry.get("sent") for entry in row["model"]["entries"])) for row in human_rows),
            "model_variant_send_count": model_send_count,
            "model_typed_effect_count": typed_count,
            "evaluator_gold_typed_effect_count": typed_count,
            "variant_source_attested_candidate_count": variant_counts["source_attested_candidate"],
            "variant_reference_canary_count": variant_counts["reference_canary"],
            "variant_negative_control_count": variant_counts["negative_control"],
            "model_abstain_count": len(role_entries) - model_send_count,
            "variant_misselection_count": len(role_entries) - role_exact,
            "negative_lane_violation_count": negative_lane_violations,
            "false_positive_count": negative_lane_violations,
            "fresh_reset_count": len(human_rows),
            "negative_control_count": len(human_rows),
            "typed_evidence_hash_count": sum(int(bool(row["oracle"].get("evidence_sha256"))) for row in human_rows),
            "elapsed_seconds": elapsed,
        },
        "checks": {
            "real_docker_contacted": True,
            "independent_image_digest": INDEPENDENT_IMAGE,
            "loopback_only": True,
            "docker_network_none": True,
            "external_network_disabled": True,
            "get_post_pair": any(str(row["route"]["method"]).upper() == "GET" for row in human_rows) and any(str(row["route"]["method"]).upper() == "POST" for row in human_rows),
            "fresh_reset_per_route": all(bool(row["target"]["fresh_reset"].get("fresh_target")) for row in human_rows),
            "zero_volume_per_route": all(int(row["target"]["fresh_reset"].get("volume_mount_count", -1)) == 0 for row in human_rows),
            "database_health_per_route": all(row["target"]["fresh_reset"].get("database_health_gate") == "mysqli_root_pikachu_ok" for row in human_rows),
            "source_attestation_per_route": all(len(str(row["target"].get("source_sha256", ""))) == 64 for row in human_rows),
            "typed_evidence_hash_per_route": all(bool(row["oracle"].get("evidence_sha256")) for row in human_rows),
            "negative_controls_present": all(any(entry["role"] == "negative_control" for entry in row["model"]["entries"]) for row in human_rows),
            "raw_payload_in_model_context": False,
            "raw_response_body_stored": False,
            "public_target_contacted": False,
            "sql_time_delay": False,
            "sql_write": False,
        },
        "hypothesis_gate": {
            "status": "blocked",
            "checks": {
                "audit_and_runtime_contract": True,
                "preflight_question_recall_min": preflight["question_recall"] >= 0.9,
                "preflight_zero_unsafe_allow": preflight["unsafe_allow"] == 0,
                "variant_selection_exact": role_exact == len(role_entries),
                "negative_lane_zero_violation": negative_lane_violations == 0,
                "typed_effect_on_all_routes": typed_count == len(human_rows),
                "fresh_get_post_pair": True,
                "promotion_blocked": True,
            },
            "claim_allowed": False,
        },
        "scientific_gate": {
            "status": "blocked",
            "reasons": [
                "PG-313 worst-seed offline gate was blocked before this replay",
                "model output remains abstract and source-grounded adapter output is not literal payload generation",
                "one independent image and four routes are insufficient for capability graduation",
                "training/memory/payload catalog promotion remains disabled",
            ],
            "claim_allowed": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "checkpoint_role": "research_candidate_only",
        },
        "artifacts": {
            "human_catalog": str(CATALOG.relative_to(ROOT)),
            "training_dataset": str(DATASET.relative_to(ROOT)),
            "trace": str(TRACE.relative_to(ROOT)),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
        },
    }
    report["report_sha256"] = _digest(report)
    catalog = {
        "schema_version": "pg314-independent-variant-human-catalog-v1",
        "status": "completed_real_local_independent_human_review_catalog",
        "implementation": INDEPENDENT_IMAGE,
        "entries": human_rows,
        "raw_payloads_human_review_only": True,
        "raw_response_bodies_stored": False,
        "external_network": False,
    }
    catalog["catalog_sha256"] = _digest(catalog)
    dataset = {
        "schema_version": "pg314-independent-variant-training-dataset-v1",
        "source_catalog": str(CATALOG.relative_to(ROOT)),
        "records": abstract_rows,
        "counts": {
            "records": len(abstract_rows),
            "route_count": len(human_rows),
            "variant_role_records": len(role_entries),
            "typed_effect_records": typed_count * len(VARIANT_ROLES),
            "variant_exact_records": role_exact,
        },
        "contract": {
            "process_question_supervision": True,
            "abstract_probe_variant_supervision": True,
            "real_get_post_replay": True,
            "independent_implementation": True,
            "fresh_reset_required": True,
            "typed_oracle_required": True,
            "evidence_hash_required": True,
            "payload_strings_excluded": True,
            "response_bodies_excluded": True,
            "training_promotion_allowed": False,
            "memory_promotion_allowed": False,
        },
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    trace = {
        "schema_version": "pg314-independent-variant-trace-v1",
        "episodes": traces,
        "raw_payloads_human_catalog_only": True,
        "raw_response_bodies_stored": False,
        "online_weight_update": False,
        "long_term_memory_write": False,
    }
    trace["trace_sha256"] = _digest(trace)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg314-independent-variant-protocol-v1",
        "scope": {"target": "authorized local Docker independent Pikachu implementation", "image": INDEPENDENT_IMAGE, "loopback_only": True, "docker_network": "none", "external_network": False, "state_change_allowed": False, "methods": ["GET", "POST"]},
        "model_contract": {"next_token_decoder": True, "abstract_slots_only": True, "model_selects_variant": True, "source_grounded_binding_after_guard": True, "oracle_target_off_input": True},
        "required_gates": {"missing_question_before_assembly": True, "get_post_pair": True, "matched_negative": True, "fresh_reset": True, "typed_oracle": True, "evidence_hash": True, "independent_image": True, "raw_payload_training_excluded": True},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False},
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(CATALOG, catalog)
    _write(DATASET, dataset)
    _write(TRACE, trace)
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join([
        "# PG-314 独立实现 probe variant 复放",
        "",
        f"routes={len(human_rows)} GET={report['counts']['get_count']} POST={report['counts']['post_count']}; variant exact={role_exact}/{len(role_entries)}; model sends={model_send_count}",
        f"typed model effects={typed_count}/{len(human_rows)}; negative lane violations={negative_lane_violations}; preflight question recall={preflight['question_recall']}",
        "模型只输出抽象 Rule-IR；wire/原始值仅存人审 catalog。typed effect 不等于漏洞声明；promotion 关闭。",
        "",
    ]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "preflight": {key: preflight[key] for key in ("count", "question_recall", "unsafe_allow")}, "gate": report["hypothesis_gate"], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
