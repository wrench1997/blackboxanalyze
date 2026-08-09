"""Run the PG-28 fresh-container GET/POST dual-channel pilot.

This adapter performs only local, non-executing marker probes against the
authorized VulnerableApp container.  Response bodies and markers exist only
in memory long enough to derive bounded hashes and allow-listed regex
evidence.  The resulting Catalog is evaluation-only; no training writer is
called and the hard gate is recorded even when it remains preflight-only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import ReadOnlySafeCatalogCollector, build_catalog, sha256_json
from app.hardcore_experiment_gate import evaluate_hardcore_catalog
from app.pg25d_acceptance_gate import evaluate_catalog as evaluate_payload_acceptance
from app.regex_evidence_oracle import evaluate_allowlisted_regex
from app.safe_marker import fresh_marker, marker_sha256


RESET_SCRIPT = ROOT / "scripts" / "reset_pg25d_vulnerableapp.ps1"
REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
OUT_CATALOG = ROOT / "research" / "pg_pk_28_get_post_dual_catalog_v1.json"
OUT_ACCEPTANCE = ROOT / "research" / "pg_pk_28_get_post_dual_acceptance_v1.json"
OUT_GATE = ROOT / "research" / "pg_pk_28_get_post_dual_hardcore_gate_v1.json"
BASE_URI = "http://127.0.0.1:19090/VulnerableApp"
ROUTE = "/XSSWithHtmlTagInjection/LEVEL_1"
TARGET_ID = "owasp_vulnerableapp_2_1_44"
SOURCE_ID = "sasanlabs-vulnerableapp-2.1.44-amd64-pg28"
IMAGE_DIGEST = "sha256:7bc084dac341f089c6e788d2369a27f599c902d742c5e113d7bb50661cd92406"
COLLECTOR_SHA256 = hashlib.sha256(b"pg28-get-post-dual-collector-v1").hexdigest()
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg28-xss-safe-reflection-oracle-v1").hexdigest()


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_type_class(value: str) -> str:
    value = str(value or "").casefold()
    if value in {"html", "json", "text", "xml"}:
        return value
    if value == "text/html":
        return "html"
    if value == "application/json":
        return "json"
    if value.startswith("text/"):
        return "text"
    if value.endswith("+xml") or value == "application/xml":
        return "xml"
    return "other"


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


def _request_projection(*, method: str, marker: str, sent_marker: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make one bounded request and immediately discard raw response data."""

    method = method.upper()
    if method == "GET":
        url = f"{BASE_URI}{ROUTE}?{urllib.parse.urlencode({'probe': marker})}" if sent_marker else f"{BASE_URI}{ROUTE}"
        request = urllib.request.Request(url, method="GET", headers={"Accept": "text/plain"})
    elif method == "POST":
        url = f"{BASE_URI}{ROUTE}"
        form = urllib.parse.urlencode({"probe": marker} if sent_marker else {}).encode("ascii")
        request = urllib.request.Request(
            url,
            data=form,
            method="POST",
            headers={"Accept": "text/plain", "Content-Type": "application/x-www-form-urlencoded"},
        )
    else:  # pragma: no cover - the adapter only calls the two declared channels
        raise ValueError("PG-28 adapter only supports GET and POST")

    status = 0
    header_names: set[str] = set()
    content_type = "unknown"
    body = b""
    transport_error = False
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = int(response.status)
            header_names = {str(key).casefold() for key in response.headers.keys()}
            content_type = _content_type_class(response.headers.get_content_type())
            body = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        header_names = {str(key).casefold() for key in exc.headers.keys()}
        content_type = _content_type_class(exc.headers.get_content_type())
        body = exc.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        transport_error = True

    body_hash = _digest_bytes(body)
    body_text = body.decode("utf-8", errors="replace")
    regex = evaluate_allowlisted_regex(
        text=body_text,
        pattern_id="escaped_marker_reflection",
        marker=marker,
    )
    reflected = bool(sent_marker and regex.get("matched"))
    marker_location = "html_text" if reflected else "none"
    response = {
        "status_code": status,
        "status_class": "transport_error" if transport_error else _status_class(status),
        "content_type_class": content_type,
        "body_length_bucket": _length_bucket(len(body)),
        "body_sha256": body_hash,
        "semantic_body_sha256": body_hash,
        "shape": {"kind": content_type, "field_count": 0, "scalar_count": 0},
        "header_names": sorted(header_names & {"content-type", "location", "allow"}),
        "marker": {"reflected": reflected, "location": marker_location, "count": int(regex.get("match_count", 0)) if reflected else 0},
        "frame_policy": "unknown",
        "transport_error": transport_error,
        "status_changed": False,
        "state_changed": False,
        "location_origin_changed": False,
    }
    return response, regex


def _payload_manifest(*, method: str, marker: str, seed: int, role: str) -> dict[str, Any]:
    method = method.upper()
    placement = "query" if method == "GET" else "form"
    descriptor = {
        "route_template_id": "vulnerableapp-xss-html-level1",
        "method": method,
        "placement": placement,
        "marker_sha256": marker_sha256(marker),
        "seed": int(seed),
        "role": role,
    }
    manifest = {
        "manifest_id": f"pg28-{method.casefold()}-{role}-{seed}",
        "payload_sha256": sha256_json(descriptor),
        "probe_ref": "inert-marker" if role == "candidate" else "marker-absent-baseline",
        "probe_kind": "http_canary",
        "route_template_id": "vulnerableapp-xss-html-level1",
        "method": method,
        "placement": placement,
        "encoding_chain": ["identity"],
        "encoding_depth": 0,
        "marker_sha256": marker_sha256(marker),
        "max_bytes": 96,
        "safety": {
            "does_not_execute": True,
            "no_external_network": True,
            "no_script_execution": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }
    if method == "POST":
        manifest["form_field_names"] = ["probe"]
        manifest["form_content_type"] = "application/x-www-form-urlencoded"
    return manifest


def _oracle(*, marker: str, regex: dict[str, Any], reflected: bool, sample_role: str) -> dict[str, Any]:
    return {
        "oracle_id": "pg28-xss-reflection-only-eval-v1",
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "family": "xss",
        "modality": "negative_control" if sample_role == "negative_control" else "reflection",
        "candidate_signal": bool(reflected),
        "positive": False,
        "positive_authority": False,
        "confirmed_effect": "none",
        "signals": {
            "marker_reflected": bool(reflected),
            "regex_evidence": regex,
            "typed_dom_effect_observed": False,
            "execution_probe_sent": False,
        },
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
            "raw_body_stored": False,
        },
    }


def _rule_ir(method: str) -> dict[str, Any]:
    return {
        "rule_key": f"xss.reflection.{method.casefold()}.surface-v1",
        "grammar_version": "rule-ir-v1",
        "family_candidate": "xss",
        "operator_set": ["and", "eq", "present"],
        "required_slots": ["surface", "transport", "oracle"],
        "bound_slots": ["surface", "transport", "oracle"],
        "executable": False,
    }


def _reset(source: dict[str, Any]) -> tuple[dict[str, Any], str]:
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RESET_SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    container_id = subprocess.check_output(
        ["docker", "inspect", "pg25-vulnerableapp", "--format", "{{.Id}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    short = container_id[:12]
    return {
        "reset_id": f"pg28-reset-{short}",
        "kind": "container_recreate",
        "target_instance_id": f"pg28-instance-{short}",
        "state_epoch": f"pg28-epoch-{short}",
        "reset_adapter_sha256": source["reset_adapter_sha256"],
        "baseline_projection_sha256": "0" * 64,
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
    }, short


def _source() -> dict[str, Any]:
    return {
        "target_id": TARGET_ID,
        "app_family": "owasp_vulnerableapp",
        "source_id": SOURCE_ID,
        "source_type": "authorized_local_container",
        "origin_ref": "pg_pk_28_get_post_dual_protocol_v1",
        "license": "local-container",
        "authorization": "workspace_local_only",
        "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": 19090},
        "container_image_digest": IMAGE_DIGEST,
        "collector_sha256": COLLECTOR_SHA256,
        "reset_adapter_sha256": hashlib.sha256(RESET_SCRIPT.read_bytes()).hexdigest(),
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "read_only": True,
        "external_network": False,
    }


def main() -> int:
    source = _source()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector = ReadOnlySafeCatalogCollector(source, registry=registry)
    records: list[dict[str, Any]] = []

    # Three independently recreated target instances and three seeds satisfy
    # the method/independence preflight.  The absence of an execution oracle
    # is intentional: it must keep this round out of training.
    for seed in (101, 202, 303):
        reset, short = _reset(collector.source)
        markers = {method: fresh_marker(f"PG28{method}") for method in ("GET", "POST")}
        baseline: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for method in ("GET", "POST"):
            baseline[method] = _request_projection(method=method, marker=markers[method], sent_marker=False)
        reset["baseline_projection_sha256"] = sha256_json({method: projection for method, (projection, _) in baseline.items()})

        for method in ("GET", "POST"):
            marker = markers[method]
            control_response, control_regex = baseline[method]
            candidate_response, candidate_regex = _request_projection(method=method, marker=marker, sent_marker=True)
            control_id = f"pg28-{seed}-{short}-{method.casefold()}-control"
            candidate_id = f"pg28-{seed}-{short}-{method.casefold()}-candidate"
            control = collector.collect(
                sample_id=control_id,
                sample_role="negative_control",
                sampling_seed=seed,
                reset=reset,
                payload_manifest=_payload_manifest(method=method, marker=marker, seed=seed, role="control"),
                response_projection=control_response,
                oracle_projection=_oracle(marker=marker, regex=control_regex, reflected=False, sample_role="negative_control"),
                rule_ir=_rule_ir(method),
            )
            candidate = collector.collect(
                sample_id=candidate_id,
                sample_role="candidate",
                sampling_seed=seed,
                reset=reset,
                payload_manifest=_payload_manifest(method=method, marker=marker, seed=seed, role="candidate"),
                response_projection=candidate_response,
                oracle_projection=_oracle(
                    marker=marker,
                    regex=candidate_regex,
                    reflected=bool(candidate_response["marker"]["reflected"]),
                    sample_role="candidate",
                ),
                rule_ir=_rule_ir(method),
                negative_control={
                    "control_sample_id": control["sample_id"],
                    "control_evidence_hash": control["evidence"]["evidence_hash"],
                    "intervention": "marker-present-vs-absent",
                    "verdict": "confirmed_negative",
                    "same_source": True,
                    "same_surface": True,
                },
            )
            records.extend([control, candidate])
            print(
                f"seed={seed} target={short} method={method} "
                f"baseline_status={control_response['status_code']} "
                f"probe_status={candidate_response['status_code']} "
                f"reflected={candidate_response['marker']['reflected']}"
            )

    catalog = build_catalog("pg28-get-post-dual-catalog-v1", collector.source, records)
    acceptance = evaluate_payload_acceptance(catalog)
    hardcore = evaluate_hardcore_catalog(catalog, family="xss")
    now = datetime.now(timezone.utc).isoformat()
    catalog["protocol"] = "pg_pk_28_get_post_dual_protocol_v1"
    catalog["collected_at"] = now
    catalog["training_artifact_generated"] = False
    catalog["training_eligible"] = False
    catalog["hardcore_gate_status"] = hardcore["status"]
    catalog.pop("catalog_sha256", None)
    catalog["catalog_sha256"] = sha256_json(catalog)
    report = {
        "schema_version": "pg-pk-28-get-post-dual-run-v1",
        "protocol": "pg_pk_28_get_post_dual_protocol_v1",
        "catalog_id": catalog["catalog_id"],
        "collected_at": now,
        "fresh_reset_count": 3,
        "methods": ["GET", "POST"],
        "training_artifact_generated": False,
        "training_eligible": False,
        "acceptance": acceptance,
        "hardcore_gate": hardcore,
        "raw_bodies_retained": False,
        "raw_markers_retained": False,
    }
    OUT_CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_ACCEPTANCE.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_GATE.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"catalog": str(OUT_CATALOG), "acceptance": acceptance, "hardcore_gate": hardcore}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
