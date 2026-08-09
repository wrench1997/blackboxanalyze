"""Collect bounded, non-mutating PG-25D observations from the local lab.

This runner deliberately uses only GET/HEAD/OPTIONS-style canaries.  It never
stores request markers or response bodies: the emitted artifact contains only
hashes, length buckets, typed headers/status, and the bounded oracle projection.
Every row recreates the pinned container before taking a baseline and probe.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
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
RESET_SCRIPT = ROOT / "scripts" / "reset_pg25d_vulnerableapp.ps1"
REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
OUT_PATH = ROOT / "research" / "pg_pk_25d_vulnerableapp_catalog_v1.json"
BASE_URI = "http://127.0.0.1:19090/VulnerableApp/"
IMAGE_DIGEST = "sha256:7bc084dac341f089c6e788d2369a27f599c902d742c5e113d7bb50661cd92406"
COLLECTOR_SHA256 = "c0d6ce34fa26b55d3d4cba4f1d2f5c604f2dff0ca405f655a275a9aee85b0468"
RESET_ADAPTER_SHA256 = hashlib.sha256(RESET_SCRIPT.read_bytes()).hexdigest()
ORACLE_CONTRACT_SHA256 = "3f13c399cebea2db7d40163529617814f82c44b480e791a0953dcbe4484b69f6"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bucket(length: int) -> str:
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


def _request(url: str, *, method: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read() if method != "HEAD" else b""
            status = int(response.status)
            response_headers = {str(k).casefold() for k in response.headers.keys()}
            content_type = response.headers.get_content_type() if response.headers else "unknown"
            marker = str((headers or {}).get("X-PG25-Canary", ""))
            reflected = bool(marker and marker.encode("utf-8") in body)
            return {
                "status_code": status,
                "status_class": _status_class(status),
                "content_type_class": content_type if content_type in {"html", "json", "text", "xml"} else "other",
                "body_length_bucket": _bucket(len(body)),
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "semantic_body_sha256": hashlib.sha256(body).hexdigest(),
                "shape": {"kind": content_type, "field_count": 0, "scalar_count": 0},
                "header_names": sorted(response_headers & {"content-type", "location", "allow", "cache-control"}),
                "marker": {"reflected": reflected, "location": "html_text" if reflected else "none", "count": 1 if reflected else 0},
                "frame_policy": "unknown",
                "transport_error": False,
                "status_changed": False,
                "state_changed": False,
                "location_origin_changed": False,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read() if method != "HEAD" else b""
        return {
            "status_code": int(exc.code),
            "status_class": _status_class(int(exc.code)),
            "content_type_class": "other",
            "body_length_bucket": _bucket(len(body)),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "semantic_body_sha256": hashlib.sha256(body).hexdigest(),
            "shape": {"kind": "http_error", "field_count": 0, "scalar_count": 0},
            "header_names": [],
            "marker": {"reflected": False, "location": "none", "count": 0},
            "frame_policy": "unknown",
            "transport_error": False,
            "status_changed": False,
            "state_changed": False,
            "location_origin_changed": False,
        }
    except (OSError, urllib.error.URLError):
        zero = hashlib.sha256(b"").hexdigest()
        return {
            "status_code": 0,
            "status_class": "transport_error",
            "content_type_class": "unknown",
            "body_length_bucket": "0",
            "body_sha256": zero,
            "semantic_body_sha256": zero,
            "shape": {"kind": "transport_error", "field_count": 0, "scalar_count": 0},
            "header_names": [],
            "marker": {"reflected": False, "location": "none", "count": 0},
            "frame_policy": "unknown",
            "transport_error": True,
            "status_changed": False,
            "state_changed": False,
            "location_origin_changed": False,
        }


def _reset() -> tuple[str, str]:
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
    return f"reset-{short}", f"instance-{short}"


def _source(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_id": "owasp_vulnerableapp_2_1_44",
        "app_family": "owasp_vulnerableapp",
        "source_id": "sasanlabs-vulnerableapp-2.1.44-amd64",
        "source_type": "authorized_local_container",
        "origin_ref": "pg_pk_25d_vulnerableapp_deployment_v1",
        "license": "local-container",
        "authorization": "workspace_local_only",
        "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": 19090},
        "container_image_digest": IMAGE_DIGEST,
        "collector_sha256": COLLECTOR_SHA256,
        "reset_adapter_sha256": RESET_ADAPTER_SHA256,
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "read_only": True,
        "external_network": False,
    }


def _manifest(label: str, *, placement: str, encoding: str, marker: str) -> dict[str, Any]:
    return {
        "manifest_id": f"pg25d-manifest-{label}",
        "payload_sha256": _digest(f"safe-canary|{label}|{placement}|{encoding}|{marker}"),
        "probe_ref": "pg25d-safe-canary-v1",
        "probe_kind": "http_canary" if placement != "none" else "abstract_channel_class",
        "route_template_id": "vulnerableapp-root-v1",
        "method": "GET",
        "placement": placement,
        "encoding_chain": [encoding],
        "encoding_depth": 0 if encoding == "identity" else 1,
        "marker_sha256": _digest(marker),
        "max_bytes": 128,
        "safety": {
            "does_not_execute": True,
            "no_external_network": True,
            "no_script_execution": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }


def _oracle() -> dict[str, Any]:
    return {
        "oracle_id": "pg25d-no-positive-signal-v1",
        "oracle_contract_sha256": ORACLE_CONTRACT_SHA256,
        "family": "unknown_surface",
        "modality": "reflection",
        "candidate_signal": False,
        "positive": False,
        "positive_authority": False,
        "confirmed_effect": "none",
        "signals": {"marker_reflected": False, "state_transition": False},
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
            "raw_body_stored": False,
        },
    }


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    source = _source(registry)
    collector = ReadOnlySafeCatalogCollector(source, registry=registry)
    rows: list[dict[str, Any]] = []
    cases = [
        ("head-baseline", "none", "identity", "PG25_CANARY_HEAD"),
        ("query-identity", "query", "identity", "PG25_CANARY_IDENTITY"),
        ("query-percent", "query", "url_percent", "PG25_CANARY_PERCENT"),
    ]
    for label, placement, encoding, marker in cases:
        reset_id, instance_id = _reset()
        baseline = _request(BASE_URI, method="GET")
        baseline_hash = sha256_json(baseline)
        if placement == "none":
            url = BASE_URI
            headers: dict[str, str] = {}
        else:
            value = marker if encoding == "identity" else "".join(f"%{byte:02X}" for byte in marker.encode("utf-8"))
            url = BASE_URI + "?pg25_canary=" + value
            headers = {"X-PG25-Canary": marker} if encoding == "identity" else {}
        response = _request(url, method="GET", headers=headers)
        if response["transport_error"]:
            for _ in range(10):
                time.sleep(1)
                response = _request(url, method="GET", headers=headers)
                if not response["transport_error"]:
                    break
        if response["transport_error"]:
            raise RuntimeError(f"PG-25D transport error for {label}")
        reset = {
            "reset_id": reset_id,
            "kind": "stop_remove_recreate",
            "target_instance_id": instance_id,
            "state_epoch": instance_id.replace("instance-", "epoch-"),
            "reset_adapter_sha256": RESET_ADAPTER_SHA256,
            "baseline_projection_sha256": baseline_hash,
            "fresh_target": True,
            "completed": True,
            "evaluator_state_hidden": True,
            "state_change_allowed": False,
            "external_network": False,
        }
        rule_ir = {
            "rule_key": "unknown_surface.no-positive-signal",
            "grammar_version": "rule-ir-v1",
            "family_candidate": "unknown_surface",
            "operator_set": ["and"],
            "required_slots": [],
            "bound_slots": [],
            "executable": False,
        }
        rows.append(
            collector.collect(
                sample_id=f"pg25d-{label}",
                sample_role="negative_control",
                sampling_seed=25,
                reset=reset,
                payload_manifest=_manifest(label, placement=placement, encoding=encoding, marker=marker),
                response_projection=response,
                oracle_projection=_oracle(),
                rule_ir=rule_ir,
            )
        )
    catalog = build_catalog("pg25d-vulnerableapp-safe-canaries-v1", collector.source, rows)
    catalog["collection"] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "probe_policy": "GET-only inert canaries; no raw request/response retained",
        "training_eligible": False,
    }
    OUT_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT_PATH), "row_count": len(rows), "training_eligible": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
