"""Collect a typed, read-only Clickjacking oracle pair from PG-25D.

The route level and expected outcome are evaluator-side only.  The model-facing
record receives only an abstract Rule IR and bounded frame-policy projection;
raw security-header values and page bodies are discarded.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import ReadOnlySafeCatalogCollector, build_catalog, sha256_json  # noqa: E402
from app.pg25d_clickjacking_oracle import build_clickjacking_oracle, classify_frame_policy  # noqa: E402
from app.regex_evidence_oracle import evaluate_allowlisted_regex  # noqa: E402


RESET_SCRIPT = ROOT / "scripts" / "reset_pg25d_vulnerableapp.ps1"
REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
OUT_PATH = ROOT / "research" / "pg_pk_25d_clickjacking_catalog_v1.json"
BASE_URI = "http://127.0.0.1:19090/VulnerableApp/"
ORACLE_CONTRACT_SHA256 = "3f13c399cebea2db7d40163529617814f82c44b480e791a0953dcbe4484b69f6"
RESET_ADAPTER_SHA256 = hashlib.sha256(RESET_SCRIPT.read_bytes()).hexdigest()
COLLECTOR_SHA256 = "c0d6ce34fa26b55d3d4cba4f1d2f5c604f2dff0ca405f655a275a9aee85b0468"
IMAGE_DIGEST = "sha256:7bc084dac341f089c6e788d2369a27f599c902d742c5e113d7bb50661cd92406"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _head_projection(url: str) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error: Exception | None = None
    for _ in range(12):
        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=10) as response:
                raw_headers = {str(key).casefold(): str(value) for key, value in response.headers.items()}
                frame_policy = classify_frame_policy(raw_headers)
                content_type = response.headers.get_content_type() if response.headers else "unknown"
                projection = {
                    "status_code": int(response.status),
                    "status_class": f"{int(response.status) // 100}xx",
                    "content_type_class": content_type if content_type in {"html", "json", "text", "xml"} else "other",
                    "body_length_bucket": "0",
                    "body_sha256": hashlib.sha256(b"").hexdigest(),
                    "semantic_body_sha256": hashlib.sha256(b"").hexdigest(),
                    "shape": {"kind": "head", "field_count": 0, "scalar_count": 0},
                    "header_names": sorted(set(raw_headers) & {"content-type", "x-frame-options", "content-security-policy"}),
                    "marker": {"reflected": False, "location": "none", "count": 0},
                    "frame_policy": frame_policy,
                    "transport_error": False,
                    "status_changed": False,
                    "state_changed": False,
                    "location_origin_changed": False,
                }
                if frame_policy == "allowall":
                    regex_id = "header_xfo_allowall"
                    regex_text = raw_headers.get("x-frame-options", "")
                elif frame_policy in {"sameorigin", "deny"}:
                    regex_id = "header_xfo_protected"
                    regex_text = raw_headers.get("x-frame-options", "")
                elif frame_policy == "ancestors_none":
                    regex_id = "header_csp_ancestors_none"
                    regex_text = raw_headers.get("content-security-policy", "")
                else:
                    regex_id = "header_xfo_allowall"
                    regex_text = raw_headers.get("x-frame-options", "")
                return projection, evaluate_allowlisted_regex(text=regex_text, pattern_id=regex_id)
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"local HEAD probe did not become ready: {url}") from last_error


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


def _manifest(sample_id: str) -> dict[str, Any]:
    return {
        "manifest_id": f"pg25d-clickjacking-{sample_id}",
        "payload_sha256": _digest(f"pg25d-clickjacking-header|{sample_id}"),
        "probe_ref": "pg25d-clickjacking-header-v1",
        "probe_kind": "header_canary",
        "route_template_id": "vulnerableapp-clickjacking-level",
        "method": "HEAD",
        "placement": "none",
        "encoding_chain": ["identity"],
        "encoding_depth": 0,
        "marker_sha256": _digest("pg25d-clickjacking-header-marker"),
        "max_bytes": 64,
        "safety": {
            "does_not_execute": True,
            "no_external_network": True,
            "no_script_execution": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }


def _rule_ir() -> dict[str, Any]:
    return {
        "rule_key": "clickjacking.frame-protection",
        "grammar_version": "rule-ir-v1",
        "family_candidate": "clickjacking",
        "operator_set": ["eq"],
        "required_slots": ["frame_policy"],
        "bound_slots": ["frame_policy"],
        "executable": False,
    }


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector = ReadOnlySafeCatalogCollector(_source(registry), registry=registry)
    control_reset, control_instance = _reset()
    control_baseline, _ = _head_projection(BASE_URI)
    control_response, control_regex = _head_projection(BASE_URI + "ClickjackingVulnerability/LEVEL_3")
    control = collector.collect(
        sample_id="pg25d-clickjacking-control",
        sample_role="negative_control",
        sampling_seed=25,
        reset={
            "reset_id": control_reset,
            "kind": "stop_remove_recreate",
            "target_instance_id": control_instance,
            "state_epoch": control_instance.replace("instance-", "epoch-"),
            "reset_adapter_sha256": RESET_ADAPTER_SHA256,
            "baseline_projection_sha256": sha256_json(control_baseline),
            "fresh_target": True,
            "completed": True,
            "evaluator_state_hidden": True,
            "state_change_allowed": False,
            "external_network": False,
        },
        payload_manifest=_manifest("control"),
        response_projection=control_response,
        oracle_projection=build_clickjacking_oracle(
            oracle_contract_sha256=ORACLE_CONTRACT_SHA256,
            frame_policy=control_response["frame_policy"],
            expected_vulnerable=False,
            regex_evidence=control_regex,
        ),
        rule_ir=_rule_ir(),
    )

    positive_reset, positive_instance = _reset()
    positive_baseline, _ = _head_projection(BASE_URI)
    positive_response, positive_regex = _head_projection(BASE_URI + "ClickjackingVulnerability/LEVEL_2")
    positive = collector.collect(
        sample_id="pg25d-clickjacking-candidate",
        sample_role="candidate",
        sampling_seed=25,
        reset={
            "reset_id": positive_reset,
            "kind": "stop_remove_recreate",
            "target_instance_id": positive_instance,
            "state_epoch": positive_instance.replace("instance-", "epoch-"),
            "reset_adapter_sha256": RESET_ADAPTER_SHA256,
            "baseline_projection_sha256": sha256_json(positive_baseline),
            "fresh_target": True,
            "completed": True,
            "evaluator_state_hidden": True,
            "state_change_allowed": False,
            "external_network": False,
        },
        payload_manifest=_manifest("candidate"),
        response_projection=positive_response,
        oracle_projection=build_clickjacking_oracle(
            oracle_contract_sha256=ORACLE_CONTRACT_SHA256,
            frame_policy=positive_response["frame_policy"],
            expected_vulnerable=True,
            regex_evidence=positive_regex,
        ),
        rule_ir=_rule_ir(),
        negative_control={
            "control_sample_id": control["sample_id"],
            "control_evidence_hash": control["evidence"]["evidence_hash"],
            "intervention": "protected-frame-policy-level",
            "verdict": "confirmed_negative",
            "same_source": True,
            "same_surface": True,
        },
    )
    catalog = build_catalog("pg25d-vulnerableapp-clickjacking-v1", collector.source, [control, positive])
    catalog["collection"] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "row_count": 2,
        "oracle_family": "clickjacking",
        "probe_policy": "HEAD-only response-header projection; no page body or active payload",
        "training_eligible": False,
    }
    OUT_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT_PATH),
        "row_count": 2,
        "confirmed_positive": positive["decision"]["evidence_status"] == "confirmed_positive",
        "training_action": positive["decision"]["training_action"],
        "frame_policy_positive": positive_response["frame_policy"],
        "frame_policy_control": control_response["frame_policy"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
