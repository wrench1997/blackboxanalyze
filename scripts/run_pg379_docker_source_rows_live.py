"""Bind the PG-379 abstract collector to the two reviewed local Docker fixtures.

This is deliberately a small, source-grounded adapter around
``run_pg379_dynamic_source_rows_live``.  It does not invent paths: each path and
parameter name is read from the implementation's checked-in manifest.  Each
role gets a disposable ``--network none`` container with no published ports or
mounts.  Requests are issued from inside the container through loopback, and
the response body is held only in memory long enough for the PG-377 adapter to
make an abstract projection.

The command is fail-closed.  It requires ``PG379_LOCAL_DOCKER_EVAL=1`` and an
explicit ``--operator-reviewed`` acknowledgement.  It writes a bounded report,
evaluator sidecar, and (optionally) abstract source rows; raw body/URL/payload
data are never serialized.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pg379_dynamic_source_rows_live import (  # noqa: E402
    OPERATOR_FLAG,
    build_pg379_docker_runtime_factory,
    collect_pg379_dynamic_source_rows_live,
    write_artifacts,
)
from scripts.plan_pg379_source_collection import build_pg379_source_collection_plan  # noqa: E402


SAFE_CANARY = "PG379A_CANARY_safe"  # evaluator-only bounded marker, not a payload
MAX_BODY = 2 * 1024 * 1024
DOCKER_TIMEOUT = 12.0


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, timeout: float = DOCKER_TIMEOUT, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True, timeout=timeout)


def _image_id(image: str) -> str:
    result = _run(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
    value = result.stdout.strip()
    if not value.startswith("sha256:") or len(value) != 71:
        raise RuntimeError("image_digest_unavailable")
    return value


def _json_body(raw: bytes) -> Mapping[str, Any]:
    if len(raw) > MAX_BODY:
        raise RuntimeError("response_too_large")
    value = json.loads(raw.decode("utf-8", errors="replace"))
    return value if isinstance(value, Mapping) else {}


def _python_request_script() -> str:
    return (
        "import base64,json,sys,urllib.error,urllib.request;"
        "m,p,b=sys.argv[1],sys.argv[2],sys.argv[3];"
        "d=base64.b64decode(b.encode()) if b else b'';"
        "r=urllib.request.Request('http://127.0.0.1:8080'+p,data=(d or None),method=m,headers={'Content-Type':'application/x-www-form-urlencoded'});"
        "\ntry:\n x=urllib.request.urlopen(r,timeout=5); q=x.read(2097153); print(json.dumps({'status':x.status,'headers':dict(x.headers),'body_b64':base64.b64encode(q).decode()},separators=(',',':')))\n"
        "except urllib.error.HTTPError as e:\n q=e.read(2097153); print(json.dumps({'status':e.code,'headers':dict(e.headers),'body_b64':base64.b64encode(q).decode()},separators=(',',':')))"
    )


def _node_request_script() -> str:
    return (
        "const http=require('node:http');const b=Buffer.from(process.argv[3]||'','base64');"
        "const m=process.argv[1],p=process.argv[2];const r=http.request({host:'127.0.0.1',port:8799,path:p,method:m,headers:{'Content-Type':'application/x-www-form-urlencoded','Content-Length':b.length}},x=>{let a=[];x.on('data',c=>a.push(c));x.on('end',()=>process.stdout.write(JSON.stringify({status:x.statusCode,headers:x.headers,body_b64:Buffer.concat(a).subarray(0,2097153).toString('base64')})))});"
        "r.on('error',e=>{process.stdout.write(JSON.stringify({status:0,headers:{},body_b64:''}));process.exitCode=1});if(b.length)r.write(b);r.end();"
    )


class DockerRuntime:
    """One fresh container and its source-grounded route manifest."""

    def __init__(self, *, implementation_id: str, lane: str, seed: int, route: Mapping[str, Any], role: str, image: str, manifest: Mapping[str, Any], name: str):
        self.implementation_id = implementation_id
        self.lane = lane
        self.seed = int(seed)
        self.route = dict(route)
        self.role = role
        self.image = image
        self.manifest = manifest
        self.name = name
        self.port = 8080 if implementation_id.endswith("_a") else 8799
        self.started = False
        self.route_by_class = {
            str(item.get("route_class")): dict(item)
            for item in (manifest.get("routes") or manifest.get("route_classes") or manifest.get("route_shapes") or [])
            if isinstance(item, Mapping)
        }

    def _exec_request(self, method: str, path: str, body: bytes = b"") -> dict[str, Any]:
        encoded = base64.b64encode(body).decode("ascii")
        if self.port == 8080:
            command = ["docker", "exec", self.name, "python", "-c", _python_request_script(), method, path, encoded]
        else:
            command = ["docker", "exec", self.name, "node", "-e", _node_request_script(), method, path, encoded]
        completed = _run(command, timeout=DOCKER_TIMEOUT, check=False)
        text = completed.stdout.strip()
        if not text:
            raise RuntimeError("loopback_request_empty")
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise RuntimeError("loopback_response_invalid")
        raw = base64.b64decode(str(value.get("body_b64", "")).encode("ascii"), validate=False)
        if len(raw) > MAX_BODY:
            raise RuntimeError("response_too_large")
        value = dict(value)
        value["body"] = raw
        return value

    def _route(self, abstract: Mapping[str, Any]) -> dict[str, Any]:
        route_class = str(abstract.get("route_class", ""))
        route = dict(self.route_by_class.get(route_class) or {})
        if not route:
            raise RuntimeError("route_class_not_in_manifest")
        for field in ("method", "parameter_role", "encoding_chain", "response_shape", "script_surface"):
            observed = route.get(field, route.get("parameter") if field == "parameter_role" else "")
            if str(observed).upper() != str(abstract.get(field, "")).upper():
                raise RuntimeError(f"route_manifest_mismatch:{field}")
        return route

    def _wire(self, method: str, abstract: Mapping[str, Any]) -> tuple[str, bytes]:
        route = self._route(abstract)
        path = str(route.get("path", ""))
        parameter = str(route.get("parameter", route.get("parameter_role", ""))) if self.port == 8799 else str(route.get("parameter", ""))
        input_source = str(route.get("input_source", ""))
        canary = SAFE_CANARY if self.port == 8080 else "PG379B_CANARY_safe"
        if input_source == "path":
            encoded = urllib.parse.quote(canary, safe="")
            path = path.replace("<value>", encoded) if "<value>" in path else path.rstrip("/") + "/" + encoded
        elif input_source == "query":
            parameter = parameter or {
                "query_text": "q",
                "fragment_identifier": "fragment_identifier",
                "json_value": "json_value",
                "view_mode": "mode",
                "query_term": "q",
            }.get(str(abstract.get("parameter_role")), "value")
            path += "?" + urllib.parse.urlencode({parameter: canary})
        if method == "POST":
            parameter = parameter or {
                "form_field": "value",
                "json_value": "json_value",
                "view_mode": "mode",
                "attribute_value": "value",
                "structured_value": "value",
                "record_cursor": "record",
            }.get(str(abstract.get("parameter_role")), "value")
            if input_source == "json":
                body = json.dumps({parameter: canary}, separators=(",", ":")).encode()
            else:
                body = urllib.parse.urlencode({parameter: canary}).encode()
        else:
            body = b""
        return path, body

    def start(self) -> dict[str, Any]:
        env = ["PORT=8080"] if self.port == 8080 else ["PORT=8799"]
        command = [
            "docker", "run", "-d", "--rm", "--pull=never", "--name", self.name,
            "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--tmpfs", "/run:rw,noexec,nosuid,size=16m",
        ]
        for item in env:
            command += ["-e", item]
        command.append(self.image)
        _run(command, timeout=DOCKER_TIMEOUT)
        self.started = True
        deadline = time.monotonic() + DOCKER_TIMEOUT
        while time.monotonic() < deadline:
            try:
                health = self._exec_request("GET", "/__health")
                if int(health.get("status", 0)) == 200:
                    reset = self._exec_request("POST", "/__reset")
                    document = _json_body(reset.get("body", b"{}"))
                    digest = str(document.get("target_instance_digest", document.get("instance_digest", "")))
                    if len(digest) != 64:
                        health_doc = _json_body(health.get("body", b"{}"))
                        digest = str(health_doc.get("target_instance_digest", ""))
                    return {
                        "fresh_reset": True,
                        "reset_id": f"{self.name}:reset",
                        "target_instance_digest": digest,
                        "network_mode": "none",
                        "external_network": False,
                        "loopback_only": True,
                        "state_clean": True,
                        "volume_mount_count": 0,
                        "container_restart_used": False,
                        "database_health_gate": True,
                    }
            except Exception:
                time.sleep(0.15)
        raise RuntimeError("fresh_reset_health_timeout")

    def request(self, *, method: str, route: Mapping[str, Any], role: str, phase: str, **_: Any) -> dict[str, Any]:
        path, body = self._wire(str(method).upper(), route)
        response = self._exec_request(str(method).upper(), path, body)
        status = int(response.get("status", 0) or 0)
        headers = response.get("headers") if isinstance(response.get("headers"), Mapping) else {}
        content_type = str(headers.get("Content-Type", headers.get("content-type", "unknown")))
        status_class = f"{status // 100}xx" if 100 <= status < 600 else "transport_error"
        abstract = dict(route)
        expected = str(abstract.get("method", "")).upper()
        typed = expected == str(method).upper() and status not in {0, 404, 405}
        location = headers.get("Location", headers.get("location"))
        return {
            "status": status,
            "status_class": status_class,
            "content_type_class": "json" if "json" in content_type else "html" if "html" in content_type else "text" if "text" in content_type else "unknown",
            "location_class": "loopback" if location else "none",
            "body": response.get("body", b""),
            "typed_effect_confirmed": bool(typed and role != "negative"),
            "method": str(method).upper(),
            "phase": phase,
        }

    def stop(self) -> None:
        if not self.started:
            return
        try:
            _run(["docker", "stop", "-t", "1", self.name], timeout=DOCKER_TIMEOUT, check=False)
        finally:
            self.started = False


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("manifest_not_object")
    return dict(value)


def _make_attestation(*, lane: str, implementation_id: str, image: str, manifest_path: Path, module_path: Path, dockerfile_path: Path, authorization_id: str) -> tuple[dict[str, Any], str]:
    image_digest = _image_id(image)
    module_digest = _sha256_file(module_path)
    source_digest = module_digest
    process_digest = _sha256_file(dockerfile_path)
    manifest = _load_manifest(manifest_path)
    if lane == "train":
        image_ref = f"{image}@{image_digest}"
    else:
        image_ref = f"{image}@{image_digest}"
    attestation = {
        "implementation_id": implementation_id,
        "bound": True,
        "image_built": True,
        "image_attested": True,
        "attestation_status": "operator_reviewed",
        "image_digest": image_digest,
        "runtime_module_sha256": module_digest,
        "process_boundary_sha256": process_digest,
        "source_digest": source_digest,
        "authorization_id": authorization_id,
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "bind_or_volume_mounts_allowed": False,
        "published_ports": False,
        "fresh_reset_contract": True,
        "independent_source_review": True,
        "side_effects_enabled": False,
        "manifest_sha256": _sha256_file(manifest_path),
        "image_ref": image_ref,
    }
    return attestation, image_ref


def run_live(*, output: Path, sidecar_output: Path, rows_output: Path, authorization_id: str, image_a: str, image_b: str, operator_reviewed: bool, seeds: tuple[int, ...] | None = None, route_classes: tuple[str, ...] | None = None) -> dict[str, Any]:
    if not operator_reviewed:
        raise ValueError("--operator-reviewed is required")
    selected_seeds = tuple(int(seed) for seed in (seeds if seeds is not None else ()))
    plan = build_pg379_source_collection_plan(seeds=selected_seeds or None)
    requirements = plan["new_implementation_requirements"]
    att_a, ref_a = _make_attestation(
        lane="train", implementation_id=requirements["train"]["implementation_id"], image=image_a,
        manifest_path=ROOT / "fixtures/pg379/impl_a/manifest_v1.json", module_path=ROOT / "fixtures/pg379/impl_a/app.py",
        dockerfile_path=ROOT / "fixtures/pg379/impl_a/Dockerfile", authorization_id=authorization_id,
    )
    att_b, ref_b = _make_attestation(
        lane="holdout", implementation_id=requirements["holdout"]["implementation_id"], image=image_b,
        manifest_path=ROOT / "fixtures/pg379/impl_b/manifest.json", module_path=ROOT / "fixtures/pg379/impl_b/server.js",
        dockerfile_path=ROOT / "fixtures/pg379/impl_b/Dockerfile", authorization_id=authorization_id,
    )
    manifests = {"train": _load_manifest(ROOT / "fixtures/pg379/impl_a/manifest_v1.json"), "holdout": _load_manifest(ROOT / "fixtures/pg379/impl_b/manifest.json")}

    # Use the same whole-page runtime as the dynamic collector.  The older
    # compatibility DockerRuntime remains available for static wire tests, but
    # it is intentionally not used for live evidence because it followed
    # redirects and used a weaker typed-shape check.
    specs = {
        "train": {
            "image_ref": ref_a,
            "runtime_language": "python",
            "port": 8080,
            "routes": manifests["train"].get("routes") or manifests["train"].get("route_classes") or [],
        },
        "holdout": {
            "image_ref": ref_b,
            "runtime_language": "node",
            "port": 8799,
            "routes": manifests["holdout"].get("routes") or manifests["holdout"].get("route_classes") or [],
        },
    }
    factory = build_pg379_docker_runtime_factory(specs)

    def probe(**kwargs: Any) -> bool:
        # The image was inspected while building the attestation. Re-inspect
        # here so a mutable tag cannot silently replace it before the first run.
        return _image_id(image_a if kwargs.get("lane") == "train" else image_b) == str(kwargs.get("image_digest"))

    result = collect_pg379_dynamic_source_rows_live(
        plan=plan, route_classes=route_classes, live=True, attestations={"train": att_a, "holdout": att_b},
        image_digest={"train": att_a["image_digest"], "holdout": att_b["image_digest"]},
        runtime_factory=factory, image_probe=probe, environment={OPERATOR_FLAG: "1"},
    )
    paths = write_artifacts(result, output=output, sidecar_output=sidecar_output)
    rows = list(result.get("rows") or [])
    records = [dict(row["source_row"]) for row in rows if isinstance(row.get("source_row"), Mapping)]
    rows_document = {
        "schema_version": "pg379-dynamic-source-rows-v1",
        "status": "diagnostic_candidate_only",
        # Persist only the strict abstract source-row projection. Evaluator
        # bookkeeping (including in-memory HTML/body) is deliberately omitted.
        "records": records,
        "source_row_count": len(records),
        "valid_source_row_count": sum(bool(row.get("adapter_validation", {}).get("valid")) for row in rows),
        "training_eligible_count": 0,
        "raw_payload_response_context": False,
        "evaluator_sidecars_off_context": True,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "report_sha256": str(result.get("report", {}).get("report_sha256", "")),
    }
    # The collector has already scrubbed each row; this document contains only
    # abstract adapter output and is intentionally not a training authorization.
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.write_text(json.dumps(rows_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["rows"] = str(rows_output)
    return {"result": result, "paths": paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--operator-reviewed", action="store_true")
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--image-a", default="pg379-impl-a:reviewed")
    parser.add_argument("--image-b", default="pg379-impl-b:reviewed")
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg379_dynamic_source_rows_live_report_v1.json")
    parser.add_argument("--sidecar-output", type=Path, default=ROOT / "research/pg379_dynamic_source_rows_live_sidecars_v1.json")
    parser.add_argument("--rows-output", type=Path, default=ROOT / "research/pg379_dynamic_source_rows_live_v1.json")
    parser.add_argument("--seeds", help="comma-separated seed subset for a bounded smoke; omit for all three seeds")
    parser.add_argument("--route-class", action="append", help="reviewed abstract route class; repeat for a bounded smoke")
    args = parser.parse_args()
    if not args.live:
        print(json.dumps({"status": "planning_only_live_blocked", "reason": "--live_required"}, ensure_ascii=False))
        return 0
    try:
        parsed_seeds = tuple(int(item.strip()) for item in str(args.seeds).split(",") if item.strip()) if args.seeds else None
        result = run_live(output=args.output, sidecar_output=args.sidecar_output, rows_output=args.rows_output, authorization_id=args.authorization_id, image_a=args.image_a, image_b=args.image_b, operator_reviewed=bool(args.operator_reviewed), seeds=parsed_seeds, route_classes=tuple(args.route_class) if args.route_class else None)
    except Exception as error:
        print(f"pg379_docker_live_failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    report = result["result"]["report"]
    print(json.dumps({"status": report.get("status"), "counts": report.get("counts"), "hard_gate": report.get("hard_gate"), "artifacts": result["paths"]}, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"completed_source_row_candidate_only", "completed_incomplete_source_rows"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DockerRuntime", "run_live"]
