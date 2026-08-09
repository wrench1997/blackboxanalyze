"""PG-214: clean-database SQL replay on the repaired local Pikachu image.

PG-212 proved that the original pinned image could not execute PHP-FPM.  This
experiment uses a locally derived, digest-pinned runtime with PHP 8.0 FPM and
mysqli.  Every GET/POST route episode gets a new ``--rm`` container with no
volume or bind mount; a Docker restart is deliberately not used as a database
reset.  The AI and an independent reference probe both send bounded syntax
shape probes.  Response shape is recorded, but no SQL AST/result oracle is
claimed and no vulnerability/payload is promoted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_pg212() -> Any:
    path = ROOT / "scripts" / "run_pg212_pikachu_sql_response_shape_loop.py"
    spec = importlib.util.spec_from_file_location("pg212_for_pg214", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-212 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG212 = _load_pg212()

RESEARCH = ROOT / "research"
IMAGE = "sift/pikachu-fixed@sha256:cca4288b6b701725e7a771f47ce7fcafd6cea9bd7622fa34ef2ed0b440f472c6"
BASE_IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
REPORT_PATH = RESEARCH / "pg214_pikachu_fixed_sql_loop_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg214_pikachu_fixed_sql_loop_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg214_pikachu_fixed_sql_loop_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg214_pikachu_fixed_sql_loop_report_v1.md"
SEEDS = (21401, 21402)
BASE_PORT = 3625


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _docker(*args: str) -> str:
    result = subprocess.run(
        ["docker", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _database_health(name: str) -> bool:
    """Run the image's own mysqli credentials check without persisting output."""

    code = "$db=@new mysqli('127.0.0.1','root','root','pikachu',3306); exit($db->connect_errno ? 1 : 0);"
    result = subprocess.run(
        ["docker", "exec", name, "php", "-r", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.returncode == 0


def _start(seed: int, run_index: int) -> tuple[str, int, str, dict[str, Any]]:
    name = f"sift-pg214-{seed}-{run_index}"
    if _exists(name):
        raise RuntimeError(f"PG-214 refuses to reuse target {name}")
    port = BASE_PORT + int(run_index)
    _docker(
        "run",
        "--detach",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--label",
        "sift.pg214=true",
        "--label",
        f"sift.pg214.reset_epoch={seed}-{run_index}",
        "--publish",
        f"127.0.0.1:{port}:8090",
        IMAGE,
    )
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500 and _database_health(name):
                container_id = _docker("inspect", "--format", "{{.Id}}", name)
                mounts = json.loads(_docker("inspect", "--format", "{{json .Mounts}}", name) or "[]")
                image_ref = _docker("inspect", "--format", "{{.Config.Image}}", name)
                image_id = _docker("inspect", "--format", "{{.Image}}", name)
                if mounts:
                    raise RuntimeError("PG-214 clean reset requires zero mounts/volumes")
                if image_ref != IMAGE:
                    raise RuntimeError("PG-214 image digest attestation mismatch")
                reset = {
                    "reset_id": f"pg214-reset-{seed}-{run_index}",
                    "reset_epoch": f"{seed}-{run_index}",
                    "fresh_target": True,
                    "completed": True,
                    "container_recreated": True,
                    "container_restart_used": False,
                    "container_id_sha256": hashlib.sha256(container_id.encode("utf-8")).hexdigest(),
                    "image": image_ref,
                    "image_id_sha256": image_id.removeprefix("sha256:"),
                    "volume_mount_count": len(mounts),
                    "database_health_gate": "mysqli_root_pikachu_ok",
                    "database_clean_contract": "fresh_writable_layer_no_volume_no_stateful_probe",
                    "baseline_from_original_derived_image": True,
                    "state_change_allowed": False,
                    "evaluator_state_hidden": True,
                    "external_network": False,
                }
                return name, port, container_id, reset
        except (httpx.HTTPError, subprocess.SubprocessError):
            pass
        time.sleep(1.0)
    if _exists(name):
        _docker("stop", "--timeout", "5", name)
    raise RuntimeError(f"PG-214 target {name} did not pass HTTP + database health gates")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def main() -> int:
    routes = PG212._routes()
    model, vocabulary = PG212.PG208._load_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    from app.payload_learner import PayloadLearner

    learner = PayloadLearner(seed=214)
    episodes: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        for route in routes:
            name = ""
            try:
                name, port, container_id, reset = _start(seed, run_index)
                target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
                targets.append(
                    {
                        "seed": seed,
                        "route": route["path"],
                        "method": route["method"],
                        "target_instance_hash": target_hash,
                        "fresh_container": True,
                        "loopback_port": port,
                        "image": IMAGE,
                        "base_image": BASE_IMAGE,
                        "reset_id": reset["reset_id"],
                        "database_health_gate": reset["database_health_gate"],
                    }
                )
                client = httpx.Client(
                    base_url=f"http://127.0.0.1:{port}",
                    timeout=12.0,
                    follow_redirects=False,
                    cookies={},
                )
                try:
                    episodes.append(
                        PG212._route_episode(
                            model,
                            vocabulary,
                            device,
                            learner,
                            client,
                            route,
                            seed=seed,
                            target_hash=target_hash,
                            reset=reset,
                            target_url=f"http://127.0.0.1:{port}",
                        )
                    )
                finally:
                    client.close()
            finally:
                if name:
                    _stop(name)
            run_index += 1

    counts = {
        "fresh_container_count": len(targets),
        "episode_count": len(episodes),
        "get_episode_count": sum(int(row["method"] == "GET") for row in episodes),
        "post_episode_count": sum(int(row["method"] == "POST") for row in episodes),
        "database_health_gate_count": sum(int(row["reset"].get("database_health_gate") == "mysqli_root_pikachu_ok") for row in episodes),
        "database_clean_reset_verified_count": sum(int(row["database_clean_reset_verified"]) for row in episodes),
        "reference_send_count": sum(int(row["reference"]["sent"]) for row in episodes),
        "ai_candidate_send_count": sum(int(row["ai"].get("sent")) for row in episodes),
        "database_unavailable_count": sum(int(row["database_unavailable"]) for row in episodes),
        "sql_evaluator_typed_available_count": sum(int((row["control"].get("oracle") or {}).get("typed_available")) for row in episodes),
        "ai_reference_shape_agreement_count": sum(int(row["comparison"].get("ai_reference_shape_agreement")) for row in episodes),
        "abstain_count": sum(int(row["ai"].get("model_decision", {}).get("effective_action") == "abstain") for row in episodes),
        "confirmed_positive_count": 0,
        "false_positive_count": 0,
    }
    report = {
        "protocol_id": "pg-pk-214-pikachu-fixed-sql-loop-v1",
        "schema_version": "pg214-pikachu-fixed-sql-loop-report-v1",
        "status": "completed_backend_response_shape_evaluator_only",
        "device": str(device),
        "model": {"variant": "xxl_field_token_adapter", "base_parameter_count": 101487169, "online_weight_update": False},
        "runtime": {
            "image": IMAGE,
            "base_image": BASE_IMAGE,
            "derived_runtime": "php8.0-fpm8.0-mysql_nginx_compatibility_patch",
            "php_fpm": "php8.0-fpm",
            "database": "bundled_mysql_5.7.27",
            "database_health_gate": "mysqli_root_pikachu_ok",
        },
        "routes": {"count": len(routes), "get_count": sum(int(route["method"] == "GET") for route in routes), "post_count": sum(int(route["method"] == "POST") for route in routes)},
        "targets": targets,
        "episodes": episodes,
        "counts": counts,
        "promotion": {
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "payload_generation_claim_allowed": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "safety": {
            "loopback_only": True,
            "pinned_image": IMAGE,
            "base_image_pinned": BASE_IMAGE,
            "fresh_container_per_episode": True,
            "fresh_reset_per_route": True,
            "no_volume_or_bind_mount": True,
            "docker_restart_used_count": 0,
            "get_post_replay": True,
            "time_delay_used": False,
            "database_write": False,
            "external_network_target": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg214-pikachu-fixed-sql-loop-protocol-v1",
        "ai_participates_in_send": True,
        "independent_reference_sent": True,
        "evaluator": "response_shape_only; backend connection health is not SQL AST/result evidence",
        "backend_health_gate": "mysqli root/pikachu connection must succeed before route replay",
        "backend_failure_is_not_vulnerability": True,
        "allowed_probe_classes": ["syntax_shape", "encoded_syntax_shape"],
        "forbidden_probe_classes": ["time_delay", "local_side_channel", "write", "destructive"],
        "fresh_reset_required": True,
        "fresh_reset_unit": "one disposable no-volume derived-runtime container per route episode",
        "database_clean_required": True,
        "database_clean_attestation": ["pinned_derived_image_digest", "fresh_container_recreated", "zero_volume_or_bind_mount", "mysqli_health_gate", "state_change_allowed_false", "baseline_state_fingerprint"],
        "docker_restart_allowed": "not used; fresh container is the reset primitive",
        "matched_negative_control_required": True,
        "evidence_hash_required": True,
        "raw_payload_and_response_excluded": True,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg214-pikachu-fixed-sql-loop-trace-v1", "evaluation_only": True, "targets": targets, "episodes": episodes, "training_eligible": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    lines = [
        "# PG-214 Pikachu fixed-runtime SQL loop",
        "",
        f"device={device}; fresh containers={len(targets)}; episodes={len(episodes)}; GET={counts['get_episode_count']}; POST={counts['post_episode_count']}",
        f"mysqli health gates={counts['database_health_gate_count']}; clean resets={counts['database_clean_reset_verified_count']}; AI sends={counts['ai_candidate_send_count']}; reference sends={counts['reference_send_count']}",
        f"typed response-shape evaluator available={counts['sql_evaluator_typed_available_count']}; AI/reference shape agreement={counts['ai_reference_shape_agreement_count']}; database unavailable={counts['database_unavailable_count']}",
        "",
        "每个 route episode 都从派生镜像的全新无 volume 容器开始，并通过 mysqli(root/pikachu) 健康门；没有用 docker restart 伪装数据库 reset。结果只证明 GET/POST 能到达后端并产生响应形状，不证明 SQL AST、查询结果或漏洞 payload 成功。",
        "",
    ]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "counts": counts, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
