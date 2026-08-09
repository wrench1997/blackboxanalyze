"""Independent audit for the PG-280 remote Docker adapter probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "research" / "pg280_remote_docker_probe_v2.json"
AUDIT = ROOT / "research" / "pg280_remote_docker_probe_audit_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def main() -> None:
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    scope = dict(probe.get("scope") or {})
    commands = list(scope.get("command_allowlist") or [])
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    check("probe_hash", probe.get("evidence_sha256") == sha({key: value for key, value in probe.items() if key != "evidence_sha256"}))
    check("schema", probe.get("schema_version") == "pg280-authorized-remote-docker-adapter-v1")
    check("fixed_remote_host", scope.get("remote_host") == "112.111.7.91:60228")
    check("explicit_authorization", scope.get("authorization") == "operator_allowlisted_remote_docker")
    check("command_allowlist", commands == [["command", "-v", "docker"], ["docker", "version", "--format={{json .Server}}"], ["docker", "ps", "--format={{.Names}}"]])
    check("mutations_disabled", scope.get("mutating_docker_commands_allowed") is False and scope.get("container_creation_allowed") is False and scope.get("container_restart_allowed") is False and scope.get("container_removal_allowed") is False)
    check("no_training_side_effect", probe.get("training_or_replay_started") is False and int(probe.get("real_application_gold_rows", 0) or 0) == 0)
    check("status_consistent", probe.get("status") in {"available", "unavailable", "unreachable", "ssh_unavailable"})
    if probe.get("status") != "available":
        check("unavailable_has_no_server", probe.get("docker_server") is False and probe.get("running_containers") == [])

    audit = {
        "audit_id": "pg280-authorized-remote-docker-adapter-audit-v1",
        "status": "passed" if not failures else "failed",
        "probe": PROBE.relative_to(ROOT).as_posix(),
        "audit_checks": {name: name not in failures for name in ["probe_hash", "schema", "fixed_remote_host", "explicit_authorization", "command_allowlist", "mutations_disabled", "no_training_side_effect", "status_consistent", "unavailable_has_no_server"]},
        "failures": failures,
        "interpretation": "适配层只证明固定远程主机的 Docker 可用性投影；Docker 不可用时不创建容器、不发包、不生成真实 gold。",
    }
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

