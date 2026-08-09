"""Run the PG-280 read-only probe against the fixed authorized SSH host.

This command intentionally does not start or reset Docker.  It records only a
bounded availability projection so an unavailable daemon cannot be mistaken
for a real target or a training example.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.authorized_remote_docker import probe_authorized_remote_docker


OUTPUT = ROOT / "research" / "pg280_remote_docker_probe_v2.json"


def main() -> None:
    result = probe_authorized_remote_docker()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "docker_binary": result["docker_binary"], "docker_server": result["docker_server"], "running_containers": result["running_containers"], "evidence_sha256": result["evidence_sha256"], "output": OUTPUT.relative_to(ROOT).as_posix()}, ensure_ascii=False, indent=2))
    if result["status"] == "ssh_unavailable":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
