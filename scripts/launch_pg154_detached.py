"""Launch PG-154 outside the terminal job for a long GPU run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
log_dir = root / "artifacts" / "pg154-multisource-action-replay-v1"
log_dir.mkdir(parents=True, exist_ok=True)
log = (log_dir / "run_detached.log").open("w", encoding="utf-8")
flags = 0
if os.name == "nt":
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
process = subprocess.Popen(
    [sys.executable, "-u", str(root / "scripts" / "run_pg154_multisource_action_replay.py")],
    cwd=root,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    creationflags=flags,
    close_fds=True,
)
print(process.pid)
log.close()
