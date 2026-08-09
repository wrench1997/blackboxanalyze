"""Launch PG-151 outside the terminal job so long GPU runs survive polling timeouts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
log_dir = root / "artifacts" / "pg151-moe-capacity-v1"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "run_detached.log"
log = log_path.open("w", encoding="utf-8")
flags = 0
if os.name == "nt":
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
process = subprocess.Popen(
    [sys.executable, "-u", str(root / "scripts" / "run_pg151_moe_capacity.py")],
    cwd=root,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    creationflags=flags,
    close_fds=True,
)
print(process.pid)
log.close()
