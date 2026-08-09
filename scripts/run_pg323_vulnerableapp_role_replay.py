"""Replay PG-323 decoy/ASK checkpoints on the independent VulnerableApp."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load() -> Any:
    path = ROOT / "scripts" / "run_pg322_vulnerableapp_role_replay.py"
    spec = importlib.util.spec_from_file_location("pg322_vapp_replay_for_pg323", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-322 VulnerableApp replay")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("PG-322", "PG-323").replace("pg322", "pg323")
    if isinstance(value, list):
        return [_replace(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item) for key, item in value.items()}
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> int:
    if os.environ.get("PG323_VAPP_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-323 VulnerableApp replay requires PG323_VAPP_LOCAL_DOCKER_EVAL=1")
    os.environ["PG322_VAPP_LOCAL_DOCKER_EVAL"] = "1"
    module = _load()
    module.CHECKPOINT_DIR = ROOT / "artifacts" / "pg323-decoy-ask-anchor" / "seeds"
    # The PG-323 wrapper reuses the PG-322 trainer's checkpoint stem so that
    # replay loads the exact files produced by the training run.  Keep the
    # experiment/report identity PG-323 while mapping the on-disk prefix.
    module.CHECKPOINT_PREFIX = "pg322_cross_impl_decoy_seed_"
    module.REPORT = ROOT / "research" / "pg323_vulnerableapp_role_replay_report_v1.json"
    module.CATALOG = ROOT / "research" / "pg323_vulnerableapp_role_catalog_v1.json"
    module.TRACE = ROOT / "research" / "pg323_vulnerableapp_role_trace_v1.json"
    module.PROTOCOL = ROOT / "research" / "pg323_vulnerableapp_role_protocol_v1.json"
    result = int(module.main())
    for path, key in ((module.REPORT, "report_sha256"), (module.CATALOG, "catalog_sha256"), (module.TRACE, "trace_sha256"), (module.PROTOCOL, "protocol_sha256")):
        data = _replace(json.loads(path.read_text(encoding="utf-8-sig")))
        data[key] = ""
        data[key] = _digest(data)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = json.loads(module.REPORT.read_text(encoding="utf-8-sig"))
    print(json.dumps({"status": report.get("status"), "counts": report.get("counts"), "worst_seed_metrics": report.get("worst_seed_metrics"), "gate": report.get("hypothesis_gate"), "report": str(module.REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
