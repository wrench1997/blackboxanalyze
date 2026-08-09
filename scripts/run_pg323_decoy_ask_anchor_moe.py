"""Run the PG-322 decoder training loop on PG-323 ASK/decoy anchors."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load() -> object:
    path = ROOT / "scripts" / "run_pg322_cross_impl_decoy_moe.py"
    spec = importlib.util.spec_from_file_location("pg322_training_for_pg323", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-322 training loop")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load()
    module.DATASET = ROOT / "research" / "pg323_decoy_ask_anchor_dataset_v1.json"
    module.AUDIT = ROOT / "research" / "pg323_decoy_ask_anchor_dataset_audit_v1.json"
    module.BASE_DIR = ROOT / "artifacts" / "pg322-cross-impl-decoy" / "seeds"
    module.BASE_PREFIX = "pg322_cross_impl_decoy_seed_"
    module.OUT_DIR = ROOT / "artifacts" / "pg323-decoy-ask-anchor" / "seeds"
    module.CHECKPOINT = ROOT / "artifacts" / "pg323-decoy-ask-anchor" / "pg323_decoy_ask_anchor_moe_local_morning.pt"
    module.REPORT = ROOT / "research" / "pg323_decoy_ask_anchor_moe_training_report_v1_local_morning.json"
    os.environ.setdefault("BLACKBOX_LOCAL_MORNING_TRAIN", "1")
    result = int(module.main())
    report_path = module.REPORT
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    def replace(value):
        if isinstance(value, str):
            return value.replace("PG-322", "PG-323").replace("pg322", "pg323")
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value
    report = replace(report)
    report["status"] = "completed_local_morning_pg323_decoy_ask_anchor"
    report["report_sha256"] = ""
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
