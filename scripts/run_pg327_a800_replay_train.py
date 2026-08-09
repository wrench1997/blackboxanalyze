"""Run the PG-327 replay-mix candidate on the authorized remote A800 lane.

This is a training-only candidate run.  It consumes the already audited,
abstract PG-323 dataset and frozen PG-322 checkpoints; it never emits a wire,
contacts a target, or promotes memory/catalog entries.  The wrapper keeps the
parent loop reusable while stamping the resulting artifacts with explicit
device and provenance metadata.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load() -> object:
    path = ROOT / "scripts" / "run_pg322_cross_impl_decoy_moe.py"
    spec = importlib.util.spec_from_file_location("pg322_training_for_pg327", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-322 training loop")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_checkpoint(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema_version"] = "pg327-a800-replay-candidate-checkpoint-v1"
    assignment = dict(payload.get("assignment") or {})
    assignment.update(
        {
            "protocol_id": "pg-pk-327-a800-replay-train-v1",
            "execution_mode": "remote_a800_gpu0",
            "device": "cuda:0",
            "promotion_blocked": True,
        }
    )
    payload["assignment"] = assignment
    payload["promotion_blocked"] = True
    torch.save(payload, path)
    return _sha256(path)


def main() -> int:
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1":
        raise RuntimeError("PG-327 requires BLACKBOX_REMOTE_A800_TRAIN=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("PG-327 requires CUDA_VISIBLE_DEVICES=0")

    module = _load()
    module.DATASET = ROOT / "research" / "pg323_decoy_ask_anchor_dataset_v1.json"
    module.AUDIT = ROOT / "research" / "pg323_decoy_ask_anchor_dataset_audit_v1.json"
    module.BASE_DIR = ROOT / "artifacts" / "pg322-cross-impl-decoy" / "seeds"
    module.BASE_PREFIX = "pg322_cross_impl_decoy_seed_"
    module.OUT_DIR = ROOT / "artifacts" / "pg327-a800-replay" / "seeds"
    module.CHECKPOINT = ROOT / "artifacts" / "pg327-a800-replay" / "pg327_a800_replay_candidate.pt"
    module.REPORT = ROOT / "research" / "pg327_a800_replay_training_report_v1.json"
    os.environ["BLACKBOX_REMOTE_A800_TRAIN"] = "1"
    result = int(module.main())

    checkpoint_hashes: dict[str, str] = {}
    for path in sorted(module.OUT_DIR.glob("*.pt")):
        checkpoint_hashes[str(path.relative_to(ROOT))] = _rewrite_checkpoint(path)
    checkpoint_hashes[str(module.CHECKPOINT.relative_to(ROOT))] = _rewrite_checkpoint(module.CHECKPOINT)

    report = json.loads(module.REPORT.read_text(encoding="utf-8-sig"))
    report["protocol_id"] = "pg-pk-327-a800-replay-train-v1"
    report["schema_version"] = "pg327-a800-replay-training-report-v1"
    report["status"] = "completed_remote_a800_pg327_candidate"
    report["sources"]["base_checkpoint_dir"] = str(module.BASE_DIR.relative_to(ROOT))
    report["training"].update(
        {
            "execution_mode": "remote_a800_gpu0",
            "device": "cuda:0",
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "promotion_reason": "PG-327 strict failure-action, role-bound evidence and paired before/after replay are still incomplete",
        }
    )
    report["provenance"] = {
        "training_script_sha256": _sha256(ROOT / "scripts" / "run_pg327_a800_replay_train.py"),
        "parent_loop_sha256": _sha256(ROOT / "scripts" / "run_pg322_cross_impl_decoy_moe.py"),
        "model_impl_sha256": _sha256(ROOT / "app" / "pg295_causal_moe.py"),
        "dataset_file_sha256": _sha256(module.DATASET),
        "audit_file_sha256": _sha256(module.AUDIT),
        "checkpoint_sha256": checkpoint_hashes,
        "captured_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    report["promotion"] = {
        "training_allowed": True,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "checkpoint_role": "research_candidate_only",
        "promotion_blocked": True,
    }
    report["report_sha256"] = ""
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    module.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "device": report["training"]["device"],
                "gpu": report["training"]["gpu_name"],
                "metrics": report.get("metrics"),
                "promotion_blocked": report["promotion"]["promotion_blocked"],
                "checkpoint": str(module.CHECKPOINT.relative_to(ROOT)),
                "report": str(module.REPORT.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())

