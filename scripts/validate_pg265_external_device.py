# -*- coding: utf-8 -*-
"""Fail-closed validation for the explicitly selected remote training GPU."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_rows() -> list[dict[str, str]]:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4:
            rows.append({"index": fields[0], "name": fields[1], "memory_total_mib": fields[2], "memory_used_mib": fields[3]})
    return rows


def main() -> int:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_rows = _gpu_rows()
    dataset = RESEARCH / "pg264_pikachu_growth_collection_dataset_v1.json"
    audit = RESEARCH / "pg264_pikachu_growth_collection_audit_v1.json"
    artifact = ROOT / "artifacts" / "pg249-pikachu-route-seed-capacity-v1" / "frozen_xxl_capacity_hidden4096.pt"
    audit_payload = json.loads(audit.read_text(encoding="utf-8")) if audit.exists() else {}
    checks = {
        "explicit_single_gpu_selection": visible == "0",
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "one_visible_device": int(torch.cuda.device_count()) == 1,
        "selected_device_is_a800": bool(gpu_rows and "A800" in gpu_rows[0]["name"]),
        "selected_device_vram_ge_16gib": bool(gpu_rows and int(gpu_rows[0]["memory_total_mib"]) >= 16384),
        "pg264_audit_complete": bool(audit_payload.get("all_required_fields_complete")) and int(audit_payload.get("audited_record_count", 0) or 0) == 32,
        "dataset_present": dataset.exists(),
        "frozen_artifact_present": artifact.exists(),
    }
    result = {
        "schema_version": "pg265-external-device-validation-v1",
        "status": "ready" if all(checks.values()) else "blocked",
        "visible_devices": visible,
        "torch": {"version": torch.__version__, "cuda_version": getattr(torch.version, "cuda", None), "device_count": int(torch.cuda.device_count())},
        "gpu_rows": gpu_rows,
        "checks": checks,
        "artifacts": {"dataset_sha256": _sha256(dataset) if dataset.exists() else "", "audit_sha256": _sha256(audit) if audit.exists() else "", "frozen_artifact_sha256": _sha256(artifact) if artifact.exists() else ""},
        "other_gpu_mutation_allowed": False,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
    }
    (RESEARCH / "pg265_external_device_validation_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())

