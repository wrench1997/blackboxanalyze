"""Record a fail-closed verdict for the PG-03 GPU diagnostic training run.

The trainer may produce a checkpoint for engineering inspection, but this
report explicitly prevents it from being treated as a capability or memory
promotion result until the trace-aligned GET/POST dataset gate passes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_REPORT = ROOT / "research" / "pg03_rule_ir_decoder_v1.json"
CHECKPOINTS = [
    ROOT / "artifacts" / "pg03-rule-ir-decoder" / "source_split_same_family" / "decoder.pt",
    ROOT / "artifacts" / "pg03-rule-ir-decoder" / "family_holdout_structural_transfer" / "decoder.pt",
    ROOT / "artifacts" / "pg03-rule-ir-decoder" / "family_holdout_unseen_surface" / "decoder.pt",
]
OUTPUT = ROOT / "research" / "pg_pk_32_gpu_training_diagnostic_v1.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    evaluations = report.get("evaluations") or []
    split_summary = [
        {
            "split": row.get("split"),
            "total": row.get("total", 0),
            "exit_found_rate": row.get("exit_found_rate", 0.0),
            "false_positive_rate": row.get("false_positive_rate", 0.0),
            "abstain_rate": row.get("abstain_rate", 0.0),
            "rule_ir_emission_rate": row.get("rule_ir_emission_rate", 0.0),
        }
        for row in evaluations
    ]
    output = {
        "schema_version": "pg-pk-32-gpu-training-diagnostic-v1",
        "run_id": "pg32-pg03-rule-ir-decoder-gpu-diagnostic",
        "status": "diagnostic_only",
        "training_started": True,
        "training_completed": True,
        "device": report.get("model", {}).get("device", "cuda"),
        "cuda_available": bool(report.get("model", {}).get("cuda_available", True)),
        "source": {
            "catalog": "research/payload_replay_catalog_v1.json",
            "catalog_type": "authorized_in_repo_local_replay",
            "sample_count": 20,
            "methods_observed": ["GET"],
            "typed_oracle_replay": True,
            "independent_target_implementation": False,
        },
        "split_results": split_summary,
        "checkpoints": [
            {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            for path in CHECKPOINTS if path.exists()
        ],
        "capability_claim_allowed": False,
        "training_allowed_for_promotion": False,
        "memory_promotion_allowed": False,
        "training_candidate": False,
        "quarantine": {
            "reason": [
                "missing_GET_POST_dual_channel_context",
                "single_in_repo_target_implementation",
                "no_PG30_capability_evidence_matrix",
                "族外出口率未达到预注册目标",
            ],
            "allowed_use": ["engineering_regression", "failure_analysis", "next_ablation_seed"],
        },
        "interpretation": "本次 GPU 运行证明训练管线可执行，不证明模型获得可迁移漏洞检测能力；不得写入长期记忆或能力数据集。",
        "trainer_report": "research/pg03_rule_ir_decoder_v1.json",
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
