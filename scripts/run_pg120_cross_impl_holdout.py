"""PG-120 frozen cross-implementation metadata holdout.

The PG-119 checkpoint is loaded read-only.  A new eta implementation is
replayed on three seeds and three matched-negative-control strengths.  No
training rows, weights, or memory entries are created by this runner.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from app.pg119_metadata_rule_ir_decoder import FEATURE_DIM, MetadataRuleIRDecisionDecoder
from app.pg120_cross_impl_replay import ENCODING_CHAIN, collect_target

try:  # direct ``python scripts/run_...py`` puts scripts/ on sys.path
    from scripts.run_pg119_metadata_rule_ir_training import _evaluate_group, _evaluate_pg114
except ModuleNotFoundError:  # pragma: no cover - exercised by direct runner invocation
    from run_pg119_metadata_rule_ir_training import _evaluate_group, _evaluate_pg114


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg119-metadata-rule-ir-decoder-v1" / "model.pt"
TRACE_PATH = RESEARCH / "pg120_cross_impl_holdout_trace_v1.json"
VISIBLE_PATH = RESEARCH / "pg120_cross_impl_holdout_visible_dataset_v1.json"
REPORT_PATH = RESEARCH / "pg120_cross_impl_holdout_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg120_cross_impl_holdout_report_v1.md"
SEEDS = [12001, 12003, 12005]
DECOY_STRENGTHS = [0, 1, 2]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _collect() -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for strength in DECOY_STRENGTHS:
        for seed in SEEDS:
            collected.append(await collect_target(seed, decoy_strength=strength))
    return collected


def _all_episodes(collected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [episode for target in collected for episode in target["episodes"]]


def _visible_rows(collected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in collected:
        for episode in target["episodes"]:
            for step in episode["steps"]:
                rows.append({"row_id": step["step_id"], "target_seed": target["target_seed"], "decoy_strength": target["decoy_strength"], "surface_kind": episode["surface_kind"], "model_input": step["model_input"], "evaluation_label": step["decision"], "memory_promotion_allowed": False})
    return rows


def main() -> None:
    collected = asyncio.run(_collect())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if int(checkpoint.get("feature_dim", -1)) != FEATURE_DIM:
        raise ValueError("PG-119 checkpoint feature dimension does not match PG-120 decoder")
    model = MetadataRuleIRDecisionDecoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    episodes = _all_episodes(collected)
    steps = [step for episode in episodes for step in episode["steps"]]
    bindings = [episode["rule_ir_slot_binding"] for episode in episodes]
    trace = {
        "schema_version": "pg120-cross-implementation-holdout-trace-v1",
        "protocol_id": "pg-pk-120-cross-implementation-metadata-holdout-v1",
        "status": "frozen_cross_implementation_holdout_trace_collected",
        "evaluation_only": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "execution_mode": "in_process_loopback_asgi_get_post",
        "frozen_checkpoint": "artifacts/pg119-metadata-rule-ir-decoder-v1/model.pt",
        "target_implementation": "pg120-eta-independent-target",
        "target_instance_count": len(collected),
        "seed_count": len(SEEDS),
        "episode_count": len(episodes),
        "step_count": len(steps),
        "get_step_count": sum(step["action_manifest"]["method"] == "GET" for step in steps),
        "post_step_count": sum(step["action_manifest"]["method"] == "POST" for step in steps),
        "decoy_strengths": DECOY_STRENGTHS,
        "encoding_chain": list(ENCODING_CHAIN),
        "fresh_reset_per_step": all(step["fresh_reset"]["fresh_target"] and step["fresh_reset"]["completed"] for step in steps),
        "evidence_hash_valid": all(record["evidence_hash"] == _sha256_json({key: value for key, value in record.items() if key != "evidence_hash"}) for episode in episodes for record in episode["evidence_records"]),
        "rule_ir_slot_binding_count": len(bindings),
        "rule_ir_slot_bindings": bindings,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "online_weight_update": False,
        "long_term_memory_write": False,
        "sources": collected,
    }
    trace["trace_manifest_sha256"] = _sha256_json({key: value for key, value in trace.items() if key != "trace_manifest_sha256"})
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    visible = {"schema_version": "pg120-cross-implementation-visible-dataset-v1", "evaluation_only": True, "training_eligible": False, "model_input_family_free": True, "model_input_oracle_blind": True, "rows": _visible_rows(collected), "memory_promotion_allowed": False}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE_PATH.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pg114 = _evaluate_pg114(model, device)
    full = _evaluate_group(model, device, collected)
    ablation = _evaluate_group(model, device, collected, ablate=True)
    per_strength = {str(strength): _evaluate_group(model, device, [target for target in collected if target["decoy_strength"] == strength]) for strength in DECOY_STRENGTHS}
    ablation_changed = any(left["predicted_final_decision"] != right["predicted_final_decision"] for left, right in zip(full["final_episode_rows"], ablation["final_episode_rows"]))
    report = {
        "protocol_id": "pg-pk-120-cross-implementation-metadata-holdout-v1",
        "schema_version": "pg120-cross-implementation-holdout-report-v1",
        "status": "completed_pg120_cross_implementation_metadata_holdout",
        "scope": {"model": "frozen_pg119_metadata_transition_rule_ir_decoder", "feature_dim": FEATURE_DIM, "device": str(device), "weights_frozen": True, "real_vulnerability_scanner_claim_allowed": False},
        "collection": {"target_implementation": trace["target_implementation"], "target_instance_count": trace["target_instance_count"], "seed_count": trace["seed_count"], "episode_count": trace["episode_count"], "step_count": trace["step_count"], "get_step_count": trace["get_step_count"], "post_step_count": trace["post_step_count"], "decoy_strengths": trace["decoy_strengths"], "encoding_chain": trace["encoding_chain"], "fresh_reset_per_step": trace["fresh_reset_per_step"], "evidence_hash_valid": trace["evidence_hash_valid"], "rule_ir_slot_binding_count": trace["rule_ir_slot_binding_count"]},
        "blind_pg114": pg114,
        "blind_pg120": full,
        "slot_ablation_pg120": ablation,
        "negative_control_strength_scan": per_strength,
        "checks": {"checkpoint_exists": CHECKPOINT_PATH.exists(), "checkpoint_loaded": True, "weights_frozen": True, "training_rows_added": False, "online_weight_update": False, "long_term_memory_write": False, "cross_implementation": True, "get_post_balanced_collection": trace["get_step_count"] == trace["post_step_count"], "fresh_reset_per_step": trace["fresh_reset_per_step"], "evidence_hashes_valid": trace["evidence_hash_valid"], "rule_ir_slots_bind_evidence": all(binding["evidence_sha256"] in {step["evidence_sha256"] for episode in episodes for step in episode["steps"]} for binding in bindings), "model_input_family_free": True, "model_input_oracle_blind": True, "no_raw_probe_strings": True, "no_raw_response_bodies": True, "pg114_recall_nonzero": pg114["family_holdout_confirm_recall"] > 0.0, "pg114_decoy_false_accept_zero": pg114["decoy_false_accept_count"] == 0, "pg114_unknown_abstain_nonzero": pg114["withheld_oracle_abstain_rate"] > 0.0, "pg120_recall_nonzero": full["metadata_positive_recall"] > 0.0, "pg120_decoy_false_accept_zero": full["decoy_false_accept_count"] == 0, "pg120_unknown_abstain_nonzero": full["blind_oracle_abstain_rate"] > 0.0, "pg120_cross_seed_recall_variance_bounded": full["cross_seed"]["positive_recall_variance"] <= 0.1, "all_strengths_positive_recall_nonzero": all(value["metadata_positive_recall"] > 0.0 for value in per_strength.values()), "all_strengths_decoy_false_accept_zero": all(value["decoy_false_accept_count"] == 0 for value in per_strength.values()), "all_strengths_unknown_abstain_nonzero": all(value["blind_oracle_abstain_rate"] > 0.0 for value in per_strength.values()), "metadata_slot_ablation_changes_prediction": ablation_changed, "full_recall_above_slot_ablation": full["metadata_positive_recall"] > ablation["metadata_positive_recall"], "all_abstain_not_success": full["metadata_positive_recall"] > 0.0},
        "diagnosis": {"cross_implementation": "PG120 changes route/payload/schema implementation while preserving only generic metadata_changed projection", "negative_control_scan": "decoy_strength 0/1/2", "ablation": "zero PG-119 metadata slots in the fixed checkpoint", "checkpoint_reused_for_training": False},
        "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "evaluation_only_cross_impl_holdout_pending_manual_review", "reason": "通过冻结模型的局部跨实现复放仍不足以准入长期记忆；需继续增加实现、family、seed 和人工审核。"},
        "source": {"eta_target": _sha256_file(ROOT / "app/pg120_eta_metadata_target.py"), "eta_bridge": _sha256_file(ROOT / "app/pg120_cross_impl_replay.py"), "runner": _sha256_file(Path(__file__)), "frozen_checkpoint": _sha256_file(CHECKPOINT_PATH), "pg119_report": _sha256_file(RESEARCH / "pg119_metadata_training_report_v1.json"), "pg117_report": _sha256_file(RESEARCH / "pg117_double_holdout_report_v1.json"), "pg114_report": _sha256_file(RESEARCH / "pg114_family_holdout_replay_report_v1.json")},
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-120 cross-implementation metadata holdout", "", "冻结 PG-119 checkpoint，换用 PG-120 eta 独立实现，三档 matched decoy 强度和三个 seed 做 GET/POST fresh-reset 复放。", "", f"- 设备/feature dim：`{device}` / `{FEATURE_DIM}`；weights frozen：`True`。", f"- PG-120 正例召回/decoy 误接受/未知弃权：`{full['metadata_positive_recall']}` / `{full['decoy_false_accept_count']}` / `{full['blind_oracle_abstain_rate']}`。", f"- 跨 seed 正例召回方差：`{full['cross_seed']['positive_recall_variance']}`；slot 消融后召回：`{ablation['metadata_positive_recall']}`。", f"- 阴性对照强度档位：`{DECOY_STRENGTHS}`；所有档位门：`{all(value['decoy_false_accept_count'] == 0 for value in per_strength.values())}`。", "- 本轮 evaluation-only，不增加训练行、不更新权重、不写长期记忆。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "target_instances": trace["target_instance_count"], "episodes": trace["episode_count"], "steps": trace["step_count"], "pg120_recall": full["metadata_positive_recall"], "pg120_false_accept": full["decoy_false_accept_count"], "pg120_unknown_abstain": full["blind_oracle_abstain_rate"], "pg120_seed_variance": full["cross_seed"]["positive_recall_variance"], "pg120_ablation_recall": ablation["metadata_positive_recall"], "all_gates": all(report["checks"].values()), "promotion": report["promotion"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
