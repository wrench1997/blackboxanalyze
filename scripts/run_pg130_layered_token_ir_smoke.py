"""PG-130 local smoke audit for source -> Rule IR -> IR-token compression."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.layered_ir_tokenizer import layered_compress, validate_layered_compression


RESEARCH = ROOT / "research"
SOURCE_TRACE = RESEARCH / "pg127_key_feature_assembly_trace_v1.json"
REPORT = RESEARCH / "pg130_layered_token_ir_report_v1.json"
PROTOCOL = RESEARCH / "pg130_layered_token_ir_protocol_v1.json"
PROPOSAL = RESEARCH / "pg130_layered_token_ir_proposal_v1.json"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    trace = json.loads(SOURCE_TRACE.read_text(encoding="utf-8"))
    holdout_targets = trace["sources"]["pg127_long_holdout"]
    audited_steps = 0
    source_tokens = 0
    ir_tokens = 0
    failure_adjusted = 0
    forward_baseline = 0
    methods: set[str] = set()
    manifests: list[str] = []
    for target in holdout_targets:
        for episode in target["episodes"]:
            for step in episode["steps"]:
                method = str(step["action_manifest"]["method"]).upper()
                methods.add(method)
                html = f'<form method="{method}"><input name="abstract_probe"></form><script>fetch("local")</script>'
                javascript = 'if (document.querySelector("form")) { fetch("local"); }'
                result = layered_compress(html_snapshot=html, javascript_snapshot=javascript, action_manifests=[step["action_manifest"]], response_projection=step["response_projection"], failure_signature=step["failure_signature"])
                validate_layered_compression(result)
                audited_steps += 1
                source_tokens += sum(layer["token_count"] for layer in result["layers"]["source_token_layers"])
                ir_tokens += int(result["layers"]["ir_tokens"]["token_count"])
                phase = next(slot["value"] for slot in result["layers"]["rule_ir"]["slots"] if slot["slot_id"] == "failure.recovery_phase")
                failure_adjusted += phase == "failure_adjusted"
                forward_baseline += phase == "forward_baseline"
                manifests.append(result["manifest_sha256"])
    report = {"protocol_id": "pg-pk-130-layered-token-ir-v1", "schema_version": "pg130-layered-token-ir-report-v1", "status": "completed_pg130_layered_token_ir_smoke", "source_trace": str(SOURCE_TRACE.relative_to(ROOT)), "audited_steps": audited_steps, "source_token_total": source_tokens, "ir_token_total": ir_tokens, "average_source_tokens": round(source_tokens / audited_steps, 6), "average_ir_tokens": round(ir_tokens / audited_steps, 6), "failure_adjusted_steps": failure_adjusted, "forward_baseline_steps": forward_baseline, "methods": sorted(methods), "distinct_manifest_hashes": len(set(manifests)), "raw_source_saved": False, "raw_javascript_saved": False, "script_execution": False, "external_network": False, "oracle_authority_included": False, "training_eligible": False, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "all_steps_validated": audited_steps == 72, "get_post_covered": methods == {"GET", "POST"}, "layer_contract_passed": True, "claim_scope": "tokenization/IR compression engineering contract only; not vulnerability detection capability"}
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg-pk-130-layered-token-ir-v1", "schema_version": "pg130-layered-token-ir-protocol-v1", "objective": "将本地 HTML、GET/POST manifest 和 JavaScript 快照压缩为 source token，再压缩为 Rule IR 和 IR token。", "layers": ["html_get_post_javascript_source_tokens", "family_free_rule_ir_slots", "ir_tokens_with_slot_weights"], "safety": {"local_only": True, "raw_source_retained": False, "raw_javascript_retained": False, "script_execution": False, "external_network": False, "typed_oracle_authority_in_model_input": False, "memory_promotion_allowed": False}, "acceptance": {"fresh_trace_source": "pg127 unseen seed holdout", "required_methods": ["GET", "POST"], "required_ir_hash": True, "all_steps_validated": True}}
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-130-layered-token-ir-v1", "proposal_id": "pg130-layered-token-ir-proposal-v1", "question": "分层 source token → Rule IR → IR token 是否能在不保留原始页面/脚本的前提下覆盖 GET/POST 失败轨迹？", "prediction": {"all_steps_validated": True, "get_post_covered": True, "raw_source_saved": False, "layer_contract_passed": True}, "next": "把 IR token 接入 history-sensitive 顺序 holdout；在此之前不把 tokenization smoke 当模型能力或训练增益。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "audited_steps": audited_steps, "source_token_total": source_tokens, "ir_token_total": ir_tokens, "failure_adjusted_steps": failure_adjusted, "forward_baseline_steps": forward_baseline, "get_post_covered": methods == {"GET", "POST"}, "all_steps_validated": audited_steps == 72, "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
