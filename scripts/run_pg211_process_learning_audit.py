"""PG-211: audit whether the AI learned the request/feedback process.

PG-210 already proved that an AI-selected candidate was sent to fresh local
Pikachu containers.  That is not the same as learning: a frozen adapter can
emit the same action for every route and still pass a matched reference test.
This audit is intentionally causal and report-only.  It checks the persisted
send gate, route/field binding, decision diversity, and feedback/weight
dependency.  It never reconstructs or stores a probe string or response body.

The expected result for the current checkpoint is ``attached_but_not_learned``:
the request path is real and hash-bound, while online policy updates and
evaluator-dependent decisions are absent.  A later training run must make the
negative fields in this report change before it can claim process learning.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
REPORT_IN = ROOT / "research" / "pg210_ai_reference_payload_validation_report_v1.json"
VIEW_IN = ROOT / "research" / "pg210_request_anatomy_view_v1.json"
REPORT_OUT = ROOT / "research" / "pg211_process_learning_audit_report_v1.json"
PROTOCOL_OUT = ROOT / "research" / "pg211_process_learning_audit_protocol_v1.json"
MARKDOWN_OUT = ROOT / "research" / "pg211_process_learning_audit_report_v1.md"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _request_matches_episode(episode: Mapping[str, Any]) -> bool:
    request = dict((episode.get("ai") or {}).get("request") or {})
    if not request or not bool((episode.get("ai") or {}).get("sent")):
        return False
    return (
        request.get("method") == episode.get("method")
        and request.get("path") == episode.get("path")
        and request.get("placement") == ("query" if episode.get("method") == "GET" else "form")
        and sorted(request.get("field_names") or []) == sorted(episode.get("fields") or [])
        and bool(request.get("binding_sha256"))
        and bool(request.get("payload_sha256"))
        and bool(request.get("probe_sha256"))
    )


def audit(report: Mapping[str, Any], view: Mapping[str, Any]) -> dict[str, Any]:
    episodes = list(report.get("episodes") or [])
    sent = [row for row in episodes if bool((row.get("ai") or {}).get("sent"))]
    decisions = [dict((row.get("ai") or {}).get("model_decision") or {}) for row in episodes]
    signatures = [
        (
            str(decision.get("effective_action", "")),
            str(decision.get("encoding", "")),
            str(decision.get("failure", "")),
        )
        for decision in decisions
    ]
    feedback = [dict((row.get("ai") or {}).get("ai_feedback") or {}) for row in episodes]
    request_rows = list(view.get("rows") or [])
    bound_request_count = sum(int(_request_matches_episode(row)) for row in episodes)
    view_bound_count = sum(
        int(bool(row.get("ai_sent")) and bool((row.get("ai_request") or {}).get("binding_sha256")))
        for row in request_rows
    )
    policy_uses_evaluator = sum(int(bool(row.get("policy_uses_evaluator"))) for row in feedback)
    decision_depends_on_history = sum(
        int(bool(decision.get("history_dependent"))) for decision in decisions
    )
    # PG-210 does not persist a history token, so the absence is measured
    # explicitly rather than inferred from a successful oracle outcome.
    history_fields = sum(int("history_len" in (decision.get("features") or {})) for decision in decisions)
    online_update = bool((report.get("model") or {}).get("online_weight_update", False))
    unique_candidate_ids = len({str((row.get("ai") or {}).get("candidate_id", "")) for row in sent})
    route_count = len({(row.get("method"), row.get("path"), tuple(row.get("fields") or [])) for row in episodes})
    all_effect_agree = all(bool(row.get("ai_reference_effect_agreement")) for row in episodes) if episodes else False

    attached = bool(
        sent
        and bound_request_count == len(sent)
        and view_bound_count == len(sent)
        and all_effect_agree
    )
    learned = bool(
        attached
        and online_update
        and policy_uses_evaluator > 0
        and history_fields == len(decisions)
        and len(set(signatures)) > 1
    )
    status = "learned_process" if learned else ("attached_but_not_learned" if attached else "send_binding_incomplete")
    return {
        "protocol_id": "pg-pk-211-process-learning-audit-v1",
        "schema_version": "pg211-process-learning-audit-report-v1",
        "status": status,
        "source_report": str(REPORT_IN.relative_to(ROOT)),
        "source_request_view": str(VIEW_IN.relative_to(ROOT)),
        "counts": {
            "episode_count": len(episodes),
            "route_count": route_count,
            "ai_sent_count": len(sent),
            "route_request_binding_count": bound_request_count,
            "request_view_binding_count": view_bound_count,
            "independent_effect_agreement_count": sum(int(bool(row.get("ai_reference_effect_agreement"))) for row in episodes),
            "unique_model_decision_signature_count": len(set(signatures)),
            "unique_ai_candidate_id_count": unique_candidate_ids,
            "feedback_policy_uses_evaluator_count": policy_uses_evaluator,
            "history_dependent_decision_count": decision_depends_on_history,
            "history_feature_present_count": history_fields,
        },
        "model": {
            "variant": str((report.get("model") or {}).get("variant", "unknown")),
            "base_parameter_count": int((report.get("model") or {}).get("base_parameter_count", 0) or 0),
            "online_weight_update": online_update,
        },
        "decision_signature_histogram": [
            {"effective_action": action, "encoding": encoding, "failure": failure, "count": count}
            for (action, encoding, failure), count in sorted(Counter(signatures).items())
        ],
        "gates": {
            "real_ai_request_proven": attached,
            "process_learning_proven": learned,
            "matched_reference_only_is_insufficient": True,
            "requires_online_update": True,
            "requires_evaluator_feedback_dependency": True,
            "requires_history_features": True,
            "requires_decision_diversity_or_counterfactual_gain": True,
        },
        "conclusion": (
            "AI 的请求已经真实发出并且与路由/字段/哈希绑定；当前 checkpoint 仍是冻结决策头，"
            "反馈没有进入策略，不能声称模型从失败中学会了下一步。"
            if not learned
            else "AI 的过程学习门通过。"
        ),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


def main() -> int:
    report = _load(REPORT_IN)
    view = _load(VIEW_IN)
    result = audit(report, view)
    result["report_sha256"] = _digest(result)
    REPORT_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "protocol_id": result["protocol_id"],
        "schema_version": "pg211-process-learning-audit-protocol-v1",
        "input_reports": [str(REPORT_IN.relative_to(ROOT)), str(VIEW_IN.relative_to(ROOT))],
        "real_send_gate": ["ai.sent", "method/path/placement/fields match", "binding_sha256", "payload_sha256", "probe_sha256"],
        "learning_gate": ["online_weight_update", "policy_uses_evaluator", "history feature", "decision diversity or counterfactual gain"],
        "matched_reference_does_not_equal_learning": True,
        "raw_payload_and_response_excluded": True,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    PROTOCOL_OUT.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = result["counts"]
    lines = [
        "# PG-211 process learning audit",
        "",
        f"status={result['status']}; AI 实际发包={counts['ai_sent_count']}; 路由/字段/哈希绑定={counts['route_request_binding_count']}; 独立参考一致={counts['independent_effect_agreement_count']}",
        f"决策签名种类={counts['unique_model_decision_signature_count']}; evaluator 反馈回流={counts['feedback_policy_uses_evaluator_count']}; history 特征={counts['history_feature_present_count']}; online update={result['model']['online_weight_update']}",
        "",
        result["conclusion"],
        "",
        "本审计只保存计数、决策签名和哈希；不保存原始 payload 或响应正文。",
        "",
    ]
    MARKDOWN_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["status"], "counts": counts, "report": str(REPORT_OUT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
