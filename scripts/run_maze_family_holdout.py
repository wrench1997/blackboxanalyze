"""Run the safe family-holdout protocol for the rule-maze oracle layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.dom_oracle import run_dom_oracle
from app.maze_labs import public_maze_labs
from app.maze_solver import assess_rule_exit
from app.sql_ast_oracle import run_sql_ast_oracle


ROOT = Path(__file__).resolve().parents[1]
JSON_OUTPUT = ROOT / "research" / "maze_lab_holdout_v1.json"
MARKDOWN_OUTPUT = ROOT / "research" / "maze_lab_holdout_v1.md"
TRAINING_FAMILIES = {"access_control", "url_redirect", "logic"}
HOLDOUT_FAMILIES = {"xss", "injection"}


def _training_evidence(family: str) -> dict[str, Any]:
    if family == "access_control":
        return {"candidate_signal": True, "protected_resource_transition": True, "session_signal": True}
    if family == "url_redirect":
        return {"candidate_signal": True, "location_origin_changed": True}
    return {"candidate_signal": True, "invariant_violation": True, "state_replay": True}


def _holdout_evidence(lab_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if lab_id in {"maze_dom_sink", "maze_dom_double_decode", "maze_dom_template_sink"}:
        value = '<span data-sift-marker="sift-marker">inert</span>'
        if lab_id == "maze_dom_double_decode":
            value = "&amp;lt;span data-sift-marker=\"sift-marker\"&amp;gt;inert&amp;lt;/span&amp;gt;"
            transforms = ["html_entity_decode", "html_entity_decode"]
        else:
            transforms = []
        sink = "template.innerHTML" if lab_id == "maze_dom_template_sink" else "innerHTML"
        first = run_dom_oracle(value, sink=sink, transforms=transforms, marker="sift-marker").to_dict()
        second = run_dom_oracle(value, sink=sink, transforms=transforms, marker="sift-marker").to_dict()
        return first, second
    class_by_lab = {
        "maze_sql_boundary": "operator_like",
        "maze_sql_error_channel": "syntax_error",
        "maze_sql_blind_channel": "blind_boolean",
        "maze_sql_timing_channel": "time_delay",
        "maze_sql_local_side_channel": "local_side_channel",
    }
    fragment_class = class_by_lab.get(lab_id, "subquery_like")
    first = run_sql_ast_oracle(fragment_class).to_dict()["evidence"]
    second = run_sql_ast_oracle(fragment_class).to_dict()["evidence"]
    return first, second


def _safety_ok(evidence: dict[str, Any]) -> bool:
    return all(not bool(evidence.get(key, False)) for key in (
        "script_execution",
        "network_access",
        "navigation",
        "database_touched",
        "real_sleep_performed",
    ))


def main() -> None:
    rows: list[dict[str, Any]] = []
    for lab in public_maze_labs():
        family = str(lab["family"])
        split = "holdout" if family in HOLDOUT_FAMILIES else "training_contract"
        if split == "holdout":
            evidence, recheck = _holdout_evidence(str(lab["id"]))
            result = assess_rule_exit(family, visible_evidence=evidence, rechecks=[recheck])
        else:
            evidence = _training_evidence(family)
            result = assess_rule_exit(family, visible_evidence=evidence, rechecks=[dict(evidence)])
            recheck = dict(evidence)
        rows.append({
            "lab_id": lab["id"],
            "scenario_id": lab["scenario_id"],
            "family": family,
            "split": split,
            "exit_oracle": lab["exit_oracle"],
            "status": result["status"],
            "observable": result["observable"],
            "evaluator_confirmed": result["evaluator_confirmed"],
            "recheck_count": result["recheck_count"],
            "safety_ok": _safety_ok(evidence) and _safety_ok(recheck),
            "modality": evidence.get("modality", "semantic_contract"),
            "evidence": evidence,
        })

    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    modalities = sorted({str(row["modality"]) for row in holdout_rows})
    report = {
        "protocol": "sift-rule-maze-family-holdout-v1",
        "training_families": sorted(TRAINING_FAMILIES),
        "holdout_families": sorted(HOLDOUT_FAMILIES),
        "policy_labels_visible": False,
        "database_executed": False,
        "browser_script_executed": False,
        "external_network": False,
        "rows": rows,
        "summary": {
            "lab_count": len(rows),
            "holdout_lab_count": len(holdout_rows),
            "holdout_observable_success": sum(bool(row["observable"]) for row in holdout_rows),
            "holdout_evaluator_confirmed": sum(bool(row["evaluator_confirmed"]) for row in holdout_rows),
            "holdout_safety_pass": all(bool(row["safety_ok"]) for row in holdout_rows),
            "holdout_modalities": modalities,
        },
        "interpretation": "这是 oracle/协议层族外测试，不是神经模型泛化分数；evaluator_confirmed 必须由 fresh target 外部提供。",
    }
    JSON_OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    lines = [
        "# 规则迷宫族外读出 v1",
        "",
        "这是 oracle/协议层测试，不是模型分数。策略未看到漏洞族标签，靶场 evaluator 也没有被伪造。",
        "",
        f"- 训练契约族：`{', '.join(sorted(TRAINING_FAMILIES))}`",
        f"- 族外族：`{', '.join(sorted(HOLDOUT_FAMILIES))}`",
        f"- 族外靶场：`{summary['holdout_lab_count']}`，可观察出口：`{summary['holdout_observable_success']}`",
        f"- evaluator 确认：`{summary['holdout_evaluator_confirmed']}`（预期为 0）",
        f"- 安全门禁：`{'PASS' if summary['holdout_safety_pass'] else 'FAIL'}`",
        f"- 观测通道：`{', '.join(summary['holdout_modalities'])}`",
        "",
        "SQL 的 error、blind/row-shape、bounded timing 都是确定性模拟标记；没有数据库执行、网络访问或真实 sleep。",
        "DOM 的 sink/DOM 差分在浏览器端使用 detached node，在 Python 端使用 HTMLParser 复核；没有脚本执行。",
    ]
    MARKDOWN_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "json": str(JSON_OUTPUT.relative_to(ROOT)), "markdown": str(MARKDOWN_OUTPUT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
