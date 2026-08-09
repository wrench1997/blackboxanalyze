"""Render PG-237's fresh AI/reference/negative request anatomy without raw values."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "research" / "pg237_pikachu_result_fixture_replay_trace_v1.json"
OUTPUT = ROOT / "research" / "pg237_human_wire_view_v1.md"


def _field(route: str) -> str:
    lowered = route.casefold()
    return "id" if lowered.endswith("sqli_id.php") else "name"


def _shape(method: str, route: str) -> str:
    field = _field(route)
    value = "<RUNTIME_SQL_BOUND_PROBE>"
    if method == "GET":
        return f"GET <LOOPBACK_ORIGIN>{route}?{field}={value}&submit=submit"
    return "\n".join([f"POST <LOOPBACK_ORIGIN>{route}", "Content-Type: application/x-www-form-urlencoded", "", f"{field}={value}&submit=submit"])


def main() -> int:
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    rows = trace.get("results", [])
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in rows:
        key = (str(row.get("method")), str(row.get("route")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    lines = [
        "# PG-237 Pikachu AI payload wire view",
        "",
        "> 这是两个新 seed 的本地、只读、fresh-container 回放。AI candidate、独立 reference、matched negative 和结果 fixture 都实际发包；运行时值只在 loopback 容器内绑定，本文只展示可读 wire 形状和哈希前缀。",
        "",
        "## AI 参与的流程",
        "",
        "1. AI 选择 `sql_channel_class` 抽象探针。",
        "2. 控制器按已观察的 method/path/field 绑定运行时 SQL 边界 probe。",
        "3. 同一 fresh 容器分别发送 reference、negative 和只读结果 fixture。",
        "4. 只有 typed effect、结果形状、阴性干净、fresh reset、数据库健康门和证据哈希同时成立，才标为训练 candidate；不成立则 abstain。",
        "",
        "## 路由与结果",
        "",
        "| method | route | field | AI sent | typed effect | result fixture | negative | payload hash | evidence hash |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in unique:
        typed = bool((row.get("typed_oracle") or {}).get("typed_effect_confirmed"))
        fixture = bool((row.get("result_oracle") or {}).get("result_fixture_verified"))
        typed_hash = str((row.get("typed_oracle") or {}).get("evidence_hash", ""))
        result_hash = str((row.get("result_oracle") or {}).get("evidence_hash", ""))
        anatomy = dict(row.get("ai_request_anatomy") or {})
        lines.append(f"| {row.get('method')} | `{row.get('route')}` | `{_field(str(row.get('route')) )}` | {int(bool(row.get('ai_sent')))} | {int(typed)} | {int(fixture)} | {int(bool(row.get('negative_sent')))} | `{str(anatomy.get('payload_sha256',''))[:12]}` | `{(result_hash or typed_hash)[:12]}` |")
    lines.extend(["", "## 可读的 wire 形状（占位）", ""])
    for row in unique:
        method = str(row.get("method", "GET")).upper()
        route = str(row.get("route", ""))
        lines.extend([f"### {method} {route}", "", "```text", _shape(method, route), "```", ""])
    lines.extend([
        "## 解释",
        "",
        "`<RUNTIME_SQL_BOUND_PROBE>` 不是可复用的原始字符串；它代表 API 在本地发送时临时绑定的受控边界探针。结果 fixture 只用于验证本地只读记录/阴性差分，不能据此宣称任意网站存在漏洞。",
        "",
        "原始 payload、原始响应正文、数据库查询文本和秘密均未写入该视图。",
        "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(str(OUTPUT.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

