"""Render a human-readable, redacted view of PG-217/218 request anatomy."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg217_pikachu_typed_sql_oracle_report_v1.json"
RESULT_REPORT = ROOT / "research" / "pg218_pikachu_result_fixture_report_v1.json"
OUTPUT = ROOT / "research" / "pg217_human_payload_view_v1.md"


def _route_kind(path: str, method: str) -> tuple[str, str]:
    lowered = path.casefold()
    if lowered.endswith("sqli_id.php"):
        return "id", "numeric syntax boundary"
    if lowered.endswith("sqli_search.php"):
        return "name", "LIKE/string syntax boundary"
    if lowered.endswith("sqli_str.php"):
        return "name", "quoted-string syntax boundary"
    if lowered.endswith("sqli_x.php"):
        return "name", "parenthesized-string syntax boundary"
    if lowered.endswith("sqli_widebyte.php"):
        return "name", "escaped/wide-byte boundary (abstain)"
    if lowered.endswith("sqli_blind_b.php"):
        return "name", "boolean-blind boundary (abstain)"
    if lowered.endswith("sqli_blind_t.php"):
        return "name", "timing channel (forbidden/abstain)"
    return "field", "unknown boundary"


def _wire_shape(method: str, path: str, field: str) -> str:
    # PG-212 uses a numeric base for the id route, while string fields use a
    # short runtime marker.  Keep both forms human-readable but redacted.
    value = "1'" if field == "id" else "<RUNTIME_CANARY>'"
    if method == "GET":
        return f"GET <LOOPBACK_ORIGIN>{path}?{field}={value}&submit=submit"
    return "\n".join([
        f"POST <LOOPBACK_ORIGIN>{path}",
        "Content-Type: application/x-www-form-urlencoded",
        "",
        f"{field}={value}&submit=submit",
    ])


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    result_report = json.loads(RESULT_REPORT.read_text(encoding="utf-8-sig"))
    rows = report["results"]
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in rows:
        key = (str(row["method"]), str(row["route"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    result_by_route = {str(row["route"]): row for row in result_report["results"]}
    lines = [
        "# PG-217/218 API payload human view",
        "",
        "> 这是本地、只读 Pikachu 回放的请求视图。`<RUNTIME_CANARY>` 代表发送时由 API 临时绑定的短 canary；原始值和响应正文不落盘。它是 SQL 输入边界检测 probe，不是数据导出/时间延迟/写入 payload。",
        "",
        "## 发送流程",
        "",
        "1. AI 根据 route 的 method/path/fields 选择 `sql_channel_class`。",
        "2. 运行时把抽象类绑定到对应字段，发送一个 syntax-boundary canary；独立 reference 再发一次同类 probe。",
        "3. 另发一个普通未知值作 negative control。",
        "4. evaluator 只比较 SQL 错误形状、negative、reference、fresh reset、数据库健康门和 SHA-256 证据。",
        "",
        "## 路由视图",
        "",
        "| 方法 | 路径 | 字段 | API 探针 | probe hash 前缀 | 本地 typed effect | 结果 fixture |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in unique:
        field, kind = _route_kind(str(row["route"]), str(row["method"]))
        anatomy = row["ai"].get("request_anatomy") or {}
        result = result_by_route.get(str(row["route"]), {})
        lines.append("| {method} | `{route}` | `{field}` | `{kind}` | `{probe}` | {typed} | {fixture} |".format(
            method=row["method"], route=row["route"], field=field, kind=kind,
            probe=str(anatomy.get("probe_sha256", ""))[:12] or "n/a",
            typed="yes" if row["typed_oracle"].get("typed_effect_confirmed") else "abstain",
            fixture="verified" if result.get("result_oracle", {}).get("result_fixture_verified") else "not used",
        ))
    lines.extend(["", "## 可读的 wire 形状（占位显示）", ""])
    for row in unique:
        field, _ = _route_kind(str(row["route"]), str(row["method"]))
        lines.extend([f"### {row['method']} {row['route']}", "", "```text", _wire_shape(str(row["method"]), str(row["route"]), field), "```", ""])
    lines.extend([
        "## 结果怎么读",
        "",
        f"PG-217 共 {report['counts']['confirmed_positive_count']}/{report['counts']['episode_count']} 个 route episodes 通过本地 typed input-boundary oracle；PG-218 中 {result_report['counts']['result_fixture_verified_count']} 个同时通过只读已知记录/负对照。",
        "",
        "`blind_b`、`blind_t` 和 `widebyte` 目前显示 abstain：不是说它们不存在问题，而是当前安全、非时间、非写入 oracle 没有足够证据。",
        "",
        "原始 probe、原始响应、SQL 查询文本和可执行利用字符串均不写入 catalog；报告只保留 request anatomy、projection hash、probe/payload hash 和 evidence hash。",
        "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(str(OUTPUT.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
