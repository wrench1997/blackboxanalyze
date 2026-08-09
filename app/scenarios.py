from __future__ import annotations

from copy import deepcopy
from typing import Any


def c(value: Any) -> dict[str, Any]:
    return {"op": "const", "value": value}


def f(path: str) -> dict[str, Any]:
    return {"op": "field", "path": path}


def prev(path: str, offset: int = 1) -> dict[str, Any]:
    return {"op": "prev", "path": path, "offset": offset}


def binop(op: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": op, "left": left, "right": right}


def logic(op: str, *args: dict[str, Any]) -> dict[str, Any]:
    return {"op": op, "args": list(args)}


SCENARIOS: dict[str, dict[str, Any]] = {
    "js_truthy_access": {
        "id": "js_truthy_access",
        "name": "Truthy Access Gate",
        "description": "研究 JavaScript truthy/falsy 语义如何把数值配置意外提升为授权条件。",
        "category": "类型混淆 · 授权绕过",
        "severity": "HIGH",
        "cwe": "CWE-863",
        "research_question": "仅凭黑盒判定结果，模型能否识别 quota 被当作布尔授权位，并找到负数同样放行的反例？",
        "hypothesis": "主动选择 0、负数与正数边界，比均匀随机采样更快区分严格角色检查与 truthiness 缺陷。",
        "game_rule": "守门人应只让管理员通行；但计分板 quota 只要不是 0，也会被误当成管理员徽章。研究目标是用最少询问还原这条隐藏放行规则。",
        "intended_rule": "role === 'admin'",
        "js_source": "export function canEnter({ role, quota }) {\n  // BUG: quota is a number, not an authorization decision.\n  const allowed = role === 'admin' || quota;\n  return Boolean(allowed);\n}",
        "tags": ["truthiness", "access-control", "boundary"],
        "fields": [
            {"path": "input.role", "label": "角色", "type": "enum", "domain": ["guest", "member", "admin"]},
            {"path": "input.quota", "label": "剩余额度", "type": "number", "domain": [-2, -1, 0, 1, 2, 8]},
        ],
        "hidden_rule": logic(
            "or",
            binop("eq", f("input.role"), c("admin")),
            logic("or", binop("lt", f("input.quota"), c(0)), binop("gt", f("input.quota"), c(0))),
        ),
    },
    "js_substring_redirect": {
        "id": "js_substring_redirect",
        "name": "Substring Redirect Guard",
        "description": "研究字符串包含检查造成的开放重定向与可信域伪装。",
        "category": "字符串匹配 · 开放重定向",
        "severity": "HIGH",
        "cwe": "CWE-601",
        "research_question": "模型能否从少量 URL 判定中区分 hostname 校验与字符串包含校验？",
        "hypothesis": "带前后缀的对抗域名能显著降低错误规则的行为等价类数量。",
        "game_rule": "传送门只应通往 trusted.com；有缺陷的守卫只检查卷轴上是否出现这串文字，攻击者可以把它藏进恶意域名。",
        "intended_rule": "new URL(next).hostname === 'trusted.com'",
        "js_source": "export function allowRedirect(next) {\n  // BUG: substring presence is not hostname equality.\n  return next.includes('trusted.com');\n}",
        "tags": ["redirect", "substring", "adversarial-input"],
        "fields": [
            {"path": "input.next", "label": "跳转地址", "type": "string", "domain": [
                "https://trusted.com/home",
                "https://trusted.com.evil.test/phish",
                "https://evil.test/?next=trusted.com",
                "https://evil.test/trusted.com",
                "https://example.com/home",
                "trusted.com@evil.test",
            ]},
        ],
        "hidden_rule": binop("contains", f("input.next"), c("trusted.com")),
    },
    "js_boundary_coupon": {
        "id": "js_boundary_coupon",
        "name": "Coupon Boundary Drift",
        "description": "研究严格大于与大于等于之间的单点边界偏移。",
        "category": "边界条件 · 业务逻辑",
        "severity": "MEDIUM",
        "cwe": "CWE-193",
        "research_question": "在极少查询预算下，主动实验能否稳定命中 total=100 的唯一行为差异？",
        "hypothesis": "候选最大分歧采样应优先于随机采样发现窄边界缺陷。",
        "game_rule": "满 100 分应获得奖励券；错误裁判只在超过 100 分时发券。唯一关键反例恰好位于 100 分。",
        "intended_rule": "member && total >= 100",
        "js_source": "export function issueCoupon({ member, total }) {\n  // BUG: the product requirement says total >= 100.\n  return member && total > 100;\n}",
        "tags": ["off-by-one", "business-logic", "active-learning"],
        "fields": [
            {"path": "input.member", "label": "会员", "type": "bool", "domain": [False, True]},
            {"path": "input.total", "label": "订单金额", "type": "number", "domain": [0, 1, 99, 100, 101, 150]},
        ],
        "hidden_rule": logic(
            "and",
            binop("eq", f("input.member"), c(True)),
            binop("gt", f("input.total"), c(100)),
        ),
    },
    "js_sequence_replay": {
        "id": "js_sequence_replay",
        "name": "Sequence Replay Window",
        "description": "研究仅检查上一步动作、却未绑定会话状态的序列授权缺陷。",
        "category": "历史依赖 · 流程绕过",
        "severity": "CRITICAL",
        "cwe": "CWE-294",
        "research_question": "引入一阶历史后，模型能否发现 commit 的许可来自前一步 verify，而不是当前可见状态？",
        "hypothesis": "Episode 隔离能消除跨会话污染，并使历史依赖规则可被准确归纳。",
        "game_rule": "玩家只要上一回合喊过 verify，本回合 commit 就成功；守卫没有把验证绑定到挑战或身份。",
        "intended_rule": "verifiedChallenge === currentChallenge && action === 'commit'",
        "js_source": "let previousAction;\nexport function step(action) {\n  const accepted = action === 'commit' && previousAction === 'verify';\n  previousAction = action; // BUG: no challenge/session binding.\n  return accepted;\n}",
        "tags": ["stateful", "replay", "history"],
        "fields": [
            {"path": "input.action", "label": "动作", "type": "enum", "domain": ["wait", "verify", "commit", "cancel"]},
        ],
        "hidden_rule": logic(
            "and",
            binop("eq", f("input.action"), c("commit")),
            binop("eq", prev("input.action", 1), c("verify")),
        ),
        "stateful": True,
        "validation_cases": [
            {"history": [], "input": {"action": "commit"}},
            {"history": [{"input": {"action": "verify"}, "context": {}, "state": {}, "output": False}], "input": {"action": "commit"}},
            {"history": [{"input": {"action": "wait"}, "context": {}, "state": {}, "output": False}], "input": {"action": "commit"}},
            {"history": [{"input": {"action": "verify"}, "context": {}, "state": {}, "output": False}], "input": {"action": "cancel"}},
        ],
    },
    "parity_color": {
        "id": "parity_color",
        "name": "颜色与奇偶门",
        "description": "输入包含颜色与数字。隐藏规则同时使用枚举条件和算术取模。",
        "fields": [
            {"path": "input.color", "label": "颜色", "type": "enum", "domain": ["red", "blue", "green"]},
            {"path": "input.number", "label": "数字", "type": "number", "domain": list(range(0, 13))},
        ],
        "hidden_rule": logic(
            "and",
            binop("eq", f("input.color"), c("red")),
            binop("eq", binop("mod", f("input.number"), c(2)), c(0)),
        ),
    },
    "access_gate": {
        "id": "access_gate",
        "name": "上下文访问控制",
        "description": "规则横跨 input 与 context，适合验证通用路径和 OR/AND 组合。",
        "fields": [
            {"path": "context.age", "label": "年龄", "type": "number", "domain": [15, 17, 18, 20, 30, 60]},
            {"path": "context.score", "label": "信誉分", "type": "number", "domain": [40, 60, 79, 80, 90, 100]},
            {"path": "input.vip", "label": "VIP", "type": "bool", "domain": [False, True]},
        ],
        "hidden_rule": logic(
            "and",
            binop("ge", f("context.age"), c(18)),
            logic("or", binop("eq", f("input.vip"), c(True)), binop("ge", f("context.score"), c(80))),
        ),
    },
    "text_shape": {
        "id": "text_shape",
        "name": "文本形态过滤",
        "description": "规则包含字符串片段与长度限制，展示非数值条件。",
        "fields": [
            {"path": "input.name", "label": "名称", "type": "string", "domain": ["ai", "agent", "train", "brain", "plain", "airlock", "robot"]},
        ],
        "hidden_rule": logic(
            "and",
            binop("contains", f("input.name"), c("ai")),
            binop("ge", {"op": "length", "arg": f("input.name")}, c(5)),
        ),
    },
    "sequence_lock": {
        "id": "sequence_lock",
        "name": "敲门序列锁",
        "description": "只有上一步是 knock 且本步是 open 才通过；它要求搜索历史状态。",
        "fields": [
            {"path": "input.action", "label": "动作", "type": "enum", "domain": ["wait", "knock", "open", "leave"]},
        ],
        "hidden_rule": logic(
            "and",
            binop("eq", f("input.action"), c("open")),
            binop("eq", prev("input.action", 1), c("knock")),
        ),
        "stateful": True,
        "validation_cases": [
            {"history": [], "input": {"action": "open"}},
            {"history": [{"input": {"action": "knock"}, "context": {}, "state": {}, "output": False}], "input": {"action": "open"}},
            {"history": [{"input": {"action": "wait"}, "context": {}, "state": {}, "output": False}], "input": {"action": "open"}},
            {"history": [{"input": {"action": "knock"}, "context": {}, "state": {}, "output": False}], "input": {"action": "leave"}},
        ],
    },
    "js_dom_sink_injection": {
        "id": "js_dom_sink_injection",
        "name": "脱离文档的 DOM Sink 迷宫",
        "description": "安全的合成 DOM 规则：只观测不可信文本是否创建节点，不执行脚本。",
        "category": "DOM 语义 · XSS",
        "severity": "HIGH",
        "cwe": "CWE-79",
        "research_question": "模型能否区分纯文本反射、实体编码文本与真正改变 DOM 结构的输入？",
        "hypothesis": "把 source→transform→sink 抽象成结构化证据后，XSS 族的误报会低于只看响应文本的策略。",
        "game_rule": "迷宫出口不是收到 200，而是安全标记到达 HTML 解析器并产生新节点；脚本执行永远不属于本实验。",
        "intended_rule": "text_is_rendered_as_text, not HTML",
        "js_source": "export function renderPreview(text) {\n  // BUG: innerHTML treats untrusted text as markup.\n  preview.innerHTML = text;\n  return preview.childElementCount > 0;\n}",
        "tags": ["xss", "dom-sink", "browser-oracle", "no-script-execution"],
        "fields": [
            {"path": "input.text", "label": "不可信文本", "type": "string", "domain": [
                "plain marker",
                "&lt;span data-sift-marker&gt;encoded&lt;/span&gt;",
                "<span data-sift-marker>inert</span>",
                "<p>nested <b>marker</b></p>"
            ]}
        ],
        "hidden_rule": binop("gt", {"op": "html_tag_count", "arg": f("input.text")}, c(0)),
        "lab_mode": "synthetic_detached_dom",
        "evaluator_oracle": "dom_sink_and_mutation_observer",
    },
    "js_sql_structure_boundary": {
        "id": "js_sql_structure_boundary",
        "name": "SQL 结构边界迷宫",
        "description": "不连接数据库的合成注入靶场：只返回查询结构是否被输入改变。",
        "category": "解释器边界 · SQL 注入",
        "severity": "HIGH",
        "cwe": "CWE-89",
        "research_question": "模型能否从安全的结构差分信号识别字符串拼接跨越了解释器边界？",
        "hypothesis": "把‘结果异常’拆成 query-structure changed 与 value-only changed 两个观测，比返回错误文本更可迁移。",
        "game_rule": "迷宫只提供结构标签，不执行 SQL、不连接外部数据库；出口是输入改变了查询语法树而非字段值。",
        "intended_rule": "parameterized_query_keeps_input_in_value_domain",
        "js_source": "export function buildProbe(fragmentClass) {\n  // BUG: this synthetic adapter models string concatenation.\n  const queryShape = fragmentClass === 'plain' ? 'value' : 'structure';\n  return { queryShape };\n}",
        "tags": ["injection", "sql", "interpreter-boundary", "synthetic-no-database"],
        "fields": [
            {"path": "input.fragment_class", "label": "输入片段类别", "type": "enum", "domain": ["plain", "quoted_value", "operator_like", "comment_like", "subquery_like", "blind_boolean", "row_shape", "syntax_error", "time_delay"]}
        ],
        "hidden_rule": logic(
            "or",
            binop("eq", f("input.fragment_class"), c("operator_like")),
            binop("eq", f("input.fragment_class"), c("comment_like")),
            binop("eq", f("input.fragment_class"), c("subquery_like")),
            binop("eq", f("input.fragment_class"), c("blind_boolean")),
            binop("eq", f("input.fragment_class"), c("row_shape")),
            binop("eq", f("input.fragment_class"), c("syntax_error")),
            binop("eq", f("input.fragment_class"), c("time_delay")),
        ),
        "lab_mode": "synthetic_interpreter_boundary",
        "evaluator_oracle": "query_ast_shape_diff",
    },
    "manual_lab": {
        "id": "manual_lab",
        "name": "手工标注外部黑盒",
        "description": "系统不执行隐藏规则。你可以把任意外部程序的返回值手工或批量录入，再让搜索器归纳。",
        "mode": "manual",
        "fields": [
            {"path": "input.x", "label": "数值 X", "type": "number", "domain": list(range(0, 21))},
            {"path": "input.tag", "label": "标签", "type": "enum", "domain": ["a", "b", "c"]},
        ],
        "hidden_rule": None,
    },
}


def public_scenarios() -> list[dict[str, Any]]:
    result = []
    for scenario in SCENARIOS.values():
        item = {key: deepcopy(value) for key, value in scenario.items() if key not in {"hidden_rule", "validation_cases"}}
        result.append(item)
    return result


def get_scenario(scenario_id: str) -> dict[str, Any] | None:
    value = SCENARIOS.get(scenario_id)
    return deepcopy(value) if value else None
