from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Callable

from .rule_ir import pretty, truthy_result


Envelope = dict[str, dict[str, Any]]


def c(value: Any) -> dict[str, Any]:
    return {"op": "const", "value": value}


def f(path: str) -> dict[str, Any]:
    return {"op": "field", "path": path}


def prev(path: str, offset: int = 1) -> dict[str, Any]:
    return {"op": "prev", "path": path, "offset": offset}


def binary(op: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": op, "left": left, "right": right}


def logic(op: str, *args: dict[str, Any]) -> dict[str, Any]:
    return {"op": op, "args": list(args)}


def envelope(**values: Any) -> Envelope:
    return {"input": dict(values), "context": {}, "state": {}}


@dataclass
class SyntheticProgram:
    family: str
    split: str
    js_source: str
    game_rule: str
    intended_rule_text: str
    intended_rule_ir: dict[str, Any]
    buggy_rule_ir: dict[str, Any]
    fields: list[dict[str, Any]]
    probes: list[dict[str, Any]]
    counterexample: dict[str, Any]
    cwe: str
    severity: str


def _numeric_boundary(rng: random.Random) -> SyntheticProgram:
    field_name = rng.choice(["total", "score", "age", "retryCount", "balance"])
    threshold = rng.randint(2, 250)
    direction = rng.choice(["min", "max"])
    if direction == "min":
        intended_op, buggy_op, intended_text, js_op = "ge", "gt", f"{field_name} >= {threshold}", ">"
        game_rule = f"玩家达到 {threshold} 分就应通过；错误裁判要求严格超过 {threshold} 分。"
    else:
        intended_op, buggy_op, intended_text, js_op = "le", "lt", f"{field_name} <= {threshold}", "<"
        game_rule = f"玩家最多可以使用 {threshold} 次；错误裁判在恰好第 {threshold} 次就提前拒绝。"
    path = f"input.{field_name}"
    domain = sorted({0, threshold - 2, threshold - 1, threshold, threshold + 1, threshold + 2, threshold * 2})
    return SyntheticProgram(
        family="numeric_boundary",
        split="train",
        js_source=f"export function decide({{{field_name}}}) {{ return {field_name} {js_op} {threshold}; }}",
        game_rule=game_rule,
        intended_rule_text=intended_text,
        intended_rule_ir=binary(intended_op, f(path), c(threshold)),
        buggy_rule_ir=binary(buggy_op, f(path), c(threshold)),
        fields=[{"path": path, "type": "number", "domain": domain}],
        probes=[{"envelope": envelope(**{field_name: value}), "history": []} for value in domain],
        counterexample={"envelope": envelope(**{field_name: threshold}), "history": []},
        cwe="CWE-193",
        severity="MEDIUM",
    )


def _truthiness_gate(rng: random.Random) -> SyntheticProgram:
    numeric = rng.choice(["quota", "credits", "retries", "balance", "flags"])
    admin = rng.choice(["admin", "root", "owner"])
    roles = ["guest", "member", admin]
    values = [-2, -1, 0, 1, 2, rng.randint(3, 20)]
    role_ref, value_ref = f("input.role"), f(f"input.{numeric}")
    intended = binary("eq", role_ref, c(admin))
    buggy = logic("or", intended, binary("ne", value_ref, c(0)))
    probes = []
    for role in roles:
        for value in values:
            probes.append({"envelope": envelope(role=role, **{numeric: value}), "history": []})
    return SyntheticProgram(
        family="truthiness_gate",
        split="train",
        js_source=f"export function allow({{role, {numeric}}}) {{ return Boolean(role === '{admin}' || {numeric}); }}",
        game_rule=f"守门人应只认 {admin} 身份，但错误实现把非零 {numeric} 也当成通行徽章。",
        intended_rule_text=f"role === '{admin}'",
        intended_rule_ir=intended,
        buggy_rule_ir=buggy,
        fields=[
            {"path": "input.role", "type": "enum", "domain": roles},
            {"path": f"input.{numeric}", "type": "number", "domain": values},
        ],
        probes=probes,
        counterexample={"envelope": envelope(role="guest", **{numeric: -1}), "history": []},
        cwe="CWE-863",
        severity="HIGH",
    )


def _substring_origin(rng: random.Random) -> SyntheticProgram:
    trusted = rng.choice(["trusted.example", "safe.local", "portal.test", "account.internal"])
    urls = [
        f"https://{trusted}/home",
        f"https://{trusted}.evil.test/phish",
        f"https://evil.test/?next={trusted}",
        f"https://evil.test/{trusted}",
        "https://example.test/home",
    ]
    intended = binary("starts_with", f("input.url"), c(f"https://{trusted}/"))
    buggy = binary("contains", f("input.url"), c(trusted))
    return SyntheticProgram(
        family="substring_origin",
        split="validation",
        js_source=f"export function redirect(url) {{ return url.includes('{trusted}'); }}",
        game_rule=f"传送门只应去往 {trusted}，错误守卫却只检查整段文字里有没有这个域名。",
        intended_rule_text=f"new URL(url).hostname === '{trusted}'",
        intended_rule_ir=intended,
        buggy_rule_ir=buggy,
        fields=[{"path": "input.url", "type": "string", "domain": urls}],
        probes=[{"envelope": envelope(url=url), "history": []} for url in urls],
        counterexample={"envelope": envelope(url=f"https://{trusted}.evil.test/phish"), "history": []},
        cwe="CWE-601",
        severity="HIGH",
    )


def _authorization_or(rng: random.Random) -> SyntheticProgram:
    resource = rng.choice(["document", "project", "invoice", "profile"])
    intended = logic(
        "and",
        binary("eq", f("input.authenticated"), c(True)),
        binary("eq", f("input.isOwner"), c(True)),
    )
    buggy = logic(
        "or",
        binary("eq", f("input.authenticated"), c(True)),
        binary("eq", f("input.isOwner"), c(True)),
    )
    probes = [
        {"envelope": envelope(authenticated=auth, isOwner=owner), "history": []}
        for auth in [False, True]
        for owner in [False, True]
    ]
    return SyntheticProgram(
        family="authorization_or",
        split="test",
        js_source=f"export function edit{resource.title()}({{authenticated, isOwner}}) {{ return authenticated || isOwner; }}",
        game_rule=f"编辑 {resource} 必须同时登录且是所有者，错误规则把两个必要条件写成了任选其一。",
        intended_rule_text="authenticated && isOwner",
        intended_rule_ir=intended,
        buggy_rule_ir=buggy,
        fields=[
            {"path": "input.authenticated", "type": "bool", "domain": [False, True]},
            {"path": "input.isOwner", "type": "bool", "domain": [False, True]},
        ],
        probes=probes,
        counterexample={"envelope": envelope(authenticated=True, isOwner=False), "history": []},
        cwe="CWE-863",
        severity="HIGH",
    )


def _postmessage_origin(rng: random.Random) -> SyntheticProgram:
    trusted_host = rng.choice(["trusted.com", "safe.local", "portal.test", "account.internal"])
    trusted_origin = f"https://{trusted_host}"
    origins = [
        trusted_origin,
        f"http://{trusted_host}",
        f"https://sub.{trusted_host}",
        f"https://evil{trusted_host}",
        f"https://{trusted_host}.evil.test",
        f"https://evil.test/{trusted_host}",
        f"wss://{trusted_host}",
        f"chrome-extension://{trusted_host}",
        "null",
        "https://evil.test",
        "http://localhost",
        "https://example.test",
        f"https://not-{trusted_host}",
        f"https://deep.sub.{trusted_host}",
        f"ftp://{trusted_host}",
        f"https://{trusted_host}-attacker.test",
    ]
    origin_ref = f("input.origin")
    intended = binary("origin_eq", origin_ref, c(trusted_origin))
    buggy = binary("ends_with", origin_ref, c(trusted_host))
    return SyntheticProgram(
        family="postmessage_origin",
        split="test",
        js_source=f"export function trustMessage(event) {{ return event.origin.endsWith('{trusted_host}'); }}",
        game_rule=f"信使必须来自完整源 {trusted_origin}，错误守卫只比较字符串后缀 {trusted_host}。",
        intended_rule_text=f"event.origin === '{trusted_origin}'",
        intended_rule_ir=intended,
        buggy_rule_ir=buggy,
        fields=[{"path": "input.origin", "type": "string", "domain": origins}],
        probes=[{"envelope": envelope(origin=origin), "history": []} for origin in origins],
        counterexample={"envelope": envelope(origin=f"https://evil{trusted_host}"), "history": []},
        cwe="CWE-346",
        severity="HIGH",
    )


def _dom_sink_injection(rng: random.Random) -> SyntheticProgram:
    marker = rng.choice(["probe", "notice", "profile", "message"])
    payloads = [
        "hello world",
        f"plain {marker}",
        "2 < 3 and 5 > 4",
        "&lt;b&gt;encoded&lt;/b&gt;",
        "no markup here",
        "<not-closed",
        "angle > only",
        "x & y",
        "{{template}}",
        "javascript:alert(1)",
        f"<b>{marker}</b>",
        f"<span data-marker='{marker}'>x</span>",
        "<img src=x>",
        "</p><section>injected</section><p>",
        "<svg><title>x</title></svg>",
        "<a href='/local'>link</a>",
        "<template>fragment</template>",
        "<x-sift-probe></x-sift-probe>",
        "<input value='probe'>",
        "<math><mtext>node</mtext></math>",
    ]
    payload_ref = f("input.payload")
    buggy = {"op": "html_creates_nodes", "arg": payload_ref}
    return SyntheticProgram(
        family="dom_sink_injection",
        split="test",
        js_source="export function render(target, input) { target.innerHTML = `<p>${input}</p>`; return target.children.length > 1; }",
        game_rule="玩家输入只能作为文字显示；错误渲染器把文字送入 HTML 解析器并允许创建新节点。",
        intended_rule_text="untrusted input creates zero DOM elements",
        intended_rule_ir=c(False),
        buggy_rule_ir=buggy,
        fields=[{"path": "input.payload", "type": "string", "domain": payloads}],
        probes=[{"envelope": envelope(payload=payload), "history": []} for payload in payloads],
        counterexample={"envelope": envelope(payload=f"<span data-marker='{marker}'>x</span>"), "history": []},
        cwe="CWE-79",
        severity="HIGH",
    )


def _string_suffix_primitive(rng: random.Random) -> SyntheticProgram:
    suffix = rng.choice(["trusted.com", "safe.local", "portal.test", "account.internal"])
    values = [
        f"https://{suffix}",
        f"http://{suffix}",
        f"https://sub.{suffix}",
        f"https://prefix{suffix}",
        f"wss://{suffix}",
        f"ftp://{suffix}",
        f"https://{suffix}.other.test",
        f"https://{suffix}-extra.test",
        f"https://other.test/{suffix}",
        "https://example.test",
        "http://localhost",
        "null",
    ]
    token_ref = f("input.endpoint")
    buggy = binary("ends_with", token_ref, c(suffix))
    intended = binary("origin_eq", token_ref, c(f"https://{suffix}"))
    return SyntheticProgram(
        family="string_suffix_primitive",
        split="train",
        js_source=f"export function hasSuffix(token) {{ return token.endsWith('{suffix}'); }}",
        game_rule=f"训练一个纯字符串原语：判断 endpoint 文本是否以后缀 {suffix} 结束，不赋予安全含义。",
        intended_rule_text=f"parsed endpoint origin equals 'https://{suffix}'",
        intended_rule_ir=intended,
        buggy_rule_ir=buggy,
        fields=[{"path": "input.endpoint", "type": "string", "domain": values}],
        probes=[{"envelope": envelope(endpoint=value), "history": []} for value in values],
        counterexample={"envelope": envelope(endpoint=f"https://prefix{suffix}"), "history": []},
        cwe="CWE-20",
        severity="LOW",
    )


def _markup_lexeme_primitive(rng: random.Random) -> SyntheticProgram:
    word = rng.choice(["alpha", "beta", "notice", "sample"])
    values = [
        word,
        f"plain {word}",
        "2 < 3 and 5 > 4",
        "&lt;i&gt;encoded&lt;/i&gt;",
        "<unfinished",
        "right > angle",
        f"<b>{word}</b>",
        f"<i>{word}</i>",
        "<img src=x>",
        "<x-token></x-token>",
        "<section>node</section>",
        "<input value='x'>",
    ]
    message_ref = f("input.message")
    buggy = {"op": "html_creates_nodes", "arg": message_ref}
    return SyntheticProgram(
        family="markup_lexeme_primitive",
        split="train",
        js_source="export function recognizesMarkup(message) { const t=document.createElement('template'); t.innerHTML=message; return t.content.childElementCount > 0; }",
        game_rule="训练一个无副作用词法原语：区分普通文字、编码后的尖括号和能够形成元素的标记。",
        intended_rule_text="all samples are treated as plain text",
        intended_rule_ir=c(False),
        buggy_rule_ir=buggy,
        fields=[{"path": "input.message", "type": "string", "domain": values}],
        probes=[{"envelope": envelope(message=value), "history": []} for value in values],
        counterexample={"envelope": envelope(message=f"<b>{word}</b>"), "history": []},
        cwe="CWE-20",
        severity="LOW",
    )


def _url_hostname_primitive(rng: random.Random) -> SyntheticProgram:
    host = rng.choice(["trusted.com", "safe.local", "portal.test", "account.internal"])
    values = [f"https://{host}", f"http://{host}", f"ftp://{host}", f"https://sub.{host}", f"https://{host}.evil.test", "null"]
    endpoint = f("input.endpoint")
    buggy = binary("eq", {"op": "url_hostname", "arg": endpoint}, c(host))
    intended = binary("origin_eq", endpoint, c(f"https://{host}"))
    return SyntheticProgram(
        family="url_hostname_primitive", split="train",
        js_source=f"export function sameHost(value) {{ return new URL(value).hostname === '{host}'; }}",
        game_rule=f"只训练 URL hostname 抽取原语：scheme 不参与 hostname 相等判断。",
        intended_rule_text=f"complete origin equals https://{host}", intended_rule_ir=intended, buggy_rule_ir=buggy,
        fields=[{"path": "input.endpoint", "type": "string", "domain": values}],
        probes=[{"envelope": envelope(endpoint=value), "history": []} for value in values],
        counterexample={"envelope": envelope(endpoint=f"http://{host}"), "history": []}, cwe="CWE-20", severity="LOW",
    )


def _html_entity_decode_primitive(rng: random.Random) -> SyntheticProgram:
    word = rng.choice(["probe", "notice", "profile", "message"])
    values = [word, f"&lt;b&gt;{word}&lt;/b&gt;", f"&amp;lt;i&amp;gt;{word}&amp;lt;/i&amp;gt;", f"<b>{word}</b>", "2 &lt; 3", "plain text"]
    value_ref = f("input.encoded")
    decoded = {"op": "html_entity_decode", "arg": value_ref}
    buggy = {"op": "html_creates_nodes", "arg": decoded}
    intended = {"op": "html_creates_nodes", "arg": value_ref}
    return SyntheticProgram(
        family="html_entity_decode_primitive", split="train",
        js_source="export function decodeThenParse(v) { const t=document.createElement('template'); t.innerHTML=decodeEntities(v); return t.content.childElementCount>0; }",
        game_rule="训练实体解码原语：编码尖括号解码后可能从文字变成标记。",
        intended_rule_text="raw encoded text is parsed without an extra decode", intended_rule_ir=intended, buggy_rule_ir=buggy,
        fields=[{"path": "input.encoded", "type": "string", "domain": values}],
        probes=[{"envelope": envelope(encoded=value), "history": []} for value in values],
        counterexample={"envelope": envelope(encoded=f"&lt;b&gt;{word}&lt;/b&gt;"), "history": []}, cwe="CWE-20", severity="LOW",
    )


def _casefold_primitive(rng: random.Random) -> SyntheticProgram:
    expected = rng.choice(["admin", "owner", "root", "operator"])
    values = [expected, expected.upper(), expected.title(), f" {expected}", f"{expected} ", "guest", "member"]
    value_ref = f("input.token")
    buggy = binary("eq", {"op": "casefold", "arg": value_ref}, c(expected.casefold()))
    intended = binary("eq", value_ref, c(expected))
    return SyntheticProgram(
        family="casefold_primitive", split="train",
        js_source=f"export function equalFold(v) {{ return v.toLocaleLowerCase() === '{expected}'; }}",
        game_rule="训练大小写折叠原语，但保留原始标识符与折叠标识符的区别。",
        intended_rule_text=f"token exactly equals {expected}", intended_rule_ir=intended, buggy_rule_ir=buggy,
        fields=[{"path": "input.token", "type": "string", "domain": values}],
        probes=[{"envelope": envelope(token=value), "history": []} for value in values],
        counterexample={"envelope": envelope(token=expected.upper()), "history": []}, cwe="CWE-20", severity="LOW",
    )


def _numeric_coercion_primitive(rng: random.Random) -> SyntheticProgram:
    limit = rng.randint(5, 40)
    values = [str(limit), f"0{limit}", f" {limit} ", f"{limit}.0", f"{limit}e0", "", "0x10", str(limit + 1), "NaN", "Infinity"]
    value_ref = f("input.amount")
    numeric = {"op": "to_number", "arg": value_ref}
    canonical = {"op": "regex", "arg": value_ref, "pattern": "^(0|[1-9][0-9]*)$"}
    buggy = binary("le", numeric, c(limit))
    intended = logic("and", canonical, binary("le", numeric, c(limit)))
    return SyntheticProgram(
        family="numeric_coercion_primitive", split="train",
        js_source=f"export function underLimit(v) {{ return Number(v) <= {limit}; }}",
        game_rule="训练 JavaScript 数值强转原语：空白、前导零、指数与十六进制文本不能静默当作规范十进制。",
        intended_rule_text=f"canonical decimal amount <= {limit}", intended_rule_ir=intended, buggy_rule_ir=buggy,
        fields=[{"path": "input.amount", "type": "string", "domain": values}],
        probes=[{"envelope": envelope(amount=value), "history": []} for value in values],
        counterexample={"envelope": envelope(amount=f" {limit} "), "history": []}, cwe="CWE-704", severity="LOW",
    )


def _url_scheme_downgrade(rng: random.Random) -> SyntheticProgram:
    host = rng.choice(["trusted.com", "safe.local", "portal.test", "account.internal"])
    values = [f"https://{host}", f"https://{host}:443", f"http://{host}", f"ftp://{host}", f"ws://{host}", f"https://sub.{host}", f"https://{host}.evil.test", "null"]
    origin = f("input.origin")
    buggy = binary("eq", {"op": "url_hostname", "arg": origin}, c(host))
    intended = binary("origin_eq", origin, c(f"https://{host}"))
    return SyntheticProgram(
        family="url_scheme_downgrade", split="test",
        js_source=f"export function acceptOrigin(v) {{ return new URL(v).hostname === '{host}'; }}",
        game_rule=f"守卫认对了主机却忘了协议；HTTP、FTP 和 WebSocket 被当作 HTTPS 源。",
        intended_rule_text=f"origin exactly equals https://{host}", intended_rule_ir=intended, buggy_rule_ir=buggy,
        fields=[{"path": "input.origin", "type": "string", "domain": values}],
        probes=[{"envelope": envelope(origin=value), "history": []} for value in values],
        counterexample={"envelope": envelope(origin=f"http://{host}"), "history": []}, cwe="CWE-346", severity="HIGH",
    )


def _dom_double_decode(rng: random.Random) -> SyntheticProgram:
    word = rng.choice(["probe", "notice", "profile", "message"])
    values = ["plain text", f"&lt;b&gt;{word}&lt;/b&gt;", f"&amp;lt;img src=x&amp;gt;", f"&amp;amp;lt;i&amp;amp;gt;{word}", "2 &lt; 3", "&quot;quoted&quot;", "<b>already markup</b>"]
    payload = f("input.payload")
    decoded_twice = {"op": "html_entity_decode", "arg": {"op": "html_entity_decode", "arg": payload}}
    buggy = {"op": "html_creates_nodes", "arg": decoded_twice}
    return SyntheticProgram(
        family="dom_double_decode", split="test",
        js_source="export function render(v) { target.innerHTML = decodeEntities(decodeEntities(v)); return target.children.length>0; }",
        game_rule="经过两层组件的重复实体解码后，原本安全的编码文本重新变成 DOM 标记。",
        intended_rule_text="encoded display text creates no DOM nodes", intended_rule_ir=c(False), buggy_rule_ir=buggy,
        fields=[{"path": "input.payload", "type": "string", "domain": values}],
        probes=[{"envelope": envelope(payload=value), "history": []} for value in values],
        counterexample={"envelope": envelope(payload=f"&amp;lt;img src=x&amp;gt;"), "history": []}, cwe="CWE-79", severity="HIGH",
    )


def _unicode_casefold_role(rng: random.Random) -> SyntheticProgram:
    privileged = rng.choice(["admin", "owner", "root", "operator"])
    roles = [privileged, privileged.upper(), privileged.title(), "guest", "member", f" {privileged}"]
    quotas = [0, -1, 1]
    role_ref, quota_ref = f("input.role"), f("input.quota")
    buggy = logic("or", binary("eq", {"op": "casefold", "arg": role_ref}, c(privileged.casefold())), binary("ne", quota_ref, c(0)))
    intended = binary("eq", role_ref, c(privileged))
    probes = [{"envelope": envelope(role=role, quota=quota), "history": []} for role in roles for quota in quotas]
    return SyntheticProgram(
        family="unicode_casefold_role", split="test",
        js_source=f"export function allow(role, quota) {{ return role.toLocaleLowerCase() === '{privileged}' || quota; }}",
        game_rule="身份标识符经过大小写折叠，又与数值 truthiness 合并，形成跨语义授权旁路。",
        intended_rule_text=f"role exactly equals {privileged}", intended_rule_ir=intended, buggy_rule_ir=buggy,
        fields=[{"path": "input.role", "type": "string", "domain": roles}, {"path": "input.quota", "type": "number", "domain": quotas}],
        probes=probes, counterexample={"envelope": envelope(role=privileged.upper(), quota=0), "history": []}, cwe="CWE-863", severity="HIGH",
    )


def _numeric_string_coercion(rng: random.Random) -> SyntheticProgram:
    limit = rng.randint(5, 40)
    values = [str(limit), f"0{limit}", f" {limit} ", f"{limit}.0", f"{limit}e0", "", "0x10", str(limit + 1), "NaN", "Infinity"]
    amount = f("input.amount")
    numeric = {"op": "to_number", "arg": amount}
    canonical = {"op": "regex", "arg": amount, "pattern": "^(0|[1-9][0-9]*)$"}
    buggy = binary("le", numeric, c(limit))
    intended = logic("and", canonical, binary("le", numeric, c(limit)))
    return SyntheticProgram(
        family="numeric_string_coercion", split="test",
        js_source=f"export function approveAmount(v) {{ return Number(v) <= {limit}; }}",
        game_rule="金额策略要求规范十进制，错误实现却先做 JavaScript Number 强转再比较。",
        intended_rule_text=f"canonical decimal amount <= {limit}", intended_rule_ir=intended, buggy_rule_ir=buggy,
        fields=[{"path": "input.amount", "type": "string", "domain": values}],
        probes=[{"envelope": envelope(amount=value), "history": []} for value in values],
        counterexample={"envelope": envelope(amount=f"0{limit}"), "history": []}, cwe="CWE-704", severity="MEDIUM",
    )


def _compound_origin_role(rng: random.Random) -> SyntheticProgram:
    host = rng.choice(["trusted.com", "safe.local", "portal.test", "account.internal"])
    privileged = rng.choice(["admin", "owner", "root"])
    origins = [f"https://{host}", f"http://{host}", f"https://evil{host}", "https://evil.test"]
    roles, quotas = [privileged, "member", "guest"], [0, 1, -1]
    origin_ref, role_ref, quota_ref = f("input.origin"), f("input.role"), f("input.quota")
    intended = logic("and", binary("origin_eq", origin_ref, c(f"https://{host}")), binary("eq", role_ref, c(privileged)))
    buggy = logic("or", binary("ends_with", origin_ref, c(host)), binary("ne", quota_ref, c(0)))
    probes = [{"envelope": envelope(origin=origin, role=role, quota=quota), "history": []} for origin in origins for role in roles for quota in quotas]
    return SyntheticProgram(
        family="compound_origin_role", split="test",
        js_source=f"export function allow(e) {{ return e.origin.endsWith('{host}') || e.quota; }}",
        game_rule="正确规则要求可信完整源且具备特权身份；错误实现把源后缀与数值 truthiness 写成任选其一。",
        intended_rule_text=f"exact HTTPS origin AND role == {privileged}", intended_rule_ir=intended, buggy_rule_ir=buggy,
        fields=[{"path": "input.origin", "type": "string", "domain": origins}, {"path": "input.role", "type": "string", "domain": roles}, {"path": "input.quota", "type": "number", "domain": quotas}],
        probes=probes, counterexample={"envelope": envelope(origin=f"https://evil{host}", role="guest", quota=0), "history": []}, cwe="CWE-863", severity="CRITICAL",
    )


def _state_replay_window(rng: random.Random) -> SyntheticProgram:
    tokens = [f"n{rng.randint(10, 99)}-{suffix}" for suffix in ("a", "b", "c", "d")]
    current = f("input.token")
    previous = prev("input.token")
    buggy = binary("eq", current, previous)
    intended = binary("ne", current, previous)
    probes = []
    for before in tokens:
        for after in tokens:
            probes.append({"envelope": envelope(token=after), "history": [envelope(token=before)]})
    return SyntheticProgram(
        family="state_replay_window", split="test",
        js_source="export function accept(token) { return token === lastToken; }",
        game_rule="一次性令牌应与上一次不同，错误窗口却只接受重复令牌，暴露跨 episode 重放缺陷。",
        intended_rule_text="current token differs from previous token", intended_rule_ir=intended, buggy_rule_ir=buggy,
        fields=[{"path": "input.token", "type": "string", "domain": tokens}],
        probes=probes, counterexample={"envelope": envelope(token=tokens[0]), "history": [envelope(token=tokens[0])]}, cwe="CWE-294", severity="HIGH",
    )


FAMILY_GENERATORS: dict[str, Callable[[random.Random], SyntheticProgram]] = {
    "numeric_boundary": _numeric_boundary,
    "truthiness_gate": _truthiness_gate,
    "substring_origin": _substring_origin,
    "authorization_or": _authorization_or,
    "postmessage_origin": _postmessage_origin,
    "dom_sink_injection": _dom_sink_injection,
    "string_suffix_primitive": _string_suffix_primitive,
    "markup_lexeme_primitive": _markup_lexeme_primitive,
    "url_hostname_primitive": _url_hostname_primitive,
    "html_entity_decode_primitive": _html_entity_decode_primitive,
    "casefold_primitive": _casefold_primitive,
    "numeric_coercion_primitive": _numeric_coercion_primitive,
    "url_scheme_downgrade": _url_scheme_downgrade,
    "dom_double_decode": _dom_double_decode,
    "unicode_casefold_role": _unicode_casefold_role,
    "numeric_string_coercion": _numeric_string_coercion,
    "compound_origin_role": _compound_origin_role,
    "state_replay_window": _state_replay_window,
}


SEMANTIC_FEATURES: dict[str, dict[str, Any]] = {
    "numeric_boundary": {
        "primitive_families": ["numeric_boundary", "boolean_logic"],
        "coercions": [],
        "observed_deviation": "strict comparison replaces an inclusive boundary",
        "security_property": "boundary behavior must match the declared inclusive policy",
    },
    "truthiness_gate": {
        "primitive_families": ["coercion", "authorization", "boolean_logic"],
        "coercions": ["numeric_to_boolean_nonzero"],
        "observed_deviation": "a numeric business value becomes a sufficient authorization condition",
        "security_property": "authorization must depend only on declared identity or capability predicates",
    },
    "substring_origin": {
        "primitive_families": ["string_match", "source_sink"],
        "coercions": ["url_object_to_raw_string"],
        "observed_deviation": "substring containment replaces structured origin equality",
        "security_property": "redirect destinations must satisfy parsed origin policy",
    },
    "authorization_or": {
        "primitive_families": ["authorization", "boolean_logic"],
        "coercions": [],
        "observed_deviation": "two necessary authorization predicates are combined as alternatives",
        "security_property": "all mandatory authorization predicates must hold",
    },
    "postmessage_origin": {
        "primitive_families": ["structured_url", "origin_policy", "source_sink"],
        "coercions": ["url_origin_to_raw_string_suffix"],
        "observed_deviation": "raw origin suffix matching replaces parsed, exact origin equality",
        "security_property": "cross-window messages must match the complete trusted origin",
    },
    "dom_sink_injection": {
        "primitive_families": ["source_sink", "html_parser", "encoding_context"],
        "coercions": ["untrusted_text_to_html_nodes"],
        "observed_deviation": "untrusted text is interpreted in an HTML parsing context",
        "security_property": "untrusted display text must not create DOM nodes",
    },
    "string_suffix_primitive": {
        "primitive_families": ["string_match", "suffix"],
        "coercions": [],
        "observed_deviation": "suffix membership accepts more values than exact equality",
        "security_property": "primitive pretraining only; no security decision is attached",
    },
    "markup_lexeme_primitive": {
        "primitive_families": ["html_parser", "encoding_context"],
        "coercions": ["text_to_html_nodes"],
        "observed_deviation": "markup syntax creates elements while plain and encoded text does not",
        "security_property": "primitive pretraining only; the parser must remain detached and script-free",
    },
    "url_hostname_primitive": {
        "primitive_families": ["structured_url", "hostname"], "coercions": ["url_to_hostname"],
        "observed_deviation": "hostname equality omits scheme and port policy", "security_property": "primitive pretraining only",
        "generalization_axis": "primitive_only",
    },
    "html_entity_decode_primitive": {
        "primitive_families": ["encoding_context", "html_parser"], "coercions": ["html_entity_decode"],
        "observed_deviation": "entity decoding changes parser-visible markup", "security_property": "primitive pretraining only",
        "generalization_axis": "primitive_only",
    },
    "casefold_primitive": {
        "primitive_families": ["string_normalization", "casefold"], "coercions": ["unicode_casefold"],
        "observed_deviation": "case folding merges distinct raw identifiers", "security_property": "primitive pretraining only",
        "generalization_axis": "primitive_only",
    },
    "numeric_coercion_primitive": {
        "primitive_families": ["coercion", "numeric_boundary", "canonicalization"], "coercions": ["string_to_javascript_number"],
        "observed_deviation": "non-canonical numeric strings become numbers before comparison", "security_property": "primitive pretraining only",
        "generalization_axis": "primitive_only",
    },
    "url_scheme_downgrade": {
        "primitive_families": ["structured_url", "origin_policy", "protocol"], "coercions": ["url_to_hostname"],
        "observed_deviation": "hostname equality drops the required HTTPS scheme", "security_property": "trusted transport and authority must both match",
        "generalization_axis": "runtime_url_semantics",
    },
    "dom_double_decode": {
        "primitive_families": ["source_sink", "html_parser", "encoding_context"], "coercions": ["double_html_entity_decode"],
        "observed_deviation": "repeated decoding reconstructs active markup", "security_property": "encoded display text must remain text through the full pipeline",
        "generalization_axis": "encoding_depth",
    },
    "unicode_casefold_role": {
        "primitive_families": ["authorization", "string_normalization", "coercion", "boolean_logic"], "coercions": ["unicode_casefold", "numeric_to_boolean_nonzero"],
        "observed_deviation": "case-folded identity and numeric truthiness become alternative authorization paths", "security_property": "authorization identifiers require declared canonicalization and exact policy",
        "generalization_axis": "language_string_semantics",
    },
    "numeric_string_coercion": {
        "primitive_families": ["coercion", "numeric_boundary", "canonicalization"], "coercions": ["string_to_javascript_number"],
        "observed_deviation": "Number coercion accepts non-canonical amount encodings", "security_property": "validate the representation before numeric comparison",
        "generalization_axis": "coercion_representation",
    },
    "compound_origin_role": {
        "primitive_families": ["structured_url", "authorization", "boolean_logic", "coercion"], "coercions": ["url_origin_to_raw_string_suffix", "numeric_to_boolean_nonzero"],
        "observed_deviation": "two mandatory predicates become unrelated alternatives", "security_property": "trusted origin and privileged identity must both hold",
        "generalization_axis": "compositional_shift",
    },
    "state_replay_window": {
        "primitive_families": ["state", "history", "replay"], "coercions": [],
        "observed_deviation": "the acceptance relation over current and previous tokens is inverted", "security_property": "one-time tokens must not repeat",
        "generalization_axis": "state_and_history_shift",
    },
}


def _verified_traces(program: SyntheticProgram, rng: random.Random, traces_per_program: int) -> list[dict[str, Any]]:
    selected = list(program.probes)
    rng.shuffle(selected)
    counterexample_key = json.dumps(program.counterexample, sort_keys=True, ensure_ascii=False)
    selected = [program.counterexample] + [row for row in selected if json.dumps(row, sort_keys=True, ensure_ascii=False) != counterexample_key]
    traces = []
    for index, case in enumerate(selected[:traces_per_program]):
        current = case["envelope"]
        history = case.get("history", [])
        buggy_output = truthy_result(program.buggy_rule_ir, current, history)
        intended_output = truthy_result(program.intended_rule_ir, current, history)
        traces.append({
            "episode_id": f"episode-{index + 1}",
            "step": 0,
            **current,
            "history": history,
            "output": buggy_output,
            "intended_output": intended_output,
            "is_counterexample": buggy_output != intended_output,
        })
    if not any(row["is_counterexample"] for row in traces):
        raise ValueError(f"generated program {program.family} has no verified counterexample")
    return traces


def generate_record(index: int, rng: random.Random, family: str, traces_per_program: int = 12) -> dict[str, Any]:
    program = FAMILY_GENERATORS[family](rng)
    traces = _verified_traces(program, rng, traces_per_program)
    counterexamples = [row for row in traces if row["is_counterexample"]]
    partial_count = max(2, min(len(traces) - 1, len(traces) // 2))
    partial_traces = traces[:partial_count]
    next_probe = counterexamples[0]
    record_id = f"syn-{program.family}-{index:07d}"
    features = SEMANTIC_FEATURES[program.family]
    common_semantic_rule = {
        "schema_version": "common-semantic-rule-v1",
        "source": {
            "language": "javascript",
            "frontend": "ir",
            "adapter_version": "synthetic-v1",
            "discovery_mode": "gray_box_reconciliation",
            "provenance": ["generated_source", "executable_rule_ir", "oracle_trace"],
            "confidence": 1.0,
            "evidence_pointers": [record_id],
        },
        "behavior": {
            "rule_ir": program.buggy_rule_ir,
            "primitive_families": features["primitive_families"],
            "human_rule": program.game_rule,
        },
        "invariant": {
            "expected": program.intended_rule_text,
            "observed_deviation": features["observed_deviation"],
            "security_property": features["security_property"],
        },
        "dependencies": {
            "inputs": [field["path"] for field in program.fields],
            "state": [],
            "history_offsets": [],
            "external_effects": [],
        },
        "language_semantics": {
            "coercions": features["coercions"],
            "numeric_model": "javascript_number_ieee754",
            "evaluation_order": "left_to_right_short_circuit",
            "exception_model": "javascript_throw",
            "overflow_behavior": "ieee754_infinity_or_precision_loss",
            "nullish_behavior": "undefined_and_null_are_distinct_but_both_nullish",
        },
        "evidence": {
            "counterexamples": counterexamples,
            "reproducible": True,
            "oracle_fingerprint": record_id,
            "coverage_scope": "declared generated finite domain",
        },
    }
    return {
        "schema_version": "sift-synthetic-v1",
        "record_id": record_id,
        "split": program.split,
        "family": program.family,
        "security": {"cwe": program.cwe, "severity": program.severity},
        "generalization": {"axis": features.get("generalization_axis", "baseline"), "complete_family_holdout_ready": True},
        "modalities": {
            "js": program.js_source,
            "game": program.game_rule,
            "trace": traces,
            "rule_ir": program.buggy_rule_ir,
            "semantic_rule": common_semantic_rule,
            "evidence": {
                "intended_rule": program.intended_rule_text,
                "counterexample": next_probe,
                "reproducible": True,
            },
        },
        "tasks": [
            {
                "task": "semantic_abstraction",
                "input": {"js": program.js_source},
                "target": {"game_rule": program.game_rule, "semantic_rule": common_semantic_rule},
            },
            {
                "task": "rule_induction",
                "input": {"traces": traces},
                "target": {"rule_ir": program.buggy_rule_ir, "pretty": pretty(program.buggy_rule_ir)},
            },
            {
                "task": "next_probe",
                "input": {"traces": partial_traces, "fields": program.fields},
                "target": {"probe": {key: next_probe[key] for key in ("input", "context", "state", "history")}},
            },
            {
                "task": "vulnerability_evidence",
                "input": {"traces": traces, "candidate_rule_ir": program.buggy_rule_ir},
                "target": {
                    "cwe": program.cwe,
                    "counterexample": next_probe,
                    "intended_rule": program.intended_rule_text,
                    "reproducible": True,
                },
            },
        ],
        "verification": {
            "engine": "app.rule_ir.truthy_result",
            "verified_trace_count": len(traces),
            "verified_counterexample_count": len(counterexamples),
        },
    }


def generate_curriculum(program_count: int, traces_per_program: int = 12, seed: int = 20260801) -> list[dict[str, Any]]:
    if program_count < 1:
        raise ValueError("program_count must be positive")
    rng = random.Random(seed)
    families = list(FAMILY_GENERATORS)
    return [
        generate_record(index + 1, rng, families[index % len(families)], traces_per_program)
        for index in range(program_count)
    ]
