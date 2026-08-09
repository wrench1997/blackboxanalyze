from __future__ import annotations

import json
import math
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

MISSING = object()


class _ElementCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements += 1


def _url_parts(value: Any) -> Any:
    if value is MISSING or not isinstance(value, str) or value == "null":
        return MISSING
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return MISSING
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname.lower()
        port = parsed.port
        if port is None or (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
            authority = hostname
        else:
            authority = f"{hostname}:{port}"
        return {"scheme": scheme, "hostname": hostname, "origin": f"{scheme}://{authority}"}
    except (TypeError, ValueError):
        return MISSING


def _html_tag_count(value: Any) -> int:
    if value is MISSING or not isinstance(value, str):
        return 0
    parser = _ElementCounter()
    try:
        parser.feed(value)
        parser.close()
    except (TypeError, ValueError):
        return 0
    return parser.elements


def _js_to_number(value: Any) -> Any:
    if value is MISSING:
        return MISSING
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return MISSING
    text = value.strip()
    if text == "":
        return 0.0
    try:
        if text.lower().startswith(("0x", "+0x", "-0x")):
            sign = -1 if text.startswith("-") else 1
            digits = text[3:] if text[:1] in "+-" else text[2:]
            return float(sign * int(digits, 16))
        return float(text)
    except ValueError:
        return MISSING


def get_path(root: Any, path: str, default: Any = MISSING) -> Any:
    """Resolve dotted paths through dict/list structures."""
    current = root
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return default
    return current


def _safe_binary(op: str, left: Any, right: Any) -> Any:
    if left is MISSING or right is MISSING:
        return False
    try:
        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op == "gt":
            return left > right
        if op == "ge":
            return left >= right
        if op == "lt":
            return left < right
        if op == "le":
            return left <= right
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "mul":
            return left * right
        if op == "div":
            return left / right if right != 0 else MISSING
        if op == "mod":
            return left % right if right != 0 else MISSING
        if op == "contains":
            return right in left
        if op == "starts_with":
            return str(left).startswith(str(right))
        if op == "ends_with":
            return str(left).endswith(str(right))
        if op == "origin_eq":
            left_parts = _url_parts(left)
            right_parts = _url_parts(right)
            return left_parts is not MISSING and right_parts is not MISSING and left_parts["origin"] == right_parts["origin"]
        if op == "in":
            return left in right
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    raise ValueError(f"unsupported binary op: {op}")


def evaluate(expr: dict[str, Any], envelope: dict[str, Any], history: list[dict[str, Any]] | None = None) -> Any:
    """Evaluate a JSON Rule IR expression against an input envelope."""
    history = history or []
    if not isinstance(expr, dict) or "op" not in expr:
        raise ValueError("expression must be an object containing 'op'")

    op = expr["op"]
    if op == "const":
        return expr.get("value")
    if op == "field":
        return get_path(envelope, expr["path"])
    if op == "prev":
        offset = int(expr.get("offset", 1))
        if offset <= 0 or len(history) < offset:
            return MISSING
        return get_path(history[-offset], expr["path"])

    if op in {"eq", "ne", "gt", "ge", "lt", "le", "add", "sub", "mul", "div", "mod", "contains", "starts_with", "ends_with", "origin_eq", "in"}:
        return _safe_binary(op, evaluate(expr["left"], envelope, history), evaluate(expr["right"], envelope, history))

    if op == "and":
        return all(bool(evaluate(arg, envelope, history)) for arg in expr.get("args", []))
    if op == "or":
        return any(bool(evaluate(arg, envelope, history)) for arg in expr.get("args", []))
    if op == "not":
        return not bool(evaluate(expr["arg"], envelope, history))
    if op == "xor":
        return sum(bool(evaluate(arg, envelope, history)) for arg in expr.get("args", [])) % 2 == 1
    if op == "length":
        value = evaluate(expr["arg"], envelope, history)
        return len(value) if value is not MISSING and hasattr(value, "__len__") else MISSING
    if op == "abs":
        value = evaluate(expr["arg"], envelope, history)
        try:
            return abs(value)
        except (TypeError, ValueError):
            return MISSING
    if op == "count":
        value = evaluate(expr["arg"], envelope, history)
        return len(value) if isinstance(value, (list, tuple, set, dict, str)) else 0
    if op in {"url_scheme", "url_hostname", "url_origin"}:
        value = evaluate(expr["arg"], envelope, history)
        parts = _url_parts(value)
        return MISSING if parts is MISSING else parts[op.removeprefix("url_")]
    if op == "html_tag_count":
        return _html_tag_count(evaluate(expr["arg"], envelope, history))
    if op == "html_creates_nodes":
        return _html_tag_count(evaluate(expr["arg"], envelope, history)) > 0
    if op == "html_entity_decode":
        value = evaluate(expr["arg"], envelope, history)
        return unescape(value) if isinstance(value, str) else MISSING
    if op == "casefold":
        value = evaluate(expr["arg"], envelope, history)
        return value.casefold() if isinstance(value, str) else MISSING
    if op == "to_number":
        return _js_to_number(evaluate(expr["arg"], envelope, history))
    if op == "regex":
        value = evaluate(expr["arg"], envelope, history)
        if value is MISSING:
            return False
        try:
            return re.search(expr.get("pattern", ""), str(value)) is not None
        except re.error:
            return False
    if op == "if":
        branch = "then" if bool(evaluate(expr["condition"], envelope, history)) else "else"
        return evaluate(expr[branch], envelope, history)

    raise ValueError(f"unsupported Rule IR op: {op}")


def complexity(expr: dict[str, Any]) -> int:
    op = expr.get("op")
    if op in {"const", "field", "prev"}:
        return 1
    if op in {"not", "length", "abs", "count", "regex", "url_scheme", "url_hostname", "url_origin", "html_tag_count", "html_creates_nodes", "html_entity_decode", "casefold", "to_number"}:
        return 1 + complexity(expr.get("arg", {"op": "const", "value": None}))
    if op in {"and", "or", "xor"}:
        return 1 + sum(complexity(arg) for arg in expr.get("args", []))
    if op == "if":
        return 1 + complexity(expr["condition"]) + complexity(expr["then"]) + complexity(expr["else"])
    if "left" in expr and "right" in expr:
        return 1 + complexity(expr["left"]) + complexity(expr["right"])
    return 1


def canonical(expr: dict[str, Any]) -> str:
    """Canonical JSON used for structural de-duplication."""
    normalized = _normalize(expr)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize(expr: Any) -> Any:
    if not isinstance(expr, dict):
        return expr
    op = expr.get("op")
    out = {key: _normalize(value) for key, value in expr.items()}
    if op in {"and", "or", "xor"}:
        flattened: list[Any] = []
        for arg in out.get("args", []):
            if isinstance(arg, dict) and arg.get("op") == op:
                flattened.extend(arg.get("args", []))
            else:
                flattened.append(arg)
        out["args"] = sorted(flattened, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if op in {"eq", "ne"}:
        left_key = json.dumps(out.get("left"), sort_keys=True, ensure_ascii=False)
        right_key = json.dumps(out.get("right"), sort_keys=True, ensure_ascii=False)
        if left_key > right_key:
            out["left"], out["right"] = out["right"], out["left"]
    return out


def pretty(expr: dict[str, Any]) -> str:
    op = expr.get("op")
    if op == "const":
        return json.dumps(expr.get("value"), ensure_ascii=False)
    if op == "field":
        return expr["path"]
    if op == "prev":
        return f"prev[{expr.get('offset', 1)}].{expr['path']}"
    symbols = {
        "eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<=",
        "add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%",
        "contains": "contains", "starts_with": "startsWith", "ends_with": "endsWith", "origin_eq": "origin==", "in": "in",
    }
    if op in symbols:
        return f"({pretty(expr['left'])} {symbols[op]} {pretty(expr['right'])})"
    if op in {"and", "or", "xor"}:
        joiner = {"and": " AND ", "or": " OR ", "xor": " XOR "}[op]
        return "(" + joiner.join(pretty(arg) for arg in expr.get("args", [])) + ")"
    if op == "not":
        return f"NOT {pretty(expr['arg'])}"
    if op == "length":
        return f"len({pretty(expr['arg'])})"
    if op == "abs":
        return f"abs({pretty(expr['arg'])})"
    if op == "count":
        return f"count({pretty(expr['arg'])})"
    if op in {"url_scheme", "url_hostname", "url_origin", "html_tag_count", "html_creates_nodes", "html_entity_decode", "casefold", "to_number"}:
        return f"{op}({pretty(expr['arg'])})"
    if op == "regex":
        return f"regex({pretty(expr['arg'])}, {expr.get('pattern')!r})"
    if op == "if":
        return f"if {pretty(expr['condition'])} then {pretty(expr['then'])} else {pretty(expr['else'])}"
    return json.dumps(expr, ensure_ascii=False)


def truthy_result(expr: dict[str, Any], envelope: dict[str, Any], history: list[dict[str, Any]] | None = None) -> bool:
    value = evaluate(expr, envelope, history)
    return False if value is MISSING else bool(value)
