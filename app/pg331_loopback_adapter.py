"""Loopback-only structural observation adapter for PG-331A.

This adapter performs a baseline GET or a neutral POST against an explicitly
loopback origin.  It keeps only abstract DOM/navigation/request/response/JS
shapes and bounded counts.  Response bodies and script text are parsed in
memory and immediately discarded; no payload, URL literal, cookie, credential,
or route text is returned.

The adapter is deliberately not a vulnerability probe.  A caller must still
provide an evaluator-side typed oracle, fresh-reset attestation and a reviewed
target projection before a source row can become training-eligible.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from html.parser import HTMLParser
from typing import Any

from .pg331_web_tokenizer import ONTOLOGY_PATH, tokenize_web_observation


SCHEMA_VERSION = "pg331-loopback-observation-v1"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_ELEMENTS = 4096
MAX_LINKS = 4096
MAX_FORMS = 256
MAX_PARAMETERS = 256
SAFE_TAGS = frozenset({"a", "button", "body", "div", "form", "head", "html", "img", "input", "label", "li", "link", "main", "meta", "nav", "option", "p", "script", "section", "style", "table", "textarea", "title", "ul"})
SURFACE_ROLE_VALUES = frozenset(
    {
        "static_label",
        "display_text",
        "display_preference",
        "query_text",
        "query_term",
        "filter_choice",
        "attribute_value",
        "path_segment",
        "fragment_identifier",
        "json_value",
        "dom_text",
        "form_field",
        "profile_key",
        "view_mode",
        "tab_name",
        "sort_direction",
        "record_cursor",
        "step_index",
        "metric_group",
        "note_text",
        "notice_state",
        "status_label",
        "list_item",
    }
)
SURFACE_ENCODING_VALUES = frozenset(
    {
        "identity",
        "url_percent",
        "form_urlencode",
        "form_urlencoded",
        "json_string",
        "json_object",
        "fragment",
        "query_parameter",
        "utf8",
    }
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _bucket_count(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return "zero" if number <= 0 else "one" if number == 1 else "two" if number == 2 else "few" if number <= 5 else "many"


def _bucket_length(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return "empty" if number <= 0 else "short" if number <= 64 else "medium" if number <= 1024 else "long"


def _shape(value: Any) -> str:
    if value is None or value == "":
        return "empty"
    text = str(value)
    if re.fullmatch(r"[0-9]+", text):
        return "numeric"
    if re.fullmatch(r"[A-Za-z]+", text):
        return "alpha"
    if re.fullmatch(r"[A-Za-z0-9_-]+", text):
        return "word_mixed"
    if "/" in text or "\\" in text:
        return "path_like"
    return "mixed"


def _enum(value: Any, default: str = "unknown") -> str:
    text = str(value if value is not None and value != "" else default).strip().casefold().replace("-", "_").replace(" ", "_")
    return text[:64] if re.fullmatch(r"[a-z0-9_.:]{1,64}", text) else default


def _parameter_role(name: Any, input_type: Any = "text") -> str:
    """Project a field name into a bounded semantic role, never a literal name.

    GET/POST parameter identity is useful for composition, but retaining the
    original key would leak route-specific strings into the model context.  A
    small, versioned role vocabulary preserves the distinction needed by the
    Rule-IR layer while keeping the raw key in the evaluator-side request only.
    """

    normalized = re.sub(r"[^a-z0-9]+", "_", str(name or "").casefold()).strip("_")
    kind = str(input_type or "text").casefold()
    if kind in {"submit", "button", "reset", "image"} or normalized in {"submit", "commit", "action", "go"}:
        return "submit_control"
    if normalized in {"csrf", "csrf_token", "xsrf", "xsrf_token", "nonce", "_token"}:
        return "anti_csrf"
    if normalized in {"id", "uid", "user_id", "item_id", "post_id", "record_id", "account_id"} or normalized.endswith("_id"):
        return "identifier"
    if normalized in {"name", "q", "query", "search", "keyword", "term", "text", "value"}:
        return "query_term"
    if normalized in {"url", "uri", "redirect", "next", "return", "return_url", "callback", "target"}:
        return "destination"
    if normalized in {"email", "username", "user", "login", "account"}:
        return "account_identifier"
    if kind == "hidden":
        return "hidden_field"
    return "named_field"


def _status_class(status: int | None) -> str:
    if status is None:
        return "transport"
    return f"{status // 100}xx" if 100 <= status < 600 else "unknown"


def _header_value(headers: Mapping[str, Any], name: str) -> str:
    wanted = str(name).casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return str(value)
    return ""


def _content_type(headers: Mapping[str, Any]) -> str:
    raw = _header_value(headers, "Content-Type").casefold()
    if not raw:
        return "absent"
    if "html" in raw:
        return "html"
    if "json" in raw:
        return "json"
    if "text" in raw or raw:
        return "text"
    return "unknown"


def _csrf_presence_class(parameters: list[Mapping[str, Any]], *, status: int | None) -> str:
    """Classify only what the captured request actually carried.

    A completed request with no CSRF-shaped parameter is an observed absence;
    a transport failure remains unknown because the request was not observed.
    """

    if status is None:
        return "unknown"
    for parameter in parameters:
        role = str(parameter.get("role", "")).casefold()
        if role == "anti_csrf":
            return "present"
    return "absent"


def _safe_target_shape(raw: str, origin: urllib.parse.SplitResult) -> tuple[str, str, str, str]:
    parsed = urllib.parse.urlsplit(urllib.parse.urljoin(origin.geturl(), raw))
    absolute = bool(parsed.netloc)
    same_origin = "yes" if not absolute or (parsed.hostname or "").casefold() in LOOPBACK_HOSTS else "unknown"
    return _shape(parsed.path), same_origin, "present" if bool(parsed.query) else "absent", "present" if bool(parsed.fragment) else "absent"


def _field_capture_manifest(observation: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Derive an explicit field-status sidecar from the same token stream."""

    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8-sig"))
    tokenized = tokenize_web_observation(observation)
    values: dict[tuple[str, str], str] = {}
    for token in tokenized.get("context_tokens") or []:
        text = str(token)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        for axis in dict(ontology.get("axes") or {}):
            prefix = f"{axis}_field_"
            if key.startswith(prefix):
                values[(str(axis), key[len(prefix) :])] = value
                break
    manifest: dict[str, dict[str, str]] = {}
    for axis, spec in dict(ontology.get("axes") or {}).items():
        manifest[str(axis)] = {}
        for field in list(spec.get("fields") or []):
            value = values.get((str(axis), str(field)), "not_observed")
            manifest[str(axis)][str(field)] = "absent" if value == "absent" else "unknown" if value == "unknown" else "not_observed" if value == "not_observed" else "observed"
    return manifest


class _PageParser(HTMLParser):
    def __init__(self, origin: urllib.parse.SplitResult):
        super().__init__(convert_charrefs=True)
        self.origin = origin
        self.elements: list[dict[str, Any]] = []
        self.links: list[dict[str, Any]] = []
        self.forms: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []
        self._title_length = 0
        self._title_shape = "empty"
        self._doctype = "unknown"
        self.script_text_lengths: list[int] = []
        self.script_text_shapes: list[str] = []
        self.script_kinds: list[str] = []
        self.script_inline_external: list[str] = []
        self.event_handler_count = 0
        self.event_handler_kinds: list[str] = []
        self.fetch_count = 0
        self.xhr_count = 0
        self.source_category = "none"
        self.sink_category = "none"
        self.syntax_shape = "empty"
        self.dynamic_code_presence = "absent"
        self.storage_api_presence = "absent"
        self.head_count = 0
        self.meta_count = 0
        self.style_count = 0
        self.script_count = 0
        self.section_count = 0
        self.section_order: list[str] = []
        self.tag_counts: dict[str, int] = {}
        self.html_lang = "unknown"
        # These values are read from bounded, visible page metadata/data
        # attributes.  They are semantic surface hints, not route names or
        # target labels; raw attribute values are never retained.
        self.surface_parameter_roles: list[str] = []
        self.surface_encoding_chains: list[str] = []
        self.surface_transport_methods: list[str] = []

    def handle_decl(self, decl: str) -> None:
        if str(decl).casefold().startswith("doctype"):
            self._doctype = "html"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = str(tag).casefold()
        attrs_map = {str(key).casefold(): value for key, value in attrs}
        role_hint = attrs_map.get("data-parameter-role")
        if role_hint is not None:
            normalized_role = str(role_hint).casefold().replace("-", "_").strip()
            self.surface_parameter_roles.append(normalized_role if normalized_role in SURFACE_ROLE_VALUES else "unknown")
        encoding_hint = attrs_map.get("data-encoding-chain")
        if encoding_hint is not None:
            normalized_encoding = str(encoding_hint).casefold().replace("-", "_").replace("+", "_then_").replace(" ", "_").strip()
            parts = tuple(part for part in normalized_encoding.split("_then_") if part)
            encoding_valid = normalized_encoding in SURFACE_ENCODING_VALUES or (len(parts) >= 2 and all(part in SURFACE_ENCODING_VALUES for part in parts))
            self.surface_encoding_chains.append(normalized_encoding if encoding_valid else "unknown")
        transport_hint = attrs_map.get("data-transport-method")
        if transport_hint is not None:
            normalized_transport = str(transport_hint).upper().strip()
            self.surface_transport_methods.append(normalized_transport if normalized_transport in {"GET", "POST"} else "UNKNOWN")
        self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1
        if tag == "html":
            self.html_lang = _enum(attrs_map.get("lang"), "unknown")
        if tag == "head":
            self.head_count += 1
        elif tag == "meta":
            self.meta_count += 1
        elif tag == "style":
            self.style_count += 1
        elif tag == "script":
            self.script_count += 1
            self.script_kinds.append("module" if str(attrs_map.get("type", "")).casefold() == "module" else "classic")
            self.script_inline_external.append("external" if attrs_map.get("src") else "inline")
        elif tag in {"section", "main", "nav", "header", "footer", "article", "aside"}:
            self.section_count += 1
            self.section_order.append(tag)
        for key in attrs_map:
            if key.startswith("on"):
                self.event_handler_count += 1
                self.event_handler_kinds.append(key[2:] or "unknown")
        if tag == "a":
            shape, same_origin, query, fragment = _safe_target_shape(str(attrs_map.get("href") or ""), self.origin)
            if len(self.links) < MAX_LINKS:
                self.links.append({"method": "GET", "target_shape": shape, "same_origin": same_origin, "query_present": query, "fragment_present": fragment})
        if tag == "form":
            if len(self.forms) < MAX_FORMS:
                self.forms.append({"method": _enum(attrs_map.get("method"), "GET").upper(), "action_shape": _shape(urllib.parse.urlsplit(urllib.parse.urljoin(self.origin.geturl(), str(attrs_map.get("action") or ""))).path), "parameters": []})
        element_ref: dict[str, Any] | None = None
        if tag in SAFE_TAGS and len(self.elements) < MAX_ELEMENTS:
            depth = len(self._stack) + 1
            element_ref = {"tag": tag if tag in SAFE_TAGS else "other", "depth": depth, "sibling_count": 0, "role": _enum(attrs_map.get("role"), tag), "aria_role": _enum(attrs_map.get("aria-role"), "absent"), "id_shape": _shape(attrs_map.get("id")), "class_shape": _shape(attrs_map.get("class")), "text_shape": "empty", "text_length": 0, "attribute_presence": sorted(attrs_map)[:32]}
            self.elements.append(element_ref)
        if tag == "input" and self.forms:
            current = self.forms[-1]
            if len(current["parameters"]) < MAX_PARAMETERS:
                input_type = _enum(attrs_map.get("type"), "text")
                current["parameters"].append({"role": _parameter_role(attrs_map.get("name"), input_type), "name_shape": _shape(attrs_map.get("name")), "value_type": input_type, "presence": "present" if "name" in attrs_map else "absent", "order": len(current["parameters"]) + 1})
        if element_ref is not None:
            self._stack.append(element_ref)
        elif tag in {"script", "style"}:
            self._stack.append({"tag": tag, "text_length": 0, "text_shape": "empty"})
        elif tag in SAFE_TAGS:
            self._stack.append({"tag": tag, "text_length": 0, "text_shape": "empty"})

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = str(tag).casefold()
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index].get("tag") == tag:
                node = self._stack.pop(index)
                if tag == "script":
                    self.script_text_lengths.append(int(node.get("text_length", 0)))
                    self.script_text_shapes.append(str(node.get("text_shape", "empty")))
                break

    def handle_data(self, data: str) -> None:
        if not data:
            return
        length = len(data)
        shape = _shape(data.strip()) if data.strip() else "empty"
        if self._stack:
            node = self._stack[-1]
            node["text_length"] = int(node.get("text_length", 0)) + length
            node["text_shape"] = shape if node.get("text_shape") in {None, "empty"} else "mixed"
            if node.get("tag") == "title":
                self._title_length += length
                self._title_shape = shape if self._title_shape == "empty" else "mixed"
        # Script data is seen by HTMLParser as data while script is on stack;
        # count only abstract lexical classes, never retain the source text.
        if self._stack and self._stack[-1].get("tag") == "script":
            folded = data.casefold()
            self.fetch_count += len(re.findall(r"\bfetch\s*\(", folded))
            self.xhr_count += len(re.findall(r"\bxmlhttprequest\b|\.open\s*\(", folded))
            if re.search(r"location(?:\.search|\.hash)?|document\.cookie|input|form", folded):
                self.source_category = "dom_or_location_input"
            if re.search(r"innerhtml|outerhtml|document\.write|insertadjacenthtml|eval\s*\(", folded):
                self.sink_category = "dom_update_or_dynamic_code"
            if re.search(r"\beval\s*\(|new\s+function|settimeout\s*\(", folded):
                self.dynamic_code_presence = "present"
            if re.search(r"localstorage|sessionstorage|indexeddb", folded):
                self.storage_api_presence = "present"
            if re.search(r"[{}();=]", data):
                self.syntax_shape = "statement_or_call"

    def observation_projection(self, *, request_method: str, request_data: Mapping[str, Any] | None, query_data: Mapping[str, Any] | None, response: Mapping[str, Any]) -> dict[str, Any]:
        elements = list(self.elements)
        for element in elements:
            element["text_length"] = _bucket_length(element.get("text_length"))
            element["text_shape"] = element.get("text_shape") or "empty"
        form_parameters: list[dict[str, Any]] = []
        for form in self.forms:
            form_parameters.extend(list(form.get("parameters") or []))
        request_parameters: list[dict[str, Any]] = []
        if request_data:
            for index, (name, value) in enumerate(request_data.items()):
                request_parameters.append({"role": _parameter_role(name, type(value).__name__), "name_shape": _shape(name), "value_type": _enum(type(value).__name__, "text"), "presence": "present", "order": index + 1})
        query_parameters: list[dict[str, Any]] = []
        if query_data:
            for index, (name, value) in enumerate(query_data.items()):
                query_parameters.append({"role": _parameter_role(name, "text"), "name_shape": _shape(name), "value_type": "text", "presence": "present", "order": index + 1})
        path_parts = [part for part in str(response.get("path", "")).split("/") if part]
        query_key_shapes = list(response.get("query_key_shapes") or [])
        parameters = (request_parameters if str(request_method).upper() == "POST" else query_parameters) or form_parameters[:MAX_PARAMETERS]
        # A visible surface may declare the semantic role of a control even
        # when this baseline request carries no parameter.  Preserve that
        # bounded hint as an abstract parameter entry so the model can tell
        # query/path/DOM/JSON surfaces apart instead of receiving a false
        # zero-parameter observation.  ``surface_observed`` explicitly
        # distinguishes it from a sent request parameter.
        if self.surface_parameter_roles:
            observed_roles = {str(item.get("role")) for item in parameters if isinstance(item, Mapping)}
            for role in dict.fromkeys(self.surface_parameter_roles[:MAX_PARAMETERS]):
                if role not in observed_roles:
                    parameters.append(
                        {
                            "role": role,
                            "name_shape": "abstract",
                            "value_type": "surface_hint",
                            "presence": "surface_observed",
                            "order": 0,
                        }
                    )
        encoding_chain = str(self.surface_encoding_chains[0]) if self.surface_encoding_chains else ("form_urlencoded" if str(request_method).upper() == "POST" else "url_percent")
        return {
            "document_structure": {"doctype": self._doctype, "html_lang": self.html_lang, "title_shape": self._title_shape, "head_count": self.head_count, "meta_count": self.meta_count, "style_count": self.style_count, "script_count": self.script_count, "section_count": self.section_count, "section_order": list(self.section_order), "repeated_element_count": max(self.tag_counts.values(), default=0), "elements": elements},
            "navigation": {"links": self.links, "link_count": len(self.links), "path_segment_count": len(path_parts), "path_segment_shape": [_shape(part) for part in path_parts], "query_key_count": int(response.get("query_key_count", 0) or 0), "query_key_shape": query_key_shapes, "navigation_event": "initial_load", "form_action_shape": _shape(self.forms[0].get("action_shape") if self.forms else "")},
            "request_transport": {"method": str(request_method).upper(), "placement": "form" if str(request_method).upper() == "POST" else "query", "content_type_class": "form_urlencoded" if str(request_method).upper() == "POST" else "none", "encoding_chain": encoding_chain, "charset_class": "utf8", "body_shape": "form" if str(request_method).upper() == "POST" and request_data else "empty", "query_count": int(response.get("query_key_count", 0) or 0) if str(request_method).upper() == "GET" else 0, "form_count": len(request_parameters) if str(request_method).upper() == "POST" else len(form_parameters), "json_field_count": 0, "multipart_part_count": 0, "header_presence_class": "basic", "cookie_presence_class": "absent", "csrf_presence_class": response.get("csrf_presence_class", "unknown"), "content_length": int(response.get("request_content_length", 0) or 0), "parameters": parameters},
            "response_transport": {"status_class": response.get("status_class", "transport"), "status_shape": response.get("status_shape", "unknown"), "content_type_class": response.get("content_type_class", "unknown"), "connection_outcome": response.get("connection_outcome", "complete"), "body_length": response.get("body_length", 0), "redirect_hop_count": response.get("redirect_hop_count", 0), "body_shape": response.get("body_shape", "empty"), "charset_class": response.get("charset_class", "unknown"), "header_presence_class": response.get("header_presence_class", "basic"), "cache_shape": response.get("cache_shape", "unknown"), "redirect_location_class": response.get("redirect_location_class", "none"), "redirect_chain_shape": response.get("redirect_chain_shape", "empty")},
            "javascript_surface": {"script_count": self.script_count, "event_handler_count": self.event_handler_count, "fetch_count": self.fetch_count, "xhr_count": self.xhr_count, "ast_node_count": sum(max(1, length // 4) for length in self.script_text_lengths), "script_kind": _enum(self.script_kinds[0] if self.script_kinds else "none", "none"), "module_presence": "present" if "module" in self.script_kinds else "absent", "inline_external_class": _enum(self.script_inline_external[0] if self.script_inline_external else "none", "none"), "source_category": self.source_category, "sink_category": self.sink_category, "parser_error_class": "none", "syntax_shape": self.syntax_shape, "ast_node_shape": _enum(self.script_text_shapes[0] if self.script_text_shapes else "empty", "empty"), "dynamic_code_presence": self.dynamic_code_presence, "storage_api_presence": self.storage_api_presence, "fetch_method": "GET" if self.fetch_count else "ABSENT", "xhr_method": "GET" if self.xhr_count else "ABSENT", "fetch_target_shape": "path_like" if self.fetch_count else "empty", "xhr_target_shape": "path_like" if self.xhr_count else "empty", "event_handler_kinds": self.event_handler_kinds[:MAX_PARAMETERS]},
            "failure_feedback": {"failure_class": response.get("failure_class", "none"), "failure_stage": response.get("failure_stage", "none"), "error_shape": response.get("error_shape", "empty"), "parse_error_class": "none", "encoding_error_class": "none", "redirect_error_class": "none", "blocked_reason_class": "none", "environment_failure_class": response.get("environment_failure_class", "none"), "previous_action": "none", "next_action": "ask", "repair_delta_axis": "none", "repair_outcome": "not_applicable", "timeout_ms": response.get("timeout_ms", 0)},
            "belief_and_replay": {"observation_presence": "present", "observation_delta_axis": "document_structure", "belief_prior_bucket": "unknown", "belief_posterior_bucket": "unknown", "belief_delta_axis": "document_structure", "history_action": "baseline_observe", "typed_available": "absent", "evidence_present": "absent", "negative_control": "unknown", "fresh_reset": "unknown", "replay_ready": "absent", "reference_present": "absent", "candidate_present": "absent", "step_budget": "unknown", "evidence_hash_present": "absent", "history_length": 1, "probe_count": 0},
        }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Mapping[str, Any], newurl: str) -> None:  # type: ignore[override]
        return None


def _assert_loopback(origin: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").casefold() not in LOOPBACK_HOSTS:
        raise ValueError("PG-331 loopback adapter accepts only localhost/127.0.0.1/::1")
    if not parsed.netloc or not parsed.port:
        raise ValueError("PG-331 loopback origin must include an explicit port")
    return parsed


def capture_loopback(
    origin: str,
    *,
    method: str = "GET",
    form_data: Mapping[str, Any] | None = None,
    timeout: float = 5.0,
    evaluator: Callable[[bytes, Mapping[str, Any], int | None], Mapping[str, Any]] | None = None,
    abstract_probe_variant: str | None = None,
) -> dict[str, Any]:
    """Capture one local page and optionally pass its bytes to a typed evaluator.

    The callback runs while the body is still in memory.  Only its bounded
    abstract projection is returned; the adapter never serializes response
    bytes.  Ordinary collection leaves the callback unset and remains ASK-only.
    """

    parsed_origin = _assert_loopback(origin)
    method = str(method).upper()
    if method not in {"GET", "POST"}:
        raise ValueError("PG-331 loopback method must be GET or POST")
    data = None
    request_url = origin
    query_values = urllib.parse.parse_qs(parsed_origin.query, keep_blank_values=True)
    query_key_count = len(query_values)
    query_key_shapes = [_shape(key) for key in query_values]
    if method == "POST":
        safe_data = {str(key): "" if value is None else str(value) for key, value in dict(form_data or {}).items()}
        data = urllib.parse.urlencode(safe_data).encode("utf-8")
    headers = {"Accept": "text/html,application/json;q=0.8,*/*;q=0.1", "User-Agent": "pg331-loopback-observer/1"}
    if abstract_probe_variant is not None:
        # The value is an allow-listed abstract reference consumed only by a
        # reviewed local evaluator; it is never serialized into model rows.
        variant = str(abstract_probe_variant).casefold()
        if not re.fullmatch(r"[a-z0-9_]{1,64}", variant):
            raise ValueError("PG-331 abstract probe variant must be a bounded symbol")
        headers["X-PG348-Probe-Variant"] = variant
    request = urllib.request.Request(request_url, data=data, method=method, headers=headers)
    opener = urllib.request.build_opener(_NoRedirect())
    status: int | None = None
    headers: Mapping[str, Any] = {}
    body = b""
    failure = "none"
    timeout_ms = 0
    try:
        response = opener.open(request, timeout=timeout)
        status = int(response.getcode())
        headers = dict(response.headers.items())
        body = response.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as error:
        status = int(error.code)
        headers = dict(error.headers.items()) if error.headers else {}
        try:
            body = error.read(MAX_BODY_BYTES + 1)
        except Exception:
            body = b""
        failure = "http_error" if status >= 400 else "redirect_not_followed"
    except TimeoutError:
        failure = "timeout"
        timeout_ms = int(timeout * 1000)
    except urllib.error.URLError:
        failure = "connection_error"
    body = body[:MAX_BODY_BYTES]
    evaluator_projection: dict[str, Any] | None = None
    if evaluator is not None:
        projected = evaluator(body, headers, status)
        if not isinstance(projected, Mapping):
            raise ValueError("PG-331 evaluator callback must return an abstract mapping")
        forbidden_keys = {"body", "raw_body", "response_body", "html", "markup", "source_code", "payload", "probe"}
        if any(str(key).casefold() in forbidden_keys for key in projected):
            raise ValueError("PG-331 evaluator callback returned raw material")
        evaluator_projection = {str(key): value for key, value in projected.items()}
    content_class = _content_type(headers)
    parser = _PageParser(parsed_origin)
    # Some authorized local fixtures (including the fixed Pikachu image) omit
    # Content-Type even when the response is HTML.  Parse an explicitly
    # HTML-shaped body in memory as a structural observation; the body is
    # never persisted or exposed to the model.  This avoids turning a missing
    # transport header into false ``unknown`` document fields.
    body_leading = body.lstrip().lower()
    looks_like_html = body_leading.startswith((b"<!doctype", b"<html", b"<head", b"<body"))
    # A few authorized fixtures omit Content-Type.  Infer only unambiguous
    # HTML/JSON structure from the bounded in-memory prefix; otherwise retain
    # unknown rather than guessing a response type.
    parse_content_class = content_class
    if content_class in {"absent", "unknown"}:
        if looks_like_html:
            parse_content_class = "html"
        elif body_leading.startswith((b"{", b"[")):
            parse_content_class = "json"
        elif body and content_class == "absent":
            parse_content_class = "text"
    if body and (parse_content_class == "html" or looks_like_html):
        try:
            parser.feed(body.decode("utf-8", "replace"))
        except Exception:
            failure = "parse_error"
    location_present = bool(headers.get("Location"))
    header_names = {str(key).casefold() for key in headers}
    cache_header_names = {"cache-control", "pragma", "expires", "etag", "age", "last-modified", "vary"}
    cache_shape = "present" if header_names & cache_header_names else "absent"
    content_type_projection = parse_content_class if content_class == "absent" and parse_content_class in {"html", "json"} else content_class
    content_type_header = _header_value(headers, "Content-Type")
    charset_match = re.search(r"charset\s*=\s*['\"]?\s*([a-z0-9._-]+)", content_type_header, flags=re.IGNORECASE)
    if charset_match:
        charset_class = "utf8" if charset_match.group(1).casefold() in {"utf-8", "utf8"} else "other"
    elif content_type_header:
        charset_class = "absent"
    else:
        charset_class = "absent" if status is not None else "unknown"
    request_parameters_for_presence = []
    if method == "POST":
        for index, (name, value) in enumerate((form_data or {}).items()):
            request_parameters_for_presence.append({"role": _parameter_role(name, type(value).__name__), "order": index + 1})
    else:
        for index, (name, value) in enumerate(query_values.items()):
            request_parameters_for_presence.append({"role": _parameter_role(name, "text"), "order": index + 1})
    response_projection = {
        "status_class": _status_class(status),
        "status_shape": "numeric" if status is not None else "unknown",
        "content_type_class": content_type_projection,
        "connection_outcome": "complete" if status is not None else "transport_error",
        "body_length": len(body),
        "body_shape": "html" if parser.tag_counts.get("html") or parser.tag_counts.get("body") else ("json" if parse_content_class == "json" else "text" if body else "empty"),
        "redirect_hop_count": 1 if status is not None and 300 <= status < 400 else 0,
        "redirect_location_class": "present" if location_present else "none",
        "redirect_chain_shape": "single_hop" if status is not None and 300 <= status < 400 else "empty",
        "cache_shape": cache_shape,
        "charset_class": charset_class,
        "header_presence_class": "basic" if headers else "absent",
        "query_key_count": query_key_count,
        "query_key_shapes": query_key_shapes,
        "request_content_length": len(data or b""),
        "failure_class": failure,
        "failure_stage": "transport" if failure != "none" else "none",
        "error_shape": _shape(failure),
        "timeout_ms": timeout_ms,
        "environment_failure_class": failure if failure in {"timeout", "connection_error"} else "none",
        "path": parsed_origin.path,
        "csrf_presence_class": _csrf_presence_class(request_parameters_for_presence, status=status),
    }
    observation = parser.observation_projection(request_method=method, request_data=form_data if method == "POST" else None, query_data=query_values if method == "GET" else None, response=response_projection)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "origin_digest": _digest({"scheme": parsed_origin.scheme, "host": parsed_origin.hostname, "port": parsed_origin.port}),
        # A connection exception is an environment observation, not proof
        # that the target accepted a request.  Keep this boolean strict so a
        # stopped service cannot inflate the smoke contact count.
        "target_contacted": status is not None,
        "raw_body_stored": False,
        "raw_payload_stored": False,
        "observation": observation,
        "field_capture_manifest": _field_capture_manifest(observation),
        "transport": {"method": method, "status_class": response_projection["status_class"], "content_type_class": content_class, "redirect_hop_count": response_projection["redirect_hop_count"], "failure_class": response_projection["failure_class"]},
    }
    if evaluator_projection is not None:
        result["evaluator_projection"] = evaluator_projection
    return result


__all__ = ["LOOPBACK_HOSTS", "SCHEMA_VERSION", "capture_loopback"]
