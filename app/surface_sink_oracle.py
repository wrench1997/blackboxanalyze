"""Language-neutral, inert sink binding for bounded local responses.

The parser observes where a canary appears (attribute, text, JSON value,
header, or script source) without mounting HTML, executing JavaScript, following
URLs, or retaining the response body.  It is an evidence oracle, not a payload
generator.
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from typing import Any, Mapping

from .maze_engine import sha256_json


SINK_ORACLE_SCHEMA = "sift-surface-sink-oracle-v1"
SINK_KINDS = ("html_attribute", "html_text", "json_value", "response_header", "script_source", "none")
_MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


class _SinkParser(HTMLParser):
    def __init__(self, marker: str) -> None:
        super().__init__(convert_charrefs=True)
        self.marker = marker
        self.attribute_hits = 0
        self.text_hits = 0
        self.script_hits = 0
        self.tags = 0
        self.attributes = 0
        self.scripts = 0
        self._script_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        self.tags += 1
        self.attributes += len(attrs)
        if normalized == "script":
            self.scripts += 1
            self._script_depth += 1
        for _key, value in attrs:
            if value is not None and self.marker in str(value):
                self.attribute_hits += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() == "script" and self._script_depth:
            self._script_depth -= 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "script" and self._script_depth:
            self._script_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.marker not in data:
            return
        if self._script_depth:
            self.script_hits += 1
        else:
            self.text_hits += 1


def _json_marker_hits(value: Any, marker: str) -> int:
    if isinstance(value, str):
        return int(value == marker)
    if isinstance(value, dict):
        return sum(_json_marker_hits(child, marker) for child in value.values())
    if isinstance(value, list):
        return sum(_json_marker_hits(child, marker) for child in value)
    return 0


def observe_surface_sink(
    body: str | bytes,
    *,
    marker: str,
    content_type: str,
    headers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a bounded sink projection without persisting ``body``."""

    if not _MARKER_RE.fullmatch(str(marker)):
        raise ValueError("surface sink oracle marker must be an inert identifier")
    raw = body if isinstance(body, bytes) else str(body).encode("utf-8", errors="replace")
    text = raw.decode("utf-8", errors="replace")
    normalized_type = str(content_type).casefold()
    parser = _SinkParser(str(marker))
    try:
        parser.feed(text)
        parser.close()
    except (TypeError, ValueError):
        pass
    json_hits = 0
    if normalized_type.startswith("application/json"):
        try:
            json_hits = _json_marker_hits(json.loads(text), str(marker))
        except (TypeError, ValueError, json.JSONDecodeError):
            json_hits = 0
    header_hits = sum(
        int(str(value) == str(marker))
        for value in (dict(headers or {}).values())
    )
    kind = "none"
    if parser.attribute_hits:
        kind = "html_attribute"
    elif parser.script_hits:
        kind = "script_source"
    elif json_hits:
        kind = "json_value"
    elif parser.text_hits:
        kind = "html_text"
    elif header_hits:
        kind = "response_header"
    projection = {
        "oracle": SINK_ORACLE_SCHEMA,
        "sink_kind": kind,
        "marker_reflected": bool(parser.attribute_hits or parser.text_hits or parser.script_hits or json_hits or header_hits),
        "marker_in_attribute": bool(parser.attribute_hits),
        "marker_in_html_text": bool(parser.text_hits),
        "marker_in_json_value": bool(json_hits),
        "marker_in_header": bool(header_hits),
        "marker_in_script_source": bool(parser.script_hits),
        "marker_count": min(parser.attribute_hits + parser.text_hits + parser.script_hits + json_hits + header_hits, 8),
        "html_tag_count": min(parser.tags, 64),
        "html_attribute_count": min(parser.attributes, 64),
        "script_count": min(parser.scripts, 32),
        "content_type_class": "json" if normalized_type.startswith("application/json") else "html" if normalized_type.startswith("text/html") else "other",
        "body_sha256": hashlib.sha256(raw).hexdigest(),
        "script_execution": False,
        "network_access": False,
        "navigation": False,
        "database_touched": False,
    }
    projection["evidence_hash"] = sha256_json(projection)
    return projection


__all__ = ["SINK_KINDS", "SINK_ORACLE_SCHEMA", "observe_surface_sink"]
