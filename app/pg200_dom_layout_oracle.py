"""Fourth, independent static DOM layout oracle for PG-200.

It exercises template/textarea/SVG/shadow-like wrappers without a browser,
JavaScript, network access, or raw markup retention.  The result is a typed
surface-effect projection, never an XSS positive.
"""

from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from typing import Any


PG200_DOM_SCHEMA = "pg200-dom-layout-oracle-v1"
PG200_LAYOUTS = frozenset({"template_shell", "textarea_shell", "svg_shell", "shadow_like_shell"})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


class _LayoutParser(HTMLParser):
    def __init__(self, marker: str) -> None:
        super().__init__(convert_charrefs=True)
        self.marker = marker
        self.stack: list[str] = []
        self.marker_hits = 0
        self.visible_hits = 0
        self.script_hits = 0
        self.layout_hits = 0
        self.element_count = 0
        self.script_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = str(tag).casefold()
        self.element_count += 1
        self.stack.append(lower)
        if lower == "script":
            self.script_count += 1
        for key, value in attrs:
            text = "" if value is None else str(value)
            if self.marker not in text:
                continue
            self.marker_hits += 1
            if lower == "script" or "script" in self.stack:
                self.script_hits += 1
            if lower in {"svg", "template", "textarea", "slot", "section", "main"}:
                self.layout_hits += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self.stack and self.stack[-1] == str(tag).casefold():
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        lower = str(tag).casefold()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index] == lower:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if self.marker not in str(data):
            return
        self.marker_hits += 1
        if "script" in self.stack:
            self.script_hits += 1
        else:
            self.visible_hits += 1
            if self.stack and self.stack[-1] in {"svg", "template", "textarea", "slot", "section", "main"}:
                self.layout_hits += 1


def run_pg200_dom_layout_oracle(markup: str, *, marker: str, layout: str) -> dict[str, Any]:
    marker = str(marker)
    layout = str(layout)
    if not marker or len(marker) > 64 or not marker.replace("-", "").replace("_", "").isalnum():
        raise ValueError("PG-200 marker must be a bounded identifier")
    if layout not in PG200_LAYOUTS:
        raise ValueError("PG-200 layout is not allow-listed")
    parser = _LayoutParser(marker)
    try:
        parser.feed(str(markup))
        parser.close()
    except (TypeError, ValueError):
        pass
    typed_effect = bool(parser.layout_hits > 0 and parser.script_hits == 0)
    projection = {
        "schema_version": PG200_DOM_SCHEMA,
        "oracle_id": "pg200-fourth-dom-layout-parser-v1",
        "layout": layout,
        "modality": "typed_dom_surface_effect" if typed_effect else "negative_or_untyped_surface",
        "marker_hits": min(parser.marker_hits, 8),
        "visible_marker_hits": min(parser.visible_hits, 8),
        "layout_marker_hits": min(parser.layout_hits, 8),
        "element_count": min(parser.element_count, 512),
        "script_tag_count": min(parser.script_count, 64),
        "dom_change": typed_effect,
        "script_execution": False,
        "network_access": False,
        "database_touched": False,
        "navigation": False,
        "raw_markup_stored": False,
        "positive": False,
        "positive_authority": False,
        "vulnerability_claim_allowed": False,
    }
    projection["evidence_hash"] = _digest(projection)
    return projection


__all__ = ["PG200_DOM_LAYOUTS", "PG200_DOM_SCHEMA", "run_pg200_dom_layout_oracle"]
