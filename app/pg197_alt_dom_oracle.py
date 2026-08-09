"""Independent static DOM projection for PG-197 source agreement.

This evaluator intentionally does not use Playwright.  It parses the in-memory
markup with Python's HTMLParser, counts only bounded marker geometry, and never
executes JavaScript or retains the markup.  Agreement with the no-JS browser
oracle is a surface-effect check, not an XSS positive.
"""

from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from typing import Any


class _MarkerParser(HTMLParser):
    def __init__(self, marker: str) -> None:
        super().__init__(convert_charrefs=True)
        self.marker = marker
        self.marker_hits = 0
        self.text_hits = 0
        self.attribute_hits = 0
        self.script_marker_hits = 0
        self.element_count = 0
        self.script_count = 0
        self._script_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.element_count += 1
        if str(tag).casefold() == "script":
            self.script_count += 1
            self._script_depth += 1
        for key, value in attrs:
            text = "" if value is None else str(value)
            if self.marker in text:
                self.marker_hits += 1
                self.attribute_hits += 1
                if str(tag).casefold() == "script" or self._script_depth > 0:
                    self.script_marker_hits += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if str(tag).casefold() == "script" and self._script_depth > 0:
            self._script_depth -= 1

    def handle_data(self, data: str) -> None:
        text = str(data)
        if self.marker in text:
            self.marker_hits += 1
            self.text_hits += 1
            if self._script_depth > 0:
                self.script_marker_hits += 1


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def run_alt_dom_oracle(markup: str, *, marker: str) -> dict[str, Any]:
    marker = str(marker)
    if not marker or len(marker) > 64 or not marker.replace("-", "").replace("_", "").isalnum():
        raise ValueError("PG-197 marker must be a bounded identifier")
    parser = _MarkerParser(marker)
    try:
        parser.feed(str(markup))
        parser.close()
    except (TypeError, ValueError):
        pass
    # Marker text inside a script is deliberately not treated as a safe DOM
    # effect; scripts are data for this parser and never execute.
    typed_surface_effect = bool(parser.marker_hits > 0 and parser.script_marker_hits == 0)
    projection = {
        "oracle_id": "pg197-static-dom-parser-v1",
        "modality": "typed_dom_surface_effect" if typed_surface_effect else "negative_or_untyped_surface",
        "marker_hits": min(parser.marker_hits, 8),
        "text_hits": min(parser.text_hits, 8),
        "attribute_hits": min(parser.attribute_hits, 8),
        "element_count": min(parser.element_count, 512),
        "script_tag_count": min(parser.script_count, 64),
        "browser_dom_observed": False,
        "dom_change": typed_surface_effect,
        "script_execution": False,
        "network_request_count": 0,
        "network_access": False,
        "navigation": False,
        "database_touched": False,
        "raw_markup_stored": False,
    }
    projection["evidence_hash"] = _digest(projection)
    return projection


__all__ = ["run_alt_dom_oracle"]
