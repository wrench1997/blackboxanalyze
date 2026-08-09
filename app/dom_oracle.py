"""Controlled, inert DOM oracle for the local rule-maze labs.

This module parses markup with :class:`html.parser.HTMLParser`; it never mounts
content into a document, executes JavaScript, follows URLs, or opens a network
connection.  The browser implementation in ``frontend/lib/dom-oracle.ts``
uses the same evidence contract with a detached DOM node.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any, Iterable


ALLOWED_SINKS = frozenset({
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "template.innerHTML",
})
ALLOWED_TRANSFORMS = frozenset({"identity", "html_entity_decode", "casefold"})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class _DetachedParser(HTMLParser):
    def __init__(self, marker: str) -> None:
        super().__init__(convert_charrefs=True)
        self.marker = marker
        self.tags: list[str] = []
        self.marker_hits = 0
        self.script_like = False

    def _inspect(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        self.tags.append(normalized)
        if normalized in {"script", "iframe", "object", "embed"}:
            self.script_like = True
        for key, value in attrs:
            if str(key).casefold() == "data-sift-marker" and str(value) == self.marker:
                self.marker_hits += 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._inspect(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._inspect(tag, attrs)


def _parse(markup: str, marker: str) -> dict[str, Any]:
    parser = _DetachedParser(marker)
    try:
        parser.feed(markup)
        parser.close()
    except (TypeError, ValueError):
        # A parser failure is evidence about the representation, not a reason
        # to execute or repair the candidate input.
        pass
    return {
        "element_count": len(parser.tags),
        "tags": list(parser.tags[:32]),
        "marker_hits": parser.marker_hits,
        "script_like_present": parser.script_like,
    }


def apply_transforms(value: str, transforms: Iterable[str]) -> str:
    current = str(value)
    for transform in transforms:
        name = str(transform)
        if name not in ALLOWED_TRANSFORMS:
            raise ValueError(f"unsupported DOM oracle transform: {name}")
        if name == "html_entity_decode":
            current = unescape(current)
        elif name == "casefold":
            current = current.casefold()
    return current


@dataclass(frozen=True)
class DomOracleResult:
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.evidence)


def run_dom_oracle(
    value: str,
    *,
    sink: str = "innerHTML",
    transforms: Iterable[str] = (),
    marker: str = "sift-marker",
) -> DomOracleResult:
    """Run one inert source→transform→sink observation.

    ``value`` is treated as untrusted text.  The wrapper is not persisted, and
    only tag/marker counts and hashes are returned.  ``browser_sink_observed``
    means the controlled adapter exercised a known HTML sink; it does not mean
    that a script executed.
    """

    sink = str(sink)
    if sink not in ALLOWED_SINKS:
        raise ValueError(f"unsupported DOM oracle sink: {sink}")
    transforms = tuple(str(item) for item in transforms)
    transformed = apply_transforms(value, transforms)
    before = _parse("", marker)
    after = _parse(transformed, marker)
    projection = {
        "oracle": "controlled_detached_dom_v1",
        "sink": sink,
        "transforms": list(transforms),
        "source_sha256": hashlib.sha256(str(value).encode("utf-8")).hexdigest(),
        "transformed_sha256": hashlib.sha256(transformed.encode("utf-8")).hexdigest(),
        "before_element_count": before["element_count"],
        "after_element_count": after["element_count"],
        "marker_hits": after["marker_hits"],
        "tag_shape": after["tags"],
        "script_like_present": after["script_like_present"],
        "browser_sink_observed": True,
        "dom_change": after["element_count"] != before["element_count"],
        "candidate_signal": after["element_count"] > 0 or after["marker_hits"] > 0,
        "script_execution": False,
        "network_access": False,
        "navigation": False,
        "database_touched": False,
    }
    projection["evidence_hash"] = _digest(projection)
    return DomOracleResult(projection)

