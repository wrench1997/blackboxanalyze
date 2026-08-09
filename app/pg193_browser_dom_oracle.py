"""Loopback-only browser DOM oracle for PG-193.

The browser receives response markup in memory with JavaScript disabled and
all network routes aborted.  It reports only bounded DOM geometry and marker
counts.  A DOM structure effect is not an XSS positive: script execution is
explicitly impossible in this oracle.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{6,64}$")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_browser_dom_oracle(markup: str, *, marker: str) -> dict[str, Any]:
    """Parse markup in a no-JS browser context and return a typed projection."""

    marker = str(marker)
    if not MARKER_RE.fullmatch(marker):
        raise ValueError("PG-193 marker is not a bounded identifier")
    # Import lazily so non-browser tests can still import the repository.
    from playwright.sync_api import sync_playwright  # type: ignore

    aborted_urls: list[str] = []
    browser = None
    context = None
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(java_script_enabled=False, service_workers="block")
            page = context.new_page()

            def abort_route(route: Any) -> None:
                aborted_urls.append(str(route.request.url))
                route.abort()

            page.route("**/*", abort_route)
            page.set_content(str(markup), wait_until="domcontentloaded")
            marker_hits = int(page.locator(f'[data-sift-marker="{marker}"]').count())
            body_text_hits = int(page.locator("body").inner_text(timeout=2000).count(marker))
            tag_count = int(page.locator("*").count())
            script_count = int(page.locator("script").count())
            projection = {
                "oracle_id": "pg193-browser-dom-nojs-v1",
                "modality": "typed_dom_surface_effect" if marker_hits > 0 else "negative_or_untyped_surface",
                "marker_hits": min(marker_hits, 8),
                "body_text_hits": min(body_text_hits, 8),
                "element_count": min(tag_count, 512),
                "script_tag_count": min(script_count, 64),
                "browser_dom_observed": True,
                "dom_change": bool(marker_hits > 0),
                "script_execution": False,
                "network_request_count": len(aborted_urls),
                "network_access": False,
                "navigation": False,
                "database_touched": False,
                "raw_markup_stored": False,
            }
            projection["evidence_hash"] = _digest(projection)
            return projection
    finally:
        # The context/browser are normally closed by the sync_playwright
        # manager, but keep explicit cleanup for failed page creation.
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass


__all__ = ["MARKER_RE", "run_browser_dom_oracle"]
