# -*- coding: utf-8 -*-
"""PG-268A: browser-discover all Pikachu request surfaces.

PG-179 crawled the local app in DOM-read-only mode but never submitted a form,
so its 112 request surfaces had no parameterized response evidence.  This
stage uses Playwright against one fresh, no-volume loopback container to
discover same-origin links, form actions, methods, encodings and field names.
It deliberately does not submit a form or store source/response bodies; the
next replay stage consumes this bounded manifest and creates a fresh container
per route.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit

try:
    from playwright.sync_api import Browser, sync_playwright
except Exception:  # pragma: no cover
    Browser = Any  # type: ignore[assignment,misc]
    sync_playwright = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_pg214() -> Any:
    path = ROOT / "scripts" / "run_pg214_pikachu_fixed_sql_loop.py"
    spec = importlib.util.spec_from_file_location("pg268_pg214_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-214 fresh Pikachu helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load_pg214()
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg268_pikachu_browser_parameterized_crawl_report_v1.json"
MANIFEST = RESEARCH / "pg268_pikachu_browser_parameterized_crawl_manifest_v1.json"
PROTOCOL = RESEARCH / "pg268_pikachu_browser_parameterized_crawl_protocol_v1.json"
MARKDOWN = RESEARCH / "pg268_pikachu_browser_parameterized_crawl_report_v1.md"
SEED = 26801
BASE_PORT = 5635
MAX_PAGES = 100


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _same_origin(url: str, origin: str) -> bool:
    left = urlsplit(url)
    right = urlsplit(origin)
    return left.scheme in {"http", "https"} and left.netloc == right.netloc


def _canonical(url: str, origin: str) -> str:
    parsed = urlsplit(url)
    path = parsed.path or "/"
    query = "&".join(f"{key}={value}" for key, value in parse_qsl(parsed.query, keep_blank_values=True))
    return urlunsplit((urlsplit(origin).scheme, urlsplit(origin).netloc, path, query, ""))


def _path_and_query(url: str) -> tuple[str, list[str]]:
    parsed = urlsplit(url)
    return parsed.path or "/", sorted({str(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)})


def _extract(page: Any, url: str, response_status: int | None) -> dict[str, Any]:
    links = page.locator("a").evaluate_all(
        """els => els.map(el => ({href: el.href || '', text: (el.innerText || '').trim().slice(0, 120)})).slice(0, 300)"""
    )
    forms = page.locator("form").evaluate_all(
        """forms => forms.map(form => ({
          action: form.action || '', method: (form.method || 'get').toUpperCase(),
          enctype: form.enctype || null,
          fields: Array.from(form.elements || []).map(el => ({
            name: el.name || '', type: (el.type || '').toLowerCase(),
            required: !!el.required, disabled: !!el.disabled
          })).filter(x => x.name).slice(0, 80)
        })).slice(0, 80)"""
    )
    return {
        "url": url,
        "path": _path_and_query(url)[0],
        "query_keys": _path_and_query(url)[1],
        "response_status": response_status,
        "final_url": page.url,
        "title": str(page.title())[:160],
        "links": [{"href": str(item.get("href", ""))[:500], "text": str(item.get("text", ""))} for item in links if isinstance(item, dict)],
        "forms": [
            {
                "action": str(item.get("action", ""))[:500],
                "method": str(item.get("method", "GET")).upper(),
                "enctype": item.get("enctype"),
                "fields": [
                    {"name": str(field.get("name", ""))[:120], "type": str(field.get("type", ""))[:40], "required": bool(field.get("required")), "disabled": bool(field.get("disabled"))}
                    for field in list(item.get("fields") or [])
                    if isinstance(field, dict) and str(field.get("name", ""))
                ],
            }
            for item in forms
            if isinstance(item, dict)
        ],
    }


def _route_catalog(pages: list[dict[str, Any]], origin: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for page in pages:
        page_path = str(page["path"])
        page_query = list(page.get("query_keys") or [])
        # A page itself is a GET surface even when it has no form.
        key = (page_path, "GET", "page")
        row = grouped.setdefault(key, {"path": page_path, "method": "GET", "source": "page", "query_params": set(), "form_params": set(), "hidden_params": set(), "submit_params": set(), "enctype": None, "observed_on": set(), "link_count": 0, "form_count": 0})
        row["query_params"].update(page_query)
        row["observed_on"].add(page_path)
        row["link_count"] += len(list(page.get("links") or []))
        # Query-bearing anchors are request surfaces too.  PG-179 counted
        # these but the first PG-268 projection only followed the route and
        # silently dropped their parameter names; retain them as GET fields.
        for link in list(page.get("links") or []):
            href = str(link.get("href", "")) if isinstance(link, dict) else ""
            if not _same_origin(href, origin):
                continue
            link_path, link_query = _path_and_query(href)
            if not link_query:
                continue
            link_key = (link_path, "GET", "anchor_query")
            link_row = grouped.setdefault(link_key, {"path": link_path, "method": "GET", "source": "anchor_query", "query_params": set(), "form_params": set(), "hidden_params": set(), "submit_params": set(), "enctype": None, "observed_on": set(), "link_count": 0, "form_count": 0})
            link_row["query_params"].update(link_query)
            link_row["observed_on"].add(page_path)
            link_row["link_count"] += 1
        for form in list(page.get("forms") or []):
            action = str(form.get("action") or "")
            if not _same_origin(action, origin):
                continue
            form_path, action_query = _path_and_query(action)
            method = str(form.get("method") or "GET").upper()
            source = "form_get" if method == "GET" else "form_post"
            form_key = (form_path, method, source)
            target = grouped.setdefault(form_key, {"path": form_path, "method": method, "source": source, "query_params": set(), "form_params": set(), "hidden_params": set(), "submit_params": set(), "enctype": form.get("enctype"), "observed_on": set(), "link_count": 0, "form_count": 0})
            target["query_params"].update(action_query)
            target["observed_on"].add(page_path)
            target["form_count"] += 1
            for field in list(form.get("fields") or []):
                name = str(field.get("name", ""))
                if not name:
                    continue
                target["form_params"].add(name)
                if str(field.get("type", "")).lower() == "hidden":
                    target["hidden_params"].add(name)
                if name.casefold() in {"submit", "button", "commit", "login"} or str(field.get("type", "")).lower() in {"submit", "button"}:
                    target["submit_params"].add(name)
    result: list[dict[str, Any]] = []
    for row in grouped.values():
        method = str(row["method"])
        form_params = sorted(row["form_params"])
        query_params = sorted(row["query_params"])
        result.append(
            {
                "path": row["path"],
                "method": method,
                "source": row["source"],
                "query_params": query_params,
                "form_params": form_params,
                "hidden_params": sorted(row["hidden_params"]),
                "submit_params": sorted(row["submit_params"]),
                "enctype": row["enctype"],
                "observed_on": sorted(row["observed_on"]),
                "link_count": int(row["link_count"]),
                "form_count": int(row["form_count"]),
                "has_parameter_context": bool(query_params or form_params),
                "replay_status": "schema_discovered_not_replayed",
                "training_eligible": False,
            }
        )
    result.sort(key=lambda item: (item["path"], item["method"], item["source"]))
    return result


def main() -> int:
    if sync_playwright is None:
        raise RuntimeError("Playwright is required for PG-268A")
    PG214.BASE_PORT = BASE_PORT
    name = ""
    pages: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        name, port, container_id, reset = PG214._start(SEED, 0)
        origin = f"http://127.0.0.1:{port}"
        browser_context = sync_playwright().start()
        browser: Browser = browser_context.chromium.launch(headless=True)
        page = browser.new_page()
        queue: deque[str] = deque([origin + "/"])
        seen: set[str] = set()
        try:
            while queue and len(seen) < MAX_PAGES:
                target = _canonical(queue.popleft(), origin)
                if target in seen or not _same_origin(target, origin):
                    continue
                seen.add(target)
                try:
                    response = page.goto(target, wait_until="domcontentloaded", timeout=12000)
                    page.wait_for_timeout(50)
                    record = _extract(page, target, response.status if response is not None else None)
                    pages.append(record)
                    for link in record["links"]:
                        href = str(link.get("href", ""))
                        if _same_origin(href, origin):
                            candidate = _canonical(href, origin)
                            if candidate not in seen and len(seen) + len(queue) < MAX_PAGES * 2:
                                queue.append(candidate)
                except Exception as exc:
                    pages.append({"url": target, "path": _path_and_query(target)[0], "query_keys": _path_and_query(target)[1], "response_status": None, "error": type(exc).__name__, "links": [], "forms": []})
        finally:
            browser.close()
            browser_context.stop()
        routes = _route_catalog(pages, origin)
        source = {
            "image": PG214.IMAGE,
            "container_id_sha256": hashlib.sha256(container_id.encode("utf-8")).hexdigest(),
            "reset": reset,
            "origin": "loopback_only",
        }
        counts = {
            "page_count": len(pages),
            "route_count": len(routes),
            "request_surface_count": len(routes),
            "get_surface_count": sum(int(row["method"] == "GET") for row in routes),
            "post_surface_count": sum(int(row["method"] == "POST") for row in routes),
            "with_parameter_context": sum(int(row["has_parameter_context"]) for row in routes),
            "missing_parameter_context": sum(int(not row["has_parameter_context"]) for row in routes),
            "form_count": sum(len(list(page.get("forms") or [])) for page in pages),
            "link_count": sum(len(list(page.get("links") or [])) for page in pages),
            "error_page_count": sum(int(bool(page.get("error"))) for page in pages),
        }
        manifest = {
            "schema_version": "pg268-pikachu-browser-parameterized-crawl-manifest-v1",
            "status": "completed_browser_dom_parameter_discovery",
            "source": source,
            "crawler": {"method": "playwright_same_origin_dom", "seed": SEED, "max_pages": MAX_PAGES, "form_submissions": 0, "mutating_actions": 0, "raw_source_stored": False, "raw_payloads_stored": False, "response_bodies_stored": False},
            "counts": counts,
            "pages": pages,
            "route_catalog": routes,
            "replay_contract": {"next_stage": "PG-268B fresh route GET/POST replay", "fresh_container_per_route": True, "database_health_gate_required": True, "parameterized_response_required": True, "redirect_chain_required": True, "training_eligible": False},
            "manifest_sha256": "",
        }
        manifest["manifest_sha256"] = _digest(manifest)
        _write(MANIFEST, manifest)
        report = {"protocol_id": "pg-pk-268a-browser-parameterized-crawl-v1", "schema_version": "pg268a-pikachu-browser-parameterized-crawl-report-v1", "status": manifest["status"], "counts": counts, "source": source, "elapsed_seconds": round(time.monotonic() - started, 3), "manifest_sha256": manifest["manifest_sha256"], "training_eligible": False, "raw_source_stored": False, "raw_response_bodies_stored": False, "report_sha256": ""}
        report["report_sha256"] = _digest(report)
        _write(REPORT, report)
        protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg268a-pikachu-browser-parameterized-crawl-protocol-v1", "stages": ["fresh_loopback_reset", "same_origin_browser_crawl", "form_schema_projection", "parameterized_replay_gate"], "form_submissions": False, "mutating_actions": False, "training_promotion_blocked": True, "raw_source_and_response_excluded": True, "protocol_sha256": ""}
        protocol["protocol_sha256"] = _digest(protocol)
        _write(PROTOCOL, protocol)
        MARKDOWN.write_text("\n".join(["# PG-268A Pikachu browser parameterized crawl", "", f"pages={counts['page_count']}; routes={counts['route_count']}; GET={counts['get_surface_count']}; POST={counts['post_surface_count']}; forms={counts['form_count']}", f"with_parameter_context={counts['with_parameter_context']}; missing={counts['missing_parameter_context']}; elapsed={report['elapsed_seconds']}s", "本阶段只做同源 DOM 发现，不提交表单；PG-268B 才逐路 fresh reset 回放并记录参数化响应、302 链和有限 oracle。", ""]), encoding="utf-8")
        print(json.dumps({"status": report["status"], "counts": counts, "manifest": str(MANIFEST.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        if name:
            PG214._stop(name)


if __name__ == "__main__":
    raise SystemExit(main())
