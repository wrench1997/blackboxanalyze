"""Generate the PG-348 page fixture group B.

The pages in this directory are deliberately boring, local-only HTML surfaces.
They provide different DOM, navigation, form, and client-state shapes for the
evaluator without making a network request or writing business state.  The
generator is deterministic so that source hashes and de-duplication statistics
can be checked after every regeneration.

This module is intentionally self-contained.  It does not import the model,
an adapter, or an evaluator and it never emits a payload or an oracle answer.
The raw HTML is an evaluator-side fixture; ``training_context_raw`` is false
for every manifest record.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping


SCHEMA = "pg348-pages-b-fixture-manifest"
VERSION = "v1"
DEFAULT_COUNT = 120
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SurfaceTemplate:
    """Description of one visibly different static page layout."""

    template_id: str
    label: str
    preferred_method: str
    implementation_group: str
    response_shape: str
    redirect_shape: str
    script_surface: str
    synthetic_oracle_kind: str
    render: Callable[[Mapping[str, str]], tuple[str, str, str]]


MECHANISMS: tuple[str, ...] = (
    "mechanism_profile_projection",
    "mechanism_query_filter",
    "mechanism_json_projection",
    "mechanism_fragment_navigation",
    "mechanism_toggle_state",
    "mechanism_step_transition",
    "mechanism_table_sort",
    "mechanism_activity_filter",
    "mechanism_notice_dismiss",
    "mechanism_metric_reorder",
    "mechanism_inbox_compose",
    "mechanism_view_preference",
)


PARAMETER_ROLES: tuple[str, ...] = (
    "profile_key",
    "query_term",
    "view_mode",
    "tab_name",
    "filter_choice",
    "sort_direction",
    "notice_state",
    "step_index",
    "metric_group",
    "note_text",
    "display_preference",
    "record_cursor",
)


def _q(value: object) -> str:
    """HTML-escape a generated value."""

    return html.escape(str(value), quote=True)


def _js(value: object) -> str:
    """Encode a generated value for a JavaScript string literal."""

    return json.dumps(str(value), ensure_ascii=False)


def _shell(meta: Mapping[str, str], body: str, css: str, script: str) -> str:
    """Wrap a template body in a self-contained document.

    There are no remote stylesheets, images, modules, timers, storage APIs, or
    network APIs in this shell.  State is shown only in the current document.
    """

    challenge_id = _q(meta["challenge_id"])
    template_id = _q(meta["surface_template_id"])
    method = _q(meta["transport_method"])
    parameter_role = _q(meta["parameter_role"])
    encoding = _q(meta["encoding_label"])
    mechanism = _q(meta["mechanism_id"])
    oracle_kind = _q(meta["synthetic_oracle_kind"])
    return f'''<!doctype html>
<html lang="en" data-fixture="pg348" data-surface-template="{template_id}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="pg348-challenge" content="{challenge_id}">
  <meta name="pg348-transport" content="{method}">
  <meta name="pg348-encoding" content="{encoding}">
  <meta name="pg348-mechanism" content="{mechanism}">
  <title>Local synthetic surface {challenge_id}</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; background: #f4f7fb; color: #172033; }}
    a {{ color: #2458a6; }}
    button, input, select, textarea {{ font: inherit; }}
    button {{ border: 1px solid #7183a5; border-radius: .5rem; background: #fff; padding: .45rem .7rem; cursor: pointer; }}
    button:hover {{ background: #eef4ff; }}
    main {{ width: min(960px, calc(100% - 2rem)); margin: 2rem auto; }}
    .surface-header {{ padding: 1.25rem; background: #152641; color: #fff; border-radius: .8rem; }}
    .surface-header h1 {{ margin: 0 0 .4rem; font-size: clamp(1.25rem, 4vw, 2rem); }}
    .surface-header p {{ margin: 0; color: #cbd9ef; }}
    .surface-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; margin-top: 1rem; }}
    .card, fieldset, .panel {{ background: #fff; border: 1px solid #d5ddec; border-radius: .7rem; padding: 1rem; box-shadow: 0 3px 12px #15264112; }}
    .card h2, .panel h2 {{ margin-top: 0; font-size: 1.05rem; }}
    form {{ display: grid; gap: .65rem; }}
    label {{ display: grid; gap: .25rem; font-size: .9rem; color: #33415c; }}
    input, select, textarea {{ width: 100%; border: 1px solid #aebbd0; border-radius: .4rem; padding: .45rem .55rem; background: #fff; }}
    output, [role="status"] {{ display: block; min-height: 1.5rem; margin-top: .7rem; color: #2458a6; }}
    .muted {{ color: #63718a; font-size: .88rem; }}
    .chip {{ display: inline-block; padding: .18rem .45rem; border-radius: 99px; background: #e3ecff; color: #254878; font-size: .78rem; }}
    .stack {{ display: grid; gap: .7rem; }}
    .timeline {{ list-style: none; padding: 0; margin: 0; display: grid; gap: .65rem; }}
    .timeline li {{ border-left: 3px solid #8eabdf; padding: .4rem .7rem; background: #f8faff; }}
    .mobile-nav {{ display: flex; justify-content: space-around; gap: .4rem; position: sticky; bottom: .5rem; padding: .5rem; background: #fff; border: 1px solid #c8d3e6; border-radius: .7rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: .45rem; border-bottom: 1px solid #e4e8f0; }}
    th {{ color: #475a7c; font-size: .82rem; }}
    .stepper {{ display: flex; gap: .4rem; list-style: none; padding: 0; }}
    .stepper li {{ flex: 1; padding: .5rem; text-align: center; border-radius: .4rem; background: #edf1f8; font-size: .82rem; }}
    .stepper li[data-active="true"] {{ background: #cfe0ff; color: #173c78; font-weight: 650; }}
    .badge {{ min-width: 1.5rem; display: inline-block; text-align: center; border-radius: 99px; background: #e8c58e; color: #4d3411; }}
    .fixture-note {{ margin-top: 1rem; font-size: .78rem; color: #68758a; }}
    {css}
  </style>
</head>
<body data-challenge-id="{challenge_id}" data-transport-method="{method}" data-parameter-role="{parameter_role}" data-encoding-chain="{encoding}">
  {body}
  <p class="fixture-note">Local synthetic preview. The page keeps state in memory and makes no external request.</p>
  <script type="application/json" data-synthetic-observation="{oracle_kind}">{{"kind":"{oracle_kind}","state":"baseline"}}</script>
  <script>
    // The evaluator observes bounded DOM state only; this fixture has no wire client.
    {script}
  </script>
</body>
</html>
'''


def _meta_header(meta: Mapping[str, str], title: str, subtitle: str) -> str:
    return f'''<header class="surface-header">
  <span class="chip">{_q(meta["surface_template_id"])}</span>
  <h1>{_q(title)}</h1>
  <p>{_q(subtitle)}</p>
</header>'''


def _render_spa_profile(meta: Mapping[str, str]) -> tuple[str, str, str]:
    body = f'''<main>
  {_meta_header(meta, "Profile workspace", "A route-aware profile shell for a local synthetic preview.")}
  <nav class="panel" aria-label="Profile sections">
    <a href="#summary">Summary</a> · <a href="#preferences">Preferences</a> · <a href="#history">History</a>
  </nav>
  <section class="surface-grid" id="summary">
    <article class="card"><h2>Profile card</h2><p class="muted">Sample profile { _q(meta["serial"]) }</p><dl><dt>Display label</dt><dd>North star</dd><dt>View</dt><dd>Overview</dd></dl></article>
    <article class="card"><h2>Preview route</h2>
      <form id="profile-route" method="GET" action="#profile-preview">
        <label for="profile-key">Profile key</label><input id="profile-key" name="profile_key" value="sample-view" autocomplete="off">
        <button type="submit">Preview section</button>
      </form>
      <output id="profile-status" aria-live="polite">Ready for a local preview.</output>
    </article>
  </section>
</main>'''
    script = '''
      const form = document.getElementById("profile-route");
      const status = document.getElementById("profile-status");
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        history.pushState({ view: "profile" }, "", "#profile-preview");
        status.textContent = "Profile preview selected (local state).";
      });
    '''
    return body, ".surface-header + .panel { margin-top: 1rem; }", script


def _render_json_form(meta: Mapping[str, str]) -> tuple[str, str, str]:
    body = f'''<main>
  {_meta_header(meta, "Structured form console", "A JSON-shaped form surface with an in-memory response panel.")}
  <section class="surface-grid">
    <article class="card"><h2>Input envelope</h2>
      <form id="json-envelope" method="POST" action="#json-submit" data-encoding="json_object">
        <label for="json-term">Query term</label><input id="json-term" name="query_term" value="sample-filter" autocomplete="off">
        <label for="json-mode">View mode</label><select id="json-mode" name="view_mode"><option>compact</option><option>expanded</option></select>
        <button type="submit">Build local envelope</button>
      </form>
    </article>
    <article class="card"><h2>Shape panel</h2><pre id="json-output" aria-live="polite">{{"state":"baseline"}}</pre><output id="json-status">Awaiting a local action.</output></article>
  </section>
</main>'''
    script = '''
      const form = document.getElementById("json-envelope");
      const output = document.getElementById("json-output");
      const status = document.getElementById("json-status");
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        output.textContent = JSON.stringify({ state: "preview", shape: "object", fields: 2 }, null, 2);
        status.textContent = "Structured preview ready (no request sent).";
      });
    '''
    return body, ".card pre { min-height: 5rem; background: #f7f9fc; padding: .7rem; border-radius: .4rem; overflow: auto; }", script


def _render_mobile_dashboard(meta: Mapping[str, str]) -> tuple[str, str, str]:
    body = f'''<main class="mobile-frame">
  {_meta_header(meta, "Pocket dashboard", "A narrow-screen card layout with a local tab state.")}
  <section class="surface-grid" aria-label="Dashboard cards">
    <article class="card"><h2>Today</h2><p><strong>08</strong> queued notes</p><p class="muted">Updated in memory</p></article>
    <article class="card"><h2>Focus</h2><p><strong>03</strong> active views</p><button type="button" id="focus-button">Mark focus</button><output id="focus-status">Focus is clear.</output></article>
  </section>
  <nav class="mobile-nav" aria-label="Pocket tabs"><a href="#home" data-tab="home">Home</a><a href="#queue" data-tab="queue">Queue</a><a href="#me" data-tab="me">Me</a></nav>
</main>'''
    script = '''
      const focusButton = document.getElementById("focus-button");
      const focusStatus = document.getElementById("focus-status");
      focusButton.addEventListener("click", () => { focusStatus.textContent = "Focus marked in local state."; });
      document.querySelectorAll("[data-tab]").forEach((tab) => tab.addEventListener("click", () => {
        focusStatus.textContent = "Tab selected: " + tab.dataset.tab + ".";
      }));
    '''
    return body, ".mobile-frame { max-width: 560px; } @media (max-width: 480px) { main { width: calc(100% - 1rem); margin: .5rem auto; } }", script


def _render_settings(meta: Mapping[str, str]) -> tuple[str, str, str]:
    body = f'''<main>
  {_meta_header(meta, "Settings studio", "A preferences panel with explicit, reversible in-memory controls.")}
  <form id="settings-form" class="panel" method="POST" action="#settings-save">
    <fieldset><legend>Display</legend>
      <label><input type="checkbox" name="compact_mode" id="compact-mode"> Compact cards</label>
      <label><input type="checkbox" name="quiet_mode" id="quiet-mode"> Quiet notices</label>
    </fieldset>
    <fieldset><legend>Accent</legend><label for="accent">Accent choice</label><select id="accent" name="display_preference"><option>blue</option><option>violet</option><option>amber</option></select></fieldset>
    <button type="submit">Preview settings</button><output id="settings-status" aria-live="polite">No changes selected.</output>
  </form>
</main>'''
    script = '''
      const settingsForm = document.getElementById("settings-form");
      const settingsStatus = document.getElementById("settings-status");
      settingsForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const compact = document.getElementById("compact-mode").checked;
        const quiet = document.getElementById("quiet-mode").checked;
        const accent = document.getElementById("accent").value;
        settingsStatus.textContent = "Preview: " + accent + "; " + (compact ? "compact" : "roomy") + "; " + (quiet ? "quiet" : "standard") + ".";
      });
    '''
    return body, "fieldset { display: grid; gap: .45rem; margin: 0 0 1rem; } fieldset label { display: flex; align-items: center; gap: .45rem; }", script


def _render_catalog(meta: Mapping[str, str]) -> tuple[str, str, str]:
    body = f'''<main>
  {_meta_header(meta, "Catalog lens", "A filter-and-sort surface with a bounded list projection.")}
  <section class="panel">
    <form id="catalog-form" method="GET" action="#catalog-view">
      <label for="catalog-filter">Filter choice</label><select id="catalog-filter" name="filter_choice"><option value="all">All</option><option value="new">New</option><option value="saved">Saved</option></select>
      <label for="catalog-sort">Sort direction</label><select id="catalog-sort" name="sort_direction"><option value="up">Ascending</option><option value="down">Descending</option></select>
      <button type="submit">Apply lens</button>
    </form>
    <div id="catalog-list" class="surface-grid" aria-live="polite"><article class="card"><h2>Card A</h2><p class="muted">baseline item</p></article><article class="card"><h2>Card B</h2><p class="muted">baseline item</p></article></div>
    <output id="catalog-status">Baseline list shown.</output>
  </section>
</main>'''
    script = '''
      const catalogForm = document.getElementById("catalog-form");
      const catalogStatus = document.getElementById("catalog-status");
      catalogForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const filter = document.getElementById("catalog-filter").value;
        const sort = document.getElementById("catalog-sort").value;
        catalogStatus.textContent = "Lens preview: " + filter + " / " + sort + ".";
      });
    '''
    return body, ".panel > form { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); align-items: end; } .panel > form button { width: max-content; }", script


def _render_timeline(meta: Mapping[str, str]) -> tuple[str, str, str]:
    body = f'''<main>
  {_meta_header(meta, "Activity timeline", "An ordered event surface with a category switch.")}
  <section class="surface-grid">
    <article class="card"><h2>Timeline filter</h2><form id="timeline-form" method="GET" action="#timeline"><label for="timeline-kind">Activity kind</label><select id="timeline-kind" name="tab_name"><option>all</option><option>review</option><option>draft</option></select><button type="submit">Show events</button></form><output id="timeline-status">Showing all events.</output></article>
    <article class="card"><h2>Events</h2><ol class="timeline" id="timeline-list"><li><strong>09:10</strong> · draft prepared</li><li><strong>10:25</strong> · review queued</li><li><strong>11:40</strong> · note archived</li></ol></article>
  </section>
</main>'''
    script = '''
      const timelineForm = document.getElementById("timeline-form");
      const timelineStatus = document.getElementById("timeline-status");
      timelineForm.addEventListener("submit", (event) => {
        event.preventDefault();
        timelineStatus.textContent = "Timeline view: " + document.getElementById("timeline-kind").value + ".";
      });
    '''
    return body, ".timeline li strong { color: #2458a6; margin-right: .25rem; }", script


def _render_notifications(meta: Mapping[str, str]) -> tuple[str, str, str]:
    body = f'''<main>
  {_meta_header(meta, "Notification desk", "A disclosure list and badge state for a local notice center.")}
  <section class="surface-grid">
    <article class="card"><h2>Inbox <span class="badge" id="notice-count">3</span></h2><details open><summary>Review reminder</summary><p class="muted">A synthetic reminder for this page.</p></details><details><summary>Quiet hours</summary><p class="muted">No external delivery is configured.</p></details><button type="button" id="clear-notices">Acknowledge visible</button><output id="notice-status">Three notices remain.</output></article>
    <article class="card"><h2>Preference</h2><form id="notice-form" method="POST" action="#notice-preference"><label><input type="checkbox" id="notice-quiet" name="notice_state"> Quiet preview</label><button type="submit">Preview preference</button></form></article>
  </section>
</main>'''
    script = '''
      const clearNotices = document.getElementById("clear-notices");
      const noticeCount = document.getElementById("notice-count");
      const noticeStatus = document.getElementById("notice-status");
      clearNotices.addEventListener("click", () => { noticeCount.textContent = "0"; noticeStatus.textContent = "Visible notices acknowledged locally."; });
      document.getElementById("notice-form").addEventListener("submit", (event) => {
        event.preventDefault();
        noticeStatus.textContent = document.getElementById("notice-quiet").checked ? "Quiet preview enabled." : "Standard preview enabled.";
      });
    '''
    return body, ".card details { margin: .55rem 0; padding: .45rem; background: #f7f9fc; border-radius: .35rem; }", script


def _render_wizard(meta: Mapping[str, str]) -> tuple[str, str, str]:
    body = f'''<main>
  {_meta_header(meta, "Step navigator", "A three-step local wizard with explicit progress state.")}
  <ol class="stepper" aria-label="Wizard progress"><li data-step="1" data-active="true">Choose</li><li data-step="2">Review</li><li data-step="3">Finish</li></ol>
  <section class="panel"><form id="wizard-form" method="POST" action="#wizard"><label for="wizard-step">Step index</label><select id="wizard-step" name="step_index"><option value="1">Choose</option><option value="2">Review</option><option value="3">Finish</option></select><button type="submit">Set local step</button></form><output id="wizard-status">Step 1 of 3.</output></section>
</main>'''
    script = '''
      const wizardForm = document.getElementById("wizard-form");
      const wizardStatus = document.getElementById("wizard-status");
      wizardForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const selected = document.getElementById("wizard-step").value;
        document.querySelectorAll("[data-step]").forEach((item) => { item.dataset.active = item.dataset.step === selected ? "true" : "false"; });
        wizardStatus.textContent = "Step " + selected + " of 3 (local state).";
      });
    '''
    return body, ".stepper + .panel { margin-top: 1rem; }", script


def _render_analytics(meta: Mapping[str, str]) -> tuple[str, str, str]:
    body = f'''<main>
  {_meta_header(meta, "Metric board", "A compact table and chart legend with a local ordering control.")}
  <section class="surface-grid">
    <article class="card"><h2>Summary</h2><table><thead><tr><th>Group</th><th>Count</th><th>Trend</th></tr></thead><tbody><tr><td>Alpha</td><td>18</td><td>steady</td></tr><tr><td>Beta</td><td>11</td><td>rising</td></tr><tr><td>Gamma</td><td>07</td><td>quiet</td></tr></tbody></table></article>
    <article class="card"><h2>View order</h2><form id="metric-form" method="GET" action="#metrics"><label for="metric-group">Metric group</label><select id="metric-group" name="metric_group"><option>Alpha</option><option>Beta</option><option>Gamma</option></select><button type="submit">Preview group</button></form><output id="metric-status">Alpha selected.</output><svg viewBox="0 0 220 50" role="img" aria-label="Synthetic bar legend"><rect x="5" y="12" width="70" height="12" fill="#8eabdf"></rect><rect x="85" y="12" width="44" height="12" fill="#c6d7f6"></rect><rect x="139" y="12" width="30" height="12" fill="#e8c58e"></rect></svg></article>
  </section>
</main>'''
    script = '''
      const metricForm = document.getElementById("metric-form");
      const metricStatus = document.getElementById("metric-status");
      metricForm.addEventListener("submit", (event) => {
        event.preventDefault();
        metricStatus.textContent = document.getElementById("metric-group").value + " selected locally.";
      });
    '''
    return body, ".card svg { width: 100%; height: 60px; margin-top: 1rem; background: #f7f9fc; border-radius: .35rem; }", script


def _render_support(meta: Mapping[str, str]) -> tuple[str, str, str]:
    body = f'''<main>
  {_meta_header(meta, "Support inbox", "A two-column inbox with an in-memory note composer.")}
  <section class="surface-grid">
    <aside class="card"><h2>Threads</h2><ul><li><a href="#thread-a">Planning note</a></li><li><a href="#thread-b">Review note</a></li><li><a href="#thread-c">Archive note</a></li></ul></aside>
    <article class="card"><h2>Compose preview</h2><form id="support-form" method="POST" action="#compose"><label for="note-text">Note text</label><textarea id="note-text" name="note_text" rows="4" placeholder="Write a local note"></textarea><button type="submit">Preview note</button></form><output id="support-status">No note previewed.</output></article>
  </section>
</main>'''
    script = '''
      const supportForm = document.getElementById("support-form");
      const supportStatus = document.getElementById("support-status");
      supportForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const hasText = document.getElementById("note-text").value.trim().length > 0;
        supportStatus.textContent = hasText ? "Local note preview available." : "A note is still empty.";
      });
    '''
    return body, ".card ul { margin: 0; padding-left: 1.2rem; display: grid; gap: .65rem; }", script


SURFACE_TEMPLATES: tuple[SurfaceTemplate, ...] = (
    SurfaceTemplate("spa_profile", "SPA/profile", "GET", "b_impl_web_shell", "html_document:200", "same_origin_fragment", "spa_history_pushstate", "synthetic_navigation_delta", _render_spa_profile),
    SurfaceTemplate("json_form", "JSON form", "POST", "b_impl_data_console", "json_object:200", "none", "json_form_validation", "synthetic_json_shape_delta", _render_json_form),
    SurfaceTemplate("mobile_dashboard", "mobile dashboard", "GET", "b_impl_mobile_shell", "html_fragment:200", "same_origin_fragment", "responsive_tab_state", "synthetic_dom_marker_delta", _render_mobile_dashboard),
    SurfaceTemplate("settings_panel", "settings panel", "POST", "b_impl_preferences", "html_fragment:200", "none", "settings_toggle_state", "synthetic_aria_state_delta", _render_settings),
    SurfaceTemplate("catalog_filter", "catalog filter", "GET", "b_impl_catalog", "html_document:200", "same_origin_fragment", "select_filter_state", "synthetic_list_shape_delta", _render_catalog),
    SurfaceTemplate("activity_timeline", "activity timeline", "GET", "b_impl_activity", "html_document:200", "same_origin_fragment", "timeline_filter_state", "synthetic_ordered_list_delta", _render_timeline),
    SurfaceTemplate("notification_center", "notification center", "POST", "b_impl_alerts", "html_fragment:200", "none", "details_badge_state", "synthetic_aria_state_delta", _render_notifications),
    SurfaceTemplate("wizard_stepper", "wizard stepper", "POST", "b_impl_wizard", "html_fragment:200", "same_origin_fragment", "stepper_progress_state", "synthetic_navigation_delta", _render_wizard),
    SurfaceTemplate("analytics_cards", "analytics cards", "GET", "b_impl_metrics", "html_table:200", "none", "table_svg_state", "synthetic_table_shape_delta", _render_analytics),
    SurfaceTemplate("support_inbox", "support inbox", "POST", "b_impl_support", "html_fragment:200", "same_origin_fragment", "composer_validation", "synthetic_dom_marker_delta", _render_support),
)


# Keep the abstract role aligned with a control that actually exists on the
# corresponding page while retaining bounded role variation within a template.
TEMPLATE_PARAMETER_ROLES: Mapping[str, tuple[str, ...]] = {
    "spa_profile": ("profile_key", "view_mode"),
    "json_form": ("query_term", "view_mode"),
    "mobile_dashboard": ("view_mode", "tab_name"),
    "settings_panel": ("display_preference", "notice_state"),
    "catalog_filter": ("filter_choice", "sort_direction"),
    "activity_timeline": ("tab_name", "record_cursor"),
    "notification_center": ("notice_state", "display_preference"),
    "wizard_stepper": ("step_index", "view_mode"),
    "analytics_cards": ("metric_group", "sort_direction"),
    "support_inbox": ("note_text", "record_cursor"),
}


def _encoding_chain(template: SurfaceTemplate, ordinal: int) -> list[str]:
    """Return a bounded abstract encoding chain (never a literal value)."""

    if template.template_id == "json_form":
        return ["json_object", "utf8"]
    if template.template_id == "mobile_dashboard":
        return ["fragment", "utf8"]
    if template.preferred_method == "GET":
        return ["query_parameter", "url_percent" if ordinal % 2 else "utf8"]
    if template.template_id in {"settings_panel", "notification_center"}:
        return ["form_urlencoded", "utf8"]
    return ["form_urlencoded", "url_percent" if ordinal % 2 else "utf8"]


def _page_record(template: SurfaceTemplate, mechanism_id: str, ordinal: int, output_root: Path) -> dict[str, object]:
    challenge_id = f"pg348-b-{ordinal:04d}"
    encoding_chain = _encoding_chain(template, ordinal)
    encoding_label = "+".join(encoding_chain)
    role_options = TEMPLATE_PARAMETER_ROLES.get(template.template_id, PARAMETER_ROLES)
    parameter_role = role_options[(ordinal - 1) % len(role_options)]
    meta = {
        "challenge_id": challenge_id,
        "surface_template_id": template.template_id,
        "transport_method": template.preferred_method,
        "encoding_label": encoding_label,
        "parameter_role": parameter_role,
        "mechanism_id": mechanism_id,
        "synthetic_oracle_kind": template.synthetic_oracle_kind,
        "serial": f"{ordinal:04d}",
    }
    body, css, script = template.render(meta)
    document = _shell(meta, body, css, script)
    filename = f"{challenge_id}.html"
    path = output_root / filename
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
    source_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()
    # Paths are relative to the fixture directory, matching the pages_a
    # manifest convention and allowing a checkout to be relocated.  The
    # generated pages are written directly below ``output_dir``.
    local_path = filename
    return {
        "challenge_id": challenge_id,
        "local_path": local_path,
        "loopback_only": True,
        "localhost_only": True,
        "external_network": False,
        "state_write": False,
        "mechanism_id": mechanism_id,
        "surface_template_id": template.template_id,
        "implementation_group": template.implementation_group,
        "transport_method": template.preferred_method,
        "parameter_role": parameter_role,
        "encoding_chain": encoding_chain,
        "response_shape": template.response_shape,
        "redirect_shape": template.redirect_shape,
        "script_surface": template.script_surface,
        "synthetic_oracle_kind": template.synthetic_oracle_kind,
        "source_hash": source_hash,
        "raw_source_for_evaluator_only": True,
        "training_context_raw": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(records)
    challenge_ids = [str(row["challenge_id"]) for row in rows]
    mechanisms = [str(row["mechanism_id"]) for row in rows]
    templates = [str(row["surface_template_id"]) for row in rows]
    source_hashes = [str(row["source_hash"]) for row in rows]
    pairs = [f"{row['mechanism_id']}|{row['surface_template_id']}" for row in rows]
    transport = Counter(str(row["transport_method"]) for row in rows)
    roles = Counter(str(row["parameter_role"]) for row in rows)
    encodings = Counter("+".join(str(item) for item in row["encoding_chain"]) for row in rows)
    transport_variants = {
        (str(row["transport_method"]), tuple(str(item) for item in row["encoding_chain"]))
        for row in rows
    }
    duplicate_stats = {
        "challenge_ids": len(challenge_ids) - len(set(challenge_ids)),
        "source_hashes": len(source_hashes) - len(set(source_hashes)),
        "mechanism_template_pairs": len(pairs) - len(set(pairs)),
    }
    unique_instances = len(set(challenge_ids))
    return {
        "challenge_instances": len(rows),
        "records": len(rows),
        "instances": len(rows),
        "templates": len(set(templates)),
        "safe_variants": len(set(pairs)),
        "static_pages": len(rows),
        "unique_challenge_ids": unique_instances,
        "unique_source_hashes": len(set(source_hashes)),
        "unique_mechanism_ids": len(set(mechanisms)),
        "mechanism_families": len(set(mechanisms)),
        "unique_surface_template_ids": len(set(templates)),
        "surface_templates": len(set(templates)),
        "unique_mechanism_template_pairs": len(set(pairs)),
        "transport_method_counts": dict(sorted(transport.items())),
        "parameter_role_counts": dict(sorted(roles.items())),
        "encoding_chain_counts": dict(sorted(encodings.items())),
        "transport_variants": len(transport_variants),
        "get_count": transport.get("GET", 0),
        "post_count": transport.get("POST", 0),
        "get_records": transport.get("GET", 0),
        "post_shape_records": transport.get("POST", 0),
        "get_post_coverage": bool(transport.get("GET") and transport.get("POST")),
        "duplicate_stats": duplicate_stats,
        "duplicate_challenge_ids": duplicate_stats["challenge_ids"],
        "duplicate_source_hashes": duplicate_stats["source_hashes"],
        "duplicate_mechanism_template_pairs": duplicate_stats["mechanism_template_pairs"],
        "deduplication_ratio": (unique_instances / len(rows)) if rows else 0.0,
        "all_urls_loopback_only": all(bool(row.get("loopback_only")) for row in rows),
    }


def generate_fixture(*, output_dir: Path, manifest_path: Path, count: int = DEFAULT_COUNT) -> dict[str, object]:
    """Generate ``count`` distinct page instances and write the manifest."""

    if count < 1:
        raise ValueError("count must be positive")
    combinations = list(itertools.product(SURFACE_TEMPLATES, MECHANISMS))
    if count > len(combinations):
        raise ValueError(f"count={count} exceeds available unique combinations={len(combinations)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for ordinal, (template, mechanism_id) in enumerate(combinations[:count], start=1):
        records.append(_page_record(template, mechanism_id, ordinal, output_dir))

    counts = _counts(records)
    dedup_stats = {
        "record_count": len(records),
        "challenge_id_unique": counts["unique_challenge_ids"],
        "local_path_unique": len({str(row["local_path"]) for row in records}),
        "source_hash_unique": counts["unique_source_hashes"],
        "template_variant_pair_unique": counts["unique_mechanism_template_pairs"],
        "mechanism_template_pair_unique": counts["unique_mechanism_template_pairs"],
        "duplicate_challenge_id_groups": [],
        "duplicate_local_path_groups": [],
        "duplicate_source_hash_groups": [],
        "duplicate_template_variant_groups": [],
        "all_content_distinct": counts["unique_source_hashes"] == len(records),
    }
    generator_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "schema_version": "pg348-pages-b-manifest-v1",
        "fixture_id": "pg348_pages_b",
        "status": "diagnostic_only",
        "registry_status": "pre_registered_candidate_only",
        "fixture_group": "pages_b",
        "root": "fixtures/pg348/pages_b",
        "description": "Local static HTML surfaces for abstract shape checks; evaluator-only fixture, not a transferable capability corpus.",
        "generator": "fixtures/pg348/pages_b/generate_pages_b.py",
        "generator_sha256": _file_hash(generator_path),
        "base_origin": "http://127.0.0.1 (synthetic-only; no server is started by this generator)",
        "scope": {
            "all_urls_loopback_only": True,
            "external_network": False,
            "persistent_business_state": False,
            "credentials": False,
            "real_attack_payloads": False,
            "raw_html_js_evaluator_only": True,
        },
        "network_policy": {
            "localhost_only": True,
            "external_network": False,
            "credentials": False,
            "callbacks": False,
            "persistent_state": False,
            "write_routes": [],
        },
        "raw_fixture_policy": {
            "raw_source_for_evaluator_only": True,
            "raw_response_in_training_context": False,
            "raw_payload_in_training_context": False,
            "oracle_answer_in_training_context": False,
        },
        "raw_source_policy": {
            "raw_source_for_evaluator_only": True,
            "training_context_raw": False,
            "raw_payloads": False,
            "raw_responses": False,
            "evaluator_answers": False,
        },
        "counts": counts,
        "dedup_stats": dedup_stats,
        "deduplication": {
            "identity_key": "challenge_id",
            "combination_key": ["mechanism_id", "surface_template_id"],
            "source_hash_algorithm": "SHA-256",
            "duplicate_stats": counts["duplicate_stats"],
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "records": records,
    }
    manifest["manifest_payload_sha256"] = _canonical_hash(manifest)
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def check_fixture(manifest_path: Path) -> dict[str, object]:
    """Validate an existing manifest and every referenced local page."""

    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_generator_hash = str(document.get("generator_sha256") or "")
    if declared_generator_hash and declared_generator_hash != _file_hash(Path(__file__).resolve()):
        raise ValueError("generator_sha256 does not match this generator")
    expected_manifest_hash = str(document.get("manifest_sha256") or "")
    payload = dict(document)
    payload.pop("manifest_sha256", None)
    if expected_manifest_hash != _canonical_hash(payload):
        raise ValueError("manifest_sha256 does not match the manifest contents")
    expected_payload_hash = str(document.get("manifest_payload_sha256") or "")
    if expected_payload_hash:
        payload_for_hash = dict(payload)
        payload_for_hash.pop("manifest_payload_sha256", None)
        payload_hash = _canonical_hash(payload_for_hash)
        if expected_payload_hash != payload_hash:
            raise ValueError("manifest_payload_sha256 does not match the manifest payload")
    records = list(document.get("records") or [])
    for record in records:
        raw_path = Path(str(record.get("local_path") or ""))
        candidates = [raw_path] if raw_path.is_absolute() else [manifest_path.parent / raw_path, REPO_ROOT / raw_path]
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if not path.is_file():
            raise ValueError(f"manifest page is missing: {record.get('local_path')}")
        if _file_hash(path) != str(record.get("source_hash") or ""):
            raise ValueError(f"source_hash mismatch: {record.get('challenge_id')}")
    counts = document.get("counts") or {}
    if int(counts.get("challenge_instances", -1)) != len(records):
        raise ValueError("counts.challenge_instances does not match records")
    return {"manifest": str(manifest_path), "records": len(records), "manifest_sha256": expected_manifest_hash, "status": "checked"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PG-348 local static page fixtures (group B)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent, help="directory for generated HTML pages")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "fixture_manifest.json", help="manifest output path")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"number of instances (default: {DEFAULT_COUNT})")
    parser.add_argument("--check", action="store_true", help="verify an existing manifest and its static pages without writing")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_fixture(args.manifest), ensure_ascii=False, indent=2))
        return 0
    manifest = generate_fixture(output_dir=args.output_dir, manifest_path=args.manifest, count=args.count)
    print(json.dumps({"manifest": str(args.manifest), "counts": manifest["counts"], "manifest_sha256": manifest["manifest_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
