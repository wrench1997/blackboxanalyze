"""Build the PG-348 page-set A synthetic static fixtures.

The page set is intentionally boring from a security perspective.  It gives
the representation/evaluator work a deterministic collection of visibly
different HTML surfaces without introducing a server, a network client, a
credential, a persistent store, or a real probe string.  Every surface is
served as a file and all links/forms point to relative ``/synthetic/*`` paths
or document fragments.

Ten independent page templates are crossed with twelve safe surface variants
to produce 120 challenge identities.  The manifest is generated from the
rendered bytes, so it can be checked in a clean checkout with::

    python generate_pages_a.py --check

Generation itself only writes below this directory (or the explicitly
provided ``--output-dir``), and never contacts a server.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "pg348-pages-a-manifest-v1"
PAGE_SCHEMA_VERSION = "pg348-static-page-v1"
FIXTURE_ID = "pg348_pages_a"
IMPLEMENTATION_GROUP = "pg348_pages_a"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

# These are deliberately conservative checks.  The fixture is static and
# should remain usable even when copied to a loopback-only test server.
FORBIDDEN_SOURCE_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"(?<!:)//[A-Za-z0-9]", re.IGNORECASE),
    re.compile(r"\b(?:fetch|XMLHttpRequest|WebSocket)\s*\(", re.IGNORECASE),
    re.compile(r"\b(?:eval|setTimeout|setInterval)\s*\(", re.IGNORECASE),
    re.compile(r"document\.(?:cookie|domain)", re.IGNORECASE),
    re.compile(r"\b(?:localStorage|sessionStorage)\b", re.IGNORECASE),
    re.compile(r"\b(?:javascript|data):", re.IGNORECASE),
)


@dataclass(frozen=True)
class VariantSpec:
    """A safe, abstract surface variation shared by all templates."""

    variant_id: str
    mechanism_id: str
    transport_method: str
    parameter_role: str
    encoding_chain: tuple[str, ...]
    response_shape: str
    redirect_shape: str
    script_surface: str
    synthetic_oracle_kind: str
    description: str


@dataclass(frozen=True)
class TemplateSpec:
    """A page layout with a distinct DOM shape and visual treatment."""

    template_id: str
    slug: str
    title: str
    kicker: str
    palette: tuple[str, str, str]
    renderer: Callable[[str, str, str, VariantSpec, str], str]
    description: str


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        "plain_label",
        "label_text",
        "GET",
        "static_label",
        ("identity",),
        "html_text",
        "none",
        "none",
        "static_dom_shape",
        "Plain text label in a semantic block.",
    ),
    VariantSpec(
        "query_field",
        "query_text_field",
        "GET",
        "query_text",
        ("identity",),
        "html_form_get",
        "none",
        "none",
        "static_dom_shape",
        "A read-only GET-shaped search form.",
    ),
    VariantSpec(
        "query_encoded",
        "query_text_field_encoded",
        "GET",
        "query_text",
        ("url_percent",),
        "html_form_get_encoded",
        "none",
        "none",
        "static_dom_shape",
        "A GET form whose value is labelled as one safe encoding step.",
    ),
    VariantSpec(
        "attribute_value",
        "attribute_value",
        "GET",
        "attribute_value",
        ("identity",),
        "html_attribute",
        "none",
        "none",
        "static_dom_shape",
        "A data attribute carrying a harmless sample marker.",
    ),
    VariantSpec(
        "path_segment",
        "path_segment",
        "GET",
        "path_segment",
        ("identity",),
        "html_local_path",
        "none",
        "none",
        "static_dom_shape",
        "A relative link with a synthetic path segment.",
    ),
    VariantSpec(
        "fragment_anchor",
        "fragment_identifier",
        "GET",
        "fragment_identifier",
        ("identity",),
        "html_fragment",
        "none",
        "none",
        "static_dom_shape",
        "A document-fragment link and matching local anchor.",
    ),
    VariantSpec(
        "json_preview",
        "json_text_preview",
        "GET",
        "json_value",
        ("json_string",),
        "html_json_script",
        "none",
        "inline_json_data",
        "static_dom_shape",
        "Non-executable application/json metadata for a preview panel.",
    ),
    VariantSpec(
        "dom_text",
        "dom_text_update",
        "GET",
        "dom_text",
        ("identity",),
        "html_dom_text",
        "none",
        "inline_dom_text",
        "static_dom_shape",
        "An inert button that updates textContent only.",
    ),
    VariantSpec(
        "form_get",
        "form_field_get",
        "GET",
        "form_field",
        ("form_urlencode",),
        "html_form_get",
        "none",
        "none",
        "static_dom_shape",
        "A disabled-by-default GET preview form.",
    ),
    VariantSpec(
        "form_post",
        "form_field_post",
        "POST",
        "form_field",
        ("form_urlencode",),
        "html_form_post_preview",
        "none",
        "none",
        "static_dom_shape",
        "A non-submitting POST-shaped preview form; no state is stored.",
    ),
    VariantSpec(
        "list_item",
        "list_item_text",
        "GET",
        "list_item",
        ("identity",),
        "html_list",
        "none",
        "none",
        "static_dom_shape",
        "A list item with a bounded sample label.",
    ),
    VariantSpec(
        "status_badge",
        "status_badge",
        "GET",
        "status_label",
        ("identity",),
        "html_status_badge",
        "none",
        "none",
        "static_dom_shape",
        "A status badge that has no side effect.",
    ),
)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _style(template: TemplateSpec, challenge_id: str, variant: VariantSpec) -> str:
    accent, ink, paper = template.palette
    return f"""<style>
:root {{ color-scheme: light; --accent: {_esc(accent)}; --ink: {_esc(ink)}; --paper: {_esc(paper)}; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.55 system-ui, sans-serif; }}
a {{ color: var(--accent); }}
header {{ padding: 2.1rem clamp(1rem, 5vw, 4rem) 1.4rem; border-bottom: 1px solid color-mix(in srgb, var(--accent) 35%, #fff); }}
main {{ max-width: 1000px; margin: 0 auto; padding: 1.5rem clamp(1rem, 5vw, 4rem) 3rem; }}
.kicker {{ margin: 0 0 .4rem; color: var(--accent); font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; }}
h1, h2, h3 {{ line-height: 1.15; }}
.lede {{ max-width: 65ch; opacity: .8; }}
.surface {{ margin-top: 1.2rem; padding: 1rem; border: 1px solid color-mix(in srgb, var(--accent) 28%, #fff); border-radius: .7rem; background: color-mix(in srgb, #fff 80%, var(--accent)); }}
.grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
.tile {{ padding: 1rem; border-radius: .6rem; background: #fff; box-shadow: 0 5px 18px rgb(0 0 0 / 7%); }}
table {{ width: 100%; border-collapse: collapse; background: #fff; }}
th, td {{ padding: .65rem .8rem; border-bottom: 1px solid #dde2e7; text-align: left; }}
th {{ color: var(--accent); }}
form {{ display: grid; gap: .65rem; }}
input {{ min-width: 0; padding: .55rem .65rem; border: 1px solid #aeb8c2; border-radius: .35rem; font: inherit; }}
button {{ padding: .5rem .8rem; border: 1px solid var(--accent); border-radius: .35rem; background: var(--accent); color: #fff; font: inherit; cursor: pointer; }}
button[type=button] {{ background: transparent; color: var(--accent); }}
code, pre {{ font: .9em ui-monospace, SFMono-Regular, Consolas, monospace; }}
pre {{ overflow: auto; padding: .8rem; background: #18212b; color: #eef4f8; border-radius: .5rem; }}
.timeline, .steps {{ display: grid; gap: .8rem; padding-left: 1.3rem; }}
.timeline li, .steps li {{ padding: .65rem .8rem; border-left: 3px solid var(--accent); background: #fff; }}
.badge {{ display: inline-flex; padding: .2rem .55rem; border-radius: 999px; background: color-mix(in srgb, var(--accent) 18%, #fff); color: var(--ink); font-size: .8rem; }}
.swatch {{ min-height: 7rem; display: grid; place-items: end start; padding: .75rem; border-radius: .5rem; background: linear-gradient(140deg, color-mix(in srgb, var(--accent) 72%, #fff), #fff); }}
footer {{ max-width: 1000px; margin: 0 auto; padding: 1rem clamp(1rem, 5vw, 4rem) 2rem; opacity: .6; font-size: .8rem; }}
</style>"""


def _head(
    template: TemplateSpec,
    challenge_id: str,
    variant: VariantSpec,
    marker: str,
    style: str,
) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="pg348-synthetic" content="localhost-only">
  <meta name="pg348-network" content="none">
  <meta name="pg348-state" content="read-only">
  <meta name="pg348-challenge" content="{_esc(challenge_id)}">
  <meta name="pg348-variant" content="{_esc(variant.variant_id)}">
  <meta name="pg348-transport" content="{_esc(variant.transport_method)}">
  <meta name="pg348-parameter-role" content="{_esc(variant.parameter_role)}">
  <meta name="pg348-encoding" content="{_esc('+'.join(variant.encoding_chain))}">
  <title>{_esc(template.title)} · {_esc(variant.variant_id)}</title>
  {style}
</head>
<body data-fixture="pg348-pages-a" data-template="{_esc(template.template_id)}" data-variant="{_esc(variant.variant_id)}" data-transport-method="{_esc(variant.transport_method)}" data-parameter-role="{_esc(variant.parameter_role)}" data-encoding-chain="{_esc('+'.join(variant.encoding_chain))}" data-marker="{_esc(marker)}" data-state-write="false">
"""


def _foot(challenge_id: str) -> str:
    return f"""<footer>
  <span>Local synthetic page · identity {_esc(challenge_id)} · no external requests or persistence.</span>
</footer>
</body>
</html>
"""


def _variant_markup(
    variant: VariantSpec,
    marker: str,
    challenge_id: str,
    *,
    compact: bool = False,
) -> str:
    """Render the same safe variant in a template-neutral block."""

    marker_e = _esc(marker)
    challenge_e = _esc(challenge_id)
    if variant.variant_id == "plain_label":
        return f'<p class="surface surface-text" data-role="static-label"><strong>Sample label:</strong> <span>{marker_e}</span></p>'
    if variant.variant_id in {"query_field", "query_encoded", "form_get"}:
        label = "One-step encoded sample" if variant.variant_id == "query_encoded" else "Local preview value"
        name = "sample" if variant.variant_id != "form_get" else "preview"
        action = "/synthetic/query" if variant.variant_id != "form_get" else "/synthetic/preview"
        encoding = "url-percent" if variant.variant_id == "query_encoded" else "form-urlencode"
        return f"""<form class="surface surface-form" method="get" action="{action}" data-encoding="{encoding}" data-state-write="false">
  <label for="{challenge_e}-field">{label}</label>
  <input id="{challenge_e}-field" name="{name}" value="{marker_e}" autocomplete="off">
  <button type="submit" aria-label="Preview only">Preview locally</button>
</form>"""
    if variant.variant_id == "attribute_value":
        return f'<p class="surface surface-attribute" data-sample-value="{marker_e}" data-role="attribute">Attribute is present; value is a safe marker.</p>'
    if variant.variant_id == "path_segment":
        return f'<p class="surface surface-path" data-role="path"><a href="/synthetic/segment/{marker_e}">Open local preview path</a></p>'
    if variant.variant_id == "fragment_anchor":
        return f'<p class="surface surface-fragment"><a href="#sample-{marker_e}">Jump to local sample</a></p><p id="sample-{marker_e}" class="tile">Fragment target: {marker_e}</p>'
    if variant.variant_id == "json_preview":
        return f'''<section class="surface surface-json" data-role="json-preview">
  <script type="application/json" id="{challenge_e}-json">{{"sample":"{marker_e}","mode":"preview"}}</script>
  <p>Non-executable JSON metadata is available for the local preview.</p>
  <pre aria-label="JSON preview">{{"sample":"{marker_e}","mode":"preview"}}</pre>
</section>'''
    if variant.variant_id == "dom_text":
        return f'''<section class="surface surface-dom" data-role="dom-text">
  <output id="{challenge_e}-output" data-initial="pending">Pending local sample</output>
  <button type="button" id="{challenge_e}-button">Reveal sample text</button>
  <script>document.getElementById("{challenge_e}-button").addEventListener("click", function () {{ document.getElementById("{challenge_e}-output").textContent = "{marker_e}"; }});</script>
</section>'''
    if variant.variant_id == "form_post":
        return f'''<form class="surface surface-form-post" method="post" action="/synthetic/preview" data-submit-policy="disabled-static" data-state-write="false">
  <label for="{challenge_e}-post-field">POST-shaped preview value</label>
  <input id="{challenge_e}-post-field" name="preview" value="{marker_e}" autocomplete="off">
  <button type="button" aria-label="POST preview is disabled">Preview locally</button>
  <small>This static fixture never submits or stores the value.</small>
</form>'''
    if variant.variant_id == "list_item":
        return f'<ul class="surface surface-list" data-role="list"><li data-sample="{marker_e}">Bounded list sample {marker_e}</li><li>Reference item</li></ul>'
    if variant.variant_id == "status_badge":
        return f'<p class="surface surface-status"><span class="badge" role="status" data-status="ready">Ready · {marker_e}</span></p>'
    raise ValueError(f"unknown safe variant: {variant.variant_id}")


def _wrap(template: TemplateSpec, challenge_id: str, variant: VariantSpec, marker: str, body: str) -> str:
    style = _style(template, challenge_id, variant)
    return _head(template, challenge_id, variant, marker, style) + body + _foot(challenge_id)


def _render_query_card(challenge_id: str, marker: str, title: str, variant: VariantSpec, template_id: str) -> str:
    return _wrap(
        TEMPLATES_BY_ID[template_id],
        challenge_id,
        variant,
        marker,
        f'''<header><p class="kicker">Signal board / {template_id}</p><h1>{_esc(title)}</h1><p class="lede">A compact card layout with a local, read-only sample surface.</p></header>
<main class="layout-query"><section class="grid"><article class="tile"><h2>Overview</h2><p>Three bounded facts are shown without a remote dependency.</p></article><article class="tile"><h2>Queue</h2><p><span class="badge">idle</span> local preview</p></article><article class="tile"><h2>Owner</h2><p>synthetic fixture</p></article></section>{_variant_markup(variant, marker, challenge_id)}</main>''',
    )


def _render_profile_table(challenge_id: str, marker: str, title: str, variant: VariantSpec, template_id: str) -> str:
    return _wrap(
        TEMPLATES_BY_ID[template_id],
        challenge_id,
        variant,
        marker,
        f'''<header><p class="kicker">Directory / {template_id}</p><h1>{_esc(title)}</h1><p class="lede">A tabular profile view with deterministic row labels.</p></header>
<main class="layout-profile"><table><caption>Local profile summary</caption><thead><tr><th scope="col">Field</th><th scope="col">Value</th><th scope="col">State</th></tr></thead><tbody><tr><th scope="row">Mode</th><td>preview</td><td><span class="badge">ready</span></td></tr><tr><th scope="row">Scope</th><td>loopback</td><td>bounded</td></tr><tr><th scope="row">Marker</th><td>{_esc(marker)}</td><td>static</td></tr></tbody></table>{_variant_markup(variant, marker, challenge_id)}</main>''',
    )


def _render_timeline(challenge_id: str, marker: str, title: str, variant: VariantSpec, template_id: str) -> str:
    return _wrap(
        TEMPLATES_BY_ID[template_id],
        challenge_id,
        variant,
        marker,
        f'''<header><p class="kicker">Chronicle / {template_id}</p><h1>{_esc(title)}</h1><p class="lede">An ordered timeline shape for deterministic navigation and history tokens.</p></header>
<main class="layout-timeline"><ol class="timeline"><li><strong>First note</strong><br>Fixture opened locally.</li><li><strong>Second note</strong><br>Preview surface selected.</li><li><strong>Third note</strong><br>Sample marker <code>{_esc(marker)}</code> retained.</li></ol>{_variant_markup(variant, marker, challenge_id)}</main>''',
    )


def _render_status_board(challenge_id: str, marker: str, title: str, variant: VariantSpec, template_id: str) -> str:
    return _wrap(
        TEMPLATES_BY_ID[template_id],
        challenge_id,
        variant,
        marker,
        f'''<header><p class="kicker">Operations / {template_id}</p><h1>{_esc(title)}</h1><p class="lede">A dashboard-shaped page with status tiles and a local preview channel.</p></header>
<main class="layout-status"><section class="grid"><article class="tile"><h2>Uptime</h2><p><strong>stable</strong></p><progress max="100" value="92">92%</progress></article><article class="tile"><h2>Tasks</h2><p><strong>04</strong> queued</p></article><article class="tile"><h2>Channel</h2><p><span class="badge">loopback</span></p></article><article class="tile"><h2>Marker</h2><p><code>{_esc(marker)}</code></p></article></section>{_variant_markup(variant, marker, challenge_id)}</main>''',
    )


def _render_faq_stack(challenge_id: str, marker: str, title: str, variant: VariantSpec, template_id: str) -> str:
    return _wrap(
        TEMPLATES_BY_ID[template_id],
        challenge_id,
        variant,
        marker,
        f'''<header><p class="kicker">Guide / {template_id}</p><h1>{_esc(title)}</h1><p class="lede">A disclosure stack with intentionally different nesting from the other templates.</p></header>
<main class="layout-faq"><details open><summary>What is this page?</summary><p>A static synthetic surface for local inspection.</p></details><details><summary>Does it store anything?</summary><p>No. The page and every control are read-only previews.</p></details><details><summary>Which marker is visible?</summary><p><code>{_esc(marker)}</code></p></details>{_variant_markup(variant, marker, challenge_id)}</main>''',
    )


def _render_checkout_preview(challenge_id: str, marker: str, title: str, variant: VariantSpec, template_id: str) -> str:
    return _wrap(
        TEMPLATES_BY_ID[template_id],
        challenge_id,
        variant,
        marker,
        f'''<header><p class="kicker">Preview desk / {template_id}</p><h1>{_esc(title)}</h1><p class="lede">A fieldset-heavy layout that never handles payment, identity, or persistent state.</p></header>
<main class="layout-checkout"><fieldset><legend>Sample order preview</legend><p><label>Item <input value="paper sample" readonly></label></p><p><label>Quantity <input type="number" value="1" readonly></label></p><p><span class="badge">no transaction</span></p></fieldset><section class="surface"><h2>Local note</h2><p>This is a visual fixture only. Marker: <code>{_esc(marker)}</code></p></section>{_variant_markup(variant, marker, challenge_id)}</main>''',
    )


def _render_docs_shell(challenge_id: str, marker: str, title: str, variant: VariantSpec, template_id: str) -> str:
    return _wrap(
        TEMPLATES_BY_ID[template_id],
        challenge_id,
        variant,
        marker,
        f'''<header><p class="kicker">Notebook / {template_id}</p><h1>{_esc(title)}</h1><p class="lede">A two-column documentation shell with local fragment navigation.</p></header>
<main class="layout-docs"><div class="grid"><aside class="tile"><h2>Sections</h2><nav aria-label="Local sections"><a href="#intro">Intro</a><br><a href="#details">Details</a><br><a href="#limits">Limits</a></nav></aside><article class="tile"><h2 id="intro">Intro</h2><p>This fixture keeps every resource on the local page.</p><h3 id="details">Details</h3><p>Template marker: <code>{_esc(marker)}</code></p><h3 id="limits">Limits</h3><p>No remote links, credentials, callbacks, or writes.</p></article></div>{_variant_markup(variant, marker, challenge_id)}</main>''',
    )


def _render_gallery_grid(challenge_id: str, marker: str, title: str, variant: VariantSpec, template_id: str) -> str:
    return _wrap(
        TEMPLATES_BY_ID[template_id],
        challenge_id,
        variant,
        marker,
        f'''<header><p class="kicker">Studio / {template_id}</p><h1>{_esc(title)}</h1><p class="lede">A visual grid made from local CSS swatches instead of remote media.</p></header>
<main class="layout-gallery"><section class="grid"><figure class="tile"><div class="swatch">Alpha</div><figcaption>Warm swatch</figcaption></figure><figure class="tile"><div class="swatch">Beta</div><figcaption>Cool swatch</figcaption></figure><figure class="tile"><div class="swatch">Gamma</div><figcaption>Neutral swatch</figcaption></figure></section>{_variant_markup(variant, marker, challenge_id)}</main>''',
    )


def _render_calendar_grid(challenge_id: str, marker: str, title: str, variant: VariantSpec, template_id: str) -> str:
    return _wrap(
        TEMPLATES_BY_ID[template_id],
        challenge_id,
        variant,
        marker,
        f'''<header><p class="kicker">Planner / {template_id}</p><h1>{_esc(title)}</h1><p class="lede">A calendar-like table with predictable row and column structure.</p></header>
<main class="layout-calendar"><table><caption>Local sample week</caption><thead><tr><th scope="col">Day</th><th scope="col">Window A</th><th scope="col">Window B</th><th scope="col">Window C</th></tr></thead><tbody><tr><th scope="row">Mon</th><td>open</td><td>quiet</td><td>open</td></tr><tr><th scope="row">Tue</th><td>quiet</td><td>open</td><td>quiet</td></tr><tr><th scope="row">Wed</th><td>open</td><td>open</td><td>quiet</td></tr><tr><th scope="row">Thu</th><td>quiet</td><td>quiet</td><td>open</td></tr></tbody></table>{_variant_markup(variant, marker, challenge_id)}</main>''',
    )


def _render_wizard_steps(challenge_id: str, marker: str, title: str, variant: VariantSpec, template_id: str) -> str:
    return _wrap(
        TEMPLATES_BY_ID[template_id],
        challenge_id,
        variant,
        marker,
        f'''<header><p class="kicker">Flow / {template_id}</p><h1>{_esc(title)}</h1><p class="lede">A stepped flow whose controls are intentionally preview-only.</p></header>
<main class="layout-wizard"><ol class="steps"><li><strong>Choose</strong><br>Select a local option.</li><li><strong>Review</strong><br>Check the marker <code>{_esc(marker)}</code>.</li><li><strong>Finish</strong><br>No submission or state change occurs.</li></ol><section class="surface"><label for="{_esc(challenge_id)}-choice">Local choice</label><select id="{_esc(challenge_id)}-choice"><option>Alpha</option><option>Beta</option><option>Gamma</option></select></section>{_variant_markup(variant, marker, challenge_id)}</main>''',
    )


def _template_specs() -> tuple[TemplateSpec, ...]:
    return (
        TemplateSpec("a_t01_query_card", "query-card", "Query Card", "Signal board", ("#2b6cb0", "#1d2733", "#f3f8fc"), _render_query_card, "Card grid with a compact local preview."),
        TemplateSpec("a_t02_profile_table", "profile-table", "Profile Table", "Directory", ("#8b5e34", "#2f251e", "#fbf5ef"), _render_profile_table, "Tabular profile with caption and row headers."),
        TemplateSpec("a_t03_timeline", "timeline", "Timeline", "Chronicle", ("#0f766e", "#172a2a", "#effaf8"), _render_timeline, "Ordered event list with local marker."),
        TemplateSpec("a_t04_status_board", "status-board", "Status Board", "Operations", ("#6b46c1", "#251c35", "#f7f2ff"), _render_status_board, "Dashboard tiles and progress indicator."),
        TemplateSpec("a_t05_faq_stack", "faq-stack", "FAQ Stack", "Guide", ("#c05621", "#351b11", "#fff8f3"), _render_faq_stack, "Disclosure-oriented guide layout."),
        TemplateSpec("a_t06_checkout_preview", "checkout-preview", "Checkout Preview", "Preview desk", ("#b83280", "#321529", "#fff4fb"), _render_checkout_preview, "Fieldset-heavy visual preview with no transaction."),
        TemplateSpec("a_t07_docs_shell", "docs-shell", "Docs Shell", "Notebook", ("#2c5282", "#1b2838", "#f1f6fc"), _render_docs_shell, "Aside and article documentation shell."),
        TemplateSpec("a_t08_gallery_grid", "gallery-grid", "Gallery Grid", "Studio", ("#2f855a", "#183322", "#f1fbf4"), _render_gallery_grid, "CSS-only swatch gallery without remote media."),
        TemplateSpec("a_t09_calendar_grid", "calendar-grid", "Calendar Grid", "Planner", ("#975a16", "#30210a", "#fffaf0"), _render_calendar_grid, "Calendar-shaped table with four rows."),
        TemplateSpec("a_t10_wizard_steps", "wizard-steps", "Wizard Steps", "Flow", ("#2c7a7b", "#172e2e", "#effafa"), _render_wizard_steps, "Three-step local preview flow."),
    )


TEMPLATES: tuple[TemplateSpec, ...] = _template_specs()
TEMPLATES_BY_ID: Mapping[str, TemplateSpec] = {item.template_id: item for item in TEMPLATES}


def _render_page(template: TemplateSpec, variant: VariantSpec, challenge_id: str, marker: str) -> str:
    rendered = template.renderer(challenge_id, marker, f"{template.title} / {variant.variant_id}", variant, template.template_id)
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def _assert_safe_page(source: str, *, challenge_id: str) -> None:
    for pattern in FORBIDDEN_SOURCE_PATTERNS:
        match = pattern.search(source)
        if match:
            raise ValueError(f"{challenge_id}: forbidden source pattern {pattern.pattern!r}")
    if 'data-state-write="false"' not in source:
        raise ValueError(f"{challenge_id}: missing read-only state declaration")
    if 'content="localhost-only"' not in source or 'content="none"' not in source:
        raise ValueError(f"{challenge_id}: missing localhost/network declaration")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _generator_sha256() -> str:
    try:
        return _sha256_bytes(Path(__file__).read_bytes())
    except OSError:
        return "unknown"


def _records_and_sources() -> tuple[list[dict[str, object]], dict[str, str]]:
    records: list[dict[str, object]] = []
    sources: dict[str, str] = {}
    for template_index, template in enumerate(TEMPLATES, start=1):
        for variant_index, variant in enumerate(VARIANTS, start=1):
            challenge_id = f"pg348-a-t{template_index:02d}-v{variant_index:02d}"
            marker = f"sample-a-{template_index:02d}-{variant_index:02d}"
            file_name = f"pg348_a_t{template_index:02d}_v{variant_index:02d}_{template.slug}_{variant.variant_id}.html"
            local_path = f"pages/{file_name}"
            source = _render_page(template, variant, challenge_id, marker)
            _assert_safe_page(source, challenge_id=challenge_id)
            source_hash = _sha256_bytes(source.encode("utf-8"))
            sources[local_path] = source
            records.append(
                {
                    "challenge_id": challenge_id,
                    "local_path": local_path,
                    "mechanism_id": f"pg348.{template.slug}.{variant.mechanism_id}",
                    "surface_template_id": template.template_id,
                    "implementation_group": IMPLEMENTATION_GROUP,
                    "transport_method": variant.transport_method,
                    "parameter_role": variant.parameter_role,
                    "encoding_chain": list(variant.encoding_chain),
                    "response_shape": variant.response_shape,
                    "redirect_shape": variant.redirect_shape,
                    "script_surface": variant.script_surface,
                    "synthetic_oracle_kind": variant.synthetic_oracle_kind,
                    "source_hash": source_hash,
                    "raw_source_for_evaluator_only": True,
                    "training_context_raw": False,
                    "localhost_only": True,
                    "external_network": False,
                    "state_write": False,
                    "template_index": template_index,
                    "variant_index": variant_index,
                    "safe_variant_id": variant.variant_id,
                }
            )
    return records, sources


def _duplicate_groups(values: Iterable[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for index, value in enumerate(values):
        groups.setdefault(value, []).append(str(index))
    return [members for members in groups.values() if len(members) > 1]


def _manifest_payload(records: Sequence[dict[str, object]], sources: Mapping[str, str]) -> dict[str, object]:
    source_hashes = [str(record["source_hash"]) for record in records]
    paths = [str(record["local_path"]) for record in records]
    challenge_ids = [str(record["challenge_id"]) for record in records]
    template_variant_pairs = [
        f"{record['surface_template_id']}::{record['safe_variant_id']}" for record in records
    ]
    dedup = {
        "record_count": len(records),
        "challenge_id_unique": len(set(challenge_ids)),
        "local_path_unique": len(set(paths)),
        "source_hash_unique": len(set(source_hashes)),
        "template_variant_pair_unique": len(set(template_variant_pairs)),
        "duplicate_challenge_id_groups": _duplicate_groups(challenge_ids),
        "duplicate_local_path_groups": _duplicate_groups(paths),
        "duplicate_source_hash_groups": _duplicate_groups(source_hashes),
        "duplicate_template_variant_groups": _duplicate_groups(template_variant_pairs),
        "all_content_distinct": len(set(source_hashes)) == len(records),
    }
    templates = [
        {
            "surface_template_id": template.template_id,
            "slug": template.slug,
            "title": template.title,
            "description": template.description,
            "instance_count": sum(1 for record in records if record["surface_template_id"] == template.template_id),
        }
        for template in TEMPLATES
    ]
    variants = [
        {
            "safe_variant_id": variant.variant_id,
            "mechanism_id": variant.mechanism_id,
            "transport_method": variant.transport_method,
            "parameter_role": variant.parameter_role,
            "encoding_chain": list(variant.encoding_chain),
            "response_shape": variant.response_shape,
            "redirect_shape": variant.redirect_shape,
            "script_surface": variant.script_surface,
            "synthetic_oracle_kind": variant.synthetic_oracle_kind,
            "description": variant.description,
        }
        for variant in VARIANTS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": FIXTURE_ID,
        "status": "fixture_only_evaluation",
        "description": "Local static HTML pages for abstract surface-shape checks; not a vulnerability or payload corpus.",
        "generator": "generate_pages_a.py",
        "generator_sha256": _generator_sha256(),
        "base_origin": "http://127.0.0.1 (synthetic-only documentation; no server is started by this generator)",
        "network_policy": {
            "localhost_only": True,
            "external_network": False,
            "credentials": False,
            "callbacks": False,
            "persistent_state": False,
            "write_routes": [],
        },
        "counts": {
            "templates": len(TEMPLATES),
            "safe_variants": len(VARIANTS),
            "challenge_instances": len(records),
            "static_pages": len(sources),
            "get_records": sum(record["transport_method"] == "GET" for record in records),
            "post_shape_records": sum(record["transport_method"] == "POST" for record in records),
        },
        "templates": templates,
        "safe_variants": variants,
        "records": list(records),
        "dedup_stats": dedup,
        "raw_source_policy": {
            "raw_source_for_evaluator_only": True,
            "training_context_raw": False,
            "raw_payloads": False,
            "raw_responses": False,
            "evaluator_answers": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")


def build_manifest() -> tuple[dict[str, object], dict[str, str]]:
    records, sources = _records_and_sources()
    manifest = _manifest_payload(records, sources)
    manifest["manifest_payload_sha256"] = _sha256_bytes(_canonical_json(manifest))
    return manifest, sources


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR, *, check: bool = False) -> dict[str, object]:
    """Generate or verify the page set and return a compact summary."""

    output_dir = Path(output_dir).resolve()
    manifest, sources = build_manifest()
    pages_dir = output_dir / "pages"
    manifest_path = output_dir / "manifest_v1.json"
    if check:
        if not manifest_path.is_file():
            raise SystemExit(f"missing manifest: {manifest_path}")
        try:
            actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read manifest: {exc}") from exc
        if actual_manifest != manifest:
            raise SystemExit("manifest does not match deterministic generator output")
        for local_path, expected in sources.items():
            page_path = output_dir / local_path
            if not page_path.is_file():
                raise SystemExit(f"missing page: {page_path}")
            actual = page_path.read_text(encoding="utf-8")
            if actual != expected:
                raise SystemExit(f"page does not match deterministic output: {page_path}")
        return {
            "status": "checked",
            "manifest": str(manifest_path),
            "pages": len(sources),
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        }

    pages_dir.mkdir(parents=True, exist_ok=True)
    for local_path, source in sources.items():
        _write_text(output_dir / local_path, source)
    _write_text(manifest_path, _canonical_json(manifest).decode("utf-8"))
    return {
        "status": "generated",
        "manifest": str(manifest_path),
        "pages": len(sources),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory containing pages/ and manifest_v1.json (default: this directory)",
    )
    parser.add_argument("--check", action="store_true", help="verify existing generated files without writing")
    args = parser.parse_args(argv)
    try:
        summary = generate(args.output_dir, check=args.check)
    except (OSError, ValueError, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
