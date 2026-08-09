"""PG-286 bounded observation-to-token projection.

This module converts already-authorized response/DOM/redirect/logic
projections into compact tokens.  It intentionally excludes family names,
oracle decisions, vulnerability labels, raw bodies, request values and AST
payload fragments.  Missing modality evidence is represented explicitly as
``evidence_status=incomplete`` so a learner can abstain instead of guessing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "pg286-bounded-observation-token-v1"
INCOMPLETE = "incomplete"
COMPLETE = "complete"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _bucket(value: Any, bounds: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536)) -> str:
    try:
        number = max(0, int(value))
    except (TypeError, ValueError):
        return "unknown"
    for upper in bounds:
        if number <= upper:
            return str(upper)
    return "large"


def _bool(value: Any) -> str:
    return "1" if bool(value) else "0"


def _status_class(projection: Mapping[str, Any]) -> str:
    value = str(projection.get("status_class", "other"))
    return value if value in {"1xx", "2xx", "3xx", "4xx", "5xx", "other"} else "other"


def _content_class(projection: Mapping[str, Any]) -> str:
    value = str(projection.get("content_type_class", projection.get("content_type", "unknown")))
    return value if value in {"html", "json", "text", "xml", "image", "other", "unknown"} else "other"


def _shape_tokens(projection: Mapping[str, Any], prefix: str = "obs") -> list[str]:
    shape = projection.get("shape")
    if not isinstance(shape, Mapping):
        shape = {}
    kind = str(shape.get("kind", projection.get("shape_class", "unknown")))
    kind = kind if kind in {"object", "array", "scalar", "html", "other", "none", "unknown"} else "other"
    return [
        f"{prefix}_shape={kind}",
        f"{prefix}_shape_fields={_bucket(shape.get('field_count', 0), (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512))}",
        f"{prefix}_shape_scalars={_bucket(shape.get('scalar_count', 0), (0, 1, 2, 4, 8, 16, 32, 64))}",
    ]


def project_response_projection(projection: Mapping[str, Any], *, prefix: str = "obs") -> list[str]:
    """Project bounded transport/body geometry without retaining values."""

    return [
        f"{prefix}_status={_status_class(projection)}",
        f"{prefix}_content={_content_class(projection)}",
        f"{prefix}_length={str(projection.get('body_length_bucket', 'unknown'))[:24]}",
        f"{prefix}_transport_error={_bool(projection.get('transport_error', False))}",
        f"{prefix}_status_changed={_bool(projection.get('status_changed', False))}",
        f"{prefix}_state_changed={_bool(projection.get('state_changed', False))}",
        f"{prefix}_location_changed={_bool(projection.get('location_origin_changed', projection.get('external_redirect', False)))}",
        f"{prefix}_marker_reflected={_bool(projection.get('marker_reflected', (projection.get('marker') or {}).get('reflected', False)))}",
        f"{prefix}_marker_location={str((projection.get('marker') or {}).get('location', 'none'))[:24]}",
        f"{prefix}_redirect_hops={_bucket(len(projection.get('redirect_chain') or projection.get('status_chain', [])), (0, 1, 2, 3, 4, 8))}",
        *_shape_tokens(projection, prefix=prefix),
    ]


def project_dom_projection(dom: Mapping[str, Any] | None, *, prefix: str = "dom") -> list[str]:
    """Project no-JS DOM geometry; omit the derived ``dom_change`` label."""

    if not isinstance(dom, Mapping):
        return [f"{prefix}_available=0"]
    return [
        f"{prefix}_available={_bool(dom.get('browser_dom_observed', False))}",
        f"{prefix}_marker_hits={_bucket(dom.get('marker_hits', 0), (0, 1, 2, 4, 8))}",
        f"{prefix}_body_text_hits={_bucket(dom.get('body_text_hits', 0), (0, 1, 2, 4, 8))}",
        f"{prefix}_elements={_bucket(dom.get('element_count', 0), (0, 1, 16, 32, 64, 128, 256, 512))}",
        f"{prefix}_scripts={_bucket(dom.get('script_tag_count', 0), (0, 1, 2, 4, 8, 16, 32, 64))}",
        f"{prefix}_network={_bool(dom.get('network_access', False))}",
        f"{prefix}_navigation={_bool(dom.get('navigation', False))}",
        "dom_script_execution=0",
    ]


def project_sql_observation(
    response: Mapping[str, Any] | None,
    *,
    ast_projection: Mapping[str, Any] | None = None,
    prefix: str = "sql",
) -> tuple[list[str], bool]:
    """Project SQL response/AST shape while withholding family/effect labels."""

    response = response if isinstance(response, Mapping) else {}
    signal = response.get("signal") if isinstance(response.get("signal"), Mapping) else response
    tokens = [
        f"{prefix}_available=1" if response else f"{prefix}_available=0",
        f"{prefix}_error_shape={_bool(signal.get('sql_error_shape', False))}",
        f"{prefix}_status_changed={_bool(signal.get('status_changed', False))}",
        f"{prefix}_marker_reflected={_bool(signal.get('marker_reflected', False))}",
    ]
    tokens.extend(project_response_projection(response.get("response_projection", response), prefix=prefix))
    ast = ast_projection if isinstance(ast_projection, Mapping) else None
    if ast is None:
        tokens.extend([f"{prefix}_ast_available=0", f"{prefix}_ast_kind=unknown", f"{prefix}_ast_boundary=unknown"])
        return tokens, False
    # Only shape/modality fields survive.  Fragment class, typed effect,
    # positive labels and raw AST leaves are deliberately not copied.
    kind = str(ast.get("kind", "unknown"))
    kind = kind if kind in {"select", "parse_error", "unknown"} else "other"
    tokens.extend([
        f"{prefix}_ast_available=1",
        f"{prefix}_ast_kind={kind}",
        f"{prefix}_ast_boundary={_bool(ast.get('interpreter_boundary', ast.get('boundary', False)))}",
        f"{prefix}_timing_observed={_bool(ast.get('timing_differential', ast.get('timeout_observed', False)))}",
        f"{prefix}_row_shape_observed={_bool(ast.get('row_shape_differential', False))}",
    ])
    return tokens, True


def project_logic_observation(projection: Mapping[str, Any] | None, *, prefix: str = "logic") -> list[str]:
    projection = projection if isinstance(projection, Mapping) else {}
    transition = str(projection.get("transition_delta", "none"))
    transition = transition if transition in {"none", "authorization", "scope", "visibility", "state", "metadata", "unknown"} else "other"
    return [
        f"{prefix}_transition={transition}",
        f"{prefix}_scope_changed={_bool(projection.get('scope_changed', False))}",
        f"{prefix}_authorization_changed={_bool(projection.get('authorization_changed', False))}",
        f"{prefix}_visibility_changed={_bool(projection.get('visibility_changed', False))}",
        f"{prefix}_state_changed={_bool(projection.get('state_changed', False))}",
    ]


def project_redirect_observation(projection: Mapping[str, Any] | None, *, prefix: str = "redirect") -> list[str]:
    """Project redirect-chain geometry without retaining locations or labels."""

    projection = projection if isinstance(projection, Mapping) else {}
    terminal = str(projection.get("terminal_status", "unknown"))
    terminal = terminal if terminal in {"1xx", "2xx", "3xx", "4xx", "5xx", "unknown"} else "unknown"
    chain_shape = str(projection.get("chain_shape", "unknown"))
    chain_shape = chain_shape if chain_shape in {"none", "same_origin", "cross_origin", "loop", "unknown"} else "unknown"
    return [
        f"{prefix}_available={_bool(projection)}",
        f"{prefix}_hops={_bucket(projection.get('hop_count', 0), (0, 1, 2, 3, 4, 8))}",
        f"{prefix}_same_origin={_bool(projection.get('same_origin', False))}",
        f"{prefix}_terminal_status={terminal}",
        f"{prefix}_chain_shape={chain_shape}",
    ]


def field_role_tokens(fields: Sequence[str] | None) -> list[str]:
    """Convert observed field names to coarse roles, not exact names."""

    roles: set[str] = set()
    for field in fields or []:
        value = str(field).casefold()
        if value in {"id", "idx", "page", "count"}:
            roles.add("numeric")
        elif value in {"url", "uri", "redirect", "next", "return"}:
            roles.add("url")
        elif value in {"submit", "action", "op"}:
            roles.add("control")
        elif value in {"name", "message", "text", "content", "title", "username", "query"}:
            roles.add("text")
        else:
            roles.add("opaque")
    return [f"field_role={role}" for role in sorted(roles)] or ["field_role=none"]


def build_observation_tokens(
    *,
    method: str,
    fields: Sequence[str] | None,
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    negative: Mapping[str, Any] | None,
    sql_response: Mapping[str, Any] | None = None,
    dom: Mapping[str, Any] | None = None,
    sql_ast: Mapping[str, Any] | None = None,
    logic: Mapping[str, Any] | None = None,
    redirect: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a context-safe observation token record."""

    method = str(method).upper() if str(method).upper() in {"GET", "POST"} else "GET"
    tokens = ["[BOS]", "ir_layer=payload_grounding", "ir_family_agnostic=1", f"method={method}", *field_role_tokens(fields)]
    tokens.extend(project_response_projection(baseline or {}, prefix="baseline"))
    tokens.extend(project_response_projection(candidate or {}, prefix="candidate"))
    tokens.extend(project_response_projection(negative or {}, prefix="negative"))
    tokens.extend(project_dom_projection(dom, prefix="dom"))
    # A generic candidate response is not automatically a SQL observation.
    # Callers must opt in with ``sql_response`` when the target-side adapter
    # actually supplies SQL-channel evidence; otherwise DOM/redirect/logic
    # rows receive a neutral ``sql_available=0`` slot instead of a false SQL
    # feature.
    sql_tokens, ast_available = project_sql_observation(sql_response, ast_projection=sql_ast, prefix="sql")
    tokens.extend(sql_tokens)
    tokens.extend(project_logic_observation(logic, prefix="logic"))
    tokens.extend(project_redirect_observation(redirect, prefix="redirect"))
    required = {
        "baseline": bool(baseline),
        "candidate": bool(candidate),
        "negative": bool(negative),
        "dom_or_sql_or_logic_or_redirect": bool(dom or sql_ast or logic or redirect),
    }
    missing = [key for key, present in required.items() if not present]
    tokens.extend(["oracle_label_in_context=0", "literal_probe_in_context=0", f"evidence_status={INCOMPLETE if missing else COMPLETE}", "[CTX_END]"])
    return {
        "schema_version": SCHEMA_VERSION,
        "context_tokens": tokens,
        "evidence_status": INCOMPLETE if missing else COMPLETE,
        "missing_modalities": missing,
        "sql_ast_available": bool(ast_available),
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_label_in_context": False,
        "token_sha256": digest(tokens),
    }


__all__ = [
    "COMPLETE",
    "INCOMPLETE",
    "SCHEMA_VERSION",
    "build_observation_tokens",
    "digest",
    "field_role_tokens",
    "project_dom_projection",
    "project_logic_observation",
    "project_redirect_observation",
    "project_response_projection",
    "project_sql_observation",
]
