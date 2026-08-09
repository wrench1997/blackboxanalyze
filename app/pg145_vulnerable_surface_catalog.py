"""PG-145 local multi-surface vulnerability-family catalog.

This is a static, loopback-only evaluator fixture.  It intentionally models
three common *families* (reflected sink, query/AST boundary, and XML entity
parser) across different source styles, but never executes a probe, SQL query,
external entity, or script.  The model view contains only bounded abstract
tokens; source text, marker strings, and oracle authority remain evaluator
side and are represented by hashes/typed categories.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Iterable


SCHEMA_VERSION = "pg145-local-multisurface-vulnerability-catalog-v1"
DATASET_SCHEMA = "pg145-local-multisurface-model-dataset-v1"
BASE_URL = "http://127.0.0.1:14500"
TARGET_COUNT = 150
STYLES = ("govuk_like", "material_like", "primer_like", "dashboard_like")
FAMILIES = ("xss_reflection", "sql_ast_boundary", "xxe_entity_parser")
METHODS = ("GET", "POST")

FAMILY_SPEC = {
    "xss_reflection": {
        "sink_class": "html_dom_sink",
        "oracle_positive": "reflected_sink_reached",
        "oracle_negative": "sink_not_reached",
        "probe_shape": "inert_marker_reflection",
    },
    "sql_ast_boundary": {
        "sink_class": "query_ast_boundary",
        "oracle_positive": "mock_ast_boundary_signal",
        "oracle_negative": "mock_ast_unchanged",
        "probe_shape": "bounded_query_shape_marker",
    },
    "xxe_entity_parser": {
        "sink_class": "xml_entity_parser",
        "oracle_positive": "mock_entity_resolution_signal",
        "oracle_negative": "mock_entity_not_resolved",
        "probe_shape": "inert_xml_entity_marker",
    },
}

STYLE_REFERENCES = {
    "govuk_like": {
        "reference_url": "https://design-system.service.gov.uk/styles/layout/",
        "layout_signature": "mobile_first_two_thirds_grid",
        "component_signature": "header_footer_form_components",
        "source_license_note": "public_design_system_mit_code",
    },
    "material_like": {
        "reference_url": "https://material-web.dev/about/intro/",
        "layout_signature": "adaptive_web_components_design_tokens",
        "component_signature": "custom_elements_form_controls",
        "source_license_note": "public_guidelines_reference_only",
    },
    "primer_like": {
        "reference_url": "https://primer.style/product/getting-started/foundations/layout/",
        "layout_signature": "responsive_page_regions_stack_grid",
        "component_signature": "header_content_panes_footer",
        "source_license_note": "public_design_guidelines_reference_only",
    },
    "dashboard_like": {
        "reference_url": "local_abstract_dashboard_style",
        "layout_signature": "sidebar_toolbar_card_grid",
        "component_signature": "filter_table_status_panel",
        "source_license_note": "generated_local_style",
    },
}


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _source_shape(family: str, style: str) -> dict[str, str]:
    # Distinct source/HTML/JS shapes are described, never serialized as raw
    # source.  The digest binds the evaluator-side template implementation.
    source_id = f"pg145::{family}::{style}::template-v1"
    return {
        "source_template_id": source_id,
        "source_hash": hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
        "style_reference": STYLE_REFERENCES[style],
        "html_shape": {
            "govuk_like": "form+fieldset+service_header",
            "material_like": "custom_field+surface_container",
            "primer_like": "page_header+content_panes",
            "dashboard_like": "sidebar+toolbar+card_grid",
        }[style],
        "javascript_shape": {
            "govuk_like": "progressive_enhancement",
            "material_like": "web_component_event",
            "primer_like": "responsive_component_state",
            "dashboard_like": "filter_table_callback",
        }[style],
    }


def _model_tokens(*, family: str, style: str, method: str, positive: bool, unknown: bool = False) -> list[str]:
    spec = FAMILY_SPEC[family]
    availability = "unknown" if unknown else "typed"
    effect = "unknown" if unknown else "candidate" if positive else "no_effect"
    placement = "query" if method == "GET" else "body_field"
    return [
        "[BOS]",
        "[STEP]",
        "[SRC_HTML]",
        f"src.html.style={style}",
        "src.html.syntax=bounded_template",
        "[SRC_JAVASCRIPT]",
        f"src.javascript.shape={_source_shape(family, style)['javascript_shape']}",
        "[SRC_TRANSPORT]",
        f"src.transport.method={method}",
        f"src.transport.placement={placement}",
        "src.transport.route=loopback_allowlisted",
        "[IR]",
        "ir.surface.family_free=true",
        f"ir.surface.sink_class={spec['sink_class']}",
        "ir.probe.shape=bounded_marker",
        f"ir.response.effect={effect}",
        f"ir.oracle.availability={availability}",
        "ir.failure.weight=1.0",
        "[OBS]",
        f"obs.oracle.availability={availability}",
        "[EOS]",
    ]


def _waf_projection(*, family: str, style: str, method: str, positive: bool) -> dict[str, Any]:
    # Local mock filter categories only.  No bypass string is stored or
    # generated; this is for studying false positives/normalization behavior.
    digest = hashlib.sha256(f"{family}:{style}:{method}".encode("utf-8")).digest()
    normalization = ("plain", "percent_once", "case_fold", "entity_once")[digest[0] % 4]
    return {
        "profile": f"mock_waf_{style}",
        "normalization_variant": normalization,
        "filter_decision": "allow" if positive else "deny_or_no_effect",
        "waf_training_scope": "local_mock_only",
        "external_bypass_payloads": False,
    }


def _target_id(index: int, family: str) -> str:
    return f"pg145-{family}-{index:03d}"


def _target_url(target_id: str) -> str:
    return f"{BASE_URL}/pg145/{target_id}"


def build_catalog() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    catalog_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    target_ids: list[str] = []
    family_counts = Counter()
    style_counts = Counter()
    method_counts = Counter()
    positive_count = negative_count = unknown_count = 0
    for index in range(TARGET_COUNT):
        family = FAMILIES[index % len(FAMILIES)]
        style = STYLES[(index // len(FAMILIES)) % len(STYLES)]
        target_id = _target_id(index + 1, family)
        target_ids.append(target_id)
        source = _source_shape(family, style)
        split = "implementation_holdout" if style == "framework_adapter" or index >= 120 else "train"
        for method in METHODS:
            pair_id = f"{target_id}-{method.lower()}"
            for positive in (True, False):
                role = "confirmed_positive_candidate" if positive else "matched_negative_control"
                spec = FAMILY_SPEC[family]
                row_key = f"{target_id}::{method}::{role}"
                evaluator = {
                    "typed_available": True,
                    "expected_positive": positive,
                    "expected_effect": spec["oracle_positive"] if positive else spec["oracle_negative"],
                    "oracle_contract_sha256": sha256_json({"family": family, "effect": spec["oracle_positive"] if positive else spec["oracle_negative"]}),
                    "evidence_hash": sha256_json({"target": target_id, "method": method, "role": role, "family": family}),
                    "raw_marker_retained": False,
                    "raw_probe_retained": False,
                    "external_network": False,
                    "database_touched": False,
                    "file_read": False,
                    "script_execution": False,
                }
                reset = {
                    "fresh": True,
                    "fresh_target": True,
                    "reset_epoch": f"{target_id}-epoch-1",
                    "evidence_hash": sha256_json({"target": target_id, "reset": 1}),
                    "external_network": False,
                    "state_change_allowed": False,
                }
                catalog_row = {
                    "schema_version": SCHEMA_VERSION,
                    "row_id": row_key,
                    "target_instance_id": target_id,
                    "target_url": _target_url(target_id),
                    "vulnerability_family": family,
                    "surface_style": style,
                    "source_template_id": source["source_template_id"],
                    "source_hash": source["source_hash"],
                    "style_reference": source["style_reference"],
                    "split": split,
                    "pair_id": pair_id,
                    "control_role": role,
                    "method": method,
                    "route": f"/pg145/{target_id}",
                    "probe_shape": spec["probe_shape"],
                    "model_tokens": _model_tokens(family=family, style=style, method=method, positive=positive),
                    "oracle": evaluator,
                    "fresh_reset": reset,
                    "waf": _waf_projection(family=family, style=style, method=method, positive=positive),
                    "training_eligible": False,
                    "memory_promotion_allowed": False,
                }
                catalog_row["row_sha256"] = sha256_json(catalog_row)
                catalog_rows.append(catalog_row)
                model_rows.append({
                    "schema_version": DATASET_SCHEMA,
                    "row_id": row_key,
                    "split": split,
                    "tokens": catalog_row["model_tokens"],
                    "token_count": len(catalog_row["model_tokens"]),
                    "labels_in_model_row": False,
                    "oracle_authority_in_model_row": False,
                    "target_identity_in_model_row": False,
                    "source_hash_in_model_row": False,
                    "action_supervision_allowed": False,
                    "safety_supervision_allowed": False,
                })
                family_counts[family] += 1
                style_counts[style] += 1
                method_counts[method] += 1
                if positive:
                    positive_count += 1
                else:
                    negative_count += 1
            unknown_key = f"{target_id}::GET::unknown_oracle"
            unknown_tokens = _model_tokens(family=family, style=style, method="GET", positive=False, unknown=True)
            unknown_row = {
                "schema_version": SCHEMA_VERSION,
                "row_id": unknown_key,
                "target_instance_id": target_id,
                "target_url": _target_url(target_id),
                "vulnerability_family": family,
                "surface_style": style,
                "source_template_id": source["source_template_id"],
                "source_hash": source["source_hash"],
                "style_reference": source["style_reference"],
                "split": split,
                "pair_id": f"{target_id}-GET",
                "control_role": "unknown_oracle",
                "method": "GET",
                "route": f"/pg145/{target_id}",
                "probe_shape": FAMILY_SPEC[family]["probe_shape"],
                "model_tokens": unknown_tokens,
                "oracle": {
                    "typed_available": False,
                    "expected_positive": "unknown",
                    "expected_effect": "unknown",
                    "oracle_contract_sha256": "unknown",
                    "evidence_hash": sha256_json({"target": target_id, "role": "unknown_oracle"}),
                    "raw_marker_retained": False,
                    "raw_probe_retained": False,
                    "external_network": False,
                    "database_touched": False,
                    "file_read": False,
                    "script_execution": False,
                },
                "fresh_reset": reset,
                "waf": _waf_projection(family=family, style=style, method="GET", positive=False),
                "training_eligible": False,
                "memory_promotion_allowed": False,
            }
            unknown_row["row_sha256"] = sha256_json(unknown_row)
            catalog_rows.append(unknown_row)
            model_rows.append({
                "schema_version": DATASET_SCHEMA,
                "row_id": unknown_key,
                "split": split,
                "tokens": unknown_tokens,
                "token_count": len(unknown_tokens),
                "labels_in_model_row": False,
                "oracle_authority_in_model_row": False,
                "target_identity_in_model_row": False,
                "source_hash_in_model_row": False,
                "action_supervision_allowed": False,
                "safety_supervision_allowed": False,
            })
            unknown_count += 1
    summary = {
        "target_instance_count": len(target_ids),
        "row_count": len(catalog_rows),
        "model_row_count": len(model_rows),
        "family_counts": dict(sorted(family_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "positive_count": positive_count,
        "matched_negative_count": negative_count,
        "unknown_oracle_count": unknown_count,
        "get_post_balance": method_counts["GET"] == method_counts["POST"],
        "fresh_reset_count": len(catalog_rows),
        "typed_oracle_count": positive_count + negative_count,
        "source_hash_count": len({row["source_hash"] for row in catalog_rows}),
        "implementation_holdout_count": sum(row["split"] == "implementation_holdout" for row in catalog_rows),
        "vulnerability_family_count": len(FAMILIES),
        "surface_style_count": len(STYLES),
    }
    return catalog_rows, model_rows, summary
