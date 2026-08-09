from __future__ import annotations

import json
from pathlib import Path

from app.pg286_observation_tokens import build_observation_tokens, field_role_tokens


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _projection(*, status: str = "2xx", changed: bool = False) -> dict:
    return {
        "status_class": status,
        "content_type_class": "html",
        "body_length_bucket": "128",
        "transport_error": False,
        "status_changed": changed,
        "state_changed": changed,
        "location_changed": False,
        "marker_reflected": changed,
        "marker_location": "body" if changed else "none",
        "redirect_hops": 0,
        "shape": {"kind": "html", "field_count": 1, "scalar_count": 1},
    }


def test_projection_is_family_agnostic_and_marks_missing_sql_ast():
    result = build_observation_tokens(
        method="POST",
        fields=["id", "submit"],
        baseline=_projection(),
        candidate=_projection(changed=True),
        negative=_projection(),
        sql_ast=None,
    )
    assert result["evidence_status"] == "incomplete"
    assert "dom_or_sql_or_logic_or_redirect" in result["missing_modalities"]
    assert "sql_ast_available=0" in result["context_tokens"]
    assert "ir_family_agnostic=1" in result["context_tokens"]
    assert result["oracle_label_in_context"] is False
    assert "literal_probe_in_context=0" in result["context_tokens"]
    assert all("family=" not in token and "positive" not in token for token in result["context_tokens"])


def test_dom_projection_can_be_complete_without_label_in_context():
    result = build_observation_tokens(
        method="GET",
        fields=["q"],
        baseline=_projection(),
        candidate=_projection(changed=True),
        negative=_projection(),
        dom={"browser_dom_observed": True, "marker_hits": 0, "body_text_hits": 0, "element_count": 3, "script_tag_count": 1},
    )
    assert result["evidence_status"] == "complete"
    assert result["missing_modalities"] == []
    assert "dom_available=1" in result["context_tokens"]
    assert "dom_marker_hits=0" in result["context_tokens"]
    assert "sql_available=0" in result["context_tokens"]
    assert all("oracle=" not in token and "typed_effect" not in token for token in result["context_tokens"])


def test_redirect_geometry_is_a_complete_shared_modality_without_location_value():
    result = build_observation_tokens(
        method="GET",
        fields=["next"],
        baseline=_projection(),
        candidate=_projection(changed=True),
        negative=_projection(),
        redirect={"hop_count": 2, "same_origin": True, "terminal_status": "3xx", "chain_shape": "same_origin"},
    )
    assert result["evidence_status"] == "complete"
    assert "redirect_hops=2" in result["context_tokens"]
    assert "redirect_chain_shape=same_origin" in result["context_tokens"]
    assert all("redirect_location=" not in token and "http" not in token for token in result["context_tokens"])


def test_field_roles_are_coarse_not_exact_field_names():
    tokens = field_role_tokens(["id", "redirect", "submit", "message", "opaque-token"])
    assert tokens.count("field_role=numeric") == 1
    assert tokens.count("field_role=url") == 1
    assert tokens.count("field_role=control") == 1
    assert tokens.count("field_role=text") == 1
    assert tokens.count("field_role=opaque") == 1
    assert all(name not in " ".join(tokens) for name in ["redirect", "opaque-token"])


def test_persisted_catalog_is_quarantined_until_live_ast_and_replay_exist():
    catalog = _load("pg286_observation_token_catalog_v1.json")
    hard = _load("pg286_observation_token_hard_negative_v1.json")
    audit = _load("pg286_observation_token_catalog_independent_audit_v1.json")
    assert audit["status"] == "passed"
    assert catalog["counts"] == {"total": 28, "sql": 14, "xss": 12, "redirect": 2, "complete": 12, "incomplete": 16, "hard_negative": 28}
    assert catalog["training_contract"]["real_sql_ast_required"] is True
    assert catalog["training_contract"]["incomplete_training_eligible"] is False
    assert audit["training_eligible_rows"] == 0
    assert audit["sql_ast_available_rows"] == 0
    assert hard["training_eligible"] is False
    assert all(row["target"]["next_action"] == "abstain" for row in hard["records"])
    assert all("sql_available=0" in row["context_tokens"] for row in catalog["records"] if row["family"] != "sql")
    assert all("sql_available=1" in row["context_tokens"] for row in catalog["records"] if row["family"] == "sql")
