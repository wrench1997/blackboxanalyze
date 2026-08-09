from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_capacity_module():
    path = ROOT / "scripts" / "audit_pg331_model_capacity.py"
    spec = importlib.util.spec_from_file_location("pg331_model_capacity_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pg331_capacity_audit_does_not_hide_legacy_window_truncation() -> None:
    report = _load_capacity_module().audit()
    assert report["status"] == "blocked"
    assert report["information_audit_status"] == "blocked"
    assert report["required_context_window"] > 72
    legacy = next(item for item in report["variants"] if item["config"]["id"] == "pg322_legacy")
    assert legacy["truncation_risk"] is True
    assert legacy["capacity_pass"] is False
    candidate = next(item for item in report["variants"] if item["config"]["id"] == "pg331_minimum")
    assert candidate["context_window_pass"] is True
    # The frozen legacy manifest predates the explicit ``absent`` inventory;
    # its candidate window is large enough but the vocabulary gate must still
    # fail closed.
    assert candidate["capacity_pass"] is False
    assert report["inventory_missing_count"] > 0
    assert report["promotion"]["training_allowed"] is False


def test_pg331_capacity_audit_accepts_real_source_row_dataset_without_truncation(tmp_path: Path) -> None:
    module = _load_capacity_module()
    dataset_path = tmp_path / "source_rows.json"
    dataset_path.write_text(
        json.dumps({"records": [{"context_tokens": ["x"] * 3200, "target_tokens": ["y"] * 10}]}),
        encoding="utf-8",
    )
    information_path = tmp_path / "information.json"
    information_path.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    report = module.audit(dataset_path=dataset_path, information_audit_path=information_path)
    assert report["dataset_context_length"]["max"] == 3200
    assert report["required_context_window"] >= 4013
    minimum = next(item for item in report["variants"] if item["config"]["id"] == "pg331_minimum")
    assert minimum["context_window_pass"] is True
    assert report["dataset"] == str(dataset_path.resolve())
    assert report["information_audit"] == str(information_path.resolve())


def test_pg331_capacity_inventory_requires_all_builder_markers(tmp_path: Path) -> None:
    module = _load_capacity_module()
    dataset_path = tmp_path / "source_rows.json"
    dataset_path.write_text(
        json.dumps({"records": [{"context_tokens": ["x"], "target_tokens": ["y"]}]}),
        encoding="utf-8",
    )
    information_path = tmp_path / "information.json"
    information_path.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    # Start from the typed append-only manifest and remove one token from each
    # hard-contract category.  The capacity gate must notice every omission
    # instead of accepting a vocabulary that can collapse absent,
    # not_observed, unknown, axis framing, or parameter-role states.
    typed_manifest = json.loads(
        (ROOT / "research" / "pg331_pikachu_typed_web_token_vocabulary_v1.json").read_text(encoding="utf-8")
    )
    removed_tokens = {
        "document_structure_field_doctype=absent",
        "document_structure_field_doctype=not_observed",
        "axis_begin=document_structure",
        "axis_end=document_structure",
        "param_role=identifier",
        "chunk_boundary=begin",
        "[BOS]",
    }
    assert removed_tokens <= set(typed_manifest["context_tokens"])
    typed_manifest["context_tokens"] = [
        token for token in typed_manifest["context_tokens"] if token not in removed_tokens
    ]
    vocabulary_path = tmp_path / "vocabulary.json"
    vocabulary_path.write_text(json.dumps(typed_manifest), encoding="utf-8")
    report = module.audit(
        dataset_path=dataset_path,
        information_audit_path=information_path,
        vocabulary_path=vocabulary_path,
    )
    assert report["required_inventory_count"] >= 758
    assert removed_tokens <= set(report["inventory_missing"])
    assert all(item["capacity_pass"] is False for item in report["variants"])


def test_pg331_capacity_inventory_matches_builder_contract(tmp_path: Path) -> None:
    """Keep capacity's required inventory in lock-step with the builder."""

    capacity = _load_capacity_module()
    builder_path = ROOT / "scripts" / "build_pg331_web_token_vocabulary.py"
    spec = importlib.util.spec_from_file_location("pg331_vocabulary_builder_test", builder_path)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    dataset_path = tmp_path / "empty.json"
    dataset_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    built = builder.build(
        dataset_path=dataset_path,
        ontology_path=ROOT / "research" / "pg331_web_token_ontology_v1.json",
        audit_path=audit_path,
        base_vocabulary_path=tmp_path / "missing-base.json",
    )
    ontology = json.loads(
        (ROOT / "research" / "pg331_web_token_ontology_v1.json").read_text(encoding="utf-8")
    )
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    assert built["counts"]["ontology_inventory"] == 737
    assert capacity._required_inventory(ontology, rules) == set(built["context_tokens"])
