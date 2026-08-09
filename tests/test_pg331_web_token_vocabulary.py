from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = ROOT / "scripts" / "build_pg331_web_token_vocabulary.py"
    spec = importlib.util.spec_from_file_location("pg331_vocab_builder_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_source_rows_extend_base_vocabulary_without_dropping_tokens(tmp_path: Path) -> None:
    module = _load_builder()
    dataset = tmp_path / "rows.json"
    dataset.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "context_tokens": ["document_presence=observed", "new_shape=alpha"],
                        "target_tokens": ["[TARGET_BOS]", "question=ask_typed", "new_target=variant"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps({"context_tokens": ["legacy_shape=kept"], "target_tokens": ["legacy_target=kept"]}),
        encoding="utf-8",
    )
    result = module.build(
        dataset_path=dataset,
        ontology_path=ROOT / "research" / "pg331_web_token_ontology_v1.json",
        audit_path=audit,
        base_vocabulary_path=base,
    )
    assert "legacy_shape=kept" in result["context_tokens"]
    assert "new_shape=alpha" in result["context_tokens"]
    assert "legacy_target=kept" in result["target_tokens"]
    assert "new_target=variant" in result["target_tokens"]
    assert result["status"] == "diagnostic_only_audit_blocked"
    assert result["training_eligibility"]["allowed"] is False
    assert result["vocabulary_policy"]["base_vocabulary_append_only"] is True


def test_default_output_rebuild_keeps_existing_tokens_append_only(tmp_path: Path) -> None:
    module = _load_builder()
    output = tmp_path / "manifest.json"
    # Simulate a previous build at the default output location.  A rebuild
    # must read that file before composing the new manifest; otherwise the
    # default invocation silently drops the old vocabulary.
    module.OUTPUT = output
    output.write_text(
        json.dumps(
            {
                "context_tokens": ["legacy_context=kept"],
                "target_tokens": ["legacy_target=kept"],
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "rows.json"
    dataset.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "context_tokens": ["new_context=observed"],
                        "target_tokens": ["new_target=observed"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    result = module.build(
        dataset_path=dataset,
        ontology_path=ROOT / "research" / "pg331_web_token_ontology_v1.json",
        audit_path=audit,
    )
    assert "legacy_context=kept" in result["context_tokens"]
    assert "legacy_target=kept" in result["target_tokens"]
    assert "new_context=observed" in result["context_tokens"]
    assert "new_target=observed" in result["target_tokens"]
    assert result["vocabulary_policy"]["base_vocabulary_append_only"] is True
