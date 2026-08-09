import json
from pathlib import Path

from scripts.extend_pg349_payload_vocabulary import extend


ROOT = Path(__file__).parents[1]


def test_extension_is_append_only_and_namespaced(tmp_path):
    base_path = ROOT / "research" / "pg349_dynamic_typed_vocabulary_v7.json"
    payload_path = ROOT / "research" / "pg349_payload_probe_vocabulary_v1.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    result = extend(base_path=base_path, payload_path=payload_path)
    assert set(base["context_tokens"]).issubset(set(result["context_tokens"]))
    assert result["counts"]["payload_probe_reserved"] > 50
    assert all(token.startswith("probe_shape_") or token == "probe_shape_vocab_version=pg349_v1" for token in set(result["context_tokens"]) - set(base["context_tokens"]))
    assert result["vocabulary_policy"]["payload_probe_shapes_abstract_only"] is True
    assert result["promotion"] == {"training": False, "memory": False, "payload": False, "vulnerability": False}


def test_extension_never_adds_literal_payload_markers():
    result = extend()
    added = set(result["context_tokens"]) - set(json.loads((ROOT / "research" / "pg349_dynamic_typed_vocabulary_v7.json").read_text(encoding="utf-8"))["context_tokens"])
    text = " ".join(sorted(added)).lower()
    assert "<script" not in text
    assert "javascript:" not in text
    assert "document.cookie" not in text
    assert "http://" not in text
    assert "response_body" not in text

