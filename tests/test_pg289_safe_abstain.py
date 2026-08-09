import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg289_audit_passes_and_has_expected_quota():
    audit = _load("pg289_safe_abstain_dataset_audit_v1.json")
    assert audit["status"] == "passed"
    assert audit["checks"]["row_quota"] is True
    assert audit["checks"]["all_signature_rows_present"] is True


def test_pg289_rows_are_training_only_abstain_decoys():
    data = _load("pg289_safe_abstain_dataset_v1.json")
    rows = data["records"]
    assert len(rows) == 1512
    assert all(row["split"] == "train" for row in rows)
    assert all(row["training_decoy"] is True for row in rows)
    assert all(row["target"]["next_action"] == "abstain" and row["target"]["safe_to_send"] is False for row in rows)


def test_pg289_context_has_no_family_or_raw_probe_fields():
    data = _load("pg289_safe_abstain_dataset_v1.json")
    forbidden = ("family=", "source=", "probe=", "oracle_label=")
    for row in data["records"]:
        assert row["family"] is None
        assert all(not any(str(token).startswith(prefix) for prefix in forbidden) for token in row["context_tokens"])
        assert row["raw_payload_strings_stored"] is False
        assert row["raw_response_bodies_stored"] is False

