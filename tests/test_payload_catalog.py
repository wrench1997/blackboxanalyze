from pathlib import Path

import pytest

from app.payload_catalog import flatten_catalog, load_catalog, policy_candidate, validate_policy_candidate


ROOT = Path(__file__).resolve().parents[1]


def test_authorized_catalog_preserves_provenance_and_hides_family_from_policy():
    catalog = load_catalog(ROOT / "research" / "payload_source_catalog_v1.json")
    rows = flatten_catalog(catalog)
    assert len(catalog["sources"]) == 10
    assert len(rows) == 24
    assert all(row["probe_artifact"]["original"] == row["payload"]["probe"] for row in rows)
    assert all("family" not in policy_candidate(row) for row in rows)
    assert all("family" not in policy_candidate(row)["source_attestation"]["source_id"] for row in rows)


def test_counterfactual_metadata_is_structural_and_not_policy_visible():
    rows = flatten_catalog(load_catalog(ROOT / "research" / "pikachu_counterfactual_catalog_v1.json"))
    controls = [row for row in rows if row.get("counterfactual")]
    assert len(controls) == 12
    assert all(row["counterfactual"]["kind"] == "negative_control" for row in controls)
    assert all("counterfactual" not in policy_candidate(row) for row in controls)


def test_tampered_source_attestation_is_rejected():
    row = flatten_catalog(load_catalog(ROOT / "research" / "payload_source_catalog_v1.json"))[0]
    candidate = policy_candidate(row)
    candidate["source_attestation"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_policy_candidate(candidate)


def test_pikachu_pair_catalog_keeps_training_pair_metadata_out_of_policy(tmp_path: Path):
    catalog = load_catalog(ROOT / "research" / "pikachu_paired_catalog_v1.json")
    rows = flatten_catalog(catalog)
    assert len(rows) == 24
    assert {row["pair"]["variant"] for row in rows} == {
        "plain", "url_percent", "html_entity", "double_html_entity",
    }
    candidate = policy_candidate(rows[0])
    assert "pair" not in candidate
    assert candidate["payload"]["target"] == "http://127.0.0.1:8766"
