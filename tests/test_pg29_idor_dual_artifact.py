import json
import re
from pathlib import Path

from app.cross_lab_safe_catalog import sha256_json
from app.hardcore_experiment_gate import evaluate_hardcore_catalog


ROOT = Path(__file__).resolve().parents[1]


def _catalog() -> dict:
    return json.loads((ROOT / "research" / "pg_pk_29_idor_get_post_dual_catalog_v1.json").read_text(encoding="utf-8"))


def test_pg29_dual_artifact_is_evaluation_only_and_hash_bound():
    catalog = _catalog()
    declared = catalog["catalog_sha256"]
    body = dict(catalog)
    body.pop("catalog_sha256")
    assert declared == sha256_json(body)
    assert catalog["training_eligible"] is False
    assert catalog["training_artifact_generated"] is False
    assert {row["payload_manifest"]["method"] for row in catalog["samples"]} == {"GET", "POST"}
    assert len({row["target_instance_id"] for row in catalog["samples"]}) == 3
    assert len({row["sampling_seed"] for row in catalog["samples"]}) == 3
    assert not any(re.search(r"PG29(?:GET|POST)_[0-9A-F]{8,}", json.dumps(row)) for row in catalog["samples"])
    assert evaluate_hardcore_catalog(catalog, family="logic_access")["status"] == "preflight_only"
