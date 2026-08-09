import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg290_balanced_dataset_is_audited_and_smaller_than_pg289():
    audit = json.loads((ROOT / "research/pg290_balanced_abstain_dataset_audit_v1.json").read_text(encoding="utf-8"))
    data = json.loads((ROOT / "research/pg290_balanced_abstain_dataset_v1.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    assert len(data["records"]) == 504
    assert len(data["records"]) < 1512
    assert all(row["target"]["next_action"] == "abstain" for row in data["records"])

