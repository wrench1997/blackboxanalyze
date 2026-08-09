import json
import re
from pathlib import Path

from app.web_feature_funnel import FORBIDDEN_MODEL_FIELDS, audit_feature_funnel, build_feature_dataset, review_feature_funnel


DATASET = Path("research/pg53_web_feature_funnel_dataset_v1.json")
REPORT = Path("research/pg53_web_feature_funnel_report_v1.json")
PROTOCOL = Path("research/pg53_web_feature_funnel_protocol_v1.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_feature_funnel_reduces_candidates_and_keeps_safe_observations():
    report = _load(REPORT)
    counts = report["audit"]["stage_counts"]
    assert counts["candidate"] == 54
    assert counts["observable_safe"] == 37
    assert counts["quality"] < counts["candidate"]
    assert counts["redundancy_pruned"] <= counts["label_utility_audit"]
    accepted = report["audit"]["accepted_features"]
    assert accepted
    assert all(name not in FORBIDDEN_MODEL_FIELDS for name in accepted)
    assert report["audit"]["training_eligible"] is False
    assert report["audit"]["long_term_memory_write"] is False
    protocol = _load(PROTOCOL)
    assert protocol["funnel"][2]["stage"] == "quality"
    assert "0.60" in protocol["funnel"][3]["rule"]
    assert "0.50" in protocol["funnel"][3]["rule"]


def test_codex_review_is_explicit_and_only_allows_downstream_ood():
    report = _load(REPORT)
    review = report["review"]
    assert review["reviewer_id"] == "codex-primary-feature-funnel-review-v1"
    assert review["passed"] is True
    assert review["decision"] == "approved_for_downstream_ood_experiment"
    assert all(review["checks"].values())
    assert re.fullmatch(r"[0-9a-f]{64}", review["review_evidence_sha256"])
    assert review["training_allowed"] is False
    assert review["memory_promotion_allowed"] is False


def test_feature_dataset_separates_model_features_from_audit_metadata():
    dataset = _load(DATASET)
    assert dataset["training_eligible"] is False
    assert dataset["evaluation_only"] is True
    assert dataset["model_feature_policy"]["oracle_is_label_not_feature"] is True
    assert dataset["model_feature_policy"]["surface_observation_model_eligible"] is False
    assert set(dataset["accepted_features"]).issubset(set(dataset["model_feature_names"]))
    assert dataset["review_decision"] == "approved_for_downstream_ood_experiment"
    for row in dataset["rows"][:20]:
        assert set(row["model_features"]).issubset(set(dataset["model_feature_names"]))
        assert not any(name.startswith("surface_") for name in row["model_features"])
        assert "family" not in row["model_features"]
        assert "source_id" not in row["model_features"]
        assert row["raw_payload_stored"] is False
        assert row["raw_response_body_stored"] is False
    text = json.dumps(dataset, ensure_ascii=False).casefold()
    for forbidden in ("<svg", "onload", "union select", "password"):
        assert forbidden not in text


def test_feature_funnel_reviewer_fails_closed_without_oracle_isolation():
    dataset = {"rows": [], "model_feature_names": [], "model_feature_policy": {}}
    audit = audit_feature_funnel(dataset)
    review = review_feature_funnel(audit)
    assert review["passed"] is False
    assert review["decision"] == "rejected_pending_funnel_repairs"
