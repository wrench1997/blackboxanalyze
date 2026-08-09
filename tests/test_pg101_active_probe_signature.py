import json
import re
from pathlib import Path

from app.active_probe_signature import (
    ActiveProbeSignatureDecoder,
    PROBE_IDS,
    aggregate_signature,
    make_probe_observation,
    model_input_has_forbidden_field,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _signature(probe_id: str) -> dict:
    return aggregate_signature([
        make_probe_observation(
            probe_id=current,
            method="GET",
            phase="confirm",
            encoding="identity",
            control_geometry={},
            candidate_geometry={"object_count": 1} if current == probe_id else {},
            control_projection={"status_code": 200, "content_type_class": "json", "body_length_bucket": "1-255", "shape": {}},
            candidate_projection={"status_code": 200, "content_type_class": "json", "body_length_bucket": "1-255", "shape": {}},
            safe_probe=True,
        )
        for current in PROBE_IDS
    ])


def test_pg101_active_signature_passes_cross_source_unknown_gate_without_promotion():
    report = _load("pg101_active_probe_signature_report_v1.json")
    protocol = _load("pg101_active_probe_signature_protocol_v1.json")
    dataset = _load("pg101_active_probe_signature_visible_dataset_v1.json")
    trace = _load("pg101_active_probe_signature_trace_v1.json")
    assert report["status"] == "passed"
    assert report["source"]["train_excludes_eval_source"] is True
    assert report["source"]["probe_bank_size"] == 9
    assert report["source"]["probe_values_persisted"] is False
    assert report["metrics"]["all_rows"] == 618
    assert report["metrics"]["train_rows"] == 32
    assert report["metrics"]["dev"]["known_confirm_recall"] == 1.0
    pg42 = report["metrics"]["pg42"]
    assert pg42["count"] == 360
    assert pg42["known_confirm_recall"] == 1.0
    assert pg42["false_accept_count"] == 0
    assert pg42["unknown_strict_abstain"] is True
    assert pg42["unknown_misname_count"] == 0
    assert pg42["implementation_min_confirm_recall"] == 1.0
    third = report["metrics"]["pg35_third_implementation"]
    assert third["count"] == 162
    assert third["known_confirm_recall"] == 1.0
    assert third["false_accept_count"] == 0
    assert report["metrics"]["order_permutation_invariant_rate"] == 1.0
    assert report["metrics"]["pg42_signature_overlap"]["known_unknown_overlap_count"] == 0
    assert report["metrics"]["pg99_static_projection_overlap_count"] == 6
    assert all(report["capability_gate"]["checks"].values())
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert protocol["model_contract"]["family_label_in_features"] is False
    assert protocol["model_contract"]["oracle_label_in_features"] is False
    assert protocol["evaluation_contract"]["unknown_must_abstain"] is True
    assert dataset["training_eligible"] is False
    assert trace["training_eligible"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert len(dataset["rows"]) == 618
    assert len(trace["steps"]) == 618
    assert all(not model_input_has_forbidden_field(row["model_input"]) for row in dataset["rows"])
    assert all(len(row["evidence_sha256"]) == 64 for row in dataset["rows"])
    assert all(row["negative_control_matched"] for row in dataset["rows"])
    text = json.dumps({"report": report, "dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("markup_candidate", "operator_like", "template_candidate", "<script", "union select"):
        assert forbidden not in text
    for digest in report["source"]["source_hashes"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_pg101_signature_decoder_maps_known_probe_and_abstains_on_unseen_bank_slot():
    known = [{"family": "xss", "signature": _signature("p0")}, {"family": "injection", "signature": _signature("p1")}]
    decoder = ActiveProbeSignatureDecoder().fit(known, allowed_families=("xss", "injection"))
    assert decoder.predict(_signature("p0"))["candidate_family"] == "xss"
    assert decoder.predict(_signature("p1"))["candidate_family"] == "injection"
    unknown = decoder.predict(_signature("p8"))
    assert unknown["decision"] == "abstain"
    assert unknown["abstain"] is True


def test_pg101_signature_input_contract_rejects_family_oracle_and_raw_fields():
    assert model_input_has_forbidden_field({"delta_pattern": [True], "probe_id": "p0"}) is False
    assert model_input_has_forbidden_field({"family": "xss"}) is True
    assert model_input_has_forbidden_field({"oracle": {"positive": True}}) is True
    assert model_input_has_forbidden_field({"body_sha256": "a" * 64}) is True
