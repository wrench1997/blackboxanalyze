from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_pg386_model_generated_fixture_payload import replay


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research/pg386_model_generated_fixture_payload_v1.json"
DATASET = ROOT / "research/pg386_fixture_payload_generation_dataset_v1.json"
SELECTOR = ROOT / "artifacts/pg385-variant-selector/pg385_variant_seed_38503.pt"
DECODER = ROOT / "artifacts/pg386-fixture-payload-decoder/pg386_payload_head_seed_38603.pt"


def test_pg386_dataset_and_report_keep_raw_values_out_of_persistent_artifacts() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    artifact = json.loads(REPORT.read_text(encoding="utf-8"))
    assert dataset["output_contract"]["raw_string_in_context"] is False
    assert dataset["output_contract"]["raw_string_persisted"] is False
    assert dataset["training_eligible"] == 0
    assert artifact["replay"]["execution"]["raw_string_stored"] is False
    assert artifact["replay"]["model_boundary"]["model_emits_fixture_bound_raw_string"] is True
    assert artifact["replay"]["model_boundary"]["model_emits_arbitrary_raw_string"] is False
    assert artifact["replay"]["counts"]["negative_violation"] == 0
    assert all("PG386_CAND_0002%25253A" not in json.dumps(value) for value in (dataset, artifact))


@pytest.mark.skipif(not SELECTOR.exists() or not DECODER.exists(), reason="PG-385 selector or PG-386 decoder checkpoint is not present")
def test_token_model_generates_fixture_bound_string_and_typed_replays() -> None:
    report, wires = replay(selector_checkpoint=SELECTOR, decoder_checkpoint=DECODER, show_wire=True)
    assert report["status"] == "completed_model_generated_fixture_payload_loopback_only"
    assert report["counts"] == {
        "rows": 4,
        "implementations": 2,
        "methods_get": 2,
        "methods_post": 2,
        "model_generated_candidate": 4,
        "model_generated_reference": 4,
        "model_generated_replay": 4,
        "candidate_typed": 4,
        "reference_typed": 4,
        "replay_typed": 4,
        "negative_violation": 0,
    }
    assert len(wires) == 16  # 3 model-generated roles + evaluator-bound negative per row
    assert all("PG386_" in wire for wire in wires)
    assert all("http://127.0.0.1:" in wire for wire in wires)
    assert all(value is False for value in report["promotion"].values())

