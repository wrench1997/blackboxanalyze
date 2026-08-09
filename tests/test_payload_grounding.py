from pathlib import Path

from app.dom_oracle import run_dom_oracle
from app.payload_catalog import flatten_catalog, load_catalog, policy_candidate
from app.payload_grounding import SourceGroundedMemory


ROOT = Path(__file__).resolve().parents[1]


def test_source_grounded_memory_transfers_dom_feature_and_abstains_unknown_surface():
    rows = flatten_catalog(load_catalog(ROOT / "research" / "payload_source_catalog_v1.json"))
    train = next(row for row in rows if row["semantic"]["family"] == "xss" and row["payload"]["probe_kind"] == "encoded_dom_markup" and row["provenance"]["source_id"].endswith("-a"))
    target = next(row for row in rows if row["semantic"]["family"] == "xss" and row["payload"]["probe_kind"] == "encoded_dom_markup" and row["provenance"]["source_id"].endswith("-b"))
    unknown = next(row for row in rows if row["semantic"]["family"] == "access_control" and row["provenance"]["source_id"].endswith("-b"))
    memory = SourceGroundedMemory(seed=5)
    evidence = run_dom_oracle(
        train["payload"]["probe"],
        transforms=["html_entity_decode", "html_entity_decode"],
        marker=train["payload"]["marker"],
    ).to_dict()
    memory.observe(policy_candidate(train), status="observable_success", evidence=evidence)
    chosen = memory.select([policy_candidate(target), policy_candidate(unknown)])
    assert chosen is not None
    assert chosen["candidate_id"] == target["sample_id"]
    assert memory.select([policy_candidate(unknown)]) is None
