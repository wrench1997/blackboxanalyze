"""Build a PG-389 abstract JS decode/filter-chain candidate dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg389_js_chain_projection import CHAIN_CASES, CHAIN_VARIANTS, SCHEMA_VERSION, project_chain_case  # noqa: E402


IMPLEMENTATIONS = ("js_chain_runtime_a", "js_chain_runtime_b")
SEEDS = (38901, 38902, 38903)
ROLES = ("candidate", "reference", "negative", "replay")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_dataset() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for implementation_index, implementation in enumerate(IMPLEMENTATIONS):
        split = "train" if implementation_index == 0 else "implementation_holdout"
        variant = CHAIN_VARIANTS[implementation_index]
        for case in CHAIN_CASES:
            projection = project_chain_case(case, variant=variant)
            for seed in SEEDS:
                for role in ROLES:
                    core = {
                        "record_ref_sha256": _sha({"implementation": implementation, "case_ref": case["case_ref"], "seed": seed, "role": role}),
                        "split": split,
                        "implementation_ref": implementation,
                        "implementation_surface_variant": variant,
                        "seed_bucket": f"seed_{seed % 2}",
                        "role": role,
                        "context_tokens": projection["context_tokens"],
                        "target_tokens": projection["target_tokens"],
                        "decode_filter_context": projection["decode_filter_context"],
                        "javascript_surface": projection["javascript_surface"],
                        "source_text_stored": False,
                        "raw_value_stored": False,
                        "typed_evaluator_observed": False,
                        "fresh_reset": False,
                        "training_eligible": False,
                        "promotion": projection["promotion"],
                    }
                    row = dict(core)
                    row["row_sha256"] = _sha(core)
                    rows.append(row)
    counts = {
        "records": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "implementation_holdout": sum(row["split"] == "implementation_holdout" for row in rows),
        "cases": len(CHAIN_CASES),
        "implementations": len(IMPLEMENTATIONS),
        "seeds": len(SEEDS),
        "roles": len(ROLES),
        "GET": sum(row["javascript_surface"]["source_kind"] == "location_search" or row["javascript_surface"]["source_kind"] == "location_hash" for row in rows),
        "POST": sum(row["javascript_surface"]["source_kind"] == "form_input" for row in rows),
        "typed_evaluator_observed": 0,
        "training_eligible": 0,
    }
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "pg389_js_decode_filter_chain_dataset_v1",
        "status": "abstract_js_chain_candidate_only",
        "description": "Abstract ordered decoder/filter/guard/sink observations for JS-context reasoning; no concrete probe or evaluator answer.",
        "rows": rows,
        "counts": counts,
        "information_preservation": {
            "axes": ["source_kind", "transport", "decoder_chain_order", "filter_stage", "guard_precedence", "sink_context", "state_policy", "failure_signature", "oracle_shape", "observation_sequence", "implementation_surface_variant"],
            "entropy_audit": "required_before_training",
            "holdout": "implementation_disjoint",
        },
        "context_firewall": {
            "raw_source": False,
            "raw_value": False,
            "raw_probe": False,
            "raw_wire": False,
            "raw_response": False,
            "url_or_route_literal": False,
            "evaluator_answer": False,
        },
        "source_contract": {
            "live_rows_emitted": False,
            "fresh_role_reset": False,
            "candidate_reference_negative_replay": False,
            "typed_evidence": False,
            "operator_reviewed": False,
        },
        "training_eligible": 0,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    artifact["dataset_sha256"] = _sha({key: value for key, value in artifact.items() if key != "dataset_sha256"})
    return artifact


def write_dataset(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_dataset(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/pg389_js_decode_filter_chain_dataset_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    output = write_dataset(args.output)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    summary = {"output": str(output), "status": artifact["status"], "counts": artifact["counts"]}
    print(json.dumps(artifact if args.json else summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
