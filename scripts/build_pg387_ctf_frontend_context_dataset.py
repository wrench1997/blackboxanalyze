"""Build the PG-387 CTF-like frontend context candidate dataset.

Rows are abstract context/Rule-IR projections only.  The artifact is a
diagnostic candidate dataset until a reviewed local implementation provides
fresh typed evidence; it intentionally contains no concrete probe, URL, wire,
response body, source text or evaluator answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.pg387_ctf_frontend_projection import CTF_CASES, SCHEMA_VERSION, project_case


IMPLEMENTATIONS = ("ctf_frontend_a", "ctf_frontend_b")
SEEDS = (38701, 38702, 38703, 38704)
ROLES = ("candidate", "reference", "negative", "replay")


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_dataset() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for implementation_index, implementation in enumerate(IMPLEMENTATIONS):
        split = "train" if implementation_index == 0 else "implementation_holdout"
        for case in CTF_CASES:
            projection = project_case(case)
            for seed in SEEDS:
                for role in ROLES:
                    row_core = {
                        "record_ref_sha256": _sha({"implementation": implementation, "case_ref": case["case_ref"], "seed": seed, "role": role}),
                        "split": split,
                        "implementation_ref": implementation,
                        "seed_bucket": f"seed_{seed % 2}",
                        "role": role,
                        "context_tokens": projection["context_tokens"],
                        "target_tokens": projection["target_tokens"],
                        "javascript_surface": projection["javascript_surface"],
                        # Semantic JS overlay is model-visible only through
                        # abstract tokens; this metadata mirrors the same
                        # labels for audit/UI and never stores source text.
                        "javascript_context": projection["javascript_context"],
                        "fresh_reset": False,
                        "typed_evaluator_observed": False,
                        "source_text_stored": False,
                        "training_eligible": False,
                        "promotion": projection["promotion"],
                    }
                    row = dict(row_core)
                    row["row_sha256"] = _sha(row_core)
                    rows.append(row)

    counts = {
        "records": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "implementation_holdout": sum(row["split"] == "implementation_holdout" for row in rows),
        "cases": len(CTF_CASES),
        "implementations": len(IMPLEMENTATIONS),
        "seeds": len(SEEDS),
        "roles": len(ROLES),
        "get": sum("GET" in row["javascript_surface"].get("sink_kind", "") for row in rows),
        "typed_evaluator_observed": 0,
        "training_eligible": 0,
    }
    # Transport coverage is kept in the abstract context tokens.  It is
    # reported explicitly so audits cannot mistake a source count for a live
    # typed replay count.
    counts["GET"] = sum("transport=GET_" in " ".join(row["context_tokens"]) for row in rows)
    counts["POST"] = sum("transport=POST_" in " ".join(row["context_tokens"]) for row in rows)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "pg387_ctf_frontend_context_dataset_v1",
        "status": "abstract_ctf_candidate_only",
        "description": "CTF-like frontend sink/loader/normalization/state context projections for ASK and constrained Rule-IR decisions.",
        "rows": rows,
        "counts": counts,
        "context_firewall": {
            "raw_source": False,
            "raw_probe": False,
            "raw_response": False,
            "url_or_route_literal": False,
            "evaluator_answer": False,
        },
        "information_preservation": {
            "axes": ["javascript_surface", "js_source", "js_parser", "js_normalization_chain", "js_filter_shape", "js_guard_shape", "js_control_flow", "js_event_shape", "js_ast_shape", "js_source_to_sink", "sink_kind", "loader_policy", "state_policy", "normalization", "transport", "response_shape", "failure_shape", "oracle_shape"],
            "field_status": "abstract_observed_or_not_observed",
            "entropy_audit": "required_before_training",
        },
        "source_contract": {
            "live_rows_emitted": False,
            "fresh_role_reset": False,
            "candidate_reference_negative_replay": False,
            "typed_evidence": False,
            "operator_reviewed": False,
        },
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    artifact["dataset_sha256"] = _sha({key: value for key, value in artifact.items() if key != "dataset_sha256"})
    return artifact


def write_dataset(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_dataset()
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/pg387_ctf_frontend_context_dataset_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    output = write_dataset(args.output)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(artifact, ensure_ascii=False))
    else:
        print(json.dumps({"output": str(output), "counts": artifact["counts"], "status": artifact["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
