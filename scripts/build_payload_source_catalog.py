"""Build the local, provenance-attested PG-02 payload catalog."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.payload_catalog import probe_digest, write_catalog
from app.payload_learner import generate_payload_candidates


SOURCE_DATE = "2026-08-02"

FAMILY_CONFIG = {
    "xss": {
        "path": "/playground",
        "origin": "app/dom_oracle.py",
        "surface": "dom_sink",
        "expected_oracle": "controlled_detached_dom_v1",
        "expected_signal": "browser_sink_observed+dom_change",
    },
    "injection": {
        "path": "/api/search",
        "origin": "app/sql_ast_oracle.py",
        "surface": "sql_ast_boundary",
        "expected_oracle": "synthetic_sql_ast_differential_v1",
        "expected_signal": "controlled_differential+interpreter_boundary",
    },
    "access_control": {
        "path": "/playground",
        "origin": "app/maze_solver.py",
        "surface": "protected_resource",
        "expected_oracle": "synthetic_rule_surface_v1",
        "expected_signal": "protected_resource_transition",
    },
    "url_redirect": {
        "path": "/playground",
        "origin": "app/maze_solver.py",
        "surface": "redirect_origin",
        "expected_oracle": "synthetic_rule_surface_v1",
        "expected_signal": "location_origin_changed",
    },
    "logic": {
        "path": "/playground",
        "origin": "app/maze_solver.py",
        "surface": "state_invariant",
        "expected_oracle": "synthetic_rule_surface_v1",
        "expected_signal": "invariant_violation+state_replay",
    },
}


def _encoding(probe_kind: str) -> str:
    return {
        "inert_dom_markup": "none_inert_markup",
        "encoded_dom_markup": "html_entity_encode_depth_2",
        "sql_channel_class": "abstract_sql_fragment_class",
        "sql_fragment_class": "abstract_sql_fragment_class",
        "http_canary": "identifier_canary",
    }.get(probe_kind, "validated_probe")


def build_catalog() -> dict:
    sources: list[dict] = []
    for family_index, (family, config) in enumerate(FAMILY_CONFIG.items(), start=1):
        for source_suffix in ("a", "b"):
            # Opaque source IDs prevent the policy from recovering a family
            # label from provenance.  The evaluator retains the family in the
            # semantic section, which never enters the policy candidate.
            source_id = f"pg02-source-{family_index:02d}-{source_suffix}"
            # Keep markers opaque; embedding ``xss``/``sql`` here would leak a
            # family label through the payload itself.
            marker = f"pg02-probe-{family_index:02d}-{source_suffix}"
            candidates = generate_payload_candidates(family, path=config["path"], marker=marker)
            samples: list[dict] = []
            for index, candidate in enumerate(candidates, start=1):
                payload = candidate["payload"]
                sample_id = f"{source_id}-{index}-{candidate['candidate_id'][:12]}"
                samples.append({
                    "sample_id": sample_id,
                    "payload": payload,
                    "probe_artifact": {
                        "original": payload["probe"],
                        "encoding": _encoding(payload["probe_kind"]),
                        "probe_sha256": probe_digest(payload["probe"]),
                    },
                    "semantic": {
                        "family": family,
                        "surface": config["surface"],
                        "expected_oracle": config["expected_oracle"],
                        "expected_signal": config["expected_signal"],
                    },
                    "evaluator_state_visible": False,
                })
            sources.append({
                "provenance": {
                    "source_id": source_id,
                    "source_type": "in_repo_synthetic",
                    "origin": "research/payload_source_catalog_v1.json",
                    "license": "in_repo_synthetic",
                    "authorization": "workspace_local_only",
                    "scope": ["http://127.0.0.1:3100"],
                    "captured_at": SOURCE_DATE,
                    "authorized_for": ["training", "local_replay", "holdout_evaluation"],
                    "external_network": False,
                    "evaluator_state_visible": False,
                },
                "samples": samples,
            })
    return {
        "schema_version": "sift-authorized-payload-catalog-v1",
        "catalog_id": "pg02-local-authorized-probes-v1",
        "sources": sources,
    }


def main() -> None:
    path = ROOT / "research" / "payload_source_catalog_v1.json"
    catalog = write_catalog(path, build_catalog())
    sample_count = sum(len(source["samples"]) for source in catalog["sources"])
    print({
        "catalog": str(path.relative_to(ROOT)),
        "source_count": len(catalog["sources"]),
        "sample_count": sample_count,
        "catalog_sha256": catalog["catalog_sha256"],
    })


if __name__ == "__main__":
    main()
