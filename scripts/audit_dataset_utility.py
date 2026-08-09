"""Audit research artifacts for useful GET/POST replay context."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dataset_utility_audit import audit_dataset


INPUTS = {
    "pg31": ROOT / "research" / "pg_pk_31_ood_rule_ir_evaluation_manifest_v1.json",
    "pg28": ROOT / "research" / "pg_pk_28_get_post_dual_catalog_v1.json",
    "pg29": ROOT / "research" / "pg_pk_29_idor_get_post_dual_catalog_v1.json",
    "pg33": ROOT / "research" / "pg_pk_33_get_post_typed_replay_catalog_v1.json",
    "pg34_independent": ROOT / "research" / "pg34_independent_fixture_catalog_v1.json",
    "pg35_independent": ROOT / "research" / "pg35_independent_fixture_catalog_v1.json",
    "pg36_independent_maze": ROOT / "research" / "pg36_independent_maze_catalog_v1.json",
    "pg37_counterfactual": ROOT / "research" / "pg37_counterfactual_catalog_v1.json",
    "pg40_semantic_router": ROOT / "research" / "pg40_semantic_router_catalog_v1.json",
    "pg42_independent_semantic": ROOT / "research" / "pg42_independent_semantic_catalog_v1.json",
    "pg48_compositional_preprobe": ROOT / "research" / "pg48_compositional_preprobe_catalog_v1.json",
    "pg50_stability_matrix": ROOT / "research" / "pg50_stability_matrix_catalog_v1.json",
    "pg51_pikachu_docker_dual_channel": ROOT / "research" / "pg51_pikachu_docker_dual_channel_catalog_v1.json",
}
OUTPUT = ROOT / "research" / "pg_pk_31_dataset_utility_audit_v1.json"


def main() -> int:
    reports = {}
    for name, path in INPUTS.items():
        if not path.exists():
            reports[name] = {"status": "missing", "path": str(path)}
            continue
        reports[name] = audit_dataset(json.loads(path.read_text(encoding="utf-8")), dataset_id=name)
    output = {
        "schema_version": "pg-pk-31-dataset-utility-audit-v1",
        "purpose": "reject schema-only or negative-only artifacts as model capability data",
        "training_started": False,
        "memory_promotion_attempted": False,
        "datasets": reports,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
