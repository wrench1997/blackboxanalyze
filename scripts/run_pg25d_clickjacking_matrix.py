"""Run a fresh-reset Clickjacking level/seed matrix on the local lab."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import ReadOnlySafeCatalogCollector, build_catalog, sha256_json  # noqa: E402
from app.pg25d_clickjacking_oracle import build_clickjacking_oracle  # noqa: E402
from collect_pg25d_clickjacking_catalog import (  # noqa: E402
    BASE_URI,
    ORACLE_CONTRACT_SHA256,
    _head_projection,
    _manifest,
    _reset,
    _rule_ir,
    _source,
)


REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
OUT_PATH = ROOT / "research" / "pg_pk_25d_clickjacking_matrix_v1.json"
SEEDS = (2501, 2502)
LEVELS = tuple(range(1, 8))
EXPECTED_VULNERABLE = {1: True, 2: True, 3: False, 4: False, 5: False, 6: True, 7: False}


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reset_record(reset_id: str, instance_id: str, baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "reset_id": reset_id,
        "kind": "stop_remove_recreate",
        "target_instance_id": instance_id,
        "state_epoch": instance_id.replace("instance-", "epoch-"),
        "reset_adapter_sha256": "a2df3b48dd80d68cb0adbdefe64dc2b26137282b204ed8a55fe630c3ff884724",
        "baseline_projection_sha256": sha256_json(baseline),
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
    }


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector = ReadOnlySafeCatalogCollector(_source(registry), registry=registry)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        control_reset, control_instance = _reset()
        control_baseline, _ = _head_projection(BASE_URI)
        control_response, control_regex = _head_projection(BASE_URI + "ClickjackingVulnerability/LEVEL_3")
        control = collector.collect(
            sample_id=f"pg25d-cj-s{seed}-l3",
            sample_role="negative_control",
            sampling_seed=seed,
            reset=_reset_record(control_reset, control_instance, control_baseline),
            payload_manifest=_manifest(f"s{seed}-l3"),
            response_projection=control_response,
            oracle_projection=build_clickjacking_oracle(
                oracle_contract_sha256=ORACLE_CONTRACT_SHA256,
                frame_policy=control_response["frame_policy"],
                expected_vulnerable=False,
                regex_evidence=control_regex,
            ),
            rule_ir=_rule_ir(),
        )
        rows.append(control)
        for level in LEVELS:
            if level == 3:
                continue
            reset_id, instance_id = _reset()
            baseline, _ = _head_projection(BASE_URI)
            response, regex = _head_projection(BASE_URI + f"ClickjackingVulnerability/LEVEL_{level}")
            expected_positive = bool(EXPECTED_VULNERABLE[level])
            row = collector.collect(
                sample_id=f"pg25d-cj-s{seed}-l{level}",
                sample_role="candidate" if expected_positive else "negative_control",
                sampling_seed=seed,
                reset=_reset_record(reset_id, instance_id, baseline),
                payload_manifest=_manifest(f"s{seed}-l{level}"),
                response_projection=response,
                oracle_projection=build_clickjacking_oracle(
                    oracle_contract_sha256=ORACLE_CONTRACT_SHA256,
                    frame_policy=response["frame_policy"],
                    expected_vulnerable=expected_positive,
                    regex_evidence=regex,
                ),
                rule_ir=_rule_ir(),
                negative_control=(
                    {
                        "control_sample_id": control["sample_id"],
                        "control_evidence_hash": control["evidence"]["evidence_hash"],
                        "intervention": "protected-frame-policy-level",
                        "verdict": "confirmed_negative",
                        "same_source": True,
                        "same_surface": True,
                    }
                    if expected_positive
                    else None
                ),
            )
            rows.append(row)
    catalog = build_catalog("pg25d-vulnerableapp-clickjacking-matrix-v1", collector.source, rows)
    catalog["collection"] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "level_count": len(LEVELS),
        "seed_count": len(SEEDS),
        "episode_count": len(rows),
        "fresh_reset_per_episode": True,
        "probe_policy": "HEAD-only; bounded frame-policy regex evidence; no active payload",
        "evaluator_ground_truth_sha256": _digest(EXPECTED_VULNERABLE),
        "training_eligible": False,
    }
    OUT_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {
        "confirmed_positive": sum(row["decision"]["evidence_status"] == "confirmed_positive" for row in rows),
        "confirmed_negative": sum(row["decision"]["evidence_status"] == "confirmed_negative" for row in rows),
    }
    print(json.dumps({"output": str(OUT_PATH), "episodes": len(rows), **counts, "training_eligible": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
