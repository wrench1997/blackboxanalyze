"""Collect a fresh v3 implementation while preserving v2 as an old canary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.heterogeneous_surface_fixture_v3 import (  # noqa: E402
    V3_PORTS,
    HeterogeneousSurfaceV3Collector,
    default_heterogeneous_surface_v3_specs,
    heterogeneous_surface_v3_source_sha256,
    make_heterogeneous_surface_v3_fixture_server,
)
from build_pg273_composition_dataset import _abstract  # noqa: E402

OLD = ROOT / "research" / "pg273_composition_dataset_v1.json"
OUTPUT = ROOT / "research" / "pg276_third_implementation_dataset_v1.json"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def collect_v3() -> tuple[list[dict[str, Any]], str]:
    source_hash = heterogeneous_surface_v3_source_sha256()
    rows: list[dict[str, Any]] = []
    for index, (port, variant) in enumerate(zip(V3_PORTS, ("alpha", "beta", "gamma"))):
        server = make_heterogeneous_surface_v3_fixture_server(port=port, variant=variant)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{port}"
            specs = default_heterogeneous_surface_v3_specs(dataset_id=f"pg276-v3-{variant}", target=base_url, marker=f"pg276-v3-{index}")
            rows.extend(asyncio.run(HeterogeneousSurfaceV3Collector(base_url=base_url, target_instance_id=f"pg276-v3-{variant}", source_hash=source_hash).collect_many(specs)))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)
    return rows, source_hash


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    v3_records, v3_hash = collect_v3()
    old_rows = list(old["records"])
    train = [dict(row, split="implementation_v1_train") for row in old_rows if row["implementation"] == "heterogeneous_surface_v1"]
    canary = [dict(row, split="implementation_v2_canary") for row in old_rows if row["implementation"] == "heterogeneous_surface_v2"]
    holdout = _abstract(v3_records, split="implementation_v3_holdout", source_hash=v3_hash, implementation="heterogeneous_surface_v3")
    payload: dict[str, Any] = {
        "schema_version": "pg276-third-implementation-dataset-v1",
        "source": {"train_implementation": "heterogeneous_surface_v1", "old_canary_implementation": "heterogeneous_surface_v2", "holdout_implementation": "heterogeneous_surface_v3", "holdout_source_sha256": v3_hash, "loopback_only": True, "external_network": False, "fresh_target": True},
        "split_contract": {"implementation_disjoint": True, "train": "v1 only", "old_canary": "v2 only", "holdout": "v3 only", "oracle_in_context": False, "raw_payload_in_context": False, "promotion_blocked": True},
        "records": train + canary + holdout,
        "counts": {"train": len(train), "old_canary": len(canary), "holdout": len(holdout), "train_positive": sum(bool(x["labels"]["expected_positive"]) for x in train), "old_canary_positive": sum(bool(x["labels"]["expected_positive"]) for x in canary), "holdout_positive": sum(bool(x["labels"]["expected_positive"]) for x in holdout)},
        "training_contract": {"generic_observation_tokens_only": True, "teacher_scores_are_labels": True, "old_canary_not_used_for_update": True, "promotion_blocked": True, "memory_promotion_blocked": True},
    }
    payload["dataset_sha256"] = digest(payload)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "counts": payload["counts"], "dataset": str(OUTPUT.relative_to(ROOT)), "dataset_sha256": payload["dataset_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
