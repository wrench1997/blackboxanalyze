"""PG-240: replay the upstream Pikachu source on the repaired local runtime.

The application tree comes from the pinned upstream commit while the PHP/
MySQL runtime is the already-attested local image used by PG-214.  This is a
cross-application-source evaluation lane, not a claim that two independent
backend implementations were compared.  Each route still gets a fresh,
no-volume container and the PG-218 result/negative/typed-oracle protocol.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG218 = _load("run_pg218_pikachu_result_fixture.py")

RESEARCH = ROOT / "research"
IMAGE = "sift/pikachu-pg240-source-native:5e1e8d9d"
SOURCE_COMMIT = "5e1e8d9d14a3ba61d62f28cf35531c4df4dd24fc"
RUNTIME_IMAGE = "sift/pikachu-fixed@sha256:cca4288b6b701725e7a771f47ce7fcafd6cea9bd7622fa34ef2ed0b440f472c6"
SEEDS = (24001, 24002)
REPORT = RESEARCH / "pg240_pikachu_source_replay_report_v1.json"
DATASET = RESEARCH / "pg240_pikachu_source_replay_dataset_v1.json"
PROTOCOL = RESEARCH / "pg240_pikachu_source_replay_protocol_v1.json"
TRACE = RESEARCH / "pg240_pikachu_source_replay_trace_v1.json"
MARKDOWN = RESEARCH / "pg240_pikachu_source_replay_report_v1.md"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _image_id() -> str:
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip().removeprefix("sha256:")


def _rename(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rename(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename(item) for item in value]
    if isinstance(value, str):
        return value.replace("pg-pk-218-pikachu-result-fixture-v1", "pg-pk-240-pikachu-source-replay-v1").replace("pg218-pikachu-result-fixture", "pg240-pikachu-source-replay")
    return value


def main() -> int:
    # PG-218 supplies the bounded AI/reference/negative/result-fixture loop.
    # Only the image, ports, seeds, and output locations are replaced.
    PG218.PG214.IMAGE = IMAGE
    PG218.PG214.BASE_PORT = 9800
    PG218.SEEDS = SEEDS
    PG218.REPORT_PATH = REPORT
    PG218.PROTOCOL_PATH = PROTOCOL
    PG218.TRACE_PATH = TRACE
    PG218.MARKDOWN_PATH = MARKDOWN
    PG218.main()

    report = _rename(json.loads(REPORT.read_text(encoding="utf-8")))
    protocol = _rename(json.loads(PROTOCOL.read_text(encoding="utf-8")))
    trace = _rename(json.loads(TRACE.read_text(encoding="utf-8")))
    image_id = _image_id()

    report.update(
        {
            "schema_version": "pg240-pikachu-source-replay-report-v1",
            "status": "completed_cross_application_source_replay",
            "seeds": list(SEEDS),
            "implementation_comparison": "application_source_holdout_only_shared_repaired_runtime",
            "source_repository": {
                "url": "https://github.com/zhuifengshaonianhanlu/pikachu",
                "commit": SOURCE_COMMIT,
                "source_tree_sha256_attested_in_route_rows": True,
            },
            "runtime": {"image": IMAGE, "image_id_sha256": image_id, "base_runtime_image": RUNTIME_IMAGE, "shared_runtime_with_pg214": True},
            "training_role": "cross_application_source_evaluation_only",
            "promotion": {**dict(report.get("promotion") or {}), "training_eligible": False, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
            "honesty": {"source_holdout_not_full_backend_independence": True, "shared_runtime_isolation": True, "typed_oracle_required_for_positive": True, "environment_failure_is_not_model_label": True},
        }
    )
    protocol.update(
        {
            "schema_version": "pg240-pikachu-source-replay-protocol-v1",
            "source_commit": SOURCE_COMMIT,
            "shared_runtime_with_pg214": True,
            "source_tree_hash_required": True,
            "fresh_container_per_episode": True,
            "candidate_reference_negative_triplet_required": True,
            "typed_result_oracle_required_for_positive": True,
            "cross_application_source_training_promotion_allowed": False,
            "training_promotion_allowed": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        }
    )
    trace.update({"schema_version": "pg240-pikachu-source-replay-trace-v1", "training_eligible": False, "training_role": "cross_application_source_evaluation_only", "source_commit": SOURCE_COMMIT, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})

    records = []
    for row in list(report.get("results") or []):
        typed = dict(row.get("typed_oracle") or {})
        result_oracle = dict(row.get("result_oracle") or {})
        records.append(
            {
                "source": "pg240_pikachu_source_replay",
                "source_commit": SOURCE_COMMIT,
                "seed": row.get("seed"),
                "route": row.get("route"),
                "method": row.get("method"),
                "family": "sql",
                "ai_sent": bool(row.get("ai_sent")),
                "reference_sent": bool(row.get("reference_sent")),
                "negative_sent": bool(row.get("negative_sent")),
                "typed_effect_confirmed": bool(typed.get("typed_effect_confirmed")),
                "result_fixture_verified": bool(result_oracle.get("result_fixture_verified")),
                "confirmed_positive": bool(result_oracle.get("confirmed_positive")),
                "route_source_sha256": row.get("route_source_sha256"),
                "training_eligible": False,
                "memory_promotion_allowed": False,
                "raw_payload_strings_stored": False,
                "raw_response_bodies_stored": False,
            }
        )
    dataset = {
        "schema_version": "pg240-pikachu-source-replay-dataset-v1",
        "records": records,
        "counts": report.get("counts") or {},
        "contract": {
            "evaluation_only": True,
            "cross_application_source_holdout": True,
            "shared_runtime_with_pg214": True,
            "typed_oracle_required_for_positive": True,
            "fresh_reset_required": True,
            "negative_control_required": True,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "training_eligible": False,
            "memory_promotion_allowed": False,
        },
    }
    dataset["dataset_sha256"] = _digest(dataset)
    report.pop("report_sha256", None)
    protocol.pop("protocol_sha256", None)
    report["report_sha256"] = _digest(report)
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    _write(TRACE, trace)
    counts = dict(report.get("counts") or {})
    MARKDOWN.write_text(
        "\n".join(
            [
                "# PG-240 Pikachu upstream-source replay",
                "",
                f"source_commit={SOURCE_COMMIT}; fresh={counts.get('fresh_container_count', 0)}; GET={counts.get('get_episode_count', 0)}; POST={counts.get('post_episode_count', 0)}",
                f"AI={counts.get('ai_send_count', 0)}; reference={counts.get('reference_send_count', 0)}; negative={counts.get('negative_send_count', 0)}; result_fixture={counts.get('result_fixture_verified_count', 0)}; typed_effect={counts.get('typed_effect_confirmed_count', 0)}",
                "",
                "这是应用源码跨实现评估，不是两个独立后端运行时的证明；共享 PG-214 修复运行层。每路由 fresh reset、正负对照、typed/result oracle 全部满足前才可标记 confirmed_positive；本轮数据默认 evaluation-only，不进入训练或长期记忆。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "source_commit": SOURCE_COMMIT, "counts": counts, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
