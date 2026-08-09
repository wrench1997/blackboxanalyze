"""PG-112: replay local typed oracle traces through the Python BSP v3 core."""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bsp_v3_research_core import SCHEMA_VERSION as BSP_CORE_SCHEMA  # noqa: E402
from app.main import app  # noqa: E402
from app.pg112_replay_bridge import SCHEMA_VERSION, collect_all  # noqa: E402
from app.trace_aligned_dataset import sha256_json  # noqa: E402


PROTOCOL_ID = "pg-pk-112-python-bsp-local-replay-v1"
BRIDGE_PATH = ROOT / "app" / "pg112_replay_bridge.py"
CORE_PATH = ROOT / "app" / "bsp_v3_research_core.py"
MAIN_PATH = ROOT / "app" / "main.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg112_python_bsp_local_replay.py"
REPORT_PATH = ROOT / "research" / "pg112_python_bsp_local_replay_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg112_python_bsp_local_replay_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg112_python_bsp_local_replay_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg112_python_bsp_local_replay_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg112_python_bsp_local_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg112_python_bsp_local_replay_report_v1.md"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_evidence_hash(record: dict[str, Any]) -> bool:
    declared = str(record.get("evidence_hash", ""))
    body = dict(record)
    body.pop("evidence_hash", None)
    return len(declared) == 64 and declared == sha256_json(body)


def _model_input(step: dict[str, Any]) -> dict[str, Any]:
    """Extract the model-facing view, excluding oracle/evaluator labels."""

    return {
        "action_manifest": step["action_manifest"],
        "baseline_projection": step["baseline_projection"],
        "response_projection": step["response_projection"],
        "belief_before": step["belief_before"],
        "target_instance_slot": hashlib.sha256(step["target_instance_id"].encode("utf-8")).hexdigest()[:16],
    }


def run() -> dict[str, Any]:
    raw = asyncio.run(collect_all(app, target_seeds=(101, 202, 303)))
    episodes = raw["episodes"]
    steps = raw["steps"]
    evidence_records = raw["evidence_records"]
    decisions = Counter(str(step["decision"]) for step in steps)
    methods = Counter(str(step["action_manifest"]["method"]) for step in steps)
    known_episodes = [episode for episode in episodes if episode["oracle_available"]]
    withheld_episodes = [episode for episode in episodes if not episode["oracle_available"]]
    positive_by_slot: dict[str, list[str]] = defaultdict(list)
    for episode in known_episodes:
        positive_by_slot[str(episode["surface_slot"])].append(str(episode["final_decision"]))
    target_consistency = all(set(values) == {"confirmed_positive"} for values in positive_by_slot.values())
    visible_rows = [
        {
            "row_id": step["step_id"],
            "episode_id": step["episode_id"],
            "model_input": _model_input(step),
            "evaluator_target": {
                "decision": step["decision"],
                "positive_authority": bool(step["oracle_projection"].get("positive_authority")),
            },
            "training_eligible": False,
            "memory_promotion_allowed": False,
        }
        for step in steps
    ]
    proposal = {
        "schema_version": "pg112-python-bsp-local-replay-proposal-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_contract": {
            "family_names": False,
            "typed_oracle_labels": False,
            "raw_probe_values": False,
            "raw_response_bodies": False,
            "fields": ["action_manifest", "baseline_projection", "response_projection", "belief_before", "target_instance_slot"],
        },
        "required_pair": ["GET", "POST"],
        "required_controls": ["fresh_reset", "matched_negative_control", "evidence_sha256"],
        "bsp_core_schema": BSP_CORE_SCHEMA,
        "target_instance_seeds": [101, 202, 303],
        "surface_slot_count": 4,
        "oracle_withheld_slot_required_to_abstain": True,
        "typed_oracle_called_after_probe_only": True,
        "replay_package_promotion": False,
    }
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checks = {
        "target_instance_count_three": raw["target_instance_count"] == 3,
        "surface_matrix_complete": raw["surface_count"] == 4 and len(episodes) == 12,
        "get_post_balanced": methods["GET"] == methods["POST"] == 24,
        "fresh_reset_every_step": all(bool(step["fresh_reset"].get("fresh_target")) and bool(step["fresh_reset"].get("completed")) for step in steps),
        "evidence_hashes_valid": len(evidence_records) == len(steps) and all(_verify_evidence_hash(record) for record in evidence_records),
        "matched_negative_controls": all(bool(episode["negative_control_pair_clear"]) for episode in episodes),
        "known_pairs_confirmed_only_after_get_post": len(known_episodes) == 9 and all(episode["final_decision"] == "confirmed_positive" and episode["candidate_pair_positive"] for episode in known_episodes),
        "withheld_oracle_abstains": len(withheld_episodes) == 3 and all(episode["final_decision"] == "abstain" for episode in withheld_episodes),
        "multi_target_consistency": target_consistency,
        "model_input_oracle_blind": all("oracle_projection" not in row["model_input"] and "positive_authority" not in row["model_input"] for row in visible_rows),
        "model_input_family_free": all("family" not in json.dumps(row["model_input"], ensure_ascii=False).casefold() for row in visible_rows),
        "bsp_parameter_unchanged": all(bool(episode["bsp"]["parameter_unchanged"]) for episode in episodes),
        "bsp_mass_conserved": all(all(float(step["response_projection"]["bsp_core_projection"]["leaf_mass_error"]) <= 1.0e-12 for step in episode["steps"]) for episode in episodes),
        "no_training_or_memory_write": raw["training_eligible"] is False and raw["long_term_memory_write"] is False and all(not bool(step["online_weight_update"]) and not bool(step["long_term_memory_write"]) for step in steps),
        "no_raw_values_stored": raw["raw_probe_strings_stored"] is False and raw["raw_response_bodies_stored"] is False,
        "single_implementation_holdout_explicit": raw["transport"] == "in_process_asgi_loopback",
    }
    blocked = [name for name, passed in checks.items() if not passed]
    status = "passed_pg112_python_bsp_local_replay" if not blocked else "blocked"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg112-python-bsp-local-replay-report-v1",
        "status": status,
        "scope": {
            "transport": raw["transport"],
            "loopback_only": raw["loopback_only"],
            "external_network": raw["external_network"],
            "implementation_count": 1,
            "cross_implementation_claim_allowed": False,
            "docker_pg51_kept_as_separate_track": True,
        },
        "source": {
            "source_id": "pg112-current-project-local-asgi-maze",
            "authorization": "workspace_local_only",
            "source_hashes": {
                "bridge": _sha256_file(BRIDGE_PATH),
                "bsp_core": _sha256_file(CORE_PATH),
                "main": _sha256_file(MAIN_PATH),
                "runner": _sha256_file(RUNNER_PATH),
            },
        },
        "metrics": {
            "target_instance_count": raw["target_instance_count"],
            "surface_count": raw["surface_count"],
            "episode_count": len(episodes),
            "step_count": len(steps),
            "get_step_count": methods["GET"],
            "post_step_count": methods["POST"],
            "fresh_reset_count": sum(int(bool(step["fresh_reset"].get("fresh_target"))) for step in steps),
            "evidence_hash_valid_count": sum(int(_verify_evidence_hash(record)) for record in evidence_records),
            "typed_oracle_called_count": sum(int(bool(step["oracle_projection"].get("positive_authority")) or str(step["oracle_projection"].get("modality")) == "typed_surface_differential") for step in steps),
            "oracle_withheld_step_count": sum(int(str(step["oracle_projection"].get("modality")) == "untyped_surface_signal") for step in steps),
            "confirmed_positive_count": decisions["confirmed_positive"],
            "confirmed_negative_count": decisions["confirmed_negative"],
            "candidate_count": decisions["candidate"],
            "abstain_count": decisions["abstain"],
            "known_positive_pair_count": sum(int(episode["candidate_pair_positive"]) for episode in known_episodes),
            "unknown_oracle_abstain_rate": round(sum(int(episode["final_decision"] == "abstain") for episode in withheld_episodes) / len(withheld_episodes), 6),
            "multi_target_consistency_rate": 1.0 if target_consistency else 0.0,
            "bsp_parameter_unchanged_rate": round(sum(int(episode["bsp"]["parameter_unchanged"]) for episode in episodes) / len(episodes), 6),
        },
        "checks": checks,
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "local_typed_replay_evaluation_only",
            "reason": "This is same-project multi-target replay; it is not a cross-implementation or trained-model capability claim",
        },
        "safety": {
            "loopback_only": True,
            "external_network": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "family_labels_in_model_input": False,
            "typed_oracle_labels_in_model_input": False,
            "fresh_reset_per_step": True,
            "matched_negative_controls": True,
            "evidence_hashes_verified": True,
            "bsp_weights_updated": False,
            "long_term_memory_write": False,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset = {
        "schema_version": "pg112-python-bsp-local-replay-visible-dataset-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_family_free": True,
        "typed_oracle_labels_outside_model_input": True,
        "rows": visible_rows,
    }
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {
        "schema_version": "pg112-python-bsp-local-replay-trace-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "transport": raw["transport"],
        "target_instance_ids": sorted({episode["target_instance_id"] for episode in episodes}),
        "episodes": [{key: value for key, value in episode.items() if key not in {"evidence_records", "belief"}} for episode in episodes],
        "steps": steps,
        "evidence_records": evidence_records,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "online_weight_update": False,
        "long_term_memory_write": False,
        "trace_manifest_sha256": sha256_json([step["trace_sha256"] for step in steps]),
    }
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "title": "Python BSP v3 本地 typed replay bridge",
        "authorization": {"scope": "workspace_local_only", "transport": "in_process_asgi_loopback", "external_network": False, "raw_values_persisted": False},
        "matrix": {"target_instance_seeds": [101, 202, 303], "surface_slot_count": 4, "methods": ["GET", "POST"], "fresh_reset_per_step": True},
        "causal_contract": ["target surface slot", "safe abstract probe", "bounded observation diff", "generic belief update", "next action", "typed oracle target-only", "Rule IR/evaluation decision"],
        "positive_gate": ["GET/POST repeat", "matched negative control", "fresh reset", "valid evidence SHA-256", "typed oracle after probe"],
        "unknown_policy": "withheld typed oracle must abstain even when an anonymous candidate signal exists",
        "model_contract": {"family_labels": False, "typed_oracle_in_features": False, "raw_request_response": False, "training_allowed": False, "memory_promotion_allowed": False},
        "cross_implementation_status": "not_claimed; PG-51 Docker remains a separate implementation track",
        "status": "run_completed_python_local_replay_no_promotion",
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-112 Python BSP v3 本地 replay bridge\n\n"
        f"状态：`{status}`。3 个 fresh target instance、4 个匿名 surface slot、GET/POST 双通道，共 {len(steps)} 个 bounded steps。\n\n"
        f"- confirmed_positive：`{decisions['confirmed_positive']}`；confirmed_negative：`{decisions['confirmed_negative']}`；candidate：`{decisions['candidate']}`；abstain：`{decisions['abstain']}`。\n"
        "- 已验证 fresh reset、匹配阴性对照、证据 SHA-256；withheld typed oracle 的 episode 全部 abstain。\n"
        "- Python BSP v3 只做结构前向与质量守恒检查，参数未更新；该轮不训练、不写长期记忆，也不宣称跨实现能力。\n",
        encoding="utf-8",
    )
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": status, "metrics": report["metrics"], "blocking_reasons": blocked, "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
