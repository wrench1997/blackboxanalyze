"""Run PG-114 family-heldout + negative-decoy replay on fresh processes."""

from __future__ import annotations

import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bsp_v3_research_core import SCHEMA_VERSION as BSP_CORE_SCHEMA  # noqa: E402
from app.pg114_ood_replay import SCHEMA_VERSION, SURFACES, collect_target  # noqa: E402
from app.trace_aligned_dataset import sha256_json  # noqa: E402


PROTOCOL_ID = "pg-pk-114-family-holdout-decoy-replay-v1"
TARGET_PATH = ROOT / "app" / "pg114_family_holdout_target.py"
BRIDGE_PATH = ROOT / "app" / "pg114_ood_replay.py"
CORE_PATH = ROOT / "app" / "bsp_v3_research_core.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg114_family_holdout_replay.py"
PG112_REPORT_PATH = ROOT / "research" / "pg112_python_bsp_local_replay_report_v1.json"
PG113_REPORT_PATH = ROOT / "research" / "pg113_cross_implementation_replay_report_v1.json"
REPORT_PATH = ROOT / "research" / "pg114_family_holdout_replay_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg114_family_holdout_replay_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg114_family_holdout_replay_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg114_family_holdout_replay_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg114_family_holdout_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg114_family_holdout_replay_report_v1.md"
TARGET_SEEDS = (11401, 11402, 11403)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start(seed: int) -> tuple[subprocess.Popen[bytes], str]:
    port = _free_port()
    env = os.environ.copy()
    env["PG114_SEED"] = str(seed)
    env["PG114_TARGET_INSTANCE"] = f"pg114-target-{seed}"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.pg114_family_holdout_target:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200 and response.json().get("implementation") == "pg114-family-holdout-target":
                return process, base_url
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    _stop(process)
    raise RuntimeError("PG-114 target did not become ready")


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def _verify(record: dict[str, Any]) -> bool:
    declared = str(record.get("evidence_hash", ""))
    body = dict(record)
    body.pop("evidence_hash", None)
    return len(declared) == 64 and declared == sha256_json(body)


def _model_input(step: dict[str, Any]) -> dict[str, Any]:
    return {"action_manifest": step["action_manifest"], "baseline_projection": step["baseline_projection"], "response_projection": step["response_projection"], "belief_before": step["belief_before"], "target_instance_slot": hashlib.sha256(step["target_instance_id"].encode("utf-8")).hexdigest()[:16]}


async def _collect() -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    for seed in TARGET_SEEDS:
        process, base_url = _start(seed)
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=5.0, follow_redirects=False) as client:
                targets.append(await collect_target(client, target_seed=seed))
        finally:
            _stop(process)
    return {"execution_mode": "fresh_loopback_uvicorn_process", "targets": targets}


def run() -> dict[str, Any]:
    raw = asyncio.run(_collect())
    episodes = [episode for target in raw["targets"] for episode in target["episodes"]]
    steps = [step for episode in episodes for step in episode["steps"]]
    evidence_records = [record for episode in episodes for record in episode["evidence_records"]]
    methods = Counter(str(step["action_manifest"]["method"]) for step in steps)
    decisions = Counter(str(step["decision"]) for step in steps)
    policy_episodes = [episode for episode in episodes if episode["surface_kind"] == "policy"]
    decoy_episodes = [episode for episode in episodes if episode["surface_kind"] == "decoy"]
    neutral_episodes = [episode for episode in episodes if episode["surface_kind"] == "neutral"]
    opaque_episodes = [episode for episode in episodes if episode["surface_kind"] == "opaque"]
    pg112 = json.loads(PG112_REPORT_PATH.read_text(encoding="utf-8"))
    pg113 = json.loads(PG113_REPORT_PATH.read_text(encoding="utf-8"))
    visible_rows = [{"row_id": step["step_id"], "episode_id": step["episode_id"], "model_input": _model_input(step), "evaluator_target": {"decision": step["decision"], "positive_authority": bool(step["oracle_projection"].get("positive_authority"))}, "training_eligible": False, "memory_promotion_allowed": False} for step in steps]
    proposal = {"schema_version": "pg114-family-holdout-replay-proposal-v1", "evaluation_only": True, "training_eligible": False, "family_holdout_semantic": "security_policy_transition", "decoy_semantic": "shape_only_change", "reference_reports": ["research/pg112_python_bsp_local_replay_report_v1.json", "research/pg113_cross_implementation_replay_report_v1.json"], "model_input_contract": {"family_names": False, "typed_oracle_labels": False, "raw_probe_values": False, "raw_response_bodies": False, "fields": ["action_manifest", "baseline_projection", "response_projection", "belief_before", "target_instance_slot"]}, "matrix": {"target_seeds": list(TARGET_SEEDS), "surface_slots": len(SURFACES), "methods": ["GET", "POST"], "fresh_reset_per_action": True}, "required_outcomes": {"family_holdout_confirm_recall_min": 1.0, "decoy_false_accept_max": 0, "withheld_oracle_abstain_min": 1.0, "all_abstain_not_success": True}, "bsp_core_schema": BSP_CORE_SCHEMA, "replay_package_promotion": False}
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    target_step_sets = [[step for episode in target["episodes"] for step in episode["steps"]] for target in raw["targets"]]
    checks = {
        "target_instance_count_three": len(raw["targets"]) == 3,
        "independent_target_identity": all(target["target_implementation"] == "pg114-family-holdout-target" and target["target_schema_version"] == "pg114-family-holdout-target-v1" for target in raw["targets"]),
        "surface_matrix_complete": len(episodes) == 12 and len({episode["surface_slot"] for episode in episodes}) == 4,
        "get_post_balanced": methods["GET"] == methods["POST"] == 24,
        "fresh_reset_every_step": all(bool(step["fresh_reset"].get("fresh_target")) and bool(step["fresh_reset"].get("completed")) and bool(step["fresh_reset"].get("evaluator_state_hidden")) for step in steps),
        "reset_epoch_unique_per_target": all(len({int(step["fresh_reset"]["reset_epoch"]) for step in target_steps}) == 16 for target_steps in target_step_sets),
        "evidence_hashes_valid": len(evidence_records) == len(steps) == 48 and all(_verify(record) for record in evidence_records),
        "target_oracle_hashes_bound": all(record["target_evidence_sha256"] == record["oracle_projection"]["source_evidence_sha256"] for record in evidence_records),
        "family_holdout_confirm_recall": len(policy_episodes) == 3 and all(episode["final_decision"] == "confirmed_positive" and episode["candidate_pair_positive"] for episode in policy_episodes),
        "decoy_false_accept_zero": len(decoy_episodes) == 3 and all(episode["final_decision"] == "confirmed_negative" and not episode["candidate_pair_positive"] for episode in decoy_episodes),
        "neutral_negative_stability": len(neutral_episodes) == 3 and all(episode["final_decision"] == "confirmed_negative" for episode in neutral_episodes),
        "withheld_oracle_abstains": len(opaque_episodes) == 3 and all(episode["final_decision"] == "abstain" for episode in opaque_episodes),
        "all_abstain_not_success": decisions["confirmed_positive"] == 3,
        "cross_seed_surface_consistency": all(len({episode["final_decision"] for episode in episodes if episode["surface_kind"] == kind}) == 1 for kind in ("policy", "decoy", "neutral", "opaque")),
        "bsp_parameters_unchanged": all(bool(episode["bsp"]["parameter_unchanged"]) for episode in episodes),
        "bsp_mass_conserved": all(float(step["response_projection"]["bsp_core_projection"]["leaf_mass_error"]) <= 1.0e-12 for step in steps),
        "model_input_oracle_blind": all("oracle_projection" not in row["model_input"] and "positive_authority" not in row["model_input"] for row in visible_rows),
        "model_input_family_free": all("family" not in json.dumps(row["model_input"], ensure_ascii=False).casefold() for row in visible_rows),
        "no_raw_values_stored": all(token not in json.dumps({"steps": steps, "evidence": evidence_records}, ensure_ascii=False).casefold() for token in ("<script", "union select", "sleep(", "javascript:", "raw_probe_value", "raw_response_body_value")),
        "no_training_or_memory_write": all(not bool(step["online_weight_update"]) and not bool(step["long_term_memory_write"]) for step in steps),
        "prior_gate_not_reused_as_training": pg112["promotion"]["training_allowed"] is False and pg113["promotion"]["training_allowed"] is False,
    }
    blocked = [name for name, passed in checks.items() if not passed]
    status = "passed_pg114_family_holdout_replay" if not blocked else "blocked"
    report = {"protocol_id": PROTOCOL_ID, "schema_version": "pg114-family-holdout-replay-report-v1", "status": status, "scope": {"execution_mode": raw["execution_mode"], "loopback_only": True, "external_network": False, "implementation_count": 3, "family_holdout_claim_allowed": not blocked, "trained_model_claim_allowed": False, "docker_claimed": False}, "source": {"reference_reports": ["research/pg112_python_bsp_local_replay_report_v1.json", "research/pg113_cross_implementation_replay_report_v1.json"], "source_hashes": {"target": _file_hash(TARGET_PATH), "bridge": _file_hash(BRIDGE_PATH), "bsp_core": _file_hash(CORE_PATH), "runner": _file_hash(RUNNER_PATH), "pg112_report": _file_hash(PG112_REPORT_PATH), "pg113_report": _file_hash(PG113_REPORT_PATH)}}, "metrics": {"target_instance_count": 3, "surface_count": 4, "episode_count": 12, "step_count": 48, "get_step_count": methods["GET"], "post_step_count": methods["POST"], "fresh_reset_count": sum(int(bool(step["fresh_reset"].get("fresh_target"))) for step in steps), "evidence_hash_valid_count": sum(int(_verify(record)) for record in evidence_records), "typed_oracle_called_count": sum(int(str(step["oracle_projection"]["modality"]) == "family_holdout_typed_differential") for step in steps), "oracle_withheld_step_count": sum(int(str(step["oracle_projection"]["modality"]) == "family_holdout_untyped_signal") for step in steps), "confirmed_positive_count": decisions["confirmed_positive"], "confirmed_negative_count": decisions["confirmed_negative"], "candidate_count": decisions["candidate"], "abstain_count": decisions["abstain"], "family_holdout_positive_episode_count": len(policy_episodes), "family_holdout_confirm_recall": sum(int(episode["final_decision"] == "confirmed_positive") for episode in policy_episodes) / len(policy_episodes), "decoy_episode_count": len(decoy_episodes), "decoy_false_accept_count": sum(int(episode["final_decision"] == "confirmed_positive") for episode in decoy_episodes), "withheld_oracle_abstain_rate": sum(int(episode["final_decision"] == "abstain") for episode in opaque_episodes) / len(opaque_episodes), "bsp_parameter_unchanged_rate": sum(int(episode["bsp"]["parameter_unchanged"]) for episode in episodes) / len(episodes)}, "cross_experiment": {"pg112_protocol": pg112["protocol_id"], "pg113_protocol": pg113["protocol_id"], "family_holdout_is_not_in_prior_surface_matrix": True, "prior_training_allowed": False, "claim": "family-heldout replay and decoy rejection only; no trained-model capability claim"}, "checks": checks, "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "family_holdout_evaluation_only", "reason": "OOD/decoy replay establishes a gate for future training data but is not itself training evidence"}, "safety": {"loopback_only": True, "external_network": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "family_labels_in_model_input": False, "typed_oracle_labels_in_model_input": False, "fresh_reset_per_action": True, "matched_negative_controls": True, "bsp_weights_updated": False, "long_term_memory_write": False}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DATASET_PATH.write_text(json.dumps({"schema_version": "pg114-family-holdout-visible-dataset-v1", "evaluation_only": True, "training_eligible": False, "model_input_family_free": True, "typed_oracle_labels_outside_model_input": True, "rows": visible_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg114-family-holdout-trace-v1", "evaluation_only": True, "training_eligible": False, "execution_mode": raw["execution_mode"], "target_instance_ids": [target["target_instance_id"] for target in raw["targets"]], "episodes": [{key: value for key, value in episode.items() if key != "evidence_records"} for episode in episodes], "steps": steps, "evidence_records": evidence_records, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False, "trace_manifest_sha256": sha256_json([step["trace_sha256"] for step in steps])}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "title": "PG-114 族外表面与负向 decoy 回放", "authorization": {"scope": "workspace_local_only", "transport": "fresh_loopback_uvicorn_process", "external_network": False, "raw_values_persisted": False}, "matrix": {"target_seeds": list(TARGET_SEEDS), "surface_slot_count": len(SURFACES), "methods": ["GET", "POST"], "fresh_reset_per_action": True}, "heldout_surface": "security_policy_transition", "decoy_surface": "shape_only_change", "positive_gate": ["GET/POST repeat", "matched negative control", "fresh reset", "target evidence SHA-256", "bridge evidence SHA-256", "typed oracle after probe"], "unknown_policy": "without typed evaluator must abstain", "model_contract": {"family_labels": False, "typed_oracle_in_features": False, "raw_request_response": False, "training_allowed": False, "memory_promotion_allowed": False}, "status": "run_completed_family_holdout_no_promotion"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-114 族外表面与负向 decoy 回放", "", f"状态：`{status}`。3 个 fresh target process、4 个匿名 surface slot、GET/POST 共 {len(steps)} 步。", "", f"族外正例确认召回：`{report['metrics']['family_holdout_confirm_recall']}`；decoy 误确认：`{report['metrics']['decoy_false_accept_count']}`；未知 oracle 弃权率：`{report['metrics']['withheld_oracle_abstain_rate']}`。", "", "本轮只验证族外回放协议、负向 decoy 和弃权，不训练、不写长期记忆，也不宣称真实网址漏洞能力。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": status, "metrics": report["metrics"], "blocking_reasons": blocked, "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
