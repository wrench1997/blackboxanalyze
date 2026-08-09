"""PG-182: replay the Pikachu-trained manifest decoder on Juice Shop.

This is a family/application holdout.  The target is a pinned local Juice Shop
container and the only parameterized surface is the allow-listed read-only
``q`` search field.  The model selects only abstract probe roles; all network
values remain inert canaries and all response bodies are projected in memory.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg179b_iterative_probe import PIKACHU_IMAGE_DIGEST, action_manifest, request_chain, surface_oracle  # noqa: E402
from app.pg181_manifest_decoder import build_model, last_logits, pre_action_tokens, restrict_manifest_action  # noqa: E402


REPORT_PATH = ROOT / "research" / "pg182_juice_shop_manifest_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg182_juice_shop_manifest_replay_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg182_juice_shop_manifest_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg182_juice_shop_manifest_replay_report_v1.md"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg181-manifest-decoder-v1" / "url_holdout" / "moe_large_seed18101.pt"
IMAGE = "bkimminich/juice-shop@sha256:28870b9d2bec49e605d6ebbf4b22ed1ec1ca0a72347ef19217bbbb21ea44e3fe"
CONTAINER_NAME = "pg182-juice-shop"
PORT = 3101
CANARY = "pg182-canary-a1"
CONTROL = "pg182-control-a1"
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg182-juice-shop-no-family-oracle-v1").hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _exists() -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{CONTAINER_NAME}$", "--format", "{{.Names}}"))


def _start() -> str:
    if _exists():
        raise RuntimeError(f"refusing to reuse {CONTAINER_NAME}")
    _docker("run", "--detach", "--rm", "--pull=never", "--name", CONTAINER_NAME, "--publish", f"127.0.0.1:{PORT}:3000", IMAGE)
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{PORT}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", CONTAINER_NAME)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError("Juice Shop did not become ready")


def _stop() -> None:
    if _exists():
        _docker("stop", "--timeout", "5", CONTAINER_NAME)


def _belief_after(signal: dict[str, Any], role: str) -> dict[str, float]:
    if bool(signal.get("candidate_signal")):
        return {"candidate_surface_signal": 0.65, "unknown_surface": 0.35}
    if role == "control":
        return {"no_surface_delta": 0.65, "unknown_surface": 0.35}
    return {"no_observed_effect": 0.60, "unknown_surface": 0.40}


def main() -> int:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    variant = str(checkpoint["variant"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(vocabulary), variant).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_instance_id = _start()
    client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=8.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    prior_records: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    controller_abstain = 0
    try:
        for step_index in range(1, 6):
            previous = history[-1] if history else None
            context = pre_action_tokens(previous, history=history[:-1])
            ids = torch.tensor([[vocabulary[token] for token in context]], dtype=torch.long)
            mask = torch.ones_like(ids, dtype=torch.bool)
            with torch.inference_mode():
                logits = last_logits(model, ids.to(device), mask.to(device))[0].detach().cpu()
            predicted, confidence = restrict_manifest_action(logits, vocabulary, single_channel=True)
            if step_index == 1 and predicted != "baseline":
                controller_abstain += 1
                steps.append({"step_index": step_index, "model_action": predicted, "confidence": round(confidence, 6), "controller_decision": "abstain", "abstain_reason": "initial_state_requires_baseline"})
                break
            if step_index > 1 and predicted == "baseline":
                controller_abstain += 1
                steps.append({"step_index": step_index, "model_action": predicted, "confidence": round(confidence, 6), "controller_decision": "abstain", "abstain_reason": "baseline_only_allowed_at_episode_start"})
                break
            if predicted == "abstain":
                controller_abstain += 1
                steps.append({"step_index": step_index, "model_action": predicted, "confidence": round(confidence, 6), "controller_decision": "abstain", "abstain_reason": "model_abstain"})
                break
            role = "control" if predicted == "matched_control" else "candidate"
            marker = CONTROL if role == "control" else CANARY
            if step_index == 1:
                result = request_chain(client, method="GET", path="/rest/products/search", marker=None)
                manifest = action_manifest(path="/rest/products/search", surface="juice_search", family="logic", method="GET", field_names=[], probe_role="negative_control", marker="pg182-baseline-ref")
                controller_decision = "send_safe_baseline"
            else:
                result = request_chain(client, method="GET", path="/rest/products/search", query={"q": marker}, marker=marker, baseline_status=200)
                manifest = action_manifest(path="/rest/products/search", surface="juice_search", family="logic", method="GET", field_names=["q"], probe_role=role, marker=marker)
                controller_decision = "send_safe_canary"
            signal = {**dict(result["signal"]), "candidate_signal": bool(result["signal"].get("candidate_signal"))}
            oracle = surface_oracle(family="logic", method="GET", signal=signal, oracle_contract_sha256=ORACLE_CONTRACT_SHA256)
            failure = failure_signature({"method": "GET", "role": role if step_index > 1 else "control", "candidate_signal": bool(signal.get("candidate_signal")), "positive": False, "positive_authority": False, "typed_available": False, "probe_round": step_index, "max_probe_rounds": 5}, prior_records=prior_records, max_steps=5, step_count=step_index)
            belief = _belief_after(signal, role)
            action_view = {"method": "GET", "placement": "none" if step_index == 1 else "query", "encoding_chain": manifest["encoding_chain"], "probe_ref": manifest["probe_ref"], "probe_sha256": manifest["payload_sha256"], "field_names": [] if step_index == 1 else ["q"], "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
            steps.append({"step_index": step_index, "model_action": predicted, "confidence": round(confidence, 6), "controller_decision": controller_decision, "action_manifest": action_view, "response_projection": result["projection"], "oracle_projection": oracle, "failure_signature": failure, "belief_after": belief, "decision": "abstain", "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": manifest, "response_projection": result["projection"], "failure_signature": failure, "belief_after": belief})
            prior_records.append({"method": "GET", "role": role if step_index > 1 else "control", "candidate_signal": bool(signal.get("candidate_signal")), "belief_after": belief})
    finally:
        client.close()
        _stop()
    replay = {"checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "variant": variant, "target": "juice_shop", "target_route": "/rest/products/search", "parameter_name": "q", "fresh_container_id_hash": hashlib.sha256(target_instance_id.encode("utf-8")).hexdigest(), "step_count": len(steps), "sent_count": sum(int(item.get("controller_decision") in {"send_safe_baseline", "send_safe_canary"}) for item in steps), "candidate_sent_count": sum(int(item.get("model_action") == "safe_candidate" and item.get("controller_decision") == "send_safe_canary") for item in steps), "controller_abstain_count": controller_abstain, "typed_positive_count": 0, "vulnerability_claim_allowed": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "steps": steps}
    report = {"protocol_id": "pg-pk-182-juice-shop-manifest-replay-v1", "schema_version": "pg182-juice-shop-manifest-replay-report-v1", "status": "completed_cross_app_evaluation_only", "training_source": "Pikachu PG-181 manifest decoder", "target": {"application": "Juice Shop", "image": IMAGE, "loopback": True, "external_network": False, "fresh_container": True}, "parameter_grounding": {"source": "app/juice_shop_shadow_collector.py allow-listed SAFE_SHADOW_QUERY_NAMES", "observed_parameter": "q", "unobserved_method_forbidden": True}, "replay": replay, "oracle": {"typed_execution_available": False, "family_specific_oracle_available": False, "positive_count": 0, "vulnerability_claim_allowed": False}, "safety": {"raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "script_execution": False, "database_write": False, "credential_access": False, "memory_promotion_allowed": False}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg182-juice-shop-manifest-replay-trace-v1", "evaluation_only": True, "training_eligible": False, "replay": replay, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False})
    protocol = {"protocol_id": "pg-pk-182-juice-shop-manifest-replay-v1", "schema_version": "pg182-juice-shop-manifest-replay-protocol-v1", "source_model": str(CHECKPOINT_PATH.relative_to(ROOT)), "target_image": IMAGE, "loopback_port": PORT, "parameter_authority": "allow-listed q from Juice Shop shadow collector", "safe_canary_only": True, "fresh_container_required": True, "model_output_allowlist": ["baseline", "matched_control", "safe_candidate", "abstain"], "gates": {"manifest_validator_before_send": True, "unobserved_parameter_forbidden": True, "family_oracle_required_for_positive": True, "unknown_oracle_action": "abstain", "training_promotion_allowed": False, "memory_promotion_allowed": False}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-182 Juice Shop manifest replay", "", f"模型={variant}；发送={replay['sent_count']}；candidate={replay['candidate_sent_count']}；controller abstain={replay['controller_abstain_count']}", "", "这是 Pikachu 训练模型的族外本地复放；Juice Shop 没有 family-specific typed oracle，因此不生成漏洞阳性。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "variant": variant, "sent_count": replay["sent_count"], "candidate_sent_count": replay["candidate_sent_count"], "controller_abstain_count": replay["controller_abstain_count"], "typed_positive_count": 0, "training_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
