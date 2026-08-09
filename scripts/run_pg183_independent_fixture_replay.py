"""PG-183: replay the frozen manifest decoder on an independent HTTP fixture.

The fixture is a separate implementation/layout from Pikachu and Juice Shop.
The run is evaluation-only: inert alphanumeric values are sent to an observed
``message`` query field, responses are projected, and no raw value is stored.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_app_surface_fixture import make_surface_fixture_server, surface_fixture_source_sha256  # noqa: E402
from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg179b_iterative_probe import action_manifest, request_chain, surface_oracle  # noqa: E402
from app.pg181_manifest_decoder import build_model, last_logits, pre_action_tokens, restrict_manifest_action  # noqa: E402


REPORT_PATH = ROOT / "research" / "pg183_independent_fixture_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg183_independent_fixture_replay_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg183_independent_fixture_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg183_independent_fixture_replay_report_v1.md"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg181-manifest-decoder-v1" / "url_holdout" / "moe_large_seed18101.pt"
PORT = 8791
BASE_URL = f"http://127.0.0.1:{PORT}"
CANARY = "pg183-canary-a1"
CONTROL = "pg183-control-a1"
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg183-independent-fixture-surface-only-v1").hexdigest()
SURFACES = (("/attribute", "fixture_attribute"), ("/json", "fixture_json"), ("/header", "fixture_header"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _belief_after(signal: dict[str, Any], role: str) -> dict[str, float]:
    if bool(signal.get("candidate_signal")):
        return {"candidate_surface_signal": 0.65, "unknown_surface": 0.35}
    if role == "control":
        return {"no_surface_delta": 0.65, "unknown_surface": 0.35}
    return {"no_observed_effect": 0.60, "unknown_surface": 0.40}


def _fixture_server() -> tuple[Any, threading.Thread]:
    server = make_surface_fixture_server()
    thread = threading.Thread(target=server.serve_forever, name="pg183-fixture", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{BASE_URL}/plain", timeout=1.0).status_code == 200:
                return server, thread
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    server.shutdown()
    thread.join(timeout=2.0)
    raise RuntimeError("PG-183 fixture did not become ready")


def _replay_surface(model: torch.nn.Module, vocabulary: dict[str, int], path: str, surface: str, device: torch.device, source_hash: str) -> dict[str, Any]:
    server, thread = _fixture_server()
    client = httpx.Client(base_url=BASE_URL, timeout=5.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    prior_records: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    controller_abstain = 0
    target_hash = hashlib.sha256(f"{source_hash}:{surface}".encode("utf-8")).hexdigest()
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
                result = request_chain(client, method="GET", path=path, marker=None)
                manifest = action_manifest(path=path, surface=surface, family="xss", method="GET", field_names=[], probe_role="negative_control", marker="pg183-baseline-ref")
                controller_decision = "send_safe_baseline"
            else:
                result = request_chain(client, method="GET", path=path, query={"message": marker}, marker=marker, baseline_status=200)
                manifest = action_manifest(path=path, surface=surface, family="xss", method="GET", field_names=["message"], probe_role=role, marker=marker)
                controller_decision = "send_safe_canary"
            signal = {**dict(result["signal"]), "candidate_signal": bool(result["signal"].get("candidate_signal"))}
            oracle = surface_oracle(family="xss", method="GET", signal=signal, oracle_contract_sha256=ORACLE_CONTRACT_SHA256)
            failure = failure_signature({"method": "GET", "role": role if step_index > 1 else "control", "candidate_signal": bool(signal.get("candidate_signal")), "positive": False, "positive_authority": False, "typed_available": False, "probe_round": step_index, "max_probe_rounds": 5}, prior_records=prior_records, max_steps=5, step_count=step_index)
            belief = _belief_after(signal, role)
            action_view = {"method": "GET", "placement": "none" if step_index == 1 else "query", "encoding_chain": manifest["encoding_chain"], "probe_ref": manifest["probe_ref"], "probe_sha256": manifest["payload_sha256"], "field_names": [] if step_index == 1 else ["message"], "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
            steps.append({"step_index": step_index, "model_action": predicted, "confidence": round(confidence, 6), "controller_decision": controller_decision, "action_manifest": action_view, "response_projection": result["projection"], "oracle_projection": oracle, "failure_signature": failure, "belief_after": belief, "decision": "abstain", "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": manifest, "response_projection": result["projection"], "failure_signature": failure, "belief_after": belief})
            prior_records.append({"method": "GET", "role": role if step_index > 1 else "control", "candidate_signal": bool(signal.get("candidate_signal")), "belief_after": belief})
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()
    return {"surface": surface, "path": path, "source_hash": source_hash, "target_instance_hash": target_hash, "step_count": len(steps), "sent_count": sum(int(item.get("controller_decision") in {"send_safe_baseline", "send_safe_canary"}) for item in steps), "candidate_sent_count": sum(int(item.get("model_action") == "safe_candidate" and item.get("controller_decision") == "send_safe_canary") for item in steps), "controller_abstain_count": controller_abstain, "typed_positive_count": 0, "vulnerability_claim_allowed": False, "steps": steps}


def main() -> int:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    variant = str(checkpoint["variant"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(vocabulary), variant).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    source_hash = surface_fixture_source_sha256()
    runs = [_replay_surface(model, vocabulary, path, surface, device, source_hash) for path, surface in SURFACES]
    checkpoint_hash = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    report = {"protocol_id": "pg-pk-183-independent-fixture-replay-v1", "schema_version": "pg183-independent-fixture-replay-report-v1", "status": "completed_independent_implementation_evaluation", "training_source": "Pikachu PG-181 frozen manifest decoder", "target": {"implementation": "in_repo_surface_fixture", "base_url": BASE_URL, "fixture_source_sha256": source_hash, "independent_from_pikachu": True, "loopback_only": True, "external_network": False, "fresh_server_per_surface": True}, "model": {"variant": variant, "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_hash, "online_weight_update": False, "memory_promotion_allowed": False}, "runs": runs, "retention": {"old_checkpoint_hash_unchanged": True, "old_training_report": "research/pg181_manifest_decoder_replay_report_v1.json", "catastrophic_forgetting_claim": False, "reason": "frozen checkpoint; independent evaluation only"}, "oracle": {"typed_execution_available": False, "family_specific_oracle_available": False, "positive_count": 0, "vulnerability_claim_allowed": False}, "safety": {"raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "script_execution": False, "database_write": False, "credential_access": False, "memory_promotion_allowed": False}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg183-independent-fixture-replay-trace-v1", "evaluation_only": True, "training_eligible": False, "model_checkpoint_sha256": checkpoint_hash, "runs": runs, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False})
    protocol = {"protocol_id": "pg-pk-183-independent-fixture-replay-v1", "schema_version": "pg183-independent-fixture-replay-protocol-v1", "source_model": str(CHECKPOINT_PATH.relative_to(ROOT)), "target_implementation": "in_repo_surface_fixture", "fixture_source_hash_required": True, "frozen_checkpoint_required": True, "independent_source_required": True, "fresh_server_per_surface": True, "parameter_authority": "independent fixture observed message field", "surface_parameter": "message", "model_output_allowlist": ["baseline", "matched_control", "safe_candidate", "abstain"], "gates": {"fresh_server_per_surface": True, "manifest_validator_before_send": True, "independent_source_required": True, "raw_probe_and_response_excluded": True, "typed_positive_required_for_vulnerability_label": True, "unknown_oracle_action": "abstain", "weight_update_during_evaluation": False, "memory_promotion_during_evaluation": False}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-183 independent implementation replay", "", f"model={variant}; surfaces={len(runs)}; sent={sum(item['sent_count'] for item in runs)}; candidate={sum(item['candidate_sent_count'] for item in runs)}", "", "冻结 Pikachu 模型在独立实现上只做安全 canary 复放；没有 typed oracle，因此不生成漏洞阳性。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "variant": variant, "surface_count": len(runs), "sent_count": sum(item["sent_count"] for item in runs), "candidate_sent_count": sum(item["candidate_sent_count"] for item in runs), "typed_positive_count": 0, "training_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
