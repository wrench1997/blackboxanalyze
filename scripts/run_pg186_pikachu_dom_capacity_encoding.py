"""PG-186: capacity/seed/encoding replay on the frozen Pikachu DOM track.

This is an evaluation sweep over already-trained PG-181 decoders.  It does
not train on the target trace.  Each episode restarts the exact owned local
container, sends only an inert DOM marker to an observed GET parameter, and
stores projections/hashes rather than raw values.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
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
from app.pg180_process_action_model import abstract_step_tokens  # noqa: E402
from app.pg181_manifest_decoder import build_model, last_logits, pre_action_tokens, restrict_manifest_action  # noqa: E402
from app.pg185_pikachu_dom_adapter import build_dom_action_manifest, build_query, project_dom_response  # noqa: E402


def _load_pg185_runner() -> Any:
    path = ROOT / "scripts" / "run_pg185_pikachu_dom_replay.py"
    spec = importlib.util.spec_from_file_location("pg185_runner_for_pg186", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-185 runner helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG185 = _load_pg185_runner()
RESEARCH = ROOT / "research"
REPORT_PATH = RESEARCH / "pg186_pikachu_dom_capacity_encoding_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg186_pikachu_dom_capacity_encoding_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg186_pikachu_dom_capacity_encoding_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg186_pikachu_dom_capacity_encoding_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg186-pikachu-dom-capacity-encoding-v1"

CHECKPOINTS = (
    ("small_seed18101", ROOT / "artifacts" / "pg181-manifest-decoder-v1" / "url_holdout" / "small_seed18101.pt"),
    ("small_seed18102", ROOT / "artifacts" / "pg181-manifest-decoder-v1" / "url_holdout" / "small_seed18102.pt"),
    ("medium_seed18101", ROOT / "artifacts" / "pg181-manifest-decoder-v1" / "url_holdout" / "medium_seed18101.pt"),
    ("medium_seed18102", ROOT / "artifacts" / "pg181-manifest-decoder-v1" / "url_holdout" / "medium_seed18102.pt"),
    ("moe_large_seed18101", ROOT / "artifacts" / "pg181-manifest-decoder-v1" / "url_holdout" / "moe_large_seed18101.pt"),
    ("moe_large_seed18102", ROOT / "artifacts" / "pg181-manifest-decoder-v1" / "url_holdout" / "moe_large_seed18102.pt"),
)
ENCODING_PLANS = (
    ("identity", ("identity",)),
    ("html_entity", ("html_entity",)),
    ("html_entity_depth2", ("html_entity", "html_entity")),
)
MAX_STEPS = 5


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _restart_and_wait() -> None:
    PG185._docker("restart", PG185.CONTAINER_NAME)
    deadline = time.monotonic() + 100.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{PG185.BASE_URL}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("PG-186 fresh restart did not become ready")


def _model_context(context: list[str], vocabulary: dict[str, int]) -> tuple[list[str], int]:
    """Map held-out encoding tokens to the trained identity token.

    The mapping is explicit OOD accounting, not silent vocabulary growth: the
    frozen PG-181 model must not acquire new tokens during evaluation.
    """

    mapped: list[str] = []
    fallback_count = 0
    for token in context:
        if token in vocabulary:
            mapped.append(token)
            continue
        if token.startswith("history::encoding::"):
            fallback = "history::encoding::identity"
        elif token.startswith("encoding::"):
            fallback = "encoding::identity"
        else:
            fallback = ""
        if fallback and fallback in vocabulary:
            mapped.append(fallback)
            fallback_count += 1
        else:
            fallback_count += 1
    if not mapped:
        mapped = ["<bos>"] if "<bos>" in vocabulary else [next(iter(vocabulary))]
    return mapped, fallback_count


def _belief(signal: dict[str, Any], typed_effect: bool, role: str) -> dict[str, float]:
    # Keep the model-visible belief vocabulary aligned with PG-181.
    if typed_effect:
        return {"candidate_surface_signal": 0.72, "unknown_surface": 0.28}
    if bool(signal.get("candidate_signal")):
        return {"candidate_surface_signal": 0.60, "unknown_surface": 0.40}
    if role == "control":
        return {"no_surface_delta": 0.65, "unknown_surface": 0.35}
    return {"no_observed_effect": 0.60, "unknown_surface": 0.40}


def _replay_episode(
    model: torch.nn.Module,
    vocabulary: dict[str, int],
    route: dict[str, Any],
    device: torch.device,
    *,
    encoding_name: str,
    encoding_chain: tuple[str, ...],
    target_hash: str,
) -> dict[str, Any]:
    path = str(route["path"])
    surface = str(route["surface"])
    fields = [str(item) for item in route["field_names"]]
    client = httpx.Client(base_url=PG185.BASE_URL, timeout=10.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    prior_records: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    fallback_count = 0
    typed_effect_count = 0
    controller_abstain = 0
    baseline_status: int | None = None
    try:
        for step_index in range(1, MAX_STEPS + 1):
            previous = history[-1] if history else None
            raw_context = pre_action_tokens(previous, history=history[:-1])
            context, fallbacks = _model_context(raw_context, vocabulary)
            fallback_count += fallbacks
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
            marker = f"pg186-{('ctrl' if role == 'control' else 'cand')}-{surface[-4:]}-{encoding_name}-{step_index}"
            actual_chain = ("identity",) if role != "candidate" else encoding_chain
            manifest = build_dom_action_manifest(path=path, surface=surface, field_names=fields, probe_role=role if step_index > 1 else "negative_control", marker=marker, encoding_chain=list(actual_chain))
            if step_index == 1:
                response = client.get(path)
                baseline_status = int(response.status_code)
                projected = project_dom_response(response, marker=None)
                controller_decision = "send_safe_baseline"
                action_role = "negative_control"
            else:
                query, oracle_marker = build_query(field_names=fields, role=role, marker=marker, encoding=encoding_name if role == "candidate" else "identity")
                response = client.get(path, params=query)
                projected = project_dom_response(response, marker=oracle_marker, baseline_status=baseline_status)
                controller_decision = "send_inert_dom_candidate" if role == "candidate" else "send_safe_canary"
                action_role = role

            oracle = dict(projected["oracle_projection"])
            signal = dict(oracle.get("signals") or {})
            signal["candidate_signal"] = bool(oracle.get("candidate_signal"))
            typed_effect = bool(projected["typed_surface_effect"])
            typed_effect_count += int(typed_effect)
            failure = failure_signature(
                {"method": "GET", "role": action_role, "candidate_signal": signal["candidate_signal"], "positive": False, "positive_authority": False, "typed_available": False, "probe_round": step_index, "max_probe_rounds": MAX_STEPS},
                prior_records=prior_records,
                max_steps=MAX_STEPS,
                step_count=step_index,
            )
            belief = _belief(signal, typed_effect, role)
            action_view = {
                "method": manifest["method"],
                "placement": manifest["placement"],
                "probe_kind": manifest["probe_kind"],
                "encoding_chain": manifest["encoding_chain"],
                "probe_ref": manifest["probe_ref"],
                "payload_sha256": manifest["payload_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "field_names": fields,
                "safety": manifest["safety"],
            }
            steps.append({"step_index": step_index, "model_action": predicted, "confidence": round(confidence, 6), "controller_decision": controller_decision, "action_manifest": action_view, "response_projection": projected["response_projection"], "oracle_projection": oracle, "typed_surface_effect": typed_effect, "failure_signature": failure, "belief_after": belief, "decision": "abstain", "vulnerability_claim_allowed": False, "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": manifest, "response_projection": projected["response_projection"], "failure_signature": failure, "belief_after": belief})
            prior_records.append({"method": "GET", "role": action_role, "candidate_signal": signal["candidate_signal"], "belief_after": belief})
    finally:
        client.close()
    return {"surface": surface, "path": path, "field_names": fields, "encoding": encoding_name, "target_instance_hash": target_hash, "step_count": len(steps), "sent_count": sum(int(item.get("controller_decision", "").startswith("send_")) for item in steps), "candidate_sent_count": sum(int(item.get("controller_decision") == "send_inert_dom_candidate") for item in steps), "typed_surface_effect_count": typed_effect_count, "controller_abstain_count": controller_abstain, "encoding_fallback_count": fallback_count, "typed_positive_count": 0, "vulnerability_claim_allowed": False, "steps": steps}


def main() -> int:
    routes = PG185._load_observed_routes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    container_id = PG185._start_container()
    runs: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    try:
        episode_index = 0
        for model_name, checkpoint_path in CHECKPOINTS:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
            variant = str(checkpoint["variant"])
            model = build_model(len(vocabulary), variant).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            model_runs: list[dict[str, Any]] = []
            for route in routes:
                for encoding_name, encoding_chain in ENCODING_PLANS:
                    if episode_index:
                        _restart_and_wait()
                    target_hash = hashlib.sha256(f"{container_id}:{episode_index}:{model_name}:{route['surface']}:{encoding_name}".encode("utf-8")).hexdigest()
                    episode = _replay_episode(model, vocabulary, route, device, encoding_name=encoding_name, encoding_chain=encoding_chain, target_hash=target_hash)
                    episode["model"] = model_name
                    episode["checkpoint_sha256"] = checkpoint_hash
                    model_runs.append(episode)
                    runs.append(episode)
                    episode_index += 1
            checkpoint_summaries.append({"model": model_name, "variant": variant, "checkpoint_sha256": checkpoint_hash, "parameter_count": int(sum(p.numel() for p in model.parameters())), "episode_count": len(model_runs), "sent_count": sum(r["sent_count"] for r in model_runs), "candidate_sent_count": sum(r["candidate_sent_count"] for r in model_runs), "typed_surface_effect_count": sum(r["typed_surface_effect_count"] for r in model_runs), "controller_abstain_count": sum(r["controller_abstain_count"] for r in model_runs), "encoding_fallback_count": sum(r["encoding_fallback_count"] for r in model_runs), "typed_positive_count": 0})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        PG185._stop_container()

    report = {
        "protocol_id": "pg-pk-186-pikachu-dom-capacity-encoding-v1",
        "schema_version": "pg186-pikachu-dom-capacity-encoding-report-v1",
        "status": "completed_frozen_capacity_seed_encoding_replay",
        "source": {"pg185_report": "research/pg185_pikachu_dom_replay_report_v1.json", "image": PG185.IMAGE, "loopback_port": PG185.PORT, "fresh_restart_per_episode": True, "route_count": len(routes), "encoding_plans": [name for name, _ in ENCODING_PLANS]},
        "device": str(device),
        "model_summaries": checkpoint_summaries,
        "counts": {"episode_count": len(runs), "sent_count": sum(r["sent_count"] for r in runs), "candidate_sent_count": sum(r["candidate_sent_count"] for r in runs), "typed_surface_effect_count": sum(r["typed_surface_effect_count"] for r in runs), "typed_positive_count": 0, "controller_abstain_count": sum(r["controller_abstain_count"] for r in runs), "encoding_fallback_count": sum(r["encoding_fallback_count"] for r in runs)},
        "runs": runs,
        "selection": {"selected_variant": None, "capacity_claim_allowed": False, "training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "frozen evaluation only; typed DOM effect is not XSS execution"},
        "safety": {"loopback_only": True, "external_network": False, "fresh_container": True, "inert_dom_markup_only": True, "script_execution": False, "database_write": False, "credentials": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg186-pikachu-dom-capacity-encoding-trace-v1", "evaluation_only": True, "training_eligible": False, "runs": runs, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False})
    protocol = {"protocol_id": "pg-pk-186-pikachu-dom-capacity-encoding-v1", "schema_version": "pg186-pikachu-dom-capacity-encoding-protocol-v1", "checkpoints": [name for name, _ in CHECKPOINTS], "encoding_plans": [name for name, _ in ENCODING_PLANS], "model_output_allowlist": ["baseline", "matched_control", "safe_candidate", "abstain"], "manifest_validator_before_send": True, "frozen_checkpoints": True, "fresh_restart_per_episode": True, "typed_dom_effect_not_vulnerability": True, "unknown_or_unseen_encoding_fallback_is_counted": True, "gates": {"loopback_only": True, "inert_dom_markup_only": True, "script_execution": False, "training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-186 Pikachu DOM capacity × encoding replay", "", f"models={len(checkpoint_summaries)}; episodes={len(runs)}; sent={report['counts']['sent_count']}; candidates={report['counts']['candidate_sent_count']}; typed_surface_effects={report['counts']['typed_surface_effect_count']}", "", "冻结 small/medium/MoE、双 seed 模型在只读 GET 表面上复放多编码 inert DOM 探针；不训练目标 trace，不把 DOM effect 当漏洞阳性。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "models": len(checkpoint_summaries), "episodes": len(runs), "sent_count": report["counts"]["sent_count"], "candidate_sent_count": report["counts"]["candidate_sent_count"], "typed_surface_effect_count": report["counts"]["typed_surface_effect_count"], "typed_positive_count": 0, "training_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
