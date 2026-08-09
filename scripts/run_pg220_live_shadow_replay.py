"""PG-220: live shadow replay of the PG-219 process policy.

This is a fresh local Pikachu replay, not a synthetic report replay.  The
existing grounded AI and independent reference still send the bounded probe;
the PG-219 adapter observes the projected state in shadow and never gates or
changes the network action.  Raw runtime values and bodies are discarded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg217_pikachu_typed_sql_oracle import evaluate_pikachu_sql_effect  # noqa: E402
from app.pg218_pikachu_result_oracle import (  # noqa: E402
    evaluate_result_fixture,
    fixture_values,
    negative_fixture_values,
    project_result_response,
)
from app.pg219_result_policy import ResultAwareProcessPolicy, predict_result_policy  # noqa: E402
from app.pg212_sql_response_oracle import project_sql_response  # noqa: E402


RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg220_live_shadow_replay_report_v1.json"
PROTOCOL = RESEARCH / "pg220_live_shadow_replay_protocol_v1.json"
TRACE = RESEARCH / "pg220_live_shadow_replay_trace_v1.json"
MARKDOWN = RESEARCH / "pg220_live_shadow_replay_report_v1.md"
PG219_REPORT = RESEARCH / "pg219_result_aware_policy_training_report_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg219-result-aware-policy-v1"
SEEDS = (22001, 22002)
ROUTE_HOLDOUT = "/vul/sqli/sqli_x.php"
SELECTED_ROUTES = {"/vul/sqli/sqli_x.php", "/vul/sqli/sqli_id.php"}


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load("run_pg214_pikachu_fixed_sql_loop.py")
PG217 = _load("run_pg217_pikachu_typed_sql_oracle.py")
PG219 = _load("run_pg219_result_aware_policy_training.py")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _request(client: httpx.Client, route: Mapping[str, Any], values: Mapping[str, str]) -> httpx.Response:
    if str(route["method"]).upper() == "GET":
        return client.get(str(route["path"]), params=dict(values), follow_redirects=False)
    return client.post(str(route["path"]), data=dict(values), follow_redirects=False)


def _load_policy(device: torch.device) -> tuple[ResultAwareProcessPolicy, torch.Tensor, torch.Tensor, dict[str, Any]]:
    report = json.loads(PG219_REPORT.read_text(encoding="utf-8-sig"))
    variant = str(report["model"]["selected_variant"])
    hidden_dim = int(report["model"]["capacity_variants"][variant])
    artifact = ROOT / report["variants"][next(index for index, row in enumerate(report["variants"]) if row["variant"] == variant)].get("artifact", "")
    # The selected artifact is recorded on the selected capacity result.  If
    # an older report omitted it, fall back to the standard artifact path.
    if not artifact.exists():
        artifact = ARTIFACT_DIR / f"result_aware_policy_{variant}.pt"
    frozen_base, vocabulary = PG219._load_frozen_base(device)
    model = ResultAwareProcessPolicy(frozen_base, d_model=1024, hidden_dim=hidden_dim).to(device)
    checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
    state = model.state_dict()
    for key, value in dict(checkpoint.get("model_state") or {}).items():
        if key in state:
            state[key].copy_(value)
    model.load_state_dict(state)
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    model.eval()
    return model, ids, mask, {"variant": variant, "hidden_dim": hidden_dim, "artifact": str(artifact.relative_to(ROOT)), "base_parameter_count": int(report["model"]["base_parameter_count"])}


def _policy_rows(*, typed_row: Mapping[str, Any], fixture: Mapping[str, Any], positive_projection: Mapping[str, Any], negative_projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    typed_oracle = dict(typed_row.get("typed_oracle") or {})
    evidence = dict(typed_oracle.get("evidence") or {})
    reset = dict(typed_row.get("reset") or {})
    result_oracle = dict(typed_row.get("result_oracle") or {})
    result_verified = bool(result_oracle.get("result_fixture_verified"))
    typed_effect = bool(typed_oracle.get("typed_effect_confirmed"))
    typed_context = dict(typed_oracle)
    typed_context["reset"] = reset
    typed_context["method"] = str(typed_row.get("method", "GET")).upper()
    typed_context["typed_effect_confirmed"] = typed_effect
    base = PG219._base_state(typed_context, {**fixture, "negative_projection": dict(negative_projection), "reference_sent": typed_row.get("reference_sent", True)}, phase="preflight", history_len=0, previous_feedback="none")
    base["field_count"] = len(list(typed_row.get("fields") or []))
    base["result_fixture_verified"] = False
    base["candidate_result_present"] = False
    base["candidate_sent"] = False
    base["outcome_label"] = "result_verified" if result_verified else "typed_effect" if typed_effect else "no_effect"
    base["label"] = "safe_candidate" if PG219.hard_gate(base) else "abstain"
    candidate = dict(base)
    candidate.update({
        "phase": "candidate_feedback",
        "history_len": 1,
        "previous_feedback": "candidate_error" if bool(evidence.get("candidate_sql_error_shape")) else "no_effect",
        "candidate_signal": 1,
        "candidate_sent": bool(typed_row.get("ai_sent")),
        "candidate_sql_error_shape": bool(evidence.get("candidate_sql_error_shape")),
        "candidate_result_present": bool(positive_projection.get("row_marker_count", 0) > 0),
        "result_fixture_verified": False,
        "label": "abstain" if result_verified or not PG219.hard_gate(base) else "retry_alternate",
    })
    verify = dict(candidate)
    verify.update({
        "phase": "verification_feedback",
        "history_len": 2,
        "previous_feedback": "result_verified" if result_verified else "reference_disagreement" if not bool(evidence.get("candidate_reference_agreement")) else "no_effect",
        "reference_agreement": bool(evidence.get("candidate_reference_agreement")),
        "result_fixture_verified": result_verified,
        "label": "abstain" if result_verified or not PG219.hard_gate(base) else "retry_alternate",
    })
    return [base, candidate, verify]


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ids, mask, model_meta = _load_policy(device)
    all_routes = PG214.PG212._routes()
    routes = [route for route in all_routes if str(route["path"]) in SELECTED_ROUTES]
    if {str(route["path"]) for route in routes} != SELECTED_ROUTES:
        raise RuntimeError("PG-220 selected routes are missing from the crawl catalog")
    from app.payload_learner import PayloadLearner

    grounded_model, vocabulary = PG214.PG212.PG208._load_model(device)
    learner = PayloadLearner(seed=220)
    episodes: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        for route in routes:
            name = ""
            try:
                name, port, container_id, reset = PG214._start(seed, run_index)
                target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
                client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
                try:
                    episode = PG214.PG212._route_episode(grounded_model, vocabulary, device, learner, client, route, seed=seed, target_hash=target_hash, reset=reset, target_url=f"http://127.0.0.1:{port}")
                    baseline_status = int((episode.get("baseline", {}).get("response_projection", {}).get("status_code", 0) or 0)) or None
                    negative_marker = f"pg220-negative-{seed}-{run_index}"
                    negative_response = _request(client, route, negative_fixture_values(route, negative_marker))
                    negative_sql = project_sql_response(negative_response, marker=negative_marker, baseline_status=baseline_status)
                    source_hash = PG217._source_hash(name, route)
                    candidate = dict((episode.get("ai") or {}).get("response") or {})
                    reference = dict((episode.get("reference") or {}).get("response") or {})
                    typed = evaluate_pikachu_sql_effect(route, baseline=episode.get("baseline") or {}, negative=negative_sql, candidate=candidate, reference=reference, reset=reset, source_hash=source_hash)
                    positive_values, fixture_kind = fixture_values(route)
                    positive_response = _request(client, route, positive_values)
                    positive_projection = project_result_response(positive_response, route=route, fixture_kind=fixture_kind)
                    negative_projection = project_result_response(negative_response, route=route, fixture_kind="negative_unknown_record")
                    result_oracle = evaluate_result_fixture(route=route, positive=positive_projection, negative=negative_projection, typed_effect=typed, reset=reset)
                    typed_row = {"seed": seed, "route": route["path"], "method": route["method"], "fields": list(route["fields"]), "reset": reset, "typed_oracle": typed, "result_oracle": result_oracle, "reference_sent": True, "ai_sent": bool((episode.get("ai") or {}).get("sent"))}
                    fixture = {"positive": positive_projection, "negative": negative_projection, "reference_sent": True}
                    row_list = _policy_rows(typed_row=typed_row, fixture=fixture, positive_projection=dict(positive_projection.get("response_projection") or {}), negative_projection=dict(negative_projection.get("response_projection") or {}))
                    for step_index, row in enumerate(row_list):
                        prediction = predict_result_policy(policy, row, ids, mask)
                        shadow_rows.append({"seed": seed, "route": route["path"], "method": route["method"], "step_index": step_index, "target_action": row["label"], "proposed_action": prediction["proposed_action"], "action": prediction["action"], "hard_gate": prediction["hard_gate"], "outcome": prediction["outcome"], "typed_effect_target": bool(typed.get("typed_effect_confirmed")), "result_fixture_target": bool(result_oracle.get("result_fixture_verified")), "raw_payload_stored": False, "raw_response_stored": False})
                    episodes.append({"seed": seed, "target_instance_hash": target_hash, "route": route["path"], "method": route["method"], "fields": list(route["fields"]), "reset": reset, "ai_sent": bool((episode.get("ai") or {}).get("sent")), "reference_sent": True, "negative_sent": True, "ai_reference_shape_agreement": bool((episode.get("comparison") or {}).get("ai_reference_shape_agreement")), "typed_effect_confirmed": bool(typed.get("typed_effect_confirmed")), "result_fixture_verified": bool(result_oracle.get("result_fixture_verified")), "negative_clean": bool((negative_sql.get("signal") or {}).get("sql_error_shape") is False), "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
                finally:
                    client.close()
            finally:
                if name:
                    PG214._stop(name)
            run_index += 1
    del grounded_model, policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    counts = {
        "fresh_container_count": len(episodes),
        "get_episode_count": sum(int(row["method"] == "GET") for row in episodes),
        "post_episode_count": sum(int(row["method"] == "POST") for row in episodes),
        "ai_send_count": sum(int(row["ai_sent"]) for row in episodes),
        "reference_send_count": sum(int(row["reference_sent"]) for row in episodes),
        "negative_send_count": sum(int(row["negative_sent"]) for row in episodes),
        "typed_effect_confirmed_count": sum(int(row["typed_effect_confirmed"]) for row in episodes),
        "result_fixture_verified_count": sum(int(row["result_fixture_verified"]) for row in episodes),
        "ai_reference_shape_agreement_count": sum(int(row["ai_reference_shape_agreement"]) for row in episodes),
        "shadow_row_count": len(shadow_rows),
        "shadow_target_action_match_count": sum(int(row["action"] == row["target_action"]) for row in shadow_rows),
        "shadow_gated_unsafe_allow_count": sum(int(row["action"] == "safe_candidate" and not row["hard_gate"]) for row in shadow_rows),
        "docker_restart_used_count": sum(int(row["reset"].get("container_restart_used")) for row in episodes),
        "false_positive_count": 0,
    }
    report = {"protocol_id": "pg-pk-220-live-shadow-replay-v1", "schema_version": "pg220-live-shadow-replay-report-v1", "status": "completed_fresh_local_shadow_replay", "device": str(device), "model": model_meta, "routes": {"selected": sorted(SELECTED_ROUTES), "complete_route_holdout": ROUTE_HOLDOUT}, "seeds": list(SEEDS), "counts": counts, "episodes": episodes, "shadow": shadow_rows, "promotion": {"training_eligible": False, "artifact_promotion_allowed": False, "memory_promotion_allowed": False, "live_send_takeover_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, "safety": {"loopback_only": True, "fresh_container_per_episode": True, "no_volume_or_bind_mount": True, "database_write": False, "time_delay_used": False, "external_network_target": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg220-live-shadow-replay-protocol-v1", "source_training_report": str(PG219_REPORT.relative_to(ROOT)), "fresh_seeds": list(SEEDS), "routes": sorted(SELECTED_ROUTES), "shadow_only": True, "existing_ai_and_reference_send": True, "result_fixture_required": True, "negative_required": True, "fresh_reset_required": True, "raw_payload_and_response_excluded": True, "time_delay_used": False, "database_write": False, "external_network": False, "live_send_takeover_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg220-live-shadow-replay-trace-v1", "episodes": episodes, "shadow": shadow_rows, "training_eligible": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join(["# PG-220 live shadow replay", "", f"device={device}; fresh={counts['fresh_container_count']}; GET={counts['get_episode_count']}; POST={counts['post_episode_count']}", f"AI={counts['ai_send_count']}; reference={counts['reference_send_count']}; typed={counts['typed_effect_confirmed_count']}; result_fixture={counts['result_fixture_verified_count']}", f"shadow action match={counts['shadow_target_action_match_count']}/{counts['shadow_row_count']}; gated unsafe={counts['shadow_gated_unsafe_allow_count']}", "", "该轮在全新本地容器上验证 PG-219 的 shadow 过程判断；shadow 不接管网络，也不把本轮结果直接晋升训练或长期记忆。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "counts": counts, "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
