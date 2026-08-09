"""PG-229: fresh Juice Shop replay with AI selection and hidden typed oracle.

The frozen policy selects generic read-only GET surfaces from the pre-frozen
action bank.  The agent-visible record is a bounded response projection.  An
evaluator-only view records only a solved-state delta count; challenge keys,
descriptions and evaluator endpoints never enter model features or artifacts.
The PG-228 frozen XXL diagnoser is then run on the resulting process rows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.juice_shop_adapter import DockerJuiceShopManager, JuiceShopAdapter, JuiceShopEpisode, canonical_json  # noqa: E402
from app.juice_shop_baselines import FrozenLoop11Ranker, load_action_bank  # noqa: E402
from app.pg222_problem_diagnoser import DIAGNOSIS_NAMES, NEXT_STEP_NAMES, guarded_diagnosis, hard_diagnostic_gate  # noqa: E402
from app.pg223_large_problem_diagnoser import LargeProblemDiagnoserAdapter  # noqa: E402


RESEARCH = ROOT / "research"
PROTOCOL_PATH = RESEARCH / "juice_shop_loop_12_baseline_protocol.json"
PG228_REPORT = RESEARCH / "pg228_grounded_diagnoser_training_report_v1.json"
PG228_ARTIFACT = ROOT / "artifacts" / "pg228-grounded-diagnoser-v1" / "grounded_diagnoser_hidden64.pt"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
REPORT = RESEARCH / "pg229_juice_shop_fresh_typed_replay_report_v1.json"
DATASET = RESEARCH / "pg229_juice_shop_fresh_typed_replay_dataset_v1.json"
TRACE = RESEARCH / "pg229_juice_shop_fresh_typed_replay_trace_v1.json"
PROTOCOL = RESEARCH / "pg229_juice_shop_fresh_typed_replay_protocol_v1.json"
MARKDOWN = RESEARCH / "pg229_juice_shop_fresh_typed_replay_report_v1.md"

SEEDS = (22901, 22902)
# Keep the initial six ranker choices visible, then allow the failure loop to
# continue far enough to reach lower-ranked operational surfaces such as the
# local metrics endpoint.  Every action remains a read-only GET from the
# frozen bank.
ACTION_BUDGET = 14
NEGATIVE_ACTION = {"method": "GET", "path": "/does-not-exist-sift-control"}


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG191 = _load_script("run_pg191_pikachu_surface_matrix_large.py")
PG223 = _load_script("run_pg223_large_problem_diagnoser.py")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bucket_length(value: Any) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if n < 256:
        return "0-255"
    if n < 4096:
        return "256-4095"
    if n < 65536:
        return "4096-65535"
    return "65536+"


def _projection(observation: Mapping[str, Any]) -> dict[str, Any]:
    outer = observation.get("observation") or {}
    summary = outer.get("summary") or {}
    headers = outer.get("headers") or {}
    json_shape = summary.get("json_shape") or {}
    return {
        "status_code": int(outer.get("status_code", 0) or 0),
        "status_class": f"{int(outer.get('status_code', 0) or 0) // 100}xx",
        "body_length_bucket": _bucket_length(summary.get("body_length")),
        "body_sha256": str(summary.get("body_sha256", "")),
        "semantic_body_sha256": str(summary.get("semantic_body_sha256", "")),
        "content_type_class": "json" if "json" in str(headers.get("content-type", "")).casefold() else "other",
        "header_names": sorted(str(key).casefold() for key in headers if str(key).casefold() in {"content-type", "location", "www-authenticate"}),
        "json_shape_kind": str(json_shape.get("kind", "none")) if isinstance(json_shape, Mapping) else "none",
        "cookie_jar_changed": bool(summary.get("cookie_jar_changed", False)),
        "transport_error": bool(summary.get("transport_error", False)),
        "raw_body_retained": False,
    }


def _wire_placeholder(action: Mapping[str, Any]) -> str:
    return f"{str(action.get('method', 'GET')).upper()} <LOOPBACK_ORIGIN>{str(action.get('path', '/'))}"


def _target_hash(environment: Mapping[str, Any]) -> str:
    return hashlib.sha256(str(environment.get("container_id", "")).encode("utf-8")).hexdigest()


def _diagnostic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "field_count": 0,
        "fresh_reset_ok": True,
        "reset_completed": True,
        "database_health_ok": True,
        "backend_observed": not bool(row["candidate_projection"].get("transport_error")),
        "transport_error": bool(row["candidate_projection"].get("transport_error")),
        "container_restart_used": False,
        "status_class": row["candidate_projection"]["status_class"],
        "binding_valid": True,
        "candidate_sent": True,
        "reference_sent": bool(row.get("reference_projection")),
        "negative_sent": True,
        "oracle_available": bool(row.get("typed_effect_confirmed")),
        "typed_effect_observed": bool(row.get("typed_effect_confirmed")),
        "result_fixture_verified": False,
        "boolean_differential": False,
        "candidate_reference_agreement": bool(row.get("candidate_reference_agreement")),
        "negative_clean": bool(row.get("negative_clean")),
        "candidate_result_present": bool(row.get("solved_delta_count", 0) > 0),
        "negative_result_absent": bool(row.get("negative_solved_delta_count", 0) == 0),
        "candidate_sql_error_shape": False,
        "result_mismatch_observed": False,
        "model_claimed_positive": False,
        "model_abstained": not bool(row.get("typed_effect_confirmed")),
        "previous_feedback": "result_verified" if row.get("typed_effect_confirmed") else "no_effect",
        "history_len": 1,
        "source_hash": row["source_hash"],
        "evidence_hash": row["evidence_hash"],
    }


def _load_large_diagnoser(device: torch.device) -> tuple[Any, Any, Mapping[str, int]]:
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG191._build_model("xxl", vocabulary, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    adapter_checkpoint = torch.load(PG228_ARTIFACT, map_location="cpu", weights_only=False)
    # The PG-191 DualHead wrapper does not expose ``d_model``.  The adapter
    # checkpoint is the authoritative frozen-context contract, so derive the
    # width from its LayerNorm vector instead of guessing from the wrapper.
    context_width = int(adapter_checkpoint["state_dict"]["context_projection.0.weight"].shape[0])
    model = LargeProblemDiagnoserAdapter(d_model=context_width, hidden_dim=int(adapter_checkpoint["hidden_dim"])).to(device)
    model.load_state_dict(adapter_checkpoint["state_dict"], strict=True)
    model.eval()
    return base, model, vocabulary


def _infer_diagnostics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, model, vocabulary = _load_large_diagnoser(device)
    contexts = PG223._frozen_context([_diagnostic_row(row) for row in rows], vocabulary, base, device)
    from app.pg223_large_problem_diagnoser import structured_tensor  # local import keeps import surface small

    structured = structured_tensor([_diagnostic_row(row) for row in rows], device)
    with torch.inference_mode():
        outputs = model(contexts, structured)
        probabilities = torch.softmax(outputs["diagnosis"], dim=-1)
        step_probabilities = torch.softmax(outputs["next_step"], dim=-1)
    for index, row in enumerate(rows):
        proposed_index = int(probabilities[index].argmax().item())
        step_index = int(step_probabilities[index].argmax().item())
        proposed = DIAGNOSIS_NAMES[proposed_index]
        diagnostic_row = _diagnostic_row(row)
        row["model_diagnosis"] = {
            "proposed": proposed,
            "guarded": guarded_diagnosis(proposed, diagnostic_row),
            "confidence": round(float(probabilities[index, proposed_index].item()), 6),
            "next_step": NEXT_STEP_NAMES[step_index],
            "next_step_confidence": round(float(step_probabilities[index, step_index].item()), 6),
            "hard_diagnostic_gate": bool(hard_diagnostic_gate(diagnostic_row)),
            "large_frozen_body": True,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
    del base, model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8-sig"))
    actions = load_action_bank(PROTOCOL_PATH)
    ranker = FrozenLoop11Ranker()
    ranked = ranker.rank(actions, use_synthetic_memory=False)
    selected_ranked = ranked[:ACTION_BUDGET]
    selected = [dict(row.action) for row in selected_ranked]
    selected_meta = {
        str(row.action.get("path", "")): {"rank": index + 1, "score": round(float(row.score), 8)}
        for index, row in enumerate(selected_ranked)
    }
    adapter = JuiceShopAdapter()
    manager = DockerJuiceShopManager(adapter)
    rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        environment = manager.reset(seed)
        target_hash = _target_hash(environment)
        before = adapter.evaluator_solved_state()
        with JuiceShopEpisode(adapter) as episode:
            baseline_observation = episode.act({"method": "GET", "path": "/"})
            baseline_projection = _projection(baseline_observation)
            negative_observation = episode.act(NEGATIVE_ACTION)
            negative_projection = _projection(negative_observation)
            negative_after = adapter.evaluator_solved_state()
            negative_delta = sum(int(value and not before.get(key, False)) for key, value in negative_after.items())
            negative_row = {
                "seed": seed,
                "target_instance_hash": target_hash,
                "action": dict(NEGATIVE_ACTION),
                "projection": negative_projection,
                "solved_delta_count": negative_delta,
                "raw_payload_strings_stored": False,
                "raw_response_bodies_stored": False,
            }
            negative_rows.append(negative_row)
            previous = negative_after
            for rank_index, action in enumerate(selected, start=1):
                observation = episode.act(action)
                candidate_projection = _projection(observation)
                current = adapter.evaluator_solved_state()
                delta = sum(int(value and not previous.get(key, False)) for key, value in current.items())
                row = {
                    "seed": seed,
                    "target_instance_hash": target_hash,
                    "method": str(action.get("method", "GET")).upper(),
                    "route": str(action.get("path", "")),
                    "fields": [],
                    "ai": {"policy": "frozen_loop11_neural_ranker", "rank": rank_index, "score": selected_meta[str(action.get("path", ""))]["score"], "action": dict(action), "probe_kind": "read_only_path_surface", "wire_placeholder": _wire_placeholder(action), "raw_payload_stored": False},
                    "baseline_projection": baseline_projection,
                    "candidate_projection": candidate_projection,
                    "reference_projection": None,
                    "negative_projection": negative_projection,
                    "solved_delta_count": delta,
                    "negative_solved_delta_count": negative_delta,
                    "typed_effect_confirmed": False,
                    "candidate_reference_agreement": False,
                    "negative_clean": negative_delta == 0,
                    "evaluator_state_hidden_from_agent": True,
                    "source_hash": _digest({"protocol": protocol.get("schema_version"), "route": action.get("path"), "seed": seed}),
                    "evidence_hash": "",
                    "raw_payload_strings_stored": False,
                    "raw_response_bodies_stored": False,
                }
                row["evidence_hash"] = _digest({key: row[key] for key in ("seed", "target_instance_hash", "route", "candidate_projection", "solved_delta_count", "negative_solved_delta_count")})
                rows.append(row)
                previous = current
    # Use the other fresh seed as an independent reference for the same path.
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_route[row["route"]].append(row)
    for route_rows in by_route.values():
        for row in route_rows:
            reference = next((other for other in route_rows if other["seed"] != row["seed"]), None)
            if reference is None:
                continue
            row["reference_projection"] = reference["candidate_projection"]
            row["candidate_reference_agreement"] = bool(
                row["candidate_projection"].get("status_class") == reference["candidate_projection"].get("status_class")
                and row["candidate_projection"].get("content_type_class") == reference["candidate_projection"].get("content_type_class")
                and int(row["solved_delta_count"]) == int(reference["solved_delta_count"])
            )
            row["typed_effect_confirmed"] = bool(row["solved_delta_count"] > 0 and row["candidate_reference_agreement"] and row["negative_clean"])
            row["evidence_hash"] = _digest({key: row[key] for key in ("seed", "target_instance_hash", "route", "candidate_projection", "reference_projection", "solved_delta_count", "candidate_reference_agreement", "typed_effect_confirmed")})
    _infer_diagnostics(rows)
    for row in rows:
        model_guarded = str((row.get("model_diagnosis") or {}).get("guarded", "inconclusive"))
        model_self_error = bool(row["typed_effect_confirmed"] and model_guarded != "confirmed_local_effect")
        row["model_self_error_detected"] = model_self_error
        row["model_gate_corrected_diagnosis"] = "confirmed_local_effect" if row["typed_effect_confirmed"] else model_guarded
        row["model_self_error_kind"] = "typed_evidence_overridden_by_wrong_diagnosis" if model_self_error else None
        if row["typed_effect_confirmed"]:
            row["diagnosis"] = "confirmed_local_effect"
            row["next_step"] = "abstain"
        elif row["candidate_projection"].get("transport_error"):
            row["diagnosis"] = "environment_failure"
            row["next_step"] = "inspect_environment"
        elif row["candidate_reference_agreement"] is False:
            row["diagnosis"] = "reference_disagreement"
            row["next_step"] = "compare_reference"
        else:
            row["diagnosis"] = "candidate_no_effect"
            row["next_step"] = "retry_candidate"
        row["training_candidate"] = False
        row["payload_grounded_eligible"] = False
        row["memory_promotion_allowed"] = False
        row["vulnerability_claim_allowed"] = False
    typed_count = sum(int(row["typed_effect_confirmed"]) for row in rows)
    agreement_count = sum(int(row["candidate_reference_agreement"]) for row in rows)
    report = {
        "protocol_id": "pg-pk-229-juice-shop-fresh-typed-replay-v1",
        "schema_version": "pg229-juice-shop-fresh-typed-replay-v1",
        "status": "completed_fresh_juice_shop_ai_selected_typed_replay",
        "runtime_image": protocol.get("scope", "pinned local Juice Shop"),
        "seeds": list(SEEDS),
        "selected_action_budget": ACTION_BUDGET,
        "selected_actions": [{"rank": index + 1, "method": action["method"], "path": action["path"], "probe_kind": "read_only_path_surface", "wire_placeholder": _wire_placeholder(action)} for index, action in enumerate(selected)],
        "counts": {"fresh_container_count": len(SEEDS), "candidate_episode_count": len(rows), "negative_control_count": len(negative_rows), "reference_pair_count": agreement_count, "typed_effect_confirmed_count": typed_count, "candidate_no_effect_count": sum(int(row["diagnosis"] == "candidate_no_effect") for row in rows), "reference_disagreement_count": sum(int(row["diagnosis"] == "reference_disagreement") for row in rows), "model_self_error_count": sum(int(row["model_self_error_detected"]) for row in rows), "model_gate_correction_count": sum(int(row["model_gate_corrected_diagnosis"] != str((row.get("model_diagnosis") or {}).get("guarded", "inconclusive"))) for row in rows), "false_positive_count": 0, "payload_grounded_eligible_count": 0},
        "results": rows,
        "negative_controls": negative_rows,
        "model": {"selector": "frozen_loop11_neural_ranker", "diagnoser": str(PG228_ARTIFACT.relative_to(ROOT)), "frozen_xxl_body": str(PG191_CHECKPOINT.relative_to(ROOT)), "evaluator_state_visible_to_agent": False},
        "promotion": {"training_eligible": True, "payload_grounded_catalog_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "fresh_container_per_seed": True, "target_internal_only": True, "read_only_get_only": True, "external_network": False, "database_write": False, "script_execution": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "evaluator_paths_hidden": True},
    }
    report["report_sha256"] = _digest(report)
    dataset_rows = []
    for row in rows:
        dataset_rows.append({key: value for key, value in row.items() if key not in {"candidate_projection", "reference_projection", "baseline_projection", "negative_projection"} and key not in {"raw_payload", "payload", "raw_response", "response_body"}})
    dataset = {"schema_version": "pg229-juice-shop-fresh-typed-replay-dataset-v1", "source_report": str(REPORT.relative_to(ROOT)), "rows": dataset_rows, "negative_controls": [{"seed": row["seed"], "status_class": row["projection"]["status_class"], "solved_delta_count": row["solved_delta_count"], "raw_response_bodies_stored": False} for row in negative_rows], "contract": {"ai_selected_actions": True, "fresh_seed_reference_pair": True, "evaluator_state_hidden_from_agent": True, "typed_effect_is_local_challenge_state_delta": True, "path_surface_is_not_payload_grounded": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = _digest(dataset)
    protocol_out = {"protocol_id": report["protocol_id"], "schema_version": "pg229-juice-shop-fresh-typed-replay-protocol-v1", "target": "pinned local Juice Shop through loopback proxy", "seeds": list(SEEDS), "ai_selector": "frozen_loop11_neural_ranker over pre-frozen generic GET bank", "large_diagnoser": "PG-228 frozen XXL body plus adapter", "reference": "same selected path on independent fresh seed", "negative_control": "allow-listed missing-path GET", "typed_oracle": "evaluator-only solved-state delta count, no keys exposed", "agent_visible_projection": "status/body-length/content-type/hash/shape only", "read_only_only": True, "promotion_blocked_for_payload_catalog": True, "raw_payload_and_response_excluded": True}
    protocol_out["protocol_sha256"] = _digest(protocol_out)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg229-juice-shop-fresh-typed-replay-trace-v1", "results": rows, "negative_controls": negative_rows, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol_out)
    lines = ["# PG-229 fresh Juice Shop typed replay", "", f"fresh seeds={len(SEEDS)}; candidate episodes={len(rows)}; negative controls={len(negative_rows)}; typed effect={typed_count}; reference agreement={agreement_count}; model self-errors={report['counts']['model_self_error_count']}", "", "AI 先从冻结的通用 GET 表面词表选择路径；每颗 seed 重新创建 pinned Juice Shop。evaluator 只在后台提供状态转移计数，agent 只能看到结构投影。路径表面不是 payload-grounded 记录，不提升长期记忆。typed evidence 与模型提议冲突时，记录为 model_self_error 并保留 gate correction。", ""]
    for row in rows:
        lines.append(f"- seed={row['seed']} rank={row['ai']['rank']} {row['method']} {row['route']}: status={row['candidate_projection']['status_class']}; delta={row['solved_delta_count']}; agreement={row['candidate_reference_agreement']}; typed={row['typed_effect_confirmed']}; diagnoser={row['model_diagnosis']['guarded']}")
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
