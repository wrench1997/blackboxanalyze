#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.juice_shop_adapter import DockerJuiceShopManager, EvidenceLedger, JuiceShopAdapter, JuiceShopEpisode  # noqa: E402
from app.active_probe import active_probe_score, choose_active_probe  # noqa: E402
from app.belief_state import MultiStepBelief  # noqa: E402
from app.rule_ir_binding import bind_rule_ir_slots, evidence_digest, shadow_evidence, validate_binding  # noqa: E402
from app.rule_ir_decoder import DECODER_FAMILIES, FEATURE_DIM, RuleIRDecoder, trace_feature_vector  # noqa: E402
from app.response_head import score_observation  # noqa: E402
from app.response_projection import ResponseProjection  # noqa: E402
from app.surface_discriminator import SurfaceDiscriminator  # noqa: E402
from train_neural_url_set_head import TinyRuleSetGPT  # noqa: E402
from run_juice_shop_shadow_replay import cleanup_shadow, shadow_probe, start_shadow  # noqa: E402


ROOT_PROTOCOL = ROOT / "research/juice_shop_loop_12_hidden_matrix_protocol_v6.json"
RUNS = ROOT / "research/juice_shop_loop_12_hidden_matrix_runs_v6.json"
CHECKPOINT = ROOT / "artifacts/neural-juice-loop-12-response-head-v3-20262113/tiny_rule_set_gpt.pt"
RULE_IR_CHECKPOINT = ROOT / "artifacts/rule-ir-decoder-loop-12-20260899-v4/rule_ir_decoder.pt"
SURFACE_CHECKPOINT = ROOT / "artifacts/surface-discriminator-loop-12-20260931/surface_discriminator.pt"
EVIDENCE_DIR = ROOT / "artifacts/juice-shop-loop-12/hidden-matrix"
POLICIES = {
    "trained_response_head_v3_rule_ir_bound_v4_belief_v2": {"seed": 20262179, "response_enabled": True, "selection": "belief_active"},
    "trained_response_head_v3_rule_ir_bound_v4_belief_v2_ablation": {"seed": 20262183, "response_enabled": False, "selection": "belief_active"},
}


def shadow_trace(path: str, raw: dict) -> dict:
    headers = dict(raw.get("headers") or {})
    return {
        "input": {
            "action": {"method": "GET", "path": path},
            "response": {
                "status_code": int(raw.get("status_code", 0) or 0),
                "content_type": headers.get("content-type", ""),
                "body_shape": raw.get("body_shape", "unknown"),
                "body_length": int(raw.get("body_length", 0) or 0),
                "transport_error": raw.get("transport_error", ""),
            },
        },
        "context": {},
        "state": {},
        "history": [],
        "output": int(raw.get("status_code", 0) or 0) // 100 == 2,
    }


def load_catalog() -> dict[str, dict[str, str]]:
    rows = json.loads((ROOT / "research/juice_shop_loop_12_catalog_v3.json").read_text(encoding="utf-8"))["challenges"]
    return {row["key"]: {"family": row["family"], "split": row["split"]} for row in rows}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=list(POLICIES))
    args = parser.parse_args()
    protocol = json.loads(ROOT_PROTOCOL.read_text(encoding="utf-8"))
    runs = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else {"schema_version": "sift-juice-shop-loop-12-hidden-matrix-runs-v5", "protocol": str(ROOT_PROTOCOL.relative_to(ROOT)), "runs": {}}
    if args.policy in runs["runs"]:
        raise RuntimeError("refusing to overwrite an existing matrix policy run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRuleSetGPT().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model_state"])
    model.eval()
    decoder_checkpoint = torch.load(RULE_IR_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(decoder_checkpoint.get("feature_dim", FEATURE_DIM)) != FEATURE_DIM:
        raise RuntimeError("Rule IR checkpoint feature dimension does not match the shadow decoder")
    decoder = RuleIRDecoder().to(device)
    decoder.load_state_dict(decoder_checkpoint["model_state"])
    decoder.eval()
    decoder_abstain_threshold = float(decoder_checkpoint.get("abstain_threshold", 0.55))
    decoder_mean = torch.tensor(decoder_checkpoint["normalisation_mean"], dtype=torch.float32, device=device)
    decoder_std = torch.tensor(decoder_checkpoint["normalisation_std"], dtype=torch.float32, device=device).clamp_min(1e-4)
    surface_checkpoint = torch.load(SURFACE_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(surface_checkpoint.get("feature_dim", FEATURE_DIM)) != FEATURE_DIM:
        raise RuntimeError("surface discriminator feature dimension does not match the shadow projection")
    surface_model = SurfaceDiscriminator().to(device)
    surface_model.load_state_dict(surface_checkpoint["model_state"])
    surface_model.eval()
    surface_mean = torch.tensor(surface_checkpoint["normalisation_mean"], dtype=torch.float32, device=device)
    surface_std = torch.tensor(surface_checkpoint["normalisation_std"], dtype=torch.float32, device=device).clamp_min(1e-4)
    surface_abstain_threshold = float(surface_checkpoint.get("abstain_threshold", 0.88))
    policy_config = POLICIES[args.policy]
    policy_seed = policy_config["seed"]
    response_enabled = policy_config["response_enabled"]
    selection_policy = policy_config["selection"]
    all_actions = []
    for family in protocol["families"]:
        all_actions.extend(protocol["action_banks"][family])
    unique_actions = list(dict.fromkeys(all_actions))
    shadow_rows: dict[str, dict] = {}
    try:
        shadow = start_shadow()
        pending_rows = []
        for path in unique_actions:
            raw = shadow_probe(path)
            observation = {"action": {"method": "GET", "path": path}, "observation": {"status_code": raw.get("status_code", 0), "headers": raw.get("headers", {}), "summary": {"body_length": raw.get("body_length", 0), "body_shape": raw.get("body_shape")}}}
            projection = ResponseProjection.from_observation(observation)
            score = score_observation(model, projection, device=device, enabled=response_enabled)
            pending_rows.append({
                "path": path,
                "raw": raw,
                "projection": projection.to_dict(),
                "model_score": score,
                "evidence": shadow_evidence({"method": "GET", "path": path}, raw, projection.to_dict()),
            })
        decoder_features = torch.tensor([trace_feature_vector([shadow_trace(row["path"], row["raw"])]) for row in pending_rows], dtype=torch.float32, device=device)
        decoder_outputs = decoder.decode((decoder_features - decoder_mean) / decoder_std, abstain_threshold=decoder_abstain_threshold)
        surface_probabilities = torch.softmax(surface_model((decoder_features - surface_mean) / surface_std), dim=-1).detach().cpu()
        for row, decoder_output, probability_row in zip(pending_rows, decoder_outputs, surface_probabilities):
            surface_values = {family: round(float(value), 6) for family, value in zip(DECODER_FAMILIES, probability_row)}
            surface_confidence = max(surface_values.values()) if surface_values else 0.0
            surface_family = max(surface_values, key=surface_values.get) if surface_values else None
            row["rule_ir_decoder"] = decoder_output
            row["surface_discriminator"] = {
                "candidate_family": surface_family,
                "confidence": round(surface_confidence, 6),
                "probabilities": surface_values,
                "accepted": surface_confidence >= surface_abstain_threshold,
            }
            row["active_probe_score"] = active_probe_score(row)
            row.pop("raw", None)
            shadow_rows[row["path"]] = row
    finally:
        cleanup_shadow()

    selected: list[dict] = []
    belief = MultiStepBelief() if selection_policy == "belief_active" else None
    belief_steps: list[dict] = []
    for family in protocol["families"]:
        candidates = [shadow_rows[path] for path in protocol["action_banks"][family]]
        if selection_policy == "belief_active" and belief is not None:
            chosen = belief.choose_next_probe(candidates)
            belief_step = belief.observe(chosen["path"], chosen["surface_discriminator"]["probabilities"], evidence_hash=evidence_digest(chosen["evidence"]))
            belief_steps.append(belief_step)
        elif selection_policy == "active_entropy":
            chosen = choose_active_probe(candidates)
        else:
            chosen = max(candidates, key=lambda row: (row["model_score"], -candidates.index(row)))
            belief_step = None
        if selection_policy == "active_entropy":
            belief_step = None
        decoder_output = chosen.get("rule_ir_decoder") or {}
        if decoder_output.get("rule_ir") is not None:
            binding = bind_rule_ir_slots(decoder_output["rule_ir"], chosen["evidence"])
        else:
            binding = {
                "schema_version": "sift-rule-ir-evidence-binding-v1",
                "status": "decoder_abstained",
                "abstract_rule_ir": None,
                "bindings": {},
                "bound_slot_count": 0,
                "slot_count": 0,
                "evidence": chosen["evidence"],
                "evidence_hash": evidence_digest(chosen["evidence"]),
                "evidence_hash_algorithm": "sha256-canonical-json",
                "executable": False,
            }
        validate_binding(binding)
        selected.append({"family": family, "chosen": chosen, "candidate_count": len(candidates), "rule_ir_binding": binding, "belief_step": belief_step})

    adapter = JuiceShopAdapter()
    environment = DockerJuiceShopManager(adapter).reset(policy_seed)
    before = adapter.evaluator_solved_state()
    previous = dict(before)
    evidence_path = EVIDENCE_DIR / f"{args.policy}.jsonl"
    action_results = []
    catalog = load_catalog()
    with JuiceShopEpisode(adapter, ledger=EvidenceLedger(evidence_path, ROOT)) as episode:
        for index, row in enumerate(selected, start=1):
            action = {"method": "GET", "path": row["chosen"]["path"]}
            observation = episode.act(action)
            current = adapter.evaluator_solved_state()
            transitions = [key for key, solved in current.items() if solved and not previous.get(key, False)]
            action_results.append({
                "order": index,
                "family_condition": row["family"],
                "action": action,
                "model_score": row["chosen"]["model_score"],
                "projection": row["chosen"]["projection"],
                "rule_ir_decoder": row["chosen"].get("rule_ir_decoder"),
                "surface_discriminator": row["chosen"].get("surface_discriminator"),
                "active_probe_score": row["chosen"].get("active_probe_score"),
                "belief_step": row["belief_step"],
                "rule_ir_binding": row["rule_ir_binding"],
                "status_code": observation["observation"]["status_code"],
                "all_transitions": transitions,
                "selected_hidden_transitions": [key for key in transitions if key in catalog and catalog[key]["split"] == "hidden_test"],
                "transition_families": {key: catalog[key]["family"] for key in transitions if key in catalog},
            })
            previous = current
    family_hits = {
        family: [row for row in action_results if row["family_condition"] == family and row["selected_hidden_transitions"]]
        for family in protocol["families"]
    }
    run = {
        "policy": args.policy,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "response_slots_enabled": response_enabled,
        "rule_ir_decoder_enabled": True,
        "rule_ir_abstain_threshold": decoder_abstain_threshold,
        "surface_discriminator_enabled": True,
        "surface_discriminator_abstain_threshold": surface_abstain_threshold,
        "selection_policy": selection_policy,
        "belief": belief.snapshot() if belief is not None else None,
        "belief_step_count": len(belief_steps),
        "shadow": shadow,
        "shadow_action_count": len(unique_actions),
        "shadow_rows": shadow_rows,
        "evaluation_environment": environment,
        "action_results": action_results,
        "family_hits": family_hits,
        "hidden_family_hit_count": sum(bool(rows) for rows in family_hits.values()),
        "evidence": str(evidence_path.relative_to(ROOT)),
        "batch_not_independent_episodes": True,
    }
    runs["runs"][args.policy] = run
    RUNS.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"policy": args.policy, "hidden_family_hit_count": run["hidden_family_hit_count"], "family_hits": {key: len(value) for key, value in family_hits.items()}, "evaluation_action_count": len(action_results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
