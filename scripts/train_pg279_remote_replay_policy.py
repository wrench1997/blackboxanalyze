"""PG-279 remote replay policy training and frozen PG-278 retention check.

The policy implementation is reused only as an architectural baseline; the
dataset, report namespace and retention audit are new.  The wrapper runs on
the authorized A800 GPU0, then evaluates the updated policy on the frozen
PG-278 holdout without exposing oracle fields as model input.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg279_remote_replay_dataset_v1.json"
DATASET_AUDIT = RESEARCH / "pg279_remote_replay_dataset_audit_v1.json"
REPLAY_DATASET = RESEARCH / "pg278_multifamily_question_dataset_v1.json"
REPLAY_AUDIT = RESEARCH / "pg278_multifamily_question_dataset_audit_v1.json"
MIX_DATASET = RESEARCH / "pg279_remote_replay_training_mix_v1.json"
MIX_AUDIT = RESEARCH / "pg279_remote_replay_training_mix_audit_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg279-remote-replay-policy"
CHECKPOINT = OUTPUT_DIR / "pg279_remote_replay_policies.pt"
REPORT = RESEARCH / "pg279_remote_replay_policy_report_v1.json"
TRACE = RESEARCH / "pg279_remote_replay_policy_trace_v1.json"
PROTOCOL = RESEARCH / "pg279_remote_replay_policy_protocol_v1.json"
MARKDOWN = RESEARCH / "pg279_remote_replay_policy_report_v1.md"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def retention_matrix(mod: Any, checkpoint: Path, old_dataset_path: Path, device: torch.device) -> dict[str, Any]:
    old = json.loads(old_dataset_path.read_text(encoding="utf-8"))
    old_report = json.loads((RESEARCH / "pg278_multifamily_question_policy_report_v1.json").read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = "enriched_process_sft:27911"
    state_info = dict(payload["states"][key])
    vocab = dict(state_info["vocabulary"])
    model = mod.Policy(len(vocab))
    model.load_state_dict(state_info["state"])
    model.to(device)
    model.eval()
    holdout = [row for row in old.get("records", []) if row.get("split") == "implementation_holdout"]
    after_impl = mod.metrics(model, holdout, vocab, "enriched", device)
    after_missing = mod.missing_metrics(model, holdout, vocab, device)
    before = old_report["aggregated"]["enriched_process_sft"]
    before_values = {
        "pre_transition_accuracy": float(before["implementation_holdout"]["pre_transition_accuracy"]["min"]),
        "post_transition_accuracy": float(before["implementation_holdout"]["post_transition_accuracy"]["min"]),
        "missing_safe_rate": float(before["missing_observation"]["safe_non_supported_rate"]["min"]),
    }
    after_values = {
        "pre_transition_accuracy": float(after_impl["pre_transition_accuracy"]),
        "post_transition_accuracy": float(after_impl["post_transition_accuracy"]),
        "missing_safe_rate": float(after_missing["safe_non_supported_rate"]),
    }
    deltas = {key_name: round(after_values[key_name] - before_values[key_name], 6) for key_name in before_values}
    checks = {key_name: after_values[key_name] >= before_values[key_name] - 0.10 for key_name in before_values}
    return {"frozen_dataset": "research/pg278_multifamily_question_dataset_v1.json", "frozen_report": "research/pg278_multifamily_question_policy_report_v1.json", "before_min": before_values, "after_min": after_values, "delta": deltas, "checks": checks, "status": "passed" if all(checks.values()) else "blocked"}


def build_replay_mix() -> dict[str, Any]:
    """Create a deterministic, audited training-only replay mix.

    PG-279 remains the evaluated dataset.  PG-278 contributes only its
    implementation-train rows so the model's vocabulary and representation
    retain the prior Rule-IR contracts.  No oracle/label field is added to
    model context by this mix.
    """
    current = json.loads(DATASET.read_text(encoding="utf-8"))
    current_audit = json.loads(DATASET_AUDIT.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY_DATASET.read_text(encoding="utf-8"))
    replay_audit = json.loads(REPLAY_AUDIT.read_text(encoding="utf-8"))
    if current_audit.get("status") != "passed" or replay_audit.get("status") != "passed":
        raise RuntimeError("both PG-279 and PG-278 dataset audits must pass before replay mixing")
    rows = [dict(row) for row in current.get("records", [])]
    replay_rows = []
    for original in replay.get("records", []):
        if original.get("split") != "implementation_train":
            continue
        row = dict(original)
        row["record_id"] = f"replay:{row['record_id']}"
        row["pair_id"] = f"replay:{row['pair_id']}"
        row["paired_opposite_record_id"] = f"replay:{row['paired_opposite_record_id']}"
        row["training_lane"] = "remote_controlled_replay_with_frozen_pg278_replay"
        replay_rows.append(row)
    rows.extend(replay_rows)
    # Start from the fully audited PG-279 envelope so the baseline trainer's
    # collision and contract checks remain active for the mixed run.
    mix = dict(current)
    mix.update({
        "schema_version": "pg279-remote-replay-training-mix-v1",
        "purpose": "Training-only replay mix; PG-279 holdout remains the new-track evaluation set.",
        "primary_dataset": {"path": DATASET.relative_to(ROOT).as_posix(), "sha256": current["dataset_sha256"], "audit_sha256": current_audit["audit_sha256"]},
        "replay_dataset": {"path": REPLAY_DATASET.relative_to(ROOT).as_posix(), "sha256": replay["dataset_sha256"], "audit_sha256": replay_audit["audit_sha256"], "rows_added": len(replay_rows)},
        "records": rows,
        "evaluation_contract": {"new_track_holdout": "primary PG-279 rows with split=implementation_holdout", "frozen_retention": "PG-278 implementation holdout evaluated after training", "replay_rows_training_only": True},
        "context_firewall": {"raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "oracle_in_context": False},
    })
    mix["counts"] = {"total": len(rows), "train": sum(row.get("split") == "implementation_train" for row in rows), "holdout": sum(row.get("split") == "implementation_holdout" for row in rows), "replay_train": len(replay_rows)}
    mix["dataset_sha256"] = sha(mix)
    MIX_DATASET.write_text(json.dumps(mix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "audit_id": "pg279-remote-replay-training-mix-audit-v1",
        "status": "passed",
        "primary_audit_sha256": current_audit["audit_sha256"],
        "replay_audit_sha256": replay_audit["audit_sha256"],
        "primary_rows": len(current.get("records", [])),
        "replay_train_rows": len(replay_rows),
        "holdout_rows": sum(row.get("split") == "implementation_holdout" for row in rows),
        "training_only_replay": True,
        "context_firewall": mix["context_firewall"],
        "interpretation": "The replay mix is a training aid for forgetting control; it does not turn controlled fixtures into real-application gold.",
    }
    audit["audit_sha256"] = sha(audit)
    MIX_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return mix


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    mix = build_replay_mix()
    mod = load_module(ROOT / "scripts" / "train_pg278_multifamily_question_policy.py", "pg279_policy_baseline")
    # Train on the audited PG-279 track plus replay-only PG-278 train rows;
    # the wrapper later restores PG-279 as the report's primary binding.
    mod.DATASET = MIX_DATASET
    mod.DATASET_AUDIT = MIX_AUDIT
    mod.OUTPUT_DIR = OUTPUT_DIR
    mod.CHECKPOINT = CHECKPOINT
    mod.REPORT = REPORT
    mod.TRACE = TRACE
    mod.PROTOCOL = PROTOCOL
    mod.MARKDOWN = MARKDOWN
    mod.MODEL_SEEDS = (27911, 27912, 27913)
    mod.main()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retention = retention_matrix(mod, CHECKPOINT, RESEARCH / "pg278_multifamily_question_dataset_v1.json", device)
    report["protocol_id"] = "pg279-remote-replay-policy-v1"
    report["schema_version"] = "pg279-remote-replay-policy-report-v1"
    report["status"] = "completed_remote_loopback_replay_policy_study"
    report["source"].update({"dataset": DATASET.relative_to(ROOT).as_posix(), "dataset_sha256": mix["primary_dataset"]["sha256"], "dataset_audit": DATASET_AUDIT.relative_to(ROOT).as_posix(), "dataset_audit_sha256": mix["primary_dataset"]["audit_sha256"], "remote_host": "112.111.7.91:60228", "remote_docker_available": False, "remote_replay_only": True, "real_application_gold_rows": 0})
    report["source"]["training_replay_mix"] = {"dataset": MIX_DATASET.relative_to(ROOT).as_posix(), "dataset_sha256": mix["dataset_sha256"], "audit": MIX_AUDIT.relative_to(ROOT).as_posix(), "replay_rows": mix["replay_dataset"]["rows_added"]}
    report["retention_matrix"] = retention
    report["hypothesis_gate"]["checks"]["frozen_retention_canary"] = retention["status"] == "passed"
    report["hypothesis_gate"]["checks"]["promotion_blocked"] = True
    report["hypothesis_gate"]["status"] = "passed" if all(report["hypothesis_gate"]["checks"].values()) else "blocked"
    report["hypothesis_gate"]["claim_allowed"] = "remote_controlled_replay_process_only" if report["hypothesis_gate"]["status"] == "passed" else False
    report["promotion"] = {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "PG-279 uses remote loopback fixtures, not a real application; real_application_gold_rows=0"}
    report["formal_conclusion"] = "Remote loopback GET/POST replay can supply real transport projections, paired failure-to-repair traces and a frozen PG-278 retention check. Replay-only mixing is required to prevent representation drift; the result is controlled replay evidence only, with no Pikachu/Docker or real-application vulnerability claim authorized."
    report.pop("report_sha256", None)
    report["report_sha256"] = sha(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    trace["schema_version"] = "pg279-remote-replay-policy-trace-v1"
    trace["source_dataset"] = "research/pg279_remote_replay_dataset_v1.json"
    trace["source_dataset_sha256"] = mix["primary_dataset"]["sha256"]
    trace["report_sha256"] = report["report_sha256"]
    trace["retention_matrix"] = retention
    trace.pop("trace_sha256", None)
    trace["trace_sha256"] = sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["protocol_id"] = "pg279-remote-replay-policy-v1"
    protocol["schema_version"] = "pg279-remote-replay-policy-protocol-v1"
    protocol["report_sha256"] = report["report_sha256"]
    protocol["retention_matrix"] = retention
    protocol["training_replay_mix"] = {"dataset": MIX_DATASET.relative_to(ROOT).as_posix(), "dataset_sha256": mix["dataset_sha256"], "audit": MIX_AUDIT.relative_to(ROOT).as_posix(), "replay_rows": mix["replay_dataset"]["rows_added"]}
    protocol["next_experiment"] = "PG-280 authorized real application/Pikachu replay when a remote Docker target is available; retain the same GET/POST failure-repair, replay-mix and frozen-canary gates"
    protocol.pop("protocol_sha256", None)
    protocol["protocol_sha256"] = sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-279 远程 GET/POST 回放—失败修复与遗忘矩阵", "", f"gate={report['hypothesis_gate']['status']}", f"retention={retention['status']}", f"real_application_gold_rows=0", ""]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "cuda_assignment": report["source"]["cuda_assignment"], "hypothesis_gate": report["hypothesis_gate"], "retention": retention, "report": REPORT.relative_to(ROOT).as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
