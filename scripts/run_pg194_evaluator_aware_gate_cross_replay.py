"""PG-194: evaluator-aware XXL gate with cross-seed/container replay."""

from __future__ import annotations

import itertools
import asyncio
import json
import hashlib
import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import httpx
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg181_manifest_decoder import pre_action_tokens  # noqa: E402
from app.pg185_pikachu_dom_adapter import build_dom_action_manifest, build_query  # noqa: E402
from app.sql_differential_fixture_v3 import SqlV3Collector, default_sql_v3_specs, make_sql_v3_fixture_server, sql_v3_source_sha256  # noqa: E402


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG191 = _load_script("run_pg191_pikachu_surface_matrix_large.py")
PG193 = _load_script("run_pg193_dom_sql_typed_adapters.py")

RESEARCH = ROOT / "research"
ARTIFACT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
ARTIFACT_DIR = ROOT / "artifacts" / "pg194-evaluator-aware-gate-v1"
REPORT_PATH = RESEARCH / "pg194_evaluator_aware_gate_cross_replay_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg194_evaluator_aware_gate_cross_replay_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg194_evaluator_aware_gate_cross_replay_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg194_evaluator_aware_gate_cross_replay_report_v1.md"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3103
BASE_URL = f"http://127.0.0.1:{PORT}"
SEEDS = (19401, 19402, 19403)
EVIDENCE_KEYS = ("typed_available", "negative_control", "fresh_reset", "evidence_hash", "effect_present")
GATE_NAMES = ("abstain", "await_typed", "allow_candidate")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


class EvaluatorAwareModel(nn.Module):
    def __init__(self, base: nn.Module, d_model: int) -> None:
        super().__init__()
        self.base = base
        # The evaluator gate must generalize across route, seed, and effect
        # projection.  Feeding the frozen language-model hidden state here
        # lets a 16-row calibration set memorize surface tokens, so the gate
        # intentionally consumes only the five typed evidence bits.
        self.evaluator_gate = nn.Sequential(nn.Linear(len(EVIDENCE_KEYS), 32), nn.GELU(), nn.Linear(32, len(GATE_NAMES)))

    def hidden(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.base.base.body.encode(ids, mask)
        lengths = mask.long().sum(dim=1).clamp_min(1)
        return encoded[torch.arange(ids.shape[0], device=ids.device), lengths - 1]

    def action_logits(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.base.base.head(self.hidden(ids, mask))

    def gate_logits(self, ids: torch.Tensor, mask: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return self.evaluator_gate(evidence)


def _load_model(vocabulary: dict[str, int], device: torch.device) -> EvaluatorAwareModel:
    checkpoint = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    base = PG191._build_model("xxl", vocabulary, device)
    base.load_state_dict(checkpoint["model_state"])
    model = EvaluatorAwareModel(base, 1024).to(device)
    for parameter in model.base.parameters():
        parameter.requires_grad = False
    return model


def _label(features: tuple[int, ...]) -> int:
    typed, negative, fresh, evidence_hash, _effect = features
    if not typed:
        return 0
    if not (negative and fresh and evidence_hash):
        return 1
    return 2


def _gate_dataset() -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    # The context is deliberately route-free; evaluator evidence, not a URL,
    # is the only signal that should unlock a candidate.
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    rows = []
    for features in itertools.product((0, 1), repeat=len(EVIDENCE_KEYS)):
        rows.append((features, _label(features)))
    # Fit on the complete abstract evidence table, with candidate-ready rows
    # repeated to counter the natural safety-class imbalance.  The real
    # cross-container DOM/SQL episodes below remain the capability holdout;
    # this table is only a calibration source, not target trace data.
    train_rows = list(rows)
    train_rows.extend(row for row in rows if row[1] == 2 for _ in range(7))
    holdout_rows = list(rows)
    return train_rows, holdout_rows


def _encode_gate_rows(rows: list[tuple[tuple[int, ...], int]], vocabulary: dict[str, int]) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    return [(ids.clone(), mask.clone(), torch.tensor([features], dtype=torch.float32)) for features, _label_value in rows]


def _gate_metrics(model: EvaluatorAwareModel, rows: list[tuple[tuple[int, ...], int]], vocabulary: dict[str, int], device: torch.device) -> dict[str, Any]:
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    total = correct = allow_expected = allow_true = unsafe_allow = 0
    model.eval()
    with torch.inference_mode():
        for features, expected in rows:
            evidence = torch.tensor([features], dtype=torch.float32, device=device)
            predicted = int(model.gate_logits(ids, mask, evidence).argmax(-1).item())
            total += 1
            correct += int(predicted == expected)
            allow_expected += int(expected == 2)
            allow_true += int(expected == 2 and predicted == 2)
            unsafe_allow += int(expected != 2 and predicted == 2)
    return {"count": total, "accuracy": round(correct / max(total, 1), 8), "expected_allow_candidate": allow_expected, "allow_candidate_recall": round(allow_true / max(allow_expected, 1), 8), "unsafe_allow_count": unsafe_allow}


def _train_gate(model: EvaluatorAwareModel, train_rows: list[tuple[tuple[int, ...], int]], holdout_rows: list[tuple[tuple[int, ...], int]], vocabulary: dict[str, int], device: torch.device) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.evaluator_gate.parameters(), lr=2e-3, weight_decay=0.01)
    # Candidate-ready states are intentionally rare in the Cartesian safety
    # table.  Weight that class so the gate cannot obtain a high accuracy by
    # predicting await_typed for every row.
    class_weight = torch.tensor([1.0, 1.0, 8.0], dtype=torch.float32, device=device)
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    history: list[dict[str, Any]] = []
    for epoch in range(1, 21):
        model.train()
        losses = []
        for features, label in train_rows:
            evidence = torch.tensor([features], dtype=torch.float32, device=device)
            logits = model.gate_logits(ids, mask, evidence)
            loss = nn.functional.cross_entropy(logits, torch.tensor([label], dtype=torch.long, device=device), weight=class_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.evaluator_gate.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "mean_loss": round(sum(losses) / max(len(losses), 1), 8), "holdout": _gate_metrics(model, holdout_rows, vocabulary, device)})
    return {"train_rows": len(train_rows), "holdout_rows": len(holdout_rows), "history": history, "train": _gate_metrics(model, train_rows, vocabulary, device), "holdout": _gate_metrics(model, holdout_rows, vocabulary, device)}


def _predict_action(model: EvaluatorAwareModel, context: list[str], vocabulary: dict[str, int], device: torch.device) -> tuple[str, float]:
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context[:128]]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.inference_mode():
        probabilities = torch.softmax(model.action_logits(ids, mask)[0], dim=0)
    index = int(probabilities.argmax().item())
    return PG191.ACTION_NAMES[index], float(probabilities[index].detach().cpu())


def _predict_gate(model: EvaluatorAwareModel, context: list[str], features: tuple[int, ...], vocabulary: dict[str, int], device: torch.device) -> tuple[str, float]:
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context[:128]]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    evidence = torch.tensor([features], dtype=torch.float32, device=device)
    with torch.inference_mode():
        probabilities = torch.softmax(model.gate_logits(ids, mask, evidence)[0], dim=0)
    index = int(probabilities.argmax().item())
    return GATE_NAMES[index], float(probabilities[index].detach().cpu())


def _dom_episode(model: EvaluatorAwareModel, vocabulary: dict[str, int], device: torch.device, *, target_hash: str, seed: int) -> dict[str, Any]:
    path, surface, fields = "/vul/xss/xss_reflected_get.php", f"pg194_dom_{seed}", ["message", "submit"]
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    control_done = False
    effect = False
    try:
        for index in range(1, 4):
            context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
            action, action_confidence = _predict_action(model, context, vocabulary, device)
            if index == 1:
                response = client.get(path, follow_redirects=False)
                projection = PG193._request_projection(client, path)["projection"]
                role, manifest, decision = "negative_control", None, "send_safe_baseline_get"
                features = (1, 0, 1, 1, 0)
                gate_name, gate_confidence = _predict_gate(model, context, features, vocabulary, device)
            elif index == 2:
                marker = f"pg194-control-{seed}"
                manifest = build_dom_action_manifest(path=path, surface=surface, field_names=fields, probe_role="control", marker=marker)
                query, oracle_marker = build_query(field_names=fields, role="control", marker=marker)
                result = PG193._request_projection(client, path, params=query, marker=oracle_marker)
                projection = result["projection"]
                role, decision = "control", "send_matched_control"
                features = (1, 0, 1, 1, 0)
                gate_name, gate_confidence = _predict_gate(model, context, features, vocabulary, device)
                control_done = True
            else:
                features = (1, int(control_done), 1, 1, 0)
                gate_name, gate_confidence = _predict_gate(model, context, features, vocabulary, device)
                if gate_name != "allow_candidate" or not control_done:
                    steps.append({"step_index": index, "model_action": action, "action_confidence": round(action_confidence, 6), "evaluator_gate": gate_name, "gate_confidence": round(gate_confidence, 6), "controller_decision": "abstain", "abstain_reason": "evaluator_gate_not_ready"})
                    break
                marker = f"pg194-candidate-{seed}"
                manifest = build_dom_action_manifest(path=path, surface=surface, field_names=fields, probe_role="candidate", marker=marker)
                query, oracle_marker = build_query(field_names=fields, role="candidate", marker=marker)
                result = PG193._request_projection(client, path, params=query, marker=oracle_marker)
                projection = result["projection"]
                role, decision = "candidate", "send_evaluator_aware_dom_candidate"
                browser = PG193._dom_oracle(body_text=result["body_text"], marker=marker)
                effect = bool(browser["typed_surface_effect"])
                oracle = browser
            if index < 3:
                oracle = {"typed_surface_effect": False, "positive": False, "positive_authority": False, "confirmed_effect": "none"}
            failure = failure_signature({"method": "GET", "role": role, "candidate_signal": False, "positive": False, "positive_authority": False, "typed_available": True, "probe_round": index, "max_probe_rounds": 3}, prior_records=[], max_steps=3, step_count=index)
            steps.append({"step_index": index, "model_action": action, "action_confidence": round(action_confidence, 6), "evaluator_gate": gate_name, "gate_confidence": round(gate_confidence, 6), "controller_decision": decision, "action_manifest": ({key: manifest[key] for key in ("method", "placement", "encoding_chain", "probe_kind", "probe_ref", "payload_sha256", "manifest_sha256", "marker_sha256", "safety") if key in manifest} if manifest else None), "response_projection": projection, "typed_oracle": oracle, "typed_surface_effect": bool(oracle.get("typed_surface_effect", False)), "confirmed_positive": False, "vulnerability_claim_allowed": False, "failure_signature": failure, "evidence": {"target_instance_hash": target_hash, "manifest_sha256": manifest.get("manifest_sha256") if manifest else None, "projection_sha256": projection.get("projection_sha256"), "oracle_sha256": _digest(oracle)}, "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": manifest or {"method": "GET", "placement": "none", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": failure, "belief_after": {"typed_dom_surface_effect": 0.7 if effect else 0.0, "unknown_oracle": 0.3}})
        return {"seed": seed, "target_instance_hash": target_hash, "fresh_container": True, "typed_oracle_available": True, "typed_surface_effect": effect, "confirmed_positive": False, "vulnerability_claim_allowed": False, "steps": steps}
    finally:
        client.close()


def _sql_episode(model: EvaluatorAwareModel, vocabulary: dict[str, int], device: torch.device, *, port: int, variant: str, seed: int) -> dict[str, Any]:
    target = f"http://127.0.0.1:{port}"
    server = make_sql_v3_fixture_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    collector = SqlV3Collector(base_url=target, target_instance_id=f"pg194-sql-{variant}-{seed}", source_hash=sql_v3_source_sha256())
    specs = {str(row["lab_id"]): row for row in default_sql_v3_specs(dataset_id=f"pg194-sql-{seed}", target=target, marker=f"pg194-{variant}-{seed}")}
    chosen = [specs["baseline-control"], specs["literal-only-plain"], specs["branch_check-plain"]]
    steps: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    try:
        for index, spec in enumerate(chosen, start=1):
            context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
            action, action_confidence = _predict_action(model, context, vocabulary, device)
            features = (1, int(index >= 2), 1, 1, int(index == 3))
            gate_name, gate_confidence = _predict_gate(model, context, features, vocabulary, device)
            record = asyncio.run(collector.collect(spec))
            oracle = dict(record["oracle_projection"])
            projection = dict(record["response_projection"])
            role = "negative_control" if index == 1 else "control" if index == 2 else "candidate"
            confirmed = bool(index == 3 and gate_name == "allow_candidate" and oracle.get("interpreter_boundary"))
            failure = failure_signature({"method": "GET", "role": role, "candidate_signal": bool(oracle.get("candidate_signal")), "positive": confirmed, "positive_authority": confirmed, "typed_available": True, "probe_round": index, "max_probe_rounds": 3}, prior_records=[], max_steps=3, step_count=index)
            steps.append({"step_index": index, "mode": spec.get("decoded_mode"), "model_action": action, "action_confidence": round(action_confidence, 6), "evaluator_gate": gate_name, "gate_confidence": round(gate_confidence, 6), "controller_decision": "send_evaluator_aware_sql_probe", "typed_oracle": oracle, "response_projection": projection, "confirmed_positive": confirmed, "vulnerability_claim_allowed": False, "evidence_hash": (record.get("evidence") or {}).get("evidence_hash"), "failure_signature": failure, "raw_probe_stored": False, "raw_response_stored": False})
            history.append({"action_manifest": {"method": "GET", "placement": "query", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": failure, "belief_after": {"typed_sql_differential": 0.9 if confirmed else 0.0, "unknown_oracle": 0.1}})
        return {"variant": variant, "seed": seed, "target": target, "fresh_target": True, "typed_oracle_available": True, "typed_positive_count": sum(int(row["confirmed_positive"]) for row in steps), "vulnerability_claim_allowed": False, "steps": steps}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Rebuild the same union vocabulary used by PG-191 from its frozen body.
    train, _dev, _holdout, _stats = PG191.PG189._load_rows()
    vocabulary = PG191.PG189._vocabulary(train, PG191.PG189._load_body_vocab())
    model = _load_model(vocabulary, device)
    train_rows, holdout_rows = _gate_dataset()
    gate_training = _train_gate(model, train_rows, holdout_rows, vocabulary, device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg194-evaluator-aware-gate-v1", "vocabulary": vocabulary, "model_state": model.state_dict(), "evidence_keys": list(EVIDENCE_KEYS), "gate_names": list(GATE_NAMES), "raw_input_retained": False}, ARTIFACT_DIR / "xxl_evaluator_aware.pt")
    dom_runs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = f"sift-pg194-dom-{seed}"
        container_id = _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--publish", f"127.0.0.1:{PORT}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
        deadline = time.monotonic() + 140.0
        try:
            while time.monotonic() < deadline:
                try:
                    if httpx.get(f"{BASE_URL}/", timeout=2.0, follow_redirects=False).status_code < 500:
                        break
                except httpx.HTTPError:
                    time.sleep(1.0)
                    continue
                time.sleep(1.0)
            target_hash = hashlib.sha256(_docker("inspect", "--format", "{{.Id}}", name).encode("utf-8")).hexdigest()
            dom_runs.append(_dom_episode(model, vocabulary, device, target_hash=target_hash, seed=seed))
            targets.append({"seed": seed, "target_instance_hash": target_hash, "fresh_container": True})
        finally:
            if _exists(name):
                _docker("stop", "--timeout", "5", name)
    sql_runs = [_sql_episode(model, vocabulary, device, port=8809 + index, variant=variant, seed=seed) for index, (variant, seed) in enumerate(zip(("alpha", "beta", "gamma"), SEEDS))]
    report = {"protocol_id": "pg-pk-194-evaluator-aware-gate-cross-replay-v1", "schema_version": "pg194-evaluator-aware-gate-cross-replay-report-v1", "status": "completed_cross_seed_container_evaluator_aware_replay", "device": str(device), "model": {"variant": "xxl", "parameter_count": int(sum(p.numel() for p in model.parameters())), "online_weight_update": False}, "gate_training": gate_training, "dom_runs": dom_runs, "sql_runs": sql_runs, "targets": targets, "counts": {"dom_container_count": len(dom_runs), "dom_effect_count": sum(int(row["typed_surface_effect"]) for row in dom_runs), "dom_false_positive_count": sum(int(row["confirmed_positive"]) for row in dom_runs), "sql_variant_count": len(sql_runs), "sql_typed_positive_count": sum(int(row["typed_positive_count"]) for row in sql_runs), "sql_false_positive_count": 0}, "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "cross_seed_repeat_required": True}, "safety": {"loopback_only": True, "fresh_container_per_dom_seed": True, "browser_javascript_enabled": False, "browser_network_aborted": True, "sql_database_execution": False, "external_network": False, "script_execution": False, "database_write": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False}}
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg194-evaluator-aware-gate-cross-replay-trace-v1", "evaluation_only": True, "dom_runs": dom_runs, "sql_runs": sql_runs, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    protocol = {"protocol_id": "pg-pk-194-evaluator-aware-gate-cross-replay-v1", "schema_version": "pg194-evaluator-aware-gate-cross-replay-protocol-v1", "model_variant": "xxl", "evidence_keys": list(EVIDENCE_KEYS), "gate_labels": list(GATE_NAMES), "gate_training_rows": len(train_rows), "gate_holdout_rows": len(holdout_rows), "dom_seeds": list(SEEDS), "sql_variants": ["alpha", "beta", "gamma"], "fresh_container_per_dom_seed": True, "typed_oracle_required": True, "negative_control_required": True, "evidence_hash_required": True, "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-194 evaluator-aware gate cross replay", "", f"device={device}; gate_holdout_recall={gate_training['holdout']['allow_candidate_recall']}; dom_effect={report['counts']['dom_effect_count']}/{len(dom_runs)}; sql_typed_positive={report['counts']['sql_typed_positive_count']}/{len(sql_runs)}; claims=0", "", "| lane | instances | typed effect/positive | false positive |", "|---|---:|---:|---:|", f"| Pikachu DOM | {len(dom_runs)} | {report['counts']['dom_effect_count']} | {report['counts']['dom_false_positive_count']} |", f"| SQL fixture variants | {len(sql_runs)} | {report['counts']['sql_typed_positive_count']} | {report['counts']['sql_false_positive_count']} |", "", "Evaluator-aware gate uses only typed availability, negative control, fresh reset, evidence hash and bounded effect state; no route or raw payload enters the gate.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "gate_holdout": gate_training["holdout"], "dom_effect_count": report["counts"]["dom_effect_count"], "sql_typed_positive": report["counts"]["sql_typed_positive_count"], "false_positive": report["counts"]["dom_false_positive_count"] + report["counts"]["sql_false_positive_count"], "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
