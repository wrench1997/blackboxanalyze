"""PG-190: dual-head action/gate training with a local GET/POST replay loop.

The action head proposes one of four abstract roles.  A separately trained
safety gate can veto the role before the runner binds it to a browser-observed
route and sends a bounded, non-executing canary.  The runner returns only
response projections, failure tokens, belief summaries, and evidence hashes.
No exploit string, raw response body, script execution, database write, or
online weight update is allowed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import httpx
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402
from app.cross_lab_safe_catalog import sha256_json  # noqa: E402
from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg179b_iterative_probe import action_manifest, request_chain, surface_oracle  # noqa: E402
from app.pg181_manifest_decoder import pre_action_tokens  # noqa: E402


def _load_pg189() -> Any:
    path = ROOT / "scripts" / "run_pg189_structured_get_trace_action_training.py"
    spec = importlib.util.spec_from_file_location("pg189_for_pg190", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-189 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG189 = _load_pg189()

RESEARCH = ROOT / "research"
REPORT_PATH = RESEARCH / "pg190_dual_head_action_gate_replay_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg190_dual_head_action_gate_replay_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg190_dual_head_action_gate_replay_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg190_dual_head_action_gate_replay_report_v1.md"
CATALOG_PATH = RESEARCH / "pg179b_pikachu_iterative_catalog_v1.json"
PG189_ARTIFACT = ROOT / "artifacts" / "pg189-structured-get-trace-action-v1" / "large.pt"
ARTIFACT_DIR = ROOT / "artifacts" / "pg190-dual-head-action-gate-v1"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3106
BASE_URL = f"http://127.0.0.1:{PORT}"
MAX_STEPS = 4
SEED = 19001
ACTION_NAMES = PG189.ACTION_NAMES
ACTION_TOKENS = PG189.ACTION_TOKENS
ROUTE_ALLOWLIST = frozenset({"xss_stored_post", "sqli_id_post"})
SAFE_MARKER_RE = re.compile(r"^pg190-[a-z0-9-]{8,48}$")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def _container_name(surface: str) -> str:
    return "sift-pg190-" + re.sub(r"[^a-z0-9-]", "-", surface.casefold())


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start_container(name: str) -> str:
    if _exists(name):
        raise RuntimeError(f"refusing to reuse {name}; PG-190 requires a fresh target")
    _docker(
        "run", "--detach", "--rm", "--pull=never", "--name", name,
        "--publish", f"127.0.0.1:{PORT}:8090", IMAGE,
        "bash", "-lc", "/app/run.sh; exec tail -f /dev/null",
    )
    deadline = time.monotonic() + 140.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{BASE_URL}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", name)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"PG-190 Pikachu container {name} did not become ready")


def _stop_container(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _load_routes() -> list[dict[str, Any]]:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8-sig"))
    rows = []
    for row in data.get("route_rows", []):
        surface = str(row.get("surface", ""))
        if surface not in ROUTE_ALLOWLIST:
            continue
        fields = [str(item) for item in row.get("post_fields", [])]
        if str(row.get("parameterized_method", "")).upper() != "POST" or not fields:
            raise ValueError(f"PG-190 requires observed POST fields for {surface}")
        rows.append({
            "path": str(row["path"]),
            "surface": surface,
            "family": str(row.get("family", "unknown")),
            "post_fields": fields,
            "route_row_sha256": _sha256_json(row),
        })
    if {row["surface"] for row in rows} != set(ROUTE_ALLOWLIST):
        raise ValueError("PG-190 route allowlist is not fully grounded in the catalog")
    return rows


class DualHead(nn.Module):
    """Frozen PG-189 action body plus a trainable binary safety gate."""

    def __init__(self, base: nn.Module, d_model: int) -> None:
        super().__init__()
        self.base = base
        self.gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def hidden(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.base.body.encode(ids, mask)
        lengths = mask.long().sum(dim=1).clamp_min(1)
        return encoded[torch.arange(ids.shape[0], device=ids.device), lengths - 1]

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden(ids, mask)
        action_logits = self.base.head(hidden)
        gate_logits = self.gate(hidden)
        return action_logits, gate_logits


def _load_model(vocabulary: dict[str, int], device: torch.device) -> DualHead:
    checkpoint = torch.load(PG189_ARTIFACT, map_location="cpu", weights_only=False)
    body = CausalTraceTransformer(len(vocabulary), d_model=512, nhead=8, layers=6, max_len=128)
    base = PG189._ManifestModel(body, 512)
    base.load_state_dict(checkpoint["model_state"])
    model = DualHead(base, 512).to(device)
    for parameter in model.base.parameters():
        parameter.requires_grad = False
    return model


def _gate_batches(rows: list[dict[str, Any]], vocabulary: Mapping[str, int], *, seed: int) -> list[dict[str, Any]]:
    batches = PG189._batch(rows, vocabulary, shuffle=True, seed=seed, batch_size=8)
    for batch in batches:
        batch["gate_labels"] = torch.tensor([int(str(row["target"]) != "manifest::abstain") for row in batch["rows"]], dtype=torch.long)
    return batches


def _train_gate(model: DualHead, train: list[dict[str, Any]], dev: list[dict[str, Any]], holdout: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.gate.parameters(), lr=2e-3, weight_decay=0.01)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    batches = _gate_batches(train, vocabulary, seed=SEED)
    for epoch in range(1, 6):
        model.train()
        losses: list[float] = []
        for batch in batches:
            ids, mask = batch["ids"].to(device), batch["mask"].to(device)
            with torch.no_grad():
                hidden = model.hidden(ids, mask)
            logits = model.gate(hidden)
            loss = nn.functional.cross_entropy(logits, batch["gate_labels"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "mean_loss": round(sum(losses) / max(len(losses), 1), 8), "dev": _gate_metrics(model, dev, vocabulary, device)})
    return {"history": history, "train": _gate_metrics(model, train, vocabulary, device), "dev": _gate_metrics(model, dev, vocabulary, device), "holdout": _gate_metrics(model, holdout, vocabulary, device), "elapsed_seconds": round(time.perf_counter() - started, 3)}


def _gate_metrics(model: DualHead, rows: list[dict[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    model.eval()
    total = correct = expected_abstain = true_abstain = unsafe_allow = 0
    with torch.inference_mode():
        for batch in PG189._batch(rows, vocabulary, shuffle=False, seed=0):
            _, gate_logits = model(batch["ids"].to(device), batch["mask"].to(device))
            predicted = gate_logits.argmax(-1).detach().cpu().tolist()
            for prediction, row in zip(predicted, batch["rows"]):
                expected_allow = int(str(row["target"]) != "manifest::abstain")
                total += 1
                correct += int(prediction == expected_allow)
                expected_abstain += int(not expected_allow)
                true_abstain += int(not expected_allow and prediction == 0)
                unsafe_allow += int(not expected_allow and prediction == 1)
    return {"count": total, "accuracy": round(correct / max(total, 1), 8), "expected_abstain_count": expected_abstain, "abstain_recall": round(true_abstain / max(expected_abstain, 1), 8), "unsafe_allow_count": unsafe_allow}


def _action_and_gate(model: DualHead, context: list[str], vocabulary: Mapping[str, int], device: torch.device) -> tuple[str, float, bool, float]:
    encoded = [int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context[:128]]
    ids = torch.tensor([encoded], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.inference_mode():
        action_logits, gate_logits = model(ids, mask)
        action_probs = torch.softmax(action_logits[0], dim=0)
        gate_probs = torch.softmax(gate_logits[0], dim=0)
        action_id = int(action_probs.argmax().item())
        gate_id = int(gate_probs.argmax().item())
    action = ACTION_NAMES[action_id]
    return action, float(action_probs[action_id].detach().cpu()), bool(gate_id == 1), float(gate_probs[gate_id].detach().cpu())


def _safe_form(fields: list[str], marker: str) -> dict[str, str]:
    if not SAFE_MARKER_RE.fullmatch(marker):
        raise ValueError("PG-190 marker failed safe canary validation")
    form = {field: marker for field in fields if field != "submit"}
    if "submit" in fields:
        form["submit"] = "submit"
    return form


def _belief(signal: Mapping[str, Any]) -> dict[str, float]:
    if bool(signal.get("candidate_signal")):
        return {"candidate_surface_signal": 0.55, "unknown_oracle": 0.45}
    return {"no_observed_effect": 0.55, "unknown_oracle": 0.45}


def _replay_route(model: DualHead, vocabulary: dict[str, int], route: dict[str, Any], device: torch.device, *, target_instance_hash: str) -> dict[str, Any]:
    surface = route["surface"]
    path = route["path"]
    family = route["family"]
    fields = route["post_fields"]
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    prior_records: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    validation_failures = 0
    sent_get = sent_post = candidate_sent = controller_abstain = 0
    baseline_status: int | None = None
    try:
        for step_index in range(1, MAX_STEPS + 1):
            previous = history[-1] if history else None
            context = pre_action_tokens(previous, history=history[:-1])
            predicted, action_confidence, gate_allow, gate_confidence = _action_and_gate(model, context, vocabulary, device)
            if step_index == 1:
                if predicted != "baseline" or not gate_allow:
                    controller_abstain += 1
                    steps.append({"step_index": step_index, "model_action": predicted, "action_confidence": round(action_confidence, 6), "safety_gate": "allow" if gate_allow else "abstain", "gate_confidence": round(gate_confidence, 6), "controller_decision": "abstain", "abstain_reason": "initial_state_requires_baseline_and_allow"})
                    break
                method = "GET"
                role = "negative_control"
                marker = None
                manifest = None
                result = request_chain(client, method="GET", path=path)
                baseline_status = int(result["projection"]["status_code"])
                controller_decision = "send_safe_baseline_get"
                sent_get += 1
            else:
                if not gate_allow or predicted == "abstain":
                    controller_abstain += 1
                    steps.append({"step_index": step_index, "model_action": predicted, "action_confidence": round(action_confidence, 6), "safety_gate": "allow" if gate_allow else "abstain", "gate_confidence": round(gate_confidence, 6), "controller_decision": "abstain", "abstain_reason": "safety_gate_or_model_abstain"})
                    break
                method = "POST"
                role = "candidate" if predicted == "safe_candidate" else "control"
                marker = f"pg190-{surface[-8:].replace('_', '-')}-{step_index}"
                manifest = action_manifest(path=path, surface=surface, family=family, method="POST", field_names=fields, probe_role=role, marker=marker)
                form = _safe_form(fields, marker)
                result = request_chain(client, method="POST", path=path, form=form, marker=marker, baseline_status=baseline_status)
                controller_decision = "send_safe_candidate_post" if role == "candidate" else "send_matched_control_post"
                sent_post += 1
                candidate_sent += int(role == "candidate")
            signal = dict(result["signal"])
            oracle = surface_oracle(family=family, method=method, signal=signal, oracle_contract_sha256=hashlib.sha256(b"pg190-unknown-typed-oracle-v1").hexdigest())
            failure = failure_signature({"method": method, "role": role, "candidate_signal": bool(signal.get("candidate_signal")), "positive": False, "positive_authority": False, "typed_available": False, "probe_round": step_index, "max_probe_rounds": MAX_STEPS}, prior_records=prior_records, max_steps=MAX_STEPS, step_count=step_index)
            belief = _belief(signal)
            projection = result["projection"]
            evidence = {"manifest_sha256": manifest.get("manifest_sha256") if manifest else None, "projection_sha256": projection.get("projection_sha256"), "oracle_projection_sha256": _sha256_json(oracle), "failure_signature_sha256": _sha256_json(failure), "target_instance_hash": target_instance_hash}
            record = {"step_index": step_index, "model_action": predicted, "action_confidence": round(action_confidence, 6), "safety_gate": "allow", "gate_confidence": round(gate_confidence, 6), "controller_decision": controller_decision, "method": method, "action_manifest": ({key: manifest[key] for key in ("method", "placement", "encoding_chain", "probe_kind", "probe_ref", "payload_sha256", "manifest_sha256", "form_field_names", "marker_sha256", "safety") if key in manifest} if manifest else None), "response_projection": projection, "oracle_projection": oracle, "failure_signature": failure, "belief_after": belief, "evidence": evidence, "decision": "abstain", "vulnerability_claim_allowed": False, "online_weight_update": False, "long_term_memory_write": False}
            steps.append(record)
            history.append({"action_manifest": manifest or {"method": method, "placement": "none", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": failure, "belief_after": belief})
            prior_records.append({"method": method, "role": role, "candidate_signal": bool(signal.get("candidate_signal")), "belief_after": belief})
    except (ValueError, RuntimeError) as exc:
        validation_failures += 1
        steps.append({"step_index": len(steps) + 1, "controller_decision": "abstain", "abstain_reason": "manifest_validation_or_runner_error", "error_class": type(exc).__name__})
        controller_abstain += 1
    finally:
        client.close()
    return {"surface": surface, "path": path, "family": family, "post_fields": fields, "target_instance_hash": target_instance_hash, "fresh_container": True, "step_count": len(steps), "sent_get_count": sent_get, "sent_post_count": sent_post, "sent_count": sent_get + sent_post, "candidate_sent_count": candidate_sent, "controller_abstain_count": controller_abstain, "manifest_validation_failure_count": validation_failures, "typed_positive_count": 0, "vulnerability_claim_allowed": False, "steps": steps}


def main() -> int:
    random.seed(SEED)
    train, dev, holdout, row_stats = PG189._load_rows()
    vocabulary = PG189._vocabulary(train, PG189._load_body_vocab())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(vocabulary, device)
    gate_training = _train_gate(model, train, dev, holdout, vocabulary, device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg190-dual-head-action-gate-v1", "vocabulary": vocabulary, "model_state": model.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / "large_dual.pt")
    routes = _load_routes()
    runs: list[dict[str, Any]] = []
    container_meta: list[dict[str, Any]] = []
    for route in routes:
        name = _container_name(route["surface"])
        container_id = _start_container(name)
        target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
        try:
            runs.append(_replay_route(model, vocabulary, route, device, target_instance_hash=target_hash))
            container_meta.append({"surface": route["surface"], "container_name": name, "target_instance_hash": target_hash, "fresh_container": True, "pinned_image": IMAGE})
        finally:
            _stop_container(name)
    report = {"protocol_id": "pg-pk-190-dual-head-action-gate-replay-v1", "schema_version": "pg190-dual-head-action-gate-replay-report-v1", "status": "completed_dual_head_local_get_post_replay", "device": str(device), "source": {"pg189_artifact": str(PG189_ARTIFACT.relative_to(ROOT)), "catalog": str(CATALOG_PATH.relative_to(ROOT)), "route_allowlist": sorted(ROUTE_ALLOWLIST), "row_stats": row_stats}, "model": {"body": "large", "parameter_count": int(sum(p.numel() for p in model.parameters())), "action_vocabulary": list(ACTION_NAMES), "online_weight_update": False, "memory_promotion_allowed": False}, "gate_training": gate_training, "fresh_targets": container_meta, "counts": {"route_count": len(runs), "sent_get_count": sum(r["sent_get_count"] for r in runs), "sent_post_count": sum(r["sent_post_count"] for r in runs), "sent_count": sum(r["sent_count"] for r in runs), "candidate_sent_count": sum(r["candidate_sent_count"] for r in runs), "controller_abstain_count": sum(r["controller_abstain_count"] for r in runs), "manifest_validation_failure_count": sum(r["manifest_validation_failure_count"] for r in runs), "typed_positive_count": 0}, "runs": runs, "promotion": {"action_manifest_validation_required": True, "typed_oracle_required": True, "positive_count": 0, "training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "unknown typed oracle; safe canary replay only"}, "safety": {"loopback_only": True, "pinned_image": IMAGE, "fresh_container_per_route": True, "get_post_required": True, "post_required": True, "model_emits_abstract_action_only": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "script_execution": False, "database_write": False, "credentials": False, "external_network": False, "online_weight_update": False, "long_term_memory_write": False}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg190-dual-head-action-gate-replay-trace-v1", "evaluation_only": True, "runs": runs, "model_output_is_abstract": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False})
    protocol = {"protocol_id": "pg-pk-190-dual-head-action-gate-replay-v1", "schema_version": "pg190-dual-head-action-gate-replay-protocol-v1", "training_source": "PG-189 structured GET traces", "replay_source": "PG-179B observed POST routes", "route_count": len(routes), "methods": ["GET", "POST"], "fresh_container_per_route": True, "model_outputs": {"action": list(ACTION_NAMES), "safety_gate": ["abstain", "allow"]}, "manifest_validator_before_send": True, "typed_oracle_required_before_positive": True, "unknown_oracle_action": "abstain", "raw_payload_and_response_excluded": True, "gates": {"gate_holdout_abstain_recall_min": 0.95, "unsafe_allow_count_max": 0, "manifest_validation_failure_max": 0, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-190 dual-head action/gate local replay", "", f"device={device}; routes={len(runs)}; sent_get={report['counts']['sent_get_count']}; sent_post={report['counts']['sent_post_count']}; candidates={report['counts']['candidate_sent_count']}; abstain={report['counts']['controller_abstain_count']}; positives=0", "", "| surface | sent GET | sent POST | candidates | abstain | manifest errors |", "|---|---:|---:|---:|---:|---:|"] + [f"| {r['surface']} | {r['sent_get_count']} | {r['sent_post_count']} | {r['candidate_sent_count']} | {r['controller_abstain_count']} | {r['manifest_validation_failure_count']} |" for r in runs] + ["", "模型输出 abstract action；POST 只使用浏览器观测字段和非执行 canary。typed oracle 不可用时一律 abstain，不能宣称漏洞或生成真实利用串。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "routes": len(runs), "sent_get": report["counts"]["sent_get_count"], "sent_post": report["counts"]["sent_post_count"], "candidate_sent": report["counts"]["candidate_sent_count"], "controller_abstain": report["counts"]["controller_abstain_count"], "manifest_validation_failures": report["counts"]["manifest_validation_failure_count"], "typed_positive": 0, "gate_holdout": gate_training["holdout"], "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
