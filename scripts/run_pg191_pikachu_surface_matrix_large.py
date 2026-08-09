"""PG-191: full crawled Pikachu surface matrix plus large-model replay.

The browser crawl is the authority for parameter names.  Every parameterized
surface receives an abstract, hash-bound probe plan; eight previously unused
surfaces are replayed in fresh pinned loopback containers.  The large and XXL
PG-189 bodies share the same safety-gate training rows, then the selected
variant drives the GET/POST loop.  Values are bounded canaries only.
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


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG189 = _load_script("run_pg189_structured_get_trace_action_training.py")
PG190 = _load_script("run_pg190_dual_head_action_gate_replay.py")

RESEARCH = ROOT / "research"
CRAWL_PATH = RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json"
PG189_ARTIFACTS = {
    "large": ROOT / "artifacts" / "pg189-structured-get-trace-action-v1" / "large.pt",
    "xxl": ROOT / "artifacts" / "pg189-structured-get-trace-action-v1" / "xxl.pt",
}
ARTIFACT_DIR = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1"
REPORT_PATH = RESEARCH / "pg191_pikachu_surface_matrix_large_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg191_pikachu_surface_matrix_large_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg191_pikachu_surface_matrix_large_trace_v1.json"
MANIFEST_PATH = RESEARCH / "pg191_pikachu_surface_matrix_manifest_v1.json"
MARKDOWN_PATH = RESEARCH / "pg191_pikachu_surface_matrix_large_report_v1.md"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3101
BASE_URL = f"http://127.0.0.1:{PORT}"
SEED = 19101
MAX_STEPS = 3
ACTION_NAMES = PG189.ACTION_NAMES
ROUTE_KEYS = (
    ("/vul/csrf/csrfget/csrf_get_login.php", "GET"),
    ("/vul/dir/dir_list.php", "GET"),
    ("/vul/fileinclude/fi_local.php", "GET"),
    ("/vul/sqli/sqli_blind_t.php", "GET"),
    ("/vul/urlredirect/urlredirect.php", "GET"),
    ("/vul/xss/xss_01.php", "GET"),
    ("/vul/burteforce/bf_form.php", "POST"),
    ("/vul/xss/xssblind/xss_blind.php", "POST"),
)
SAFE_MARKER_RE = re.compile(r"^pg191-[a-z0-9-]{8,52}$")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def _container_name(surface: str) -> str:
    return "sift-pg191-" + re.sub(r"[^a-z0-9-]", "-", surface.casefold())[:38]


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start_container(name: str) -> str:
    if _exists(name):
        raise RuntimeError(f"refusing to reuse {name}; PG-191 requires a fresh target")
    _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--publish", f"127.0.0.1:{PORT}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
    deadline = time.monotonic() + 140.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{BASE_URL}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", name)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"PG-191 container {name} did not become ready")


def _stop_container(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _family_hint(path: str) -> str:
    lower = path.casefold()
    if "xss" in lower:
        return "xss"
    if "sqli" in lower:
        return "injection"
    if "urlredirect" in lower:
        return "url_redirect"
    if "csrf" in lower or "burteforce" in lower:
        return "authentication"
    if "fileinclude" in lower:
        return "input_validation"
    if "/dir/" in lower:
        return "access_control"
    return "ordinary_response"


def _load_matrix() -> list[dict[str, Any]]:
    crawl = json.loads(CRAWL_PATH.read_text(encoding="utf-8-sig"))
    rows = crawl.get("request_response_rows", [])
    dedup: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for row in rows:
        method = str(row.get("method", "")).upper()
        schema = dict(row.get("request_schema") or {})
        fields = tuple(sorted({str(item) for item in (schema.get("query_params", []) if method == "GET" else schema.get("form_params", [])) if str(item)}))
        path = str(row.get("route_path", ""))
        if method not in {"GET", "POST"} or not path.startswith("/") or not fields:
            continue
        key = (method, path, fields)
        dedup.setdefault(key, row)
    matrix: list[dict[str, Any]] = []
    for index, ((method, path, fields), row) in enumerate(sorted(dedup.items()), start=1):
        route_id = f"pg191-surface-{index:03d}"
        matrix.append({
            "route_id": route_id,
            "method": method,
            "path": path,
            "field_names": list(fields),
            "family_hint": _family_hint(path),
            "source_row_sha256": _digest(row),
            "model_input_excludes": ["path", "field_names", "family_hint", "route_id"],
            "abstract_probe_roles": ["baseline", "matched_control", "safe_candidate", "abstain"],
            "probe_kind": "http_canary",
            "expected_oracle": "typed_family_oracle_required; otherwise abstain",
            "safety": {"does_not_execute": True, "no_external_network": True, "no_script_execution": True, "no_database_write": True, "no_credential_access": True},
        })
    if len(matrix) != 44:
        raise ValueError(f"PG-191 expected 44 deduplicated parameterized surfaces, got {len(matrix)}")
    return matrix


def _select_routes(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for path, method in ROUTE_KEYS:
        matches = [row for row in matrix if row["path"] == path and row["method"] == method]
        if len(matches) != 1:
            raise ValueError(f"PG-191 expected one matrix row for {method} {path}, got {len(matches)}")
        selected.append(dict(matches[0]))
    return selected


def _build_model(variant: str, vocabulary: dict[str, int], device: torch.device) -> Any:
    checkpoint = torch.load(PG189_ARTIFACTS[variant], map_location="cpu", weights_only=False)
    expected_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    if expected_vocab != vocabulary:
        raise ValueError(f"PG-191 vocabulary drift for {variant}")
    if variant == "large":
        d_model, nhead, layers = 512, 8, 6
    else:
        d_model, nhead, layers = 1024, 16, 8
    body = CausalTraceTransformer(len(vocabulary), d_model=d_model, nhead=nhead, layers=layers, max_len=128)
    base = PG189._ManifestModel(body, d_model)
    base.load_state_dict(checkpoint["model_state"])
    model = PG190.DualHead(base, d_model).to(device)
    for parameter in model.base.parameters():
        parameter.requires_grad = False
    return model


def _train_gate_variant(model: Any, train: list[dict[str, Any]], dev: list[dict[str, Any]], holdout: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device, variant: str) -> dict[str, Any]:
    torch.manual_seed(SEED + (1 if variant == "large" else 2))
    random.seed(SEED + (1 if variant == "large" else 2))
    optimizer = torch.optim.AdamW(model.gate.parameters(), lr=2e-3, weight_decay=0.01)
    batches = PG190._gate_batches(train, vocabulary, seed=SEED, batch_size=8) if "batch_size" in PG190._gate_batches.__code__.co_varnames else PG190._gate_batches(train, vocabulary, seed=SEED)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, 6):
        model.train()
        losses: list[float] = []
        for batch in batches:
            ids, mask = batch["ids"].to(device), batch["mask"].to(device)
            with torch.no_grad():
                hidden = model.hidden(ids, mask)
            loss = nn.functional.cross_entropy(model.gate(hidden), batch["gate_labels"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "mean_loss": round(sum(losses) / max(len(losses), 1), 8), "dev": PG190._gate_metrics(model, dev, vocabulary, device)})
    return {"variant": variant, "parameter_count": int(sum(p.numel() for p in model.parameters())), "history": history, "train": PG190._gate_metrics(model, train, vocabulary, device), "dev": PG190._gate_metrics(model, dev, vocabulary, device), "holdout": PG190._gate_metrics(model, holdout, vocabulary, device), "elapsed_seconds": round(time.perf_counter() - started, 3)}


def _make_values(fields: list[str], marker: str) -> dict[str, str]:
    if not SAFE_MARKER_RE.fullmatch(marker):
        raise ValueError("PG-191 marker failed safe canary validation")
    values = {field: marker for field in fields if field != "submit"}
    if "submit" in fields:
        values["submit"] = "submit"
    return values


def _belief(signal: Mapping[str, Any]) -> dict[str, float]:
    if bool(signal.get("candidate_signal")):
        return {"candidate_surface_signal": 0.55, "unknown_oracle": 0.45}
    return {"no_observed_effect": 0.55, "unknown_oracle": 0.45}


def _action_and_gate(model: Any, context: list[str], vocabulary: Mapping[str, int], device: torch.device) -> tuple[str, float, bool, float]:
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context[:128]]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.inference_mode():
        action_logits, gate_logits = model(ids, mask)
        action_probs = torch.softmax(action_logits[0], dim=0)
        gate_probs = torch.softmax(gate_logits[0], dim=0)
    action_index = int(action_probs.argmax().item())
    gate_index = int(gate_probs.argmax().item())
    return ACTION_NAMES[action_index], float(action_probs[action_index].detach().cpu()), gate_index == 1, float(gate_probs[gate_index].detach().cpu())


def _replay_route(model: Any, vocabulary: dict[str, int], route: dict[str, Any], device: torch.device, *, target_instance_hash: str) -> dict[str, Any]:
    path, method, fields, family, surface = route["path"], route["method"], route["field_names"], route["family_hint"], route["route_id"]
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    prior: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    sent_get = sent_post = candidate_sent = abstain_count = validation_failures = 0
    baseline_status: int | None = None
    try:
        for step_index in range(1, MAX_STEPS + 1):
            context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
            action, action_conf, gate_allow, gate_conf = _action_and_gate(model, context, vocabulary, device)
            if step_index == 1:
                if action != "baseline" or not gate_allow:
                    abstain_count += 1
                    steps.append({"step_index": step_index, "model_action": action, "action_confidence": round(action_conf, 6), "safety_gate": "allow" if gate_allow else "abstain", "gate_confidence": round(gate_conf, 6), "controller_decision": "abstain", "abstain_reason": "initial_state_requires_baseline_and_allow"})
                    break
                result = request_chain(client, method="GET", path=path)
                baseline_status = int(result["projection"]["status_code"])
                actual_method = "GET"
                role = "negative_control"
                manifest = None
                controller_decision = "send_safe_baseline_get"
                sent_get += 1
            else:
                if action in {"baseline", "abstain"} or not gate_allow:
                    abstain_count += 1
                    steps.append({"step_index": step_index, "model_action": action, "action_confidence": round(action_conf, 6), "safety_gate": "allow" if gate_allow else "abstain", "gate_confidence": round(gate_conf, 6), "controller_decision": "abstain", "abstain_reason": "safety_gate_or_unsupported_followup_action"})
                    break
                actual_method = method
                role = "candidate" if action == "safe_candidate" else "control"
                marker = f"pg191-{route['route_id'][-8:]}-{step_index}"
                manifest = action_manifest(path=path, surface=surface, family=family, method=actual_method, field_names=fields, probe_role=role, marker=marker)
                values = _make_values(fields, marker)
                if actual_method == "GET":
                    result = request_chain(client, method="GET", path=path, query=values, marker=marker, baseline_status=baseline_status)
                    sent_get += 1
                else:
                    result = request_chain(client, method="POST", path=path, form=values, marker=marker, baseline_status=baseline_status)
                    sent_post += 1
                candidate_sent += int(role == "candidate")
                controller_decision = "send_safe_candidate" if role == "candidate" else "send_matched_control"
            signal = dict(result["signal"])
            oracle = surface_oracle(family=family, method=actual_method, signal=signal, oracle_contract_sha256=hashlib.sha256(b"pg191-unknown-typed-oracle-v1").hexdigest())
            failure = failure_signature({"method": actual_method, "role": role, "candidate_signal": bool(signal.get("candidate_signal")), "positive": False, "positive_authority": False, "typed_available": False, "probe_round": step_index, "max_probe_rounds": MAX_STEPS}, prior_records=prior, max_steps=MAX_STEPS, step_count=step_index)
            belief = _belief(signal)
            projection = result["projection"]
            evidence = {"route_row_sha256": route["source_row_sha256"], "target_instance_hash": target_instance_hash, "manifest_sha256": manifest.get("manifest_sha256") if manifest else None, "projection_sha256": projection["projection_sha256"], "oracle_projection_sha256": _digest(oracle), "failure_signature_sha256": _digest(failure)}
            steps.append({"step_index": step_index, "model_action": action, "action_confidence": round(action_conf, 6), "safety_gate": "allow", "gate_confidence": round(gate_conf, 6), "controller_decision": controller_decision, "method": actual_method, "action_manifest": ({key: manifest[key] for key in ("method", "placement", "encoding_chain", "probe_kind", "probe_ref", "payload_sha256", "manifest_sha256", "form_field_names", "marker_sha256", "safety") if key in manifest} if manifest else None), "response_projection": projection, "oracle_projection": oracle, "failure_signature": failure, "belief_after": belief, "evidence": evidence, "decision": "abstain", "vulnerability_claim_allowed": False, "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": manifest or {"method": actual_method, "placement": "none", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": failure, "belief_after": belief})
            prior.append({"method": actual_method, "role": role, "candidate_signal": bool(signal.get("candidate_signal")), "belief_after": belief})
    except (ValueError, RuntimeError) as exc:
        validation_failures += 1
        abstain_count += 1
        steps.append({"step_index": len(steps) + 1, "controller_decision": "abstain", "abstain_reason": "manifest_validation_or_runner_error", "error_class": type(exc).__name__})
    finally:
        client.close()
    return {"route_id": route["route_id"], "path": path, "method": method, "field_names": fields, "family_hint": family, "source_row_sha256": route["source_row_sha256"], "target_instance_hash": target_instance_hash, "fresh_container": True, "step_count": len(steps), "sent_get_count": sent_get, "sent_post_count": sent_post, "sent_count": sent_get + sent_post, "candidate_sent_count": candidate_sent, "controller_abstain_count": abstain_count, "manifest_validation_failure_count": validation_failures, "typed_positive_count": 0, "vulnerability_claim_allowed": False, "steps": steps}


def main() -> int:
    random.seed(SEED)
    matrix = _load_matrix()
    selected_routes = _select_routes(matrix)
    train, dev, holdout, row_stats = PG189._load_rows()
    vocabulary = PG189._vocabulary(train, PG189._load_body_vocab())
    _write(MANIFEST_PATH, {"schema_version": "pg191-pikachu-surface-matrix-v1", "source_crawl": str(CRAWL_PATH.relative_to(ROOT)), "parameterized_surface_count": len(matrix), "surfaces": matrix, "raw_values_stored": False, "raw_payloads_stored": False, "model_input_route_fields": False})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: dict[str, Any] = {}
    gate_results: list[dict[str, Any]] = []
    for variant in ("large", "xxl"):
        model = _build_model(variant, vocabulary, device)
        result = _train_gate_variant(model, train, dev, holdout, vocabulary, device, variant)
        models[variant] = model
        gate_results.append(result)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": "pg191-pikachu-surface-matrix-large-v1", "variant": variant, "vocabulary": vocabulary, "model_state": model.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / f"{variant}_dual.pt")
    eligible = [row for row in gate_results if row["holdout"]["abstain_recall"] >= 0.95 and row["holdout"]["unsafe_allow_count"] == 0]
    selected_variant = max(eligible, key=lambda row: (row["parameter_count"], row["holdout"]["accuracy"]))["variant"] if eligible else None
    runs: list[dict[str, Any]] = []
    target_meta: list[dict[str, Any]] = []
    if selected_variant:
        for route in selected_routes:
            name = _container_name(route["route_id"])
            container_id = _start_container(name)
            target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
            try:
                runs.append(_replay_route(models[selected_variant], vocabulary, route, device, target_instance_hash=target_hash))
                target_meta.append({"route_id": route["route_id"], "container_name": name, "target_instance_hash": target_hash, "fresh_container": True, "pinned_image": IMAGE})
            finally:
                _stop_container(name)
    report = {"protocol_id": "pg-pk-191-pikachu-surface-matrix-large-v1", "schema_version": "pg191-pikachu-surface-matrix-large-report-v1", "status": "completed_crawled_surface_matrix_and_large_replay", "device": str(device), "source": {"crawl_manifest": str(CRAWL_PATH.relative_to(ROOT)), "image": IMAGE, "loopback_port": PORT, "parameterized_surface_count": len(matrix), "selected_route_count": len(selected_routes), "training_row_stats": row_stats}, "model_variants": gate_results, "selection": {"selected_variant": selected_variant, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "gate": "holdout abstain recall >= .95 and unsafe_allow_count=0; typed oracle still required"}, "fresh_targets": target_meta, "counts": {"route_count": len(runs), "sent_get_count": sum(r["sent_get_count"] for r in runs), "sent_post_count": sum(r["sent_post_count"] for r in runs), "sent_count": sum(r["sent_count"] for r in runs), "candidate_sent_count": sum(r["candidate_sent_count"] for r in runs), "controller_abstain_count": sum(r["controller_abstain_count"] for r in runs), "manifest_validation_failure_count": sum(r["manifest_validation_failure_count"] for r in runs), "typed_positive_count": 0}, "runs": runs, "safety": {"loopback_only": True, "fresh_container_per_route": True, "pinned_image": IMAGE, "full_matrix_raw_values_stored": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "model_input_route_fields": False, "script_execution": False, "database_write": False, "credentials": False, "external_network": False, "online_weight_update": False, "long_term_memory_write": False}}
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg191-pikachu-surface-matrix-large-trace-v1", "evaluation_only": True, "matrix_surface_count": len(matrix), "selected_route_count": len(selected_routes), "runs": runs, "training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    protocol = {"protocol_id": "pg-pk-191-pikachu-surface-matrix-large-v1", "schema_version": "pg191-pikachu-surface-matrix-large-protocol-v1", "crawl_source": str(CRAWL_PATH.relative_to(ROOT)), "parameterized_surface_count": len(matrix), "selected_routes": [{"path": r["path"], "method": r["method"], "field_names": r["field_names"]} for r in selected_routes], "model_variants": {"large": "PG-189 large 19M", "xxl": "PG-189 XXL 101M"}, "fresh_container_per_route": True, "methods": ["GET", "POST"], "manifest_validator_before_send": True, "model_output_route_fields_forbidden": True, "typed_oracle_required_before_positive": True, "unknown_oracle_action": "abstain", "raw_payload_and_response_excluded": True, "gates": {"holdout_abstain_recall_min": 0.95, "unsafe_allow_count_max": 0, "manifest_validation_failure_max": 0, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-191 Pikachu crawled surface matrix", "", f"device={device}; matrix={len(matrix)}; selected={len(selected_routes)}; variant={selected_variant}; sent_get={report['counts']['sent_get_count']}; sent_post={report['counts']['sent_post_count']}; candidates={report['counts']['candidate_sent_count']}; abstain={report['counts']['controller_abstain_count']}; positives=0", "", "| route id | method | path | fields | GET | POST | candidate | abstain | manifest errors |", "|---|---|---|---|---:|---:|---:|---:|---:|"] + [f"| {r['route_id']} | {r['method']} | {r['path']} | {','.join(r['field_names'])} | {r['sent_get_count']} | {r['sent_post_count']} | {r['candidate_sent_count']} | {r['controller_abstain_count']} | {r['manifest_validation_failure_count']} |" for r in runs] + ["", "完整爬虫矩阵只保存观测字段与哈希绑定的抽象 probe plan；回放只发送 bounded canary，未知 oracle 一律 abstain。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "matrix_surfaces": len(matrix), "selected_routes": len(selected_routes), "selected_variant": selected_variant, "gate_results": [{"variant": r["variant"], "parameters": r["parameter_count"], "holdout_abstain_recall": r["holdout"]["abstain_recall"], "unsafe_allow": r["holdout"]["unsafe_allow_count"]} for r in gate_results], "sent_get": report["counts"]["sent_get_count"], "sent_post": report["counts"]["sent_post_count"], "candidate_sent": report["counts"]["candidate_sent_count"], "abstain": report["counts"]["controller_abstain_count"], "manifest_errors": report["counts"]["manifest_validation_failure_count"], "typed_positive": 0, "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
