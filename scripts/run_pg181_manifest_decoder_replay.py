"""PG-181: train a safe manifest-role decoder and use it in one local replay.

The decoder chooses only baseline/control/safe-candidate/abstain.  The replay
controller supplies the real observed parameter name from the browser crawl,
validates the manifest, and sends only an alphanumeric canary to a fresh pinned
Pikachu container.  No exploit string or raw response is persisted.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg179b_iterative_probe import (  # noqa: E402
    PIKACHU_IMAGE_DIGEST,
    action_manifest,
    request_chain,
    surface_oracle,
)
from app.pg181_manifest_decoder import (  # noqa: E402
    MANIFEST_ACTIONS,
    MANIFEST_ACTION_TOKENS,
    MODEL_VARIANTS,
    SCHEMA_VERSION,
    build_manifest_examples,
    build_model,
    collate,
    last_logits,
    manifest_encode,
    manifest_vocabulary,
    pre_action_tokens,
    restrict_manifest_action,
)
from app.failure_guided_scheduler import failure_signature  # noqa: E402


TRACE_PATH = ROOT / "research" / "pg179b_pikachu_iterative_trace_v1.json"
CRAWL_PATH = ROOT / "research" / "pg179_pikachu_browser_crawl_manifest_v1.json"
REPORT_PATH = ROOT / "research" / "pg181_manifest_decoder_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg181_manifest_decoder_replay_protocol_v1.json"
TRACE_OUT_PATH = ROOT / "research" / "pg181_manifest_decoder_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg181_manifest_decoder_replay_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg181-manifest-decoder-v1"

SEEDS = (18101, 18102)
EPOCHS = 70
PATIENCE = 10
BATCH_SIZE = 16
PORT = 8781
CONTAINER_NAME = "pg181-pikachu-replay"
IMAGE = f"tavenli/pikachu-labs@{PIKACHU_IMAGE_DIGEST}"
CANARY = "pg181-canary-a1"
CONTROL = "pg181-control-a1"
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg181-surface-signal-no-typed-effect-v1").hexdigest()
SPLITS = (
    {"name": "url_holdout", "test": ("url_redirect_get",), "dev": ("xss_blind_post",)},
    {"name": "injection_holdout", "test": ("sqli_delete_post", "sqli_header_post", "sqli_id_post", "sqli_widebyte_post"), "dev": ("url_redirect_get",)},
)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_pg180_split_helper() -> Any:
    spec = importlib.util.spec_from_file_location("pg180_for_pg181", ROOT / "scripts" / "run_pg180_process_action_model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-180 split helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _batch(rows: list[dict[str, Any]], batch_size: int, *, shuffle: bool, seed: int) -> list[list[dict[str, Any]]]:
    ordered = list(rows)
    if shuffle:
        random.Random(seed).shuffle(ordered)
    return [ordered[index:index + batch_size] for index in range(0, len(ordered), batch_size)]


def _train_one(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], vocabulary: dict[str, int], *, variant: str, seed: int, split_name: str, device: torch.device) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_model(len(vocabulary), variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4 if variant != "moe_large" else 2e-4, weight_decay=0.02)
    best_state: dict[str, torch.Tensor] | None = None
    best_dev = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for batch_rows in _batch(train_rows, BATCH_SIZE, shuffle=True, seed=seed + epoch):
            encoded = manifest_encode(batch_rows, vocabulary)
            ids, mask, _ = collate(encoded)
            ids, mask = ids.to(device), mask.to(device)
            logits = last_logits(model, ids, mask)
            action_ids = torch.tensor([vocabulary[token] for token in MANIFEST_ACTION_TOKENS], dtype=torch.long, device=device)
            target_indices = torch.tensor([list(MANIFEST_ACTION_TOKENS).index(str(row["target"])) for row in batch_rows], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits.index_select(1, action_ids), target_indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 5 == 0:
            model.eval()
            dev_losses: list[float] = []
            with torch.inference_mode():
                for batch_rows in _batch(dev_rows, BATCH_SIZE, shuffle=False, seed=seed):
                    encoded = manifest_encode(batch_rows, vocabulary)
                    ids, mask, _ = collate(encoded)
                    logits = last_logits(model, ids.to(device), mask.to(device))
                    action_ids = torch.tensor([vocabulary[token] for token in MANIFEST_ACTION_TOKENS], dtype=torch.long, device=device)
                    target_indices = torch.tensor([list(MANIFEST_ACTION_TOKENS).index(str(row["target"])) for row in batch_rows], dtype=torch.long, device=device)
                    dev_losses.append(float(F.cross_entropy(logits.index_select(1, action_ids), target_indices).detach().cpu()))
            dev_loss = statistics.mean(dev_losses)
            history.append({"epoch": epoch, "train_loss": round(statistics.mean(losses), 8), "dev_loss": round(dev_loss, 8)})
            if dev_loss < best_dev - 1e-6:
                best_dev = dev_loss
                stale = 0
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, details = _evaluate(model, test_rows, vocabulary, device)
    checkpoint = ARTIFACT_DIR / split_name / f"{variant}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "variant": variant, "seed": seed, "vocabulary": vocabulary, "model_config": MODEL_VARIANTS[variant], "model_state": model.state_dict(), "raw_input_retained": False}, checkpoint)
    result = {"split": split_name, "variant": variant, "seed": seed, "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "train_count": len(train_rows), "dev_count": len(dev_rows), "test_count": len(test_rows), "epochs_ran": history[-1]["epoch"] if history else 0, "history_tail": history[-5:], "test": metrics, "test_details": details, "checkpoint": str(checkpoint.relative_to(ROOT)), "elapsed_seconds": round(time.perf_counter() - started, 3), "raw_input_retained": False, "oracle_in_input": False, "family_in_input": False, "route_in_input": False}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _evaluate(model: torch.nn.Module, rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    details: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_rows in _batch(rows, BATCH_SIZE, shuffle=False, seed=0):
            encoded = manifest_encode(batch_rows, vocabulary)
            ids, mask, _ = collate(encoded)
            logits = last_logits(model, ids.to(device), mask.to(device)).detach().cpu()
            for index, row in enumerate(batch_rows):
                action, confidence = restrict_manifest_action(logits[index], vocabulary, single_channel=bool(row.get("single_channel")))
                details.append({"surface": row["surface"], "expected_action": str(row["target"]).split("::", 1)[1], "predicted_action": action, "confidence": round(confidence, 6), "single_channel": bool(row.get("single_channel"))})
    expected = [item["expected_action"] for item in details]
    predicted = [item["predicted_action"] for item in details]
    candidate_indices = [i for i, value in enumerate(expected) if value == "safe_candidate"]
    metrics = {"count": len(details), "accuracy": round(sum(int(a == b) for a, b in zip(predicted, expected)) / max(len(expected), 1), 6), "candidate_recall": round(sum(int(predicted[i] == "safe_candidate") for i in candidate_indices) / max(len(candidate_indices), 1), 6), "abstain_count": sum(int(value == "abstain") for value in predicted), "invalid_action_count": sum(int(value not in MANIFEST_ACTIONS) for value in predicted), "mean_confidence": round(statistics.mean(item["confidence"] for item in details), 6), "target_distribution": dict(Counter(expected)), "prediction_distribution": dict(Counter(predicted))}
    return metrics, details


def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start_container() -> str:
    if _exists(CONTAINER_NAME):
        raise RuntimeError(f"refusing to reuse {CONTAINER_NAME}")
    _docker("run", "--detach", "--rm", "--pull=never", "--name", CONTAINER_NAME, "--publish", f"127.0.0.1:{PORT}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{PORT}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", CONTAINER_NAME)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError("PG-181 container did not become ready")


def _stop_container() -> None:
    if _exists(CONTAINER_NAME):
        _docker("stop", "--timeout", "5", CONTAINER_NAME)


def _route_entry(crawl: dict[str, Any]) -> dict[str, Any]:
    entries = [row for row in crawl.get("route_catalog", []) if row.get("path") == "/vul/urlredirect/urlredirect.php" and "url" in row.get("request_schema", {}).get("get_query_params", [])]
    if len(entries) != 1:
        raise ValueError("PG-181 requires exactly one observed GET url parameter route")
    return entries[0]


def _belief_after(signal: dict[str, Any], role: str) -> dict[str, float]:
    if bool(signal.get("candidate_signal")):
        return {"candidate_surface_signal": 0.65, "unknown_surface": 0.35}
    if role == "control":
        return {"no_surface_delta": 0.65, "unknown_surface": 0.35}
    return {"no_observed_effect": 0.60, "unknown_surface": 0.40}


def _model_replay(checkpoint_path: Path, variant: str, vocabulary: dict[str, int], crawl: dict[str, Any], device: torch.device) -> dict[str, Any]:
    route = _route_entry(crawl)
    schema = route["request_schema"]
    if schema.get("get_query_params") != ["url"]:
        raise ValueError("PG-181 route field grounding changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(len(vocabulary), variant).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    container_id = _start_container()
    client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=8.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    prior_records: list[dict[str, Any]] = []
    replay_steps: list[dict[str, Any]] = []
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
                replay_steps.append({"step_index": step_index, "model_action": predicted, "confidence": round(confidence, 6), "controller_decision": "abstain", "abstain_reason": "initial_state_requires_baseline"})
                break
            if step_index > 1 and predicted == "baseline":
                controller_abstain += 1
                replay_steps.append({"step_index": step_index, "model_action": predicted, "confidence": round(confidence, 6), "controller_decision": "abstain", "abstain_reason": "baseline_only_allowed_at_episode_start"})
                break
            if predicted == "abstain":
                controller_abstain += 1
                replay_steps.append({"step_index": step_index, "model_action": predicted, "confidence": round(confidence, 6), "controller_decision": "abstain", "abstain_reason": "model_abstain"})
                break
            role = "control" if predicted == "matched_control" else "candidate"
            marker = CONTROL if role == "control" else CANARY
            if step_index == 1:
                result = request_chain(client, method="GET", path="/vul/urlredirect/urlredirect.php", marker=None)
                method = "GET"
                placement = "none"
                fields: list[str] = []
            else:
                result = request_chain(client, method="GET", path="/vul/urlredirect/urlredirect.php", query={"url": marker}, marker=marker, baseline_status=200)
                method = "GET"
                placement = "query"
                fields = ["url"]
            manifest = action_manifest(path="/vul/urlredirect/urlredirect.php", surface="url_redirect_get", family="url_redirect", method=method, field_names=fields, probe_role=role if step_index > 1 else "negative_control", marker=marker if step_index > 1 else "pg181-baseline-ref")
            signal = dict(result["signal"])
            oracle = surface_oracle(family="url_redirect", method=method, signal=signal, oracle_contract_sha256=ORACLE_CONTRACT_SHA256)
            failure = failure_signature({"method": method, "role": role if step_index > 1 else "control", "candidate_signal": bool(signal.get("candidate_signal")), "positive": False, "positive_authority": False, "typed_available": False, "probe_round": step_index, "max_probe_rounds": 5}, prior_records=prior_records, max_steps=5, step_count=step_index)
            belief = _belief_after(signal, role)
            action_view = {"method": method, "placement": placement, "encoding_chain": manifest["encoding_chain"], "probe_ref": manifest["probe_ref"], "probe_sha256": manifest["payload_sha256"], "field_count": len(fields), "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
            replay_steps.append({"step_index": step_index, "model_action": predicted, "confidence": round(confidence, 6), "controller_decision": "send_safe_baseline" if step_index == 1 else "send_safe_canary", "action_manifest": action_view, "response_projection": result["projection"], "oracle_projection": oracle, "failure_signature": failure, "belief_after": belief, "decision": "abstain", "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": manifest, "response_projection": result["projection"], "failure_signature": failure, "belief_after": belief})
            prior_records.append({"method": method, "role": role if step_index > 1 else "control", "candidate_signal": bool(signal.get("candidate_signal")), "belief_after": belief})
    finally:
        client.close()
        _stop_container()
    return {"checkpoint": str(checkpoint_path.relative_to(ROOT)), "variant": variant, "target_route": "/vul/urlredirect/urlredirect.php", "parameter_name": "url", "fresh_container_id_hash": hashlib.sha256(container_id.encode("utf-8")).hexdigest(), "step_count": len(replay_steps), "sent_count": sum(int(item.get("controller_decision") in {"send_safe_baseline", "send_safe_canary"}) for item in replay_steps), "baseline_sent_count": sum(int(item.get("controller_decision") == "send_safe_baseline") for item in replay_steps), "control_sent_count": sum(int(item.get("model_action") == "matched_control" and item.get("controller_decision") == "send_safe_canary") for item in replay_steps), "candidate_sent_count": sum(int(item.get("model_action") == "safe_candidate" and item.get("controller_decision") == "send_safe_canary") for item in replay_steps), "controller_abstain_count": controller_abstain, "typed_positive_count": 0, "vulnerability_claim_allowed": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "steps": replay_steps}


def main() -> int:
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    crawl = json.loads(CRAWL_PATH.read_text(encoding="utf-8"))
    rows = build_manifest_examples(trace)
    vocabulary = manifest_vocabulary(rows)
    # Reuse PG-180's disjoint surface sets; this run is specifically selected
    # to replay a URL surface that is absent from the training partition.
    helper = _load_pg180_split_helper()
    train_rows = [row for row in rows if row["surface"] not in {"url_redirect_get", "xss_blind_post"}]
    dev_rows = [row for row in rows if row["surface"] == "xss_blind_post"]
    test_rows = [row for row in rows if row["surface"] == "url_redirect_get"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, Any]] = []
    for variant in MODEL_VARIANTS:
        for seed in SEEDS:
            results.append(_train_one(train_rows, dev_rows, test_rows, vocabulary, variant=variant, seed=seed, split_name="url_holdout", device=device))
    chosen = next(item for item in results if item["variant"] == "moe_large" and item["seed"] == SEEDS[0])
    replay = _model_replay(ROOT / chosen["checkpoint"], chosen["variant"], vocabulary, crawl, device)
    report = {"protocol_id": "pg-pk-181-manifest-decoder-replay-v1", "schema_version": "pg181-manifest-decoder-replay-report-v1", "status": "completed_safe_model_guided_local_replay", "source": {"trace": str(TRACE_PATH.relative_to(ROOT)), "crawl": str(CRAWL_PATH.relative_to(ROOT)), "image_digest": PIKACHU_IMAGE_DIGEST, "device": str(device)}, "training": {"rows": len(rows), "run_count": len(results), "results": results, "checkpoint_selection": {"variant": chosen["variant"], "seed": chosen["seed"], "reason": "diagnostic held-out URL replay; no promotion"}}, "replay": replay, "safety": {"loopback_only": True, "external_network": False, "fresh_container": True, "safe_canary_only": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "script_execution": False, "database_write": False, "credential_access": False, "typed_positive_count": 0, "vulnerability_claim_allowed": False, "memory_promotion_allowed": False}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    _write(TRACE_OUT_PATH, {"schema_version": "pg181-manifest-decoder-replay-trace-v1", "evaluation_only": True, "training_eligible": False, "training_run_count": len(results), "replay": replay, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False})
    protocol = {"protocol_id": "pg-pk-181-manifest-decoder-replay-v1", "schema_version": "pg181-manifest-decoder-replay-protocol-v1", "input_trace": str(TRACE_PATH.relative_to(ROOT)), "parameter_manifest": str(CRAWL_PATH.relative_to(ROOT)), "variants": MODEL_VARIANTS, "seeds": list(SEEDS), "replay": {"image": IMAGE, "loopback_port": PORT, "fresh_container_per_replay": True, "only_observed_parameter": "url", "raw_value_persistence": False}, "gates": {"model_output_allowlist": list(MANIFEST_ACTIONS), "manifest_validator_before_send": True, "unobserved_field_forbidden": True, "safe_canary_only": True, "typed_positive_required_for_vulnerability_label": True, "unknown_oracle_abstain": True, "training_promotion_allowed": False, "memory_promotion_allowed": False}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    lines = ["# PG-181 manifest decoder + local replay", "", f"device={device}; trained runs={len(results)}; selected={chosen['variant']} seed={chosen['seed']}", f"replay sent={replay['sent_count']}; controller abstain={replay['controller_abstain_count']}; typed positives=0", "", "模型只选择 baseline/control/safe_candidate/abstain；字段由浏览器 manifest 提供，回放器在发送前再次校验。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "training_runs": len(results), "selected_variant": chosen["variant"], "replay_sent": replay["sent_count"], "controller_abstain": replay["controller_abstain_count"], "typed_positive_count": 0, "training_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
