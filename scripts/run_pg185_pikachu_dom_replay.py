"""PG-185: model-guided, read-only DOM-surface replay on Pikachu.

The frozen PG-181 action model chooses the next bounded role.  The controller
binds that role to two browser-observed GET surfaces and sends either a plain
canary or an inert DOM marker.  The response body is inspected in memory by a
detached DOM oracle; only projections and hashes are written to disk.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import sha256_json  # noqa: E402
from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg179b_iterative_probe import SAFE_REDIRECT_PORTS  # noqa: E402,F401
from app.pg181_manifest_decoder import build_model, last_logits, pre_action_tokens, restrict_manifest_action  # noqa: E402
from app.pg185_pikachu_dom_adapter import build_dom_action_manifest, build_query, project_dom_response  # noqa: E402


RESEARCH = ROOT / "research"
REPORT_PATH = RESEARCH / "pg185_pikachu_dom_replay_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg185_pikachu_dom_replay_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg185_pikachu_dom_replay_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg185_pikachu_dom_replay_report_v1.md"
CRAWL_PATH = RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg181-manifest-decoder-v1" / "url_holdout" / "moe_large_seed18101.pt"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
CONTAINER_NAME = "sift-pg185-pikachu-dom"
PORT = 3104
BASE_URL = f"http://127.0.0.1:{PORT}"
MAX_STEPS = 5
CANARY_PREFIX = "pg185-canary"
CONTROL_PREFIX = "pg185-control"
ROUTE_SPECS = (
    ("/vul/xss/xss_reflected_get.php", "xss_reflected_get", ("message", "submit")),
    ("/vul/xss/xss_dom_x.php", "xss_dom_x", ("text",)),
)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def _exists() -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{CONTAINER_NAME}$", "--format", "{{.Names}}"))


def _start_container() -> str:
    if _exists():
        raise RuntimeError(f"refusing to reuse {CONTAINER_NAME}; PG-185 requires a fresh target")
    _docker(
        "run", "--detach", "--rm", "--pull=never", "--name", CONTAINER_NAME,
        "--publish", f"127.0.0.1:{PORT}:8090", IMAGE,
        "bash", "-lc", "/app/run.sh; exec tail -f /dev/null",
    )
    deadline = time.monotonic() + 140.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{BASE_URL}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", CONTAINER_NAME)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError("PG-185 Pikachu container did not become ready")


def _stop_container() -> None:
    if _exists():
        _docker("stop", "--timeout", "5", CONTAINER_NAME)


def _load_observed_routes() -> list[dict[str, Any]]:
    crawl = json.loads(CRAWL_PATH.read_text(encoding="utf-8"))
    rows = crawl.get("request_response_rows", [])
    observed: list[dict[str, Any]] = []
    for path, surface, fields in ROUTE_SPECS:
        matches = [
            row for row in rows
            if row.get("method") == "GET"
            and row.get("route_path") == path
            and sorted(row.get("request_schema", {}).get("query_params", [])) == sorted(fields)
        ]
        if len(matches) != 1:
            raise ValueError(f"PG-185 expected one observed GET route for {surface}, got {len(matches)}")
        observed.append({"path": path, "surface": surface, "field_names": list(fields), "manifest_row_sha256": _sha256_json(matches[0])})
    return observed


def _belief_after(signal: dict[str, Any], typed_surface_effect: bool, role: str) -> dict[str, float]:
    if typed_surface_effect:
        # Keep the model input in the PG-181 vocabulary.  The DOM evaluator's
        # richer typed effect is retained in the evidence projection, while
        # the decoder sees the already-known abstract candidate/unknown pair.
        return {"candidate_surface_signal": 0.72, "unknown_surface": 0.28}
    if bool(signal.get("candidate_signal")):
        return {"candidate_surface_signal": 0.60, "unknown_surface": 0.40}
    if role == "control":
        return {"no_surface_delta": 0.65, "unknown_surface": 0.35}
    return {"no_observed_effect": 0.60, "unknown_surface": 0.40}


def _replay_route(
    model: torch.nn.Module,
    vocabulary: dict[str, int],
    route: dict[str, Any],
    device: torch.device,
    *,
    target_instance_hash: str,
) -> dict[str, Any]:
    path = str(route["path"])
    surface = str(route["surface"])
    fields = [str(item) for item in route["field_names"]]
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    prior_records: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    controller_abstain = 0
    typed_surface_effect_count = 0
    baseline_status: int | None = None
    try:
        for step_index in range(1, MAX_STEPS + 1):
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
            marker = f"{CONTROL_PREFIX}-{surface[-4:]}-{step_index}" if role == "control" else f"{CANARY_PREFIX}-{surface[-4:]}-{step_index}"
            manifest = build_dom_action_manifest(path=path, surface=surface, field_names=fields, probe_role=role if step_index > 1 else "negative_control", marker=marker)
            if step_index == 1:
                response = client.get(path)
                baseline_status = int(response.status_code)
                projected = project_dom_response(response, marker=None)
                controller_decision = "send_safe_baseline"
                action_role = "negative_control"
            else:
                query, oracle_marker = build_query(field_names=fields, role=role, marker=marker)
                response = client.get(path, params=query)
                projected = project_dom_response(response, marker=oracle_marker, baseline_status=baseline_status)
                controller_decision = "send_inert_dom_candidate" if role == "candidate" else "send_safe_canary"
                action_role = role

            signal = dict(projected["oracle_projection"].get("signals") or {})
            signal["candidate_signal"] = bool(projected["oracle_projection"].get("candidate_signal"))
            typed_effect = bool(projected["typed_surface_effect"])
            typed_surface_effect_count += int(typed_effect)
            failure = failure_signature(
                {
                    "method": "GET",
                    "role": action_role,
                    "candidate_signal": signal["candidate_signal"],
                    "positive": False,
                    "positive_authority": False,
                    "typed_available": False,
                    "probe_round": step_index,
                    "max_probe_rounds": MAX_STEPS,
                },
                prior_records=prior_records,
                max_steps=MAX_STEPS,
                step_count=step_index,
            )
            belief = _belief_after(signal, typed_effect, role)
            action_view = {
                "method": manifest["method"],
                "placement": manifest["placement"],
                "encoding_chain": manifest["encoding_chain"],
                "probe_kind": manifest["probe_kind"],
                "probe_ref": manifest["probe_ref"],
                "payload_sha256": manifest["payload_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "field_names": fields,
                "safety": manifest["safety"],
            }
            steps.append(
                {
                    "step_index": step_index,
                    "model_action": predicted,
                    "confidence": round(confidence, 6),
                    "controller_decision": controller_decision,
                    "action_manifest": action_view,
                    "response_projection": projected["response_projection"],
                    "oracle_projection": projected["oracle_projection"],
                    "typed_surface_effect": typed_effect,
                    "failure_signature": failure,
                    "belief_after": belief,
                    "decision": "abstain",
                    "vulnerability_claim_allowed": False,
                    "online_weight_update": False,
                    "long_term_memory_write": False,
                }
            )
            history.append(
                {
                    "action_manifest": manifest,
                    "response_projection": projected["response_projection"],
                    "failure_signature": failure,
                    "belief_after": belief,
                }
            )
            prior_records.append({"method": "GET", "role": action_role, "candidate_signal": signal["candidate_signal"], "belief_after": belief})
    finally:
        client.close()
    return {
        "surface": surface,
        "path": path,
        "field_names": fields,
        "target_instance_hash": target_instance_hash,
        "step_count": len(steps),
        "sent_count": sum(int(item.get("controller_decision", "").startswith("send_")) for item in steps),
        "candidate_sent_count": sum(int(item.get("controller_decision") == "send_inert_dom_candidate") for item in steps),
        "typed_surface_effect_count": typed_surface_effect_count,
        "controller_abstain_count": controller_abstain,
        "typed_positive_count": 0,
        "vulnerability_claim_allowed": False,
        "steps": steps,
    }


def main() -> int:
    random.seed(18501)
    routes = _load_observed_routes()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    variant = str(checkpoint["variant"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(vocabulary), variant).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    container_id = _start_container()
    checkpoint_hash = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    target_instance_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
    try:
        runs = [_replay_route(model, vocabulary, route, device, target_instance_hash=target_instance_hash) for route in routes]
    finally:
        _stop_container()

    report = {
        "protocol_id": "pg-pk-185-pikachu-dom-replay-v1",
        "schema_version": "pg185-pikachu-dom-replay-report-v1",
        "status": "completed_model_guided_read_only_dom_surface_replay",
        "source": {
            "crawl_manifest": str(CRAWL_PATH.relative_to(ROOT)),
            "image": IMAGE,
            "loopback_port": PORT,
            "fresh_container": True,
            "observed_route_count": len(routes),
        },
        "model": {
            "variant": variant,
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_hash,
            "device": str(device),
            "online_weight_update": False,
            "memory_promotion_allowed": False,
        },
        "counts": {
            "route_count": len(runs),
            "sent_count": sum(item["sent_count"] for item in runs),
            "candidate_sent_count": sum(item["candidate_sent_count"] for item in runs),
            "typed_surface_effect_count": sum(item["typed_surface_effect_count"] for item in runs),
            "typed_positive_count": 0,
            "controller_abstain_count": sum(item["controller_abstain_count"] for item in runs),
        },
        "runs": runs,
        "promotion": {
            "typed_dom_surface_effect_is_vulnerability": False,
            "vulnerability_claim_allowed": False,
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "reason": "detached DOM proves only an inert surface effect; no script execution or exploit string was used",
        },
        "safety": {
            "loopback_only": True,
            "fresh_container": True,
            "inert_dom_markup_only": True,
            "script_execution": False,
            "database_write": False,
            "credentials": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    _write(
        TRACE_PATH,
        {
            "schema_version": "pg185-pikachu-dom-replay-trace-v1",
            "evaluation_only": True,
            "training_eligible": False,
            "model_checkpoint_sha256": checkpoint_hash,
            "runs": runs,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "online_weight_update": False,
            "long_term_memory_write": False,
        },
    )
    protocol = {
        "protocol_id": "pg-pk-185-pikachu-dom-replay-v1",
        "schema_version": "pg185-pikachu-dom-replay-protocol-v1",
        "observed_parameter_authority": True,
        "routes": routes,
        "model_output_allowlist": ["baseline", "matched_control", "safe_candidate", "abstain"],
        "manifest_validator_before_send": True,
        "probe_contract": {"kind": "inert_dom_markup", "script_execution": False, "raw_value_persistence": False},
        "oracle_contract": {"detached_dom_surface_only": True, "typed_dom_effect_not_vulnerability": True, "typed_positive_count_required_for_vulnerability": False},
        "gates": {"loopback_only": True, "fresh_container": True, "unknown_oracle_action": "abstain", "training_allowed": False, "memory_promotion_allowed": False},
        "checkpoint_sha256": checkpoint_hash,
    }
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text(
        "\n".join(
            [
                "# PG-185 Pikachu DOM surface replay",
                "",
                f"model={variant}; routes={len(runs)}; sent={report['counts']['sent_count']}; candidates={report['counts']['candidate_sent_count']}; typed_dom_effects={report['counts']['typed_surface_effect_count']}",
                "",
                "模型参与角色选择，控制器只向浏览器清单中的 GET 参数发送不执行脚本的 inert DOM 标记；typed DOM effect 不等于 XSS 阳性。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol_id": report["protocol_id"],
                "variant": variant,
                "route_count": len(runs),
                "sent_count": report["counts"]["sent_count"],
                "candidate_sent_count": report["counts"]["candidate_sent_count"],
                "typed_surface_effect_count": report["counts"]["typed_surface_effect_count"],
                "typed_positive_count": 0,
                "training_allowed": False,
                "report": str(REPORT_PATH.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
