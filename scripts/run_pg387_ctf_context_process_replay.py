"""Process-only PG-387 CTF-like context replay.

This runner is a bounded local diagnostic.  It parses a handful of safe local
JS snippets into abstract context tokens, uses the projection target to choose
ASK/repair, and sends only the pre-registered PG-385 canary through the
loopback fixture for the one context whose adapter is reviewed.  Wires and
marker values are ephemeral; the report contains hashes and bounded typed
projections only.  It is not a Docker/source-attested or cross-implementation
training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg385_filter_canary_fixture import evaluate_raw_request, start_filter_canary_server  # noqa: E402
from app.pg387_ctf_frontend_projection import project_case, project_js_source  # noqa: E402


SCHEMA_VERSION = "pg387-ctf-context-process-replay-v1"
ROLES = ("candidate", "reference", "negative", "replay")
REPLAY_CASES = (
    ("client_normalizer_order", "const value = new URLSearchParams(location.search).get('q'); const normalized = decodeURIComponent(value || ''); preview.textContent = normalized;"),
    ("script_loader_policy", "const value = new URLSearchParams(location.search).get('q'); const tag = document.createElement('script'); tag.src = value; document.body.appendChild(tag);"),
    ("storage_policy_guard", "const value = new URLSearchParams(location.search).get('q'); localStorage.setItem('last', value); preview.textContent = value;"),
    ("dynamic_code_guard", "const value = new URLSearchParams(location.search).get('q'); eval(value);"),
)
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _wire_hash(method: str, raw: str) -> str:
    return _sha({"method": method, "raw": raw})


def _role_row(*, case_ref: str, role: str, context: dict[str, Any], target: dict[str, Any], reset: dict[str, Any], projection: dict[str, Any] | None, status: str, evidence: str, action_changed: bool) -> dict[str, Any]:
    return {
        "case_ref": case_ref,
        "role": role,
        "context_tokens": context["context_tokens"],
        "target_tokens": target["target_tokens"],
        "javascript_surface": context["javascript_surface"],
        "reset": {"fresh_reset": bool(reset.get("fresh_reset")), "reset_id": int(reset.get("reset_id", 0)), "state_clean": bool(reset.get("state_clean", False)), "external_network": False},
        "model_decision": {"next_action": next(token.split("=", 1)[1] for token in target["target_tokens"] if token.startswith("next_action=")), "safe_to_send": next(token.split("=", 1)[1] for token in target["target_tokens"] if token.startswith("safe_to_send=")), "source": "abstract_context_policy"},
        "typed_projection": projection,
        "status": status,
        "action_changed": bool(action_changed),
        "evidence_sha256": evidence,
        "raw_context_stored": False,
        "raw_source_stored": False,
        "raw_wire_stored": False,
    }


def run_process_replay(*, show_wire: bool = False) -> tuple[dict[str, Any], list[str]]:
    rows: list[dict[str, Any]] = []
    wires: list[str] = []
    typed_count = 0
    negative_violations = 0
    ask_count = 0
    action_changed_count = 0
    for case_ref, source in REPLAY_CASES:
        context = project_js_source(source)
        target = project_case({"case_ref": case_ref})
        for role in ROLES:
            server = None
            thread = None
            try:
                reset = {"fresh_reset": True, "reset_id": 0, "state_clean": True, "external_network": False}
                projection = None
                status = "ask_context"
                action_changed = False
                wire = ""
                if context["safe_to_send"] and case_ref == "client_normalizer_order":
                    server, thread = start_filter_canary_server()
                    reset = server.fresh_reset()
                    baseline = evaluate_raw_request(method="GET", raw_query="q=PG387_BASE_0001%3A")
                    action_changed = baseline["filter_state"] == "filtered" and "next_action=repair" in target["target_tokens"]
                    role_marker = "NEG" if role == "negative" else role.upper()
                    raw_value = f"PG387_{role_marker}_0002%25253A"
                    wire = f"GET /local-filter?q={raw_value}"
                    projection = evaluate_raw_request(method="GET", raw_query=f"q={raw_value}")
                    raw_typed = bool(projection.get("typed_effect_confirmed"))
                    if role == "negative":
                        negative_violations += int(raw_typed)
                        status = "negative_violation" if raw_typed else "negative_no_effect"
                    else:
                        typed_count += int(raw_typed)
                        status = "typed_effect" if raw_typed else "typed_missing"
                    if action_changed:
                        action_changed_count += 1
                    evidence = _sha({"case_ref": case_ref, "role": role, "reset_id": reset["reset_id"], "projection": projection, "wire_hash": _wire_hash("GET", wire)})
                    if show_wire:
                        wires.append(wire)
                else:
                    ask_count += 1
                    reset = {"fresh_reset": False, "reset_id": 0, "state_clean": False, "external_network": False}
                    evidence = _sha({"case_ref": case_ref, "role": role, "context_hash": context["projection_sha256"], "status": "ask_context"})
                rows.append(_role_row(case_ref=case_ref, role=role, context=context, target=target, reset=reset, projection=projection, status=status, evidence=evidence, action_changed=action_changed))
            finally:
                if server is not None:
                    server.shutdown()
                    server.server_close()
                if thread is not None:
                    thread.join(timeout=2)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_process_only_context_diagnostic",
        "scope": "local_loopback_process_only",
        "model_decision_source": "abstract_context_policy_not_neural_checkpoint",
        "counts": {
            "rows": len(rows),
            "cases": len(REPLAY_CASES),
            "roles": len(ROLES),
            "typed_effect": typed_count,
            "ask_context": ask_count,
            "action_changed": action_changed_count,
            "negative_violation": negative_violations,
            "fresh_reset": sum(int(row["reset"]["fresh_reset"]) for row in rows),
        },
        "context_contract": {"raw_js_source_in_context": False, "raw_wire_stored": False, "external_network": False, "persistent_state_write": False},
        "execution": {"docker_started": False, "network_contacted": False, "gpu_touched": False, "training_started": False, "second_independent_implementation": False},
        "training_eligible": 0,
        "promotion": dict(PROMOTION),
        "rows": rows,
    }
    report["report_sha256"] = _sha({key: value for key, value in report.items() if key != "report_sha256"})
    return report, wires


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/pg387_ctf_context_process_replay_v1.json")
    parser.add_argument("--show-wire", action="store_true")
    args = parser.parse_args()
    report, wires = run_process_replay(show_wire=args.show_wire)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "counts": report["counts"], "wires_ephemeral": len(wires)}, ensure_ascii=False))
    if args.show_wire:
        for wire in wires:
            print(wire)


if __name__ == "__main__":
    main()
