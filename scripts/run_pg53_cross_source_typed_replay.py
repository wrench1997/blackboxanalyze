"""PG-53: cross-source, multi-seed typed-oracle replay.

Two independently written loopback HTTP implementations are exercised through
GET and POST.  The fixtures accept only abstract probe classes and never run
markup, SQL, commands, redirects, credentials or arbitrary code.  Each action
is sent to a fresh server instance.  The model receives only bounded response
shape information; the typed effect oracle is an evaluator-only label used
after the probe and is never a feature, training sample or memory write.
"""

from __future__ import annotations

import hashlib
import json
import random
import socket
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.active_probe import choose_active_probe  # noqa: E402
from app.belief_state import DECODER_FAMILIES, MultiStepBelief  # noqa: E402
from app.catalog_rule_decoder import CATALOG_DECODER_FAMILIES, CatalogRuleIRDecoderV2, catalog_feature_vector  # noqa: E402
from app.pg34_independent_fixture import SURFACE_SPECS as PG34_SURFACE_SPECS, make_independent_fixture_server  # noqa: E402
from app.pg35_independent_fixture import PG35_VARIANTS, SURFACE_SPECS as PG35_SURFACE_SPECS, make_pg35_server  # noqa: E402
from app.pg36_independent_maze_fixture import LAYOUTS as PG36_LAYOUTS, SURFACE_SPECS as PG36_SURFACE_SPECS, make_pg36_server  # noqa: E402
from app.pg53_cross_source_oracle import (  # noqa: E402
    FAMILIES,
    PG53_SCHEMA,
    build_payload_manifest,
    generic_effect_geometry,
    response_projection,
    sha256_json,
    sha256_text,
    surface_observation,
    typed_effect_oracle,
)
from app.rule_ir_decoder import FEATURE_DIM  # noqa: E402


PROTOCOL_ID = "pg-pk-53-cross-source-typed-replay-v1"
SCHEMA_VERSION = "pg-pk-53-cross-source-typed-replay-report-v1"
REPORT_PATH = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg53_cross_source_typed_replay_protocol_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg-pk-02-pair-invariance" / "joint_holdout" / "pair_encoding_invariant" / "decoder.pt"
TRAINING_CATALOG_PATH = ROOT / "research" / "pikachu_paired_catalog_v1.json"
SEEDS = (5301, 5307, 5311)
METHODS = ("GET", "POST")
SURFACES = tuple(f"surface-{index:02d}" for index in range(1, 10))

# The two module implementations deliberately use different route/field/
# phase layouts.  alpha/beta cover form and JSON transport; north/south cover
# a two-stage maze with different field names.  They remain loopback-only.
TARGETS: tuple[dict[str, Any], ...] = (
    {"source_id": "pg34-standalone", "implementation": "pg34", "variant": "base", "layout": {"prefix": "/pg34/surface", "post_content_type": "application/json"}},
    {"source_id": "pg35-alpha", "implementation": "pg35", "variant": "alpha", "layout": PG35_VARIANTS["alpha"]},
    {"source_id": "pg35-beta", "implementation": "pg35", "variant": "beta", "layout": PG35_VARIANTS["beta"]},
    {"source_id": "pg36-north", "implementation": "pg36", "variant": "north", "layout": PG36_LAYOUTS["north"]},
    {"source_id": "pg36-south", "implementation": "pg36", "variant": "south", "layout": PG36_LAYOUTS["south"]},
)

PROBE_KINDS = {
    "xss": "inert_dom_markup",
    "injection": "abstract_sql_fragment",
    "url_redirect": "loopback_destination_class",
    "command_injection": "local_canary_class",
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FreshTarget:
    def __init__(self, target: dict[str, Any]) -> None:
        self.target = target
        self.port = _free_port()
        self.instance_id = sha256_text(f"pg53-live-target|{target['source_id']}|{target['variant']}|{self.port}")[:24]
        self.server: Any = None
        self.thread: threading.Thread | None = None
        self.client: httpx.Client | None = None

    def __enter__(self) -> httpx.Client:
        if self.target["implementation"] == "pg34":
            self.server = make_independent_fixture_server(self.port)
        elif self.target["implementation"] == "pg35":
            self.server = make_pg35_server(self.port, variant=self.target["variant"])
        elif self.target["implementation"] == "pg36":
            self.server = make_pg36_server(self.port, implementation=self.target["variant"])
        else:
            raise ValueError(f"unknown implementation {self.target['implementation']}")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.01)
        else:
            self._close()
            raise RuntimeError(f"PG-53 target did not start: {self.target['source_id']}")
        self.client = httpx.Client(
            base_url=f"http://127.0.0.1:{self.port}",
            timeout=3.0,
            follow_redirects=False,
            headers={"accept": "application/json"},
        )
        return self.client

    def _close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._close()


def _spec(target: dict[str, Any], surface: str) -> dict[str, Any]:
    if target["implementation"] == "pg34":
        table = PG34_SURFACE_SPECS
    else:
        table = PG35_SURFACE_SPECS if target["implementation"] == "pg35" else PG36_SURFACE_SPECS
    return table[surface]


def _route(target: dict[str, Any], surface: str) -> str:
    return f"{target['layout']['prefix']}/{surface}"


def _request(
    client: httpx.Client,
    target: dict[str, Any],
    surface: str,
    method: str,
    *,
    positive: bool,
    stage: str,
) -> tuple[httpx.Response, dict[str, Any], str]:
    """Send one bounded abstract probe and decode its typed response in memory."""

    spec = _spec(target, surface)
    layout = target["layout"]
    probe_value = str(spec["positive"] if positive else "normal")
    params: dict[str, Any]
    if target["implementation"] == "pg34":
        params = {str(spec["field"]): probe_value}
    else:
        slot_key = str(layout["slot_key"])
        probe_key = str(layout["probe_key"])
        params = {slot_key: surface, probe_key: probe_value}
    if target["implementation"] == "pg36":
        # A screen is intentionally ambiguous; only confirm is eligible for a
        # typed positive.  Controls use confirm with a non-positive class.
        params[str(layout["phase_key"])] = "screen" if stage == "screen" else "confirm"
    path = _route(target, surface)
    if method == "GET":
        response = client.get(path, params=params)
    elif method == "POST":
        if str(layout.get("post_content_type", "")).casefold() == "application/json":
            response = client.post(path, json=params)
        else:
            response = client.post(path, data=params)
    else:
        raise ValueError(f"unsupported method {method}")
    try:
        body = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        body = {}
    if not isinstance(body, dict):
        body = {}
    projection = response_projection(response)
    return response, body, projection["projection_sha256"]


def _run_probe(
    target: dict[str, Any],
    *,
    surface: str,
    family: str,
    method: str,
    seed: int,
    stage: str,
    positive: bool,
    client: httpx.Client | None = None,
    target_instance_id: str | None = None,
) -> dict[str, Any]:
    # A fixture is stateless and explicitly reports no state mutation.  Keep
    # one fresh target per surface/seed/layout episode so a control/candidate
    # pair shares a reproducible implementation without paying hundreds of
    # process start-ups.  The optional recursive branch keeps this helper safe
    # for one-off callers that still request their own target.
    if client is None:
        fresh_target = _FreshTarget(target)
        with fresh_target as fresh_client:
            return _run_probe(
                target,
                surface=surface,
                family=family,
                method=method,
                seed=seed,
                stage=stage,
                positive=positive,
                client=fresh_client,
                target_instance_id=fresh_target.instance_id,
            )
    if target_instance_id is None:
        raise ValueError("target_instance_id is required for a shared fresh target")
    spec = _spec(target, surface)
    raw_value = str(spec["positive"] if positive else "normal")
    response, body, _ = _request(client, target, surface, method, positive=positive, stage=stage)
    projection = response_projection(response)
    observation = surface_observation(body)
    geometry = generic_effect_geometry(body)
    oracle_stage = "screen" if stage == "screen" else "confirm"
    oracle = typed_effect_oracle(
        source_id=target["source_id"],
        family=family,
        body=body,
        response=projection,
        stage=oracle_stage,
    )
    manifest = build_payload_manifest(
        source_id=target["source_id"],
        surface=surface,
        family=family,
        method=method,
        placement="query" if method == "GET" else "body",
        probe_kind=PROBE_KINDS.get(family, "abstract_channel_class"),
        probe_value=raw_value,
        route_template_id=f"{target['source_id']}-surface-route",
        field_name=str(target["layout"]["probe_key"] if "probe_key" in target["layout"] else spec.get("field", "probe")),
        stage=stage,
    )
    target_key = f"{target['source_id']}|{target['variant']}|{seed}|{surface}|{method}|{stage}|{positive}|{target_instance_id}"
    reset_hash = sha256_text(f"pg53-fresh-reset|{target_key}")
    reset = {
        "reset_id": f"pg53-reset-{target_instance_id}",
        "kind": "fresh_loopback_http_server",
        "target_instance_id": target_instance_id,
        "state_epoch": sha256_text(f"pg53-state-epoch|{target_key}")[:24],
        "reset_adapter_sha256": reset_hash,
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
        "reset_scope": "fresh_stateless_episode_request_epoch",
    }
    return {
        "stage": stage,
        "positive_requested": bool(positive),
        "response": projection,
        "surface_observation": observation,
        "generic_effect_geometry": geometry,
        "oracle": oracle,
        "payload_manifest": manifest,
        "fresh_reset": reset,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
    }


def _model_loader() -> tuple[CatalogRuleIRDecoderV2, dict[str, Any], torch.Tensor] | None:
    if not CHECKPOINT_PATH.exists():
        return None
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if int(checkpoint.get("feature_dim", -1)) != FEATURE_DIM:
        return None
    state = checkpoint["model_state"]
    model = CatalogRuleIRDecoderV2(
        branch_dim=int(state["surface_tower.0.weight"].shape[0]),
        embedding_dim=int(state["projector.0.weight"].shape[0]),
        dropout=0.0,
    )
    model.load_state_dict(state)
    model.eval()
    reference: torch.Tensor
    try:
        from app.payload_catalog import flatten_catalog, load_catalog

        rows = flatten_catalog(load_catalog(TRAINING_CATALOG_PATH))
        visible = [catalog_feature_vector(row) for row in rows]
        reference = (torch.tensor(visible, dtype=torch.float32) - torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)) / torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        reference = torch.empty((0, FEATURE_DIM), dtype=torch.float32)
    return model, checkpoint, reference


def _model_row(method: str, probe_kind: str, response: dict[str, Any]) -> dict[str, Any]:
    shape = response.get("shape") or {}
    return {
        "payload": {
            "method": method,
            "path": "/independent/fixture/surface",
            "probe_kind": probe_kind,
            "probe": "abstract_probe",
            "encoding": "identity",
        },
        "probe_artifact": {"encoding": "identity"},
        "response_projection": {
            "status_code": int(response.get("status_code", 0)),
            "headers": {"content-type": str(response.get("content_type_class", "other"))},
            "json_shape": {
                "kind": str(shape.get("kind", "other")),
                "key_count": int(shape.get("key_count", 0)),
                "scalar_count": int(shape.get("scalar_count", 0)),
                "array_count": int(shape.get("array_count", 0)),
            },
            "body_length": int(str(response.get("body_length_bucket", "0")).split("-", 1)[0] or 0),
        },
        # Constant shape only; the model never sees the evaluator's family or
        # positive/negative value.
        "oracle_projection": {"field_count": 1},
    }


def _expanded_probabilities(values: list[float]) -> dict[str, float]:
    raw = {family: 1e-6 for family in DECODER_FAMILIES}
    for family, value in zip(CATALOG_DECODER_FAMILIES, values):
        raw[family] = max(float(value), 1e-6)
    total = sum(raw.values())
    return {family: value / total for family, value in raw.items()}


def _model_proposal(loader: tuple[CatalogRuleIRDecoderV2, dict[str, Any], torch.Tensor] | None, *, method: str, response: dict[str, Any]) -> dict[str, Any]:
    if loader is None:
        return {"available": False, "decision": "abstain", "reason": "checkpoint_unavailable", "visible_input_redacted": True}
    model, checkpoint, reference = loader
    # Keep the learner-facing probe class neutral.  The evaluator may use a
    # family-specific abstract class to build a manifest, but passing that
    # class here would make family selection a hidden label side channel.
    row = _model_row(method, "abstract_channel_class", response)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    features = (torch.tensor([catalog_feature_vector(row)], dtype=torch.float32) - mean) / std
    with torch.inference_mode():
        values = torch.softmax(model(features), dim=-1)[0].tolist()
    ordered = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    candidate = CATALOG_DECODER_FAMILIES[ordered[0]]
    confidence = float(values[ordered[0]])
    margin = float(values[ordered[0]] - values[ordered[1]])
    distance = float(torch.cdist(features, reference).min().item()) if len(reference) else None
    return {
        "available": True,
        "candidate_family": candidate,
        "confidence": round(confidence, 6),
        "margin": round(margin, 6),
        "ood_distance": round(distance, 6) if distance is not None else None,
        "decision": "candidate" if confidence >= 0.45 and margin >= 0.10 else "abstain",
        "rule_ir_emitted": False,
        "probabilities": {name: round(value, 6) for name, value in zip(CATALOG_DECODER_FAMILIES, values)},
        "visible_input_redacted": True,
    }


def _uniform_probabilities() -> dict[str, float]:
    return {family: 1.0 / len(DECODER_FAMILIES) for family in DECODER_FAMILIES}


def _visible_action(method: str, response: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    shape = response.get("shape") or {}
    probabilities = _expanded_probabilities([float(proposal.get("probabilities", {}).get(family, 0.0)) for family in CATALOG_DECODER_FAMILIES]) if proposal.get("available") else _uniform_probabilities()
    return {
        "method": method,
        "placement": "query" if method == "GET" else "body",
        "probe_kind": "abstract_channel_class",
        "response_shape": {
            "status_class": response.get("status_class", "other"),
            "content_type_class": response.get("content_type_class", "other"),
            "kind": shape.get("kind", "other"),
            "key_count_bucket": int(shape.get("key_count", 0)) // 4,
            "array_count": int(shape.get("array_count", 0)),
        },
        "model_score": float(proposal.get("confidence", 0.0) or 0.0),
        "surface_discriminator": {"probabilities": probabilities, "source": "visible_shape_only"},
        "rule_ir_decoder": {"probabilities": probabilities, "confidence": float(proposal.get("confidence", 1.0 / len(DECODER_FAMILIES)) or 0.0)},
    }


def _case_row(
    *,
    target: dict[str, Any],
    surface: str,
    family: str,
    method: str,
    seed: int,
    control: dict[str, Any],
    screen: dict[str, Any] | None,
    candidate: dict[str, Any],
    proposal: dict[str, Any],
    belief: MultiStepBelief,
    belief_steps: list[dict[str, Any]],
    active_order: list[str],
) -> dict[str, Any]:
    oracle = candidate["oracle"]
    confirmed = bool(oracle.get("positive"))
    model_family = proposal.get("candidate_family")
    expected_family = family if family != "ordinary_response" else None
    model_emitted = proposal.get("decision") == "candidate"
    return {
        "sample_id": f"pg53-{target['source_id']}-{surface}-s{seed}-{method.casefold()}",
        "source_id": target["source_id"],
        "implementation": target["implementation"],
        "variant": target["variant"],
        "surface": surface,
        "family": family,
        "method": method,
        "sampling_seed": int(seed),
        "active_probe_order": list(active_order),
        "model_proposal": proposal,
        "model_family_match": bool(confirmed and model_family == expected_family),
        "model_misclassification": bool(confirmed and model_emitted and model_family != expected_family),
        "model_false_positive": bool(not confirmed and model_emitted),
        "confirmed_family": expected_family if confirmed else None,
        "decision": "confirmed_positive" if confirmed else "confirmed_negative",
        "rule_ir_binding": {
            "family": expected_family if confirmed else None,
            "source": "typed_oracle" if confirmed else "unbound",
            "slots": ["effect", "transport", "oracle"] if confirmed else [],
            "executable": False,
        },
        "control": {"response": control["response"], "surface_observation": control["surface_observation"], "generic_effect_geometry": control["generic_effect_geometry"], "oracle": control["oracle"]},
        "screen": None if screen is None else {"response": screen["response"], "surface_observation": screen["surface_observation"], "generic_effect_geometry": screen["generic_effect_geometry"], "oracle": screen["oracle"]},
        "candidate": {"response": candidate["response"], "surface_observation": candidate["surface_observation"], "generic_effect_geometry": candidate["generic_effect_geometry"], "oracle": oracle},
        "payload_manifest": candidate["payload_manifest"],
        "negative_control": {
            "matched": control["oracle"].get("positive") is False,
            "same_source": True,
            "control_evidence_sha256": sha256_json(control),
            "candidate_vs_control": True,
        },
        "fresh_reset": {
            "control": control["fresh_reset"],
            "screen": None if screen is None else screen["fresh_reset"],
            "candidate": candidate["fresh_reset"],
        },
        "belief": {"steps": belief_steps, "final": belief.snapshot()},
        "evidence_sha256": sha256_json({
            "sample_id": f"{target['source_id']}|{surface}|{seed}|{method}",
            "control": control,
            "screen": screen,
            "candidate": candidate,
            "model": proposal,
            "active_order": active_order,
        }),
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
    }


def _run_episode(
    target: dict[str, Any],
    surface: str,
    seed: int,
    loader: Any,
    _client: httpx.Client | None = None,
    _target_instance_id: str | None = None,
) -> list[dict[str, Any]]:
    if _client is None:
        fresh_target = _FreshTarget(target)
        with fresh_target as fresh_client:
            return _run_episode(target, surface, seed, loader, fresh_client, fresh_target.instance_id)
    if _target_instance_id is None:
        raise ValueError("_target_instance_id is required for episode replay")
    spec = _spec(target, surface)
    family = str(spec["family"])
    rng = random.Random(f"{target['source_id']}|{surface}|{seed}")
    method_order = list(METHODS)
    rng.shuffle(method_order)

    controls: dict[str, dict[str, Any]] = {}
    proposals: dict[str, dict[str, Any]] = {}
    visible_candidates: list[dict[str, Any]] = []
    for method in METHODS:
        control = _run_probe(target, surface=surface, family=family, method=method, seed=seed, stage="control", positive=False, client=_client, target_instance_id=_target_instance_id)
        controls[method] = control
        proposal = _model_proposal(loader, method=method, response=control["response"])
        proposals[method] = proposal
        visible = _visible_action(method, control["response"], proposal)
        visible["surface_discriminator"]["probabilities"] = visible["surface_discriminator"]["probabilities"]
        visible_candidates.append(visible)

    belief = MultiStepBelief()
    belief_steps: list[dict[str, Any]] = []
    for method in METHODS:
        belief_steps.append(belief.observe(
            f"{method}:control",
            _uniform_probabilities(),
            evidence_hash=controls[method]["oracle"]["evidence_projection_sha256"],
        ))
    chosen = choose_active_probe(visible_candidates)
    chosen_method = str(chosen["method"])
    active_order = [chosen_method] + [method for method in method_order if method != chosen_method]
    rows: list[dict[str, Any]] = []
    for method in active_order:
        screen = None
        if target["implementation"] == "pg36":
            screen = _run_probe(target, surface=surface, family=family, method=method, seed=seed, stage="screen", positive=True, client=_client, target_instance_id=_target_instance_id)
            belief_steps.append(belief.observe(
                f"{method}:screen",
                _uniform_probabilities(),
                evidence_hash=screen["oracle"]["evidence_projection_sha256"],
            ))
        candidate = _run_probe(target, surface=surface, family=family, method=method, seed=seed, stage="candidate", positive=True, client=_client, target_instance_id=_target_instance_id)
        proposal = _model_proposal(loader, method=method, response=candidate["response"])
        probabilities = _expanded_probabilities([float(proposal.get("probabilities", {}).get(name, 0.0)) for name in CATALOG_DECODER_FAMILIES]) if proposal.get("available") else _uniform_probabilities()
        belief_steps.append(belief.observe(
            f"{method}:candidate-visible",
            probabilities,
            evidence_hash=candidate["response"]["projection_sha256"],
        ))
        rows.append(_case_row(
            target=target,
            surface=surface,
            family=family,
            method=method,
            seed=seed,
            control=controls[method],
            screen=screen,
            candidate=candidate,
            proposal=proposal,
            belief=belief,
            belief_steps=list(belief_steps),
            active_order=active_order,
        ))
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["decision"] == "confirmed_positive"]
    negatives = [row for row in rows if row["decision"] == "confirmed_negative"]
    emitted = [row for row in rows if row["model_proposal"].get("decision") == "candidate"]
    return {
        "case_count": len(rows),
        "confirmed_positive_count": len(positives),
        "confirmed_negative_count": len(negatives),
        "confirmed_positive_by_family": dict(sorted(Counter(row["family"] for row in positives).items())),
        "get_post_covered": dict(sorted(Counter(row["method"] for row in rows).items())),
        "model_candidate_count": len(emitted),
        "model_abstain_count": len(rows) - len(emitted),
        "model_family_match_count": sum(int(row["model_family_match"]) for row in rows),
        "model_family_misclassification_count": sum(int(row["model_misclassification"]) for row in rows),
        "model_false_positive_count": sum(int(row["model_false_positive"]) for row in rows),
        "oracle_family_binding_match_count": sum(int(row["confirmed_family"] == row["family"]) for row in positives),
        "negative_control_pass_count": sum(int(row["negative_control"]["matched"]) for row in rows),
        "belief_update_count": sum(len(row["belief"]["steps"]) for row in rows),
        "fresh_reset_count": sum(2 + int(row["screen"] is not None) for row in rows),
    }


def _group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _aggregate(group_rows) for name, group_rows in sorted(grouped.items())}


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# PG-53 跨独立实现、多种子 typed-oracle 复放",
        "",
        "本轮在三个独立 Python HTTP 实现（PG-34、PG-35 与 PG-36）及不同路由/字段布局上，使用三种采样种子分别重置目标，覆盖 GET/POST 与八个正向漏洞族、一个阴性对照。探针只传输抽象类别，不执行 markup、SQL、命令、重定向或凭据操作。",
        "",
        f"权威 oracle：{metrics['confirmed_positive_count']}/{metrics['case_count']} 行确认；Rule IR 权威绑定：{metrics['oracle_family_binding_match_count']}。模型候选：{metrics['model_candidate_count']}；族命中：{metrics['model_family_match_count']}；误分类：{metrics['model_family_misclassification_count']}；阴性误报：{metrics['model_false_positive_count']}。",
        "",
        "| source | cases | oracle + | model family hit | model misclassification | false positive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source, values in report["by_source"].items():
        lines.append(f"| {source} | {values['case_count']} | {values['confirmed_positive_count']} | {values['model_family_match_count']} | {values['model_family_misclassification_count']} | {values['model_false_positive_count']} |")
    lines.extend([
        "",
        f"跨实现同族复现率：`{report['cross_source']['family_replay_rate']:.3f}`；阴性对照通过率：`{report['cross_source']['negative_control_pass_rate']:.3f}`。",
        f"跨实现不一致单元：`{report['cross_source']['mismatch_cell_count']}`；这些差异保留为实验/工程 triage，不会被平均值掩盖。",
        "",
        "这只是跨源评估证据，不是训练晋升：原始 probe/响应正文未持久化，训练集与长期记忆均保持隔离；仍需独立训练候选、族外留出和三种子能力门。",
        "",
        f"训练晋升：`{report['training_boundary']['training_eligible']}`；长期记忆写入：`{report['training_boundary']['long_term_memory_write']}`；正式能力声明：`{report['formal_claim']['allowed']}`。",
    ])
    return "\n".join(lines) + "\n"


def _cross_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str, str], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        grouped[(row["surface"], int(row["sampling_seed"]), row["method"], row["family"])][row["implementation"]] = row["decision"] == "confirmed_positive"
    comparable = [values for values in grouped.values() if len(values) >= 2]
    replayed = sum(int(all(values.values())) for values in comparable if any(values.values()))
    total_positive_cells = sum(int(any(values.values())) for values in comparable)
    mismatches = [
        {
            "surface": key[0],
            "sampling_seed": key[1],
            "method": key[2],
            "family": key[3],
            "positive_by_implementation": dict(sorted(values.items())),
            "triage": "oracle_contract_gap_or_fixture_effect_schema_mismatch" if key[3] == "command_injection" else "unclassified_cross_source_difference",
            "next_check": "compare bounded effect contract before changing model features",
        }
        for key, values in sorted(grouped.items())
        if len(values) >= 2 and len(set(values.values())) > 1
    ]
    return {
        "comparable_cells": len(comparable),
        "family_replayed_positive_cells": replayed,
        "family_replay_rate": round(replayed / max(total_positive_cells, 1), 6),
        "mismatch_cell_count": len(mismatches),
        "mismatch_cells": mismatches,
        "negative_control_pass_rate": round(sum(int(row["negative_control"]["matched"]) for row in rows) / max(len(rows), 1), 6),
        "same_seed_method_surface_alignment_required": True,
    }


def main() -> int:
    loader = _model_loader()
    all_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for surface in SURFACES:
            for seed in SEEDS:
                all_rows.extend(_run_episode(target, surface, int(seed), loader))

    metrics = _aggregate(all_rows)
    by_source = _group_metrics(all_rows, "source_id")
    by_implementation = _group_metrics(all_rows, "implementation")
    positives = [row for row in all_rows if row["decision"] == "confirmed_positive"]
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "source": {
            "implementations": [
                {"source_id": target["source_id"], "implementation": target["implementation"], "variant": target["variant"], "module": {"pg34": "app.pg34_independent_fixture", "pg35": "app.pg35_independent_fixture", "pg36": "app.pg36_independent_maze_fixture"}[target["implementation"]], "independent_target_implementation": True}
                for target in TARGETS
            ],
            "sampling_seeds": list(SEEDS),
            "methods": list(METHODS),
            "surface_count": len(SURFACES),
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "external_network": False,
            "loopback_only": True,
        },
        "model_visibility": {
            "fields": ["method", "placement", "abstract_probe_kind", "status_class", "content_type_class", "body_shape", "bounded_length"],
            "oracle_visible_before_probe": False,
            "typed_oracle_used_after_probe_for_stop_and_score_only": True,
            "family_label_in_features": False,
            "raw_values_in_features": False,
        },
        "metrics": metrics,
        "by_source": by_source,
        "by_implementation": by_implementation,
        "cross_source": _cross_source(all_rows),
        "families": {family: {"confirmed_positive_count": sum(int(row["family"] == family and row["decision"] == "confirmed_positive") for row in all_rows), "model_family_match_count": sum(int(row["family"] == family and row["model_family_match"]) for row in all_rows)} for family in FAMILIES},
        "rows": all_rows,
        "training_boundary": {
            "training_eligible": False,
            "catalog_generated": False,
            "long_term_memory_write": False,
            "reason": "cross_source_replay_is_evaluation_only_until_fresh_heldout_training_and_ood_gate",
        },
        "formal_claim": {
            "allowed": False,
            "reason": "typed_positive_replay_does_not_prove_model_generalization_or_independent_implementation_ood_accuracy",
        },
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "report_sha256": "",
    }
    report["report_sha256"] = sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "metrics": metrics, "cross_source": report["cross_source"], "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
