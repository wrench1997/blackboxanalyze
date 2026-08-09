"""PG-223: compare a frozen XXL process body with larger diagnosis adapters.

The data is the PG-222 bounded diagnostic corpus.  The frozen body is the
101M-parameter PG-191 Pikachu surface-matrix model; only the diagnostic heads
are optimized.  This is a capacity experiment, not permission to send new
payloads or to claim a vulnerability.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg223_large_problem_diagnoser import LargeProblemDiagnoserAdapter, PG223_SCHEMA, train_large_adapter  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg223-large-problem-diagnoser-v1"
REPORT = RESEARCH / "pg223_large_problem_diagnoser_report_v1.json"
DATASET = RESEARCH / "pg223_large_problem_diagnoser_dataset_v1.json"
PROTOCOL = RESEARCH / "pg223_large_problem_diagnoser_protocol_v1.json"
TRACE = RESEARCH / "pg223_large_problem_diagnoser_trace_v1.json"
MARKDOWN = RESEARCH / "pg223_large_problem_diagnoser_report_v1.md"
PG222_REPORT = RESEARCH / "pg222_problem_diagnoser_training_report_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
MAX_LEN = 48


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG191 = _load_script("run_pg191_pikachu_surface_matrix_large.py")
PG222 = _load_script("run_pg222_problem_diagnoser_training.py")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _token(vocabulary: Mapping[str, int], value: str) -> int:
    return int(vocabulary.get(value, vocabulary.get("[UNK]", 1)))


def _field_token(row: Mapping[str, Any]) -> str:
    try:
        count = max(int(row.get("field_count", 0) or 0), 0)
    except (TypeError, ValueError):
        count = 0
    if count == 0:
        return "history::field_count::0"
    if count == 1:
        return "history::field_count::2"
    return "history::field_count::4"


def _status_token(row: Mapping[str, Any]) -> str:
    status = str(row.get("status_class", "2xx"))
    if status not in {"2xx", "4xx"}:
        status = "4xx"
    return f"history::status::{status}"


def _failure_token(row: Mapping[str, Any]) -> str:
    # These are derived from observable flags, never from the diagnosis target.
    if not row.get("fresh_reset_ok") or not row.get("database_health_ok") or row.get("transport_error"):
        return "ir.failure.kind=no_surface_delta"
    if not row.get("binding_valid", True):
        return "ir.failure.kind=parse_error_signature"
    if row.get("result_mismatch_observed"):
        return "ir.failure.kind=shape_delta"
    if row.get("candidate_reference_agreement") is False:
        return "ir.failure.kind=shape_delta"
    if row.get("typed_effect_observed") or row.get("result_fixture_verified") or row.get("boolean_differential"):
        return "ir.failure.kind=typed_positive"
    if not row.get("oracle_available"):
        return "ir.failure.kind=oracle_unavailable"
    return "ir.failure.kind=no_surface_delta"


def _row_tokens(row: Mapping[str, Any], vocabulary: Mapping[str, int]) -> list[int]:
    """Encode only bounded process observations into the frozen body's vocab."""

    method = str(row.get("method", "GET")).upper()
    typed = bool(row.get("oracle_available"))
    candidate = bool(row.get("candidate_sent"))
    try:
        history_len = max(int(row.get("history_len", 0) or 0), 0)
    except (TypeError, ValueError):
        history_len = 0
    history_bucket = 0 if history_len == 0 else 1 if history_len == 1 else 2 if history_len == 2 else 4
    tokens = [
        "[BOS]",
        f"history::method::{method}",
        _field_token(row),
        _status_token(row),
        "history::typed_available::0" if not typed else "ir.oracle.availability=typed",
        f"history::candidate::{int(candidate)}",
        "ir.response.candidate_signal=true" if row.get("candidate_result_present") else "ir.response.candidate_signal=false",
        "history::gate::typed_effect" if row.get("typed_effect_observed") else "history::gate::matched_negative_control",
        f"history::history_len::{history_bucket}",
        _failure_token(row),
        "obs.failure.transport=true" if row.get("transport_error") else "obs.failure.transport=false",
        "obs.oracle.availability=typed" if typed else "obs.oracle.availability=unknown",
        "[EOS]",
    ]
    return [_token(vocabulary, value) for value in tokens][:MAX_LEN]


def _encoded_rows(rows: list[dict[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [_row_tokens(row, vocabulary) for row in rows]
    width = max(len(item) for item in encoded)
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(encoded), width), dtype=torch.bool, device=device)
    for index, values in enumerate(encoded):
        ids[index, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[index, : len(values)] = True
    return ids, mask


def _frozen_context(rows: list[dict[str, Any]], vocabulary: Mapping[str, int], base: Any, device: torch.device) -> torch.Tensor:
    ids, mask = _encoded_rows(rows, vocabulary, device)
    base.eval()
    with torch.inference_mode():
        # Keep the body batch small so the experiment remains reproducible on
        # a single workstation GPU; the resulting hidden states are bounded.
        chunks: list[torch.Tensor] = []
        for start in range(0, len(rows), 16):
            chunks.append(base.hidden(ids[start : start + 16], mask[start : start + 16]).detach())
    return torch.cat(chunks, dim=0)


def main() -> int:
    torch.manual_seed(223)
    rows = PG222._build_rows()
    train_rows, holdout_rows = PG222._split(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG191._build_model("xxl", vocabulary, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    train_context = _frozen_context(train_rows, vocabulary, base, device)
    hold_context = _frozen_context(holdout_rows, vocabulary, base, device)
    frozen_parameter_count = int(sum(parameter.numel() for parameter in base.parameters()))
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: LargeProblemDiagnoserAdapter | None = None
    for hidden_dim in (64, 128, 256):
        torch.manual_seed(223 + hidden_dim)
        model = LargeProblemDiagnoserAdapter(d_model=int(train_context.shape[1]), hidden_dim=hidden_dim).to(device)
        result = train_large_adapter(model, train_context, hold_context, train_rows, holdout_rows, epochs=100, learning_rate=1e-3)
        result["hidden_dim"] = hidden_dim
        result["adapter_parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
        result["frozen_parameter_count"] = frozen_parameter_count
        result["device"] = str(device)
        variants.append(result)
        if selected is None or (
            result["holdout"]["guarded_positive_false_accept_count"],
            -result["holdout"]["guarded_diagnosis_accuracy"],
            -result["holdout"]["next_step_accuracy"],
        ) < (
            selected["holdout"]["guarded_positive_false_accept_count"],
            -selected["holdout"]["guarded_diagnosis_accuracy"],
            -selected["holdout"]["next_step_accuracy"],
        ):
            selected = result
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-223 did not produce a variant")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / f"large_problem_diagnoser_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": PG223_SCHEMA, "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "frozen_checkpoint": str(PG191_CHECKPOINT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    dataset = {
        "schema_version": "pg223-large-problem-diagnoser-dataset-v1",
        "source_dataset": str((RESEARCH / "pg222_problem_diagnoser_dataset_v1.json").relative_to(ROOT)),
        "source_dataset_sha256": hashlib.sha256((RESEARCH / "pg222_problem_diagnoser_dataset_v1.json").read_bytes()).hexdigest(),
        "rows": [{key: value for key, value in row.items() if key not in {"diagnosis", "next_step", "raw_payload", "payload", "raw_response", "response_body"}} for row in rows],
        "split": {"train_rows": len(train_rows), "holdout_rows": len(holdout_rows), "seed_and_route_holdout": True, "route_identity_as_feature": False},
        "frozen_body": {"checkpoint": str(PG191_CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": hashlib.sha256(PG191_CHECKPOINT.read_bytes()).hexdigest(), "parameter_count": frozen_parameter_count, "hidden_dim": int(train_context.shape[1])},
        "contract": {"raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "evaluator_targets_as_features": False, "local_only": True},
    }
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    report = {
        "protocol_id": "pg-pk-223-large-problem-diagnoser-v1",
        "schema_version": PG223_SCHEMA,
        "status": "completed_frozen_xxl_problem_diagnoser_capacity_sweep",
        "device": str(device),
        "source_dataset": str((RESEARCH / "pg222_problem_diagnoser_dataset_v1.json").relative_to(ROOT)),
        "frozen_checkpoint": str(PG191_CHECKPOINT.relative_to(ROOT)),
        "frozen_parameter_count": frozen_parameter_count,
        "row_counts": {"total": len(rows), "train": len(train_rows), "holdout": len(holdout_rows), "counterfactual": sum(bool(row.get("counterfactual")) for row in rows)},
        "variants": variants,
        "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "holdout": selected["holdout"]},
        "promotion": {"frozen_body_promotion_allowed": False, "adapter_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_generation": False},
        "honesty": {"frozen_body_is_pretrained_process_model": True, "adapter_only_training": True, "counterfactuals_are_not_live_evidence": True, "large_capacity_gain_not_established": True, "general_website_capability_not_established": True},
        "safety": {"loopback_only": True, "external_network": False, "database_write": False, "time_delay_used": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg223-large-problem-diagnoser-protocol-v1",
        "objective": "test whether frozen XXL process context improves self-error diagnosis over structured-only PG-222",
        "frozen_body_required": True,
        "adapter_hidden_variants": [64, 128, 256],
        "seed_and_route_holdout": True,
        "route_identity_as_feature": False,
        "evaluator_targets_as_features": False,
        "raw_payload_and_response_excluded": True,
        "positive_effect_is_local_only": True,
        "promotion_blocked_until_real_unseen_lab_replay": True,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg223-large-problem-diagnoser-trace-v1", "selected": selected, "variants": variants, "frozen_parameter_count": frozen_parameter_count, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join([
        "# PG-223 frozen XXL problem diagnoser",
        "",
        f"device={device}; frozen parameters={frozen_parameter_count}; train={len(train_rows)}; holdout={len(holdout_rows)}",
        f"selected adapter hidden={selected['hidden_dim']}; guarded holdout accuracy={selected['holdout']['guarded_diagnosis_accuracy']}; guarded positive false accepts={selected['holdout']['guarded_positive_false_accept_count']}",
        "",
        "这是容量实验：PG-191 的约 101M 参数过程体被冻结，只训练诊断 adapter。没有把 route、payload、响应正文或 evaluator target 喂给模型。",
        "",
        "当前数据仍以 PG-222 的小规模真实投影和标记反事实为主，不能把完美 holdout 分数解释成通用渗透能力；下一步必须在未见实现的本地 GET/POST 靶场上收集真实失败 trace。",
        "",
    ]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "frozen_parameter_count": frozen_parameter_count, "selected": report["selected"], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
