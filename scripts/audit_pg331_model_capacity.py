"""Read-only PG-331 vocabulary/sequence/model-capacity audit.

This does not train, contact Docker, or infer a missing field.  It answers a
different hard question from the information audit: after preserving the
whole-web token axes, can the selected decoder actually ingest the sequence
and represent the vocabulary?  A short legacy context window is reported as
truncation risk; the vocabulary is never reduced to make it fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg323_decoy_ask_anchor_dataset_v1.json"
ONTOLOGY = RESEARCH / "pg331_web_token_ontology_v1.json"
VOCABULARY = RESEARCH / "pg331_web_token_vocabulary_v1.json"
INFORMATION_AUDIT = RESEARCH / "pg331_information_preservation_audit_v1.json"
RULES = RESEARCH / "improvement_rules.json"
REPORT = RESEARCH / "pg331_model_capacity_audit_v1.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(path.name)
    return value


def _display_path(path: Path) -> str:
    """Keep artifact paths deterministic while accepting a custom dataset."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _required_inventory(ontology: Mapping[str, Any], rules: Mapping[str, Any]) -> set[str]:
    """Return the exact append-only inventory reserved by the vocabulary builder.

    Capacity must reject a manifest that can represent the observed page but
    cannot represent one of the explicit states (including ``absent``), axis
    framing markers, or the ontology-declared parameter-role taxonomy.  Keep
    this construction in lock-step with ``build_pg331_web_token_vocabulary``;
    otherwise a small capacity report could accidentally bless a lossy vocab.
    """

    inventory: set[str] = set()
    inventory.update(
        {
            "chunk_boundary=begin",
            "chunk_boundary=end",
            *{f"chunk_shape={bucket}" for bucket in ("zero", "one", "two", "few", "many")},
            *{f"chunk_index={bucket}" for bucket in ("zero", "one", "two", "few", "many")},
            *{f"chunk_count={bucket}" for bucket in ("zero", "one", "two", "few", "many")},
            *{f"chunk_digest=b{value:02x}" for value in range(256)},
        }
    )
    for axis, spec in dict(ontology.get("axes") or {}).items():
        axis_name = str(axis)
        presence = str(spec.get("presence_token", ""))
        if presence:
            inventory.update({f"{presence}=observed", f"{presence}=not_observed"})
        inventory.update({f"axis_begin={axis_name}", f"axis_end={axis_name}"})
        for field in list(spec.get("fields") or []):
            key = f"{axis_name}_field_{str(field)}"
            inventory.update({f"{key}={status}" for status in ("observed", "absent", "not_observed", "unknown")})

    role_taxonomy = dict((rules.get("pg331_vocabulary_contract_current") or {}).get("parameter_role_taxonomy") or {})
    inventory.update({f"param_role={role}" for role in list(role_taxonomy.get("roles") or []) if str(role)})
    # The builder unions the ontology inventory with both reserved sets before
    # serializing context_tokens.  Include the same reserved symbols here.
    reserved = ontology.get("reserved_tokens")
    if isinstance(reserved, Mapping):
        inventory.update(str(token) for token in (reserved.get("universal") or []))
        inventory.update(str(token) for token in (reserved.get("bucket_policy") or []))
    return inventory


def _bucket(value: int) -> str:
    return "zero" if value <= 0 else "one" if value == 1 else "two" if value == 2 else "few" if value <= 5 else "many"


def _representative_observation() -> dict[str, Any]:
    """A deterministic full-axis, multi-element page shape for capacity only."""

    elements = [
        {
            "tag": "form" if index == 0 else "input",
            "depth": 2 + (index % 3),
            "sibling_count": 4,
            "role": "form" if index == 0 else "textbox",
            "id_shape": "word_mixed",
            "class_shape": "word_mixed",
            "aria_role": "form" if index == 0 else "textbox",
            "text_shape": "alpha",
            "text_length": 32 + index,
            "attribute_presence": ["method", "name"] if index == 0 else ["type", "name"],
        }
        for index in range(8)
    ]
    links = [
        {"method": "GET", "target_shape": "path_like", "same_origin": "yes", "query_present": "present", "fragment_present": "absent"}
        for _ in range(4)
    ]
    parameters = [
        {"role": "query" if index % 2 == 0 else "form", "name_shape": "word_mixed", "value_type": "text", "presence": "present", "order": index + 1}
        for index in range(4)
    ]
    return {
        "document_structure": {"doctype": "html", "html_lang": "zh", "head_count": 1, "title_shape": "alpha", "meta_count": 6, "style_count": 2, "script_count": 4, "section_count": 5, "body_section_order": "nav_form_result", "elements": elements, "repeated_element_count": 3},
        "navigation": {"links": links, "path_segment_count": 4, "query_key_count": 3, "form_action_shape": "path_like", "navigation_event": "initial_load"},
        "request_transport": {"method": "POST", "placement": "form", "content_type_class": "form_urlencoded", "encoding_chain": "url_percent", "charset_class": "utf8", "body_shape": "structured_like", "query_count": 1, "form_count": 3, "json_field_count": 0, "multipart_part_count": 0, "header_presence_class": "basic", "cookie_presence_class": "absent", "csrf_presence_class": "present", "content_length": 128, "parameters": parameters},
        "response_transport": {"status_class": "3xx", "content_type_class": "html", "connection_outcome": "complete", "body_length": 2048, "redirect_hop_count": 2, "body_shape": "html", "charset_class": "utf8", "header_presence_class": "basic", "cache_shape": "private", "redirect_location_class": "same_origin", "redirect_chain_shape": "two_hop"},
        "javascript_surface": {"script_count": 4, "event_handler_count": 3, "fetch_count": 2, "xhr_count": 1, "ast_node_count": 96, "script_kind": "module", "module_presence": "present", "inline_external_class": "mixed", "source_category": "dom_input", "sink_category": "dom_update", "syntax_shape": "call_member", "ast_node_shape": "branch_call", "dynamic_code_presence": "absent", "storage_api_presence": "present", "fetch_method": "POST", "xhr_method": "GET", "fetch_target_shape": "path_like", "xhr_target_shape": "path_like", "event_handler_kinds": ["submit", "click", "change"]},
        "failure_feedback": {"failure_class": "none", "failure_stage": "none", "error_shape": "empty", "parse_error_class": "none", "encoding_error_class": "none", "redirect_error_class": "none", "blocked_reason_class": "none", "environment_failure_class": "none", "previous_action": "baseline_request", "next_action": "candidate_request", "repair_delta_axis": "none", "repair_outcome": "not_applicable", "timeout_ms": 0},
        "belief_and_replay": {"observation_presence": "present", "observation_delta_axis": "redirect_shape", "belief_prior_bucket": "low", "belief_posterior_bucket": "mid", "belief_delta_axis": "redirect_shape", "history_action": "candidate_request", "typed_available": "present", "evidence_present": "present", "negative_control": "present", "fresh_reset": "present", "replay_ready": "present", "reference_present": "present", "candidate_present": "present", "step_budget": "present", "evidence_hash_present": "present", "history_length": 3, "probe_count": 2},
    }


def _parameter_estimate(config: Mapping[str, int], model_vocab: int) -> dict[str, int]:
    d_model = int(config["d_model"])
    layers = int(config["layers"])
    experts = int(config["experts"])
    expert_ff = int(config["expert_ff"])
    max_len = int(config["max_length"])
    # PG-295 is a decoder-only model with one shared/tied token embedding and
    # LM head.  Count the union vocabulary once; treating context and target
    # inventories as separate heads would under/over-state the actual model
    # and could hide an accidental UNK remap.
    token_embedding = model_vocab * d_model
    position_embedding = max_len * d_model
    embeddings = token_embedding + position_embedding
    lm_head = 0
    per_layer = (4 * d_model * d_model) + (2 * d_model) + (experts * (2 * d_model * expert_ff + 2 * expert_ff)) + (d_model * experts + experts)
    parameters = embeddings + lm_head + layers * per_layer + 2 * d_model
    # A conservative fp16 activation estimate, including attention/workspace
    # multipliers; this is a planning number, not a claim about allocator use.
    activation_bytes = max_len * d_model * layers * 2 * 8
    return {
        "parameters": int(parameters),
        "embedding_parameters": int(embeddings),
        "token_embedding_parameters": int(token_embedding),
        "position_embedding_parameters": int(position_embedding),
        "lm_head_parameters": int(lm_head),
        "tied_lm_head": 1,
        "model_vocabulary_size": int(model_vocab),
        "estimated_fp16_activation_bytes": int(activation_bytes),
    }


def audit(
    *,
    dataset_path: Path | None = None,
    information_audit_path: Path | None = None,
    vocabulary_path: Path | None = None,
) -> dict[str, Any]:
    from app.pg331_web_tokenizer import tokenize_web_observation

    selected_dataset = (dataset_path or DATASET).resolve()
    selected_information = (information_audit_path or INFORMATION_AUDIT).resolve()
    selected_vocabulary = (vocabulary_path or VOCABULARY).resolve()
    dataset = _load(selected_dataset)
    ontology = _load(ONTOLOGY)
    vocabulary = _load(selected_vocabulary)
    information = _load(selected_information)
    # The parameter-role taxonomy is part of the append-only vocabulary
    # contract.  Capacity must use the same rules source as the manifest
    # builder instead of maintaining a second hand-written role list.
    rules = _load(RULES) if RULES.exists() else {}
    rows = [row for row in dataset.get("records", []) if isinstance(row, Mapping)]
    context_lengths = [len(list(row.get("context_tokens") or [])) for row in rows]
    target_lengths = [len(list(row.get("target_tokens") or [])) for row in rows]
    representative = tokenize_web_observation(_representative_observation())
    representative_context = int(representative["loss_report"]["model_token_count"])
    representative_canonical = int(representative["loss_report"]["token_count"])
    target_budget = max(max(target_lengths, default=0), 32)
    observed_max = max(max(context_lengths, default=0), representative_context)
    required_window = int(math.ceil((observed_max + target_budget) * 1.25))

    context_vocab = set(str(token) for token in vocabulary.get("context_tokens") or [])
    target_vocab = set(str(token) for token in vocabulary.get("target_tokens") or [])
    # Keep this construction identical to run_pg331_a800_next_token_smoke and
    # pg295_causal_moe: a single decoder vocabulary plus explicit PAD/UNK.
    model_vocab_tokens = context_vocab | target_vocab | {"[PAD]", "[UNK]"}
    model_vocab = len(model_vocab_tokens)
    # Keep this inventory exactly aligned with the append-only builder:
    # ontology fields (all four states), axis framing, chunk markers, the
    # versioned parameter-role taxonomy, and ontology-declared reserved
    # symbols.  A decoder that silently maps any of these to UNK cannot
    # distinguish an incomplete observation from an observed empty/negative
    # state.
    required_inventory = _required_inventory(ontology, rules)
    inventory_missing = sorted(required_inventory - context_vocab)

    configs = [
        {"id": "pg322_legacy", "d_model": 64, "layers": 2, "experts": 2, "expert_ff": 128, "max_length": 72},
        {"id": "pg331_minimum", "d_model": 128, "layers": 4, "experts": 4, "expert_ff": 512, "max_length": max(768, required_window)},
        {"id": "pg331_balanced", "d_model": 192, "layers": 6, "experts": 4, "expert_ff": 768, "max_length": max(1024, required_window)},
    ]
    variants: list[dict[str, Any]] = []
    for config in configs:
        estimates = _parameter_estimate(config, model_vocab)
        window_pass = int(config["max_length"]) >= required_window
        variants.append({"config": config, **estimates, "context_window_required": required_window, "context_window_pass": window_pass, "vocabulary_inventory_pass": not inventory_missing, "capacity_pass": bool(window_pass and not inventory_missing), "truncation_risk": bool(not window_pass)})

    report: dict[str, Any] = {
        "protocol_id": "pg-pk-331-model-capacity-audit-v1",
        "schema_version": "pg331-model-capacity-audit-v1",
        "status": "passed" if information.get("status") == "passed" and not inventory_missing and any(item["capacity_pass"] for item in variants) else "blocked",
        "information_audit_status": str(information.get("status", "missing")),
        "dataset": _display_path(selected_dataset),
        "dataset_sha256": _digest(dataset),
        "information_audit": _display_path(selected_information),
        "information_audit_sha256": _digest(information),
        "ontology": str(ONTOLOGY.relative_to(ROOT)),
        "ontology_sha256": _digest(ontology),
        "vocabulary": _display_path(selected_vocabulary),
        "vocabulary_sha256": str(vocabulary.get("vocabulary_sha256", "")),
        "input_vocabulary_size": len(context_vocab),
        "target_vocabulary_size": len(target_vocab),
        "model_vocabulary_size": model_vocab,
        "model_vocabulary_sha256": _digest(sorted(model_vocab_tokens)),
        "required_inventory_count": len(required_inventory),
        "inventory_missing_count": len(inventory_missing),
        "inventory_missing": inventory_missing,
        "dataset_context_length": {"min": min(context_lengths, default=0), "max": max(context_lengths, default=0), "mean": round(sum(context_lengths) / max(len(context_lengths), 1), 6), "count": len(context_lengths)},
        "dataset_target_length": {"min": min(target_lengths, default=0), "max": max(target_lengths, default=0), "mean": round(sum(target_lengths) / max(len(target_lengths), 1), 6), "count": len(target_lengths)},
        "representative_page": {"canonical_context_tokens": representative_canonical, "model_context_tokens": representative_context, "chunk_count": int(representative["loss_report"]["chunk_count"]), "chunk_size": int(representative["loss_report"]["chunk_size"]), "training_eligible": bool(representative["loss_report"]["training_eligible"])},
        "target_budget": target_budget,
        "required_context_window": required_window,
        "variants": variants,
        "interpretation": "词表和上下文窗口是独立硬门；旧 PG-322 max_length=72 无法容纳整网页代表序列，不能通过截断或删词伪装通过。",
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["audit_sha256"] = ""
    report["audit_sha256"] = _digest(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only PG-331 model capacity audit")
    parser.add_argument("--dataset", type=Path, default=DATASET, help="source-row dataset to measure; never silently truncated")
    parser.add_argument("--information-audit", type=Path, default=INFORMATION_AUDIT, help="matching information-preservation audit JSON")
    parser.add_argument("--vocabulary", type=Path, default=VOCABULARY, help="append-only vocabulary manifest for this dataset")
    parser.add_argument("--report", type=Path, default=REPORT, help="output report path")
    args = parser.parse_args()
    report = audit(dataset_path=args.dataset, information_audit_path=args.information_audit, vocabulary_path=args.vocabulary)
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
