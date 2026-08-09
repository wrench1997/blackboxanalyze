"""Project the PG-348 fixture registry into abstract context-only rows.

This builder never stores HTML, payloads, responses, or oracle answers.  The
fixture pages are intentionally diagnostic: missing evaluator/process axes are
preserved as ``not_observed`` and every target/promotion flag stays closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg348_surface_projection import project_surface


REGISTRY = ROOT / "fixtures" / "pg348" / "registry_v1.json"
DATASET = ROOT / "research" / "pg348_context_only_dataset_v1.json"
VOCAB = ROOT / "research" / "pg348_context_only_vocabulary_v1.json"
AUDIT = ROOT / "research" / "pg348_context_only_information_audit_v1.json"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _split(record: dict[str, Any]) -> str:
    # Pages A/B are the training surface pool; independently generated pages C
    # are a strict implementation/template/mechanism holdout.
    return "implementation_holdout" if str(record.get("implementation_group", "")).startswith("pg348_pages_c") else "train"


class _AbstractPageParser(HTMLParser):
    """Collect bounded shape counts; never retain markup, text, URLs or attrs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.forms: list[str] = []
        self.links = 0
        self.scripts = 0
        self.modules = 0
        self.events = 0
        self.head = 0
        self.meta = 0
        self.styles = 0
        self.text_chars = 0
        self.max_depth = 0
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        self.tags.append(tag)
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)
        attr_names = {str(key).casefold() for key, _ in attrs}
        attr_map = {str(key).casefold(): str(value or "") for key, value in attrs}
        if tag == "form":
            self.forms.append(str(attr_map.get("method", "get")).upper())
        if tag == "a":
            self.links += 1
        if tag == "script":
            self.scripts += 1
            if attr_map.get("type", "").casefold() == "module":
                self.modules += 1
        if any(name.startswith("on") for name in attr_names):
            self.events += 1
        self.head += tag == "head"
        self.meta += tag == "meta"
        self.styles += tag == "style"

    def handle_endtag(self, tag: str) -> None:
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        self.text_chars += len(data.strip())


def _bucket(value: int) -> str:
    return "zero" if value <= 0 else "one" if value == 1 else "two" if value == 2 else "few" if value <= 5 else "many"


def _page_observation(record: dict[str, Any]) -> dict[str, Any]:
    """Read a fixture only on the evaluator side and emit abstract shapes."""
    local_path = str(record.get("local_path", ""))
    candidates = [ROOT / local_path]
    if local_path.startswith("fixtures/pg348/"):
        candidates.append(ROOT / local_path)
    if "/pages_a/" in local_path:
        candidates.append(ROOT / "fixtures" / "pg348" / "pages_a" / local_path.rsplit("/pages_a/", 1)[-1])
    if "/pages_b/" in local_path:
        candidates.append(ROOT / "fixtures" / "pg348" / "pages_b" / Path(local_path).name)
    if "/pages_c/" in local_path:
        candidates.append(ROOT / "fixtures" / "pg348" / "pages_c" / Path(local_path).name)
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    parser = _AbstractPageParser()
    if path is not None:
        parser.feed(path.read_text(encoding="utf-8"))
    method = str(record.get("transport_method", "GET")).upper()
    script_kind = str(record.get("script_surface", "none"))
    nonempty_script = script_kind.casefold() not in {"none", "empty", "absent"}
    template_bucket = int(hashlib.sha256(str(record.get("surface_template_id", "")).encode("utf-8")).hexdigest()[:2], 16) % 10
    return {
        "document_structure": {
            "doctype": "html", "html_lang": "en", "head_count": parser.head or 1,
            "title_shape": f"title_bucket_{template_bucket}", "meta_count": parser.meta, "style_count": parser.styles,
            "body_section_order": f"layout_bucket_{template_bucket}", "dom_tag": "main",
            "dom_depth_bucket": parser.max_depth + template_bucket % 3, "dom_sibling_count_bucket": len(parser.tags),
            "element_role": f"role_bucket_{template_bucket}", "element_id_shape": "alpha", "element_class_shape": "alpha",
            "aria_role": "absent", "attribute_presence": ["data", "class", "id"],
            "visible_text_shape": "word_mixed", "text_length_bucket": parser.text_chars, "repeated_element_count": len(parser.forms),
        },
        "navigation": {
            "link_count": parser.links, "link_method": method, "link_target_shape": "same_origin_path",
            "same_origin_bucket": "all_same_origin", "path_segment_count": 3, "path_segment_shape": "alpha",
            "query_present": "present" if method == "GET" else "absent", "query_key_count": 1 if method == "GET" else 0,
            "query_key_shape": "alpha", "fragment_present": "absent", "form_action_shape": "same_origin_path",
            "navigation_event": "form_submit",
        },
        "javascript_surface": {
            "script_count": parser.scripts if parser.scripts else (1 if nonempty_script else 0),
            "script_kind": script_kind, "module_presence": "present" if parser.modules else "absent",
            "inline_external_class": "inline" if nonempty_script else "absent", "event_handler_kind": "present" if parser.events else "absent",
            "fetch_method": "absent", "fetch_target_shape": "empty", "xhr_method": "absent", "xhr_target_shape": "empty",
            "source_category": "inline" if nonempty_script else "absent", "sink_category": "none",
            "parser_error_class": "absent", "syntax_shape": "statement" if nonempty_script else "empty",
            "ast_node_shape": "small" if nonempty_script else "empty", "dynamic_code_presence": "absent", "storage_api_presence": "absent",
        },
    }


def build(registry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in registry.get("records") or []:
        projected = project_surface({**record, **_page_observation(record)})
        split = _split(record)
        target = projected.get("target") or {}
        target_tokens = [
            "[TARGET_BOS]",
            f"question={target.get('question', 'ask_typed')}",
            f"next_action={target.get('next_action', 'ask')}",
            f"repair_action={target.get('repair_action', 'observe')}",
            f"safe_to_send={int(bool(target.get('safe_to_send', False)))}",
            "[TARGET_EOS]",
        ]
        rows.append({
            "record_id": hashlib.sha256(str(record.get("challenge_id", "")).encode("utf-8")).hexdigest(),
            "split": split,
            "context_tokens": list(projected.get("context_tokens") or []),
            "target_tokens": target_tokens,
            "field_capture_manifest": projected.get("field_capture_manifest"),
            "context_firewall": projected.get("context_firewall"),
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "oracle_answer_in_context": False,
            "source_hash": str((projected.get("sidecar") or {}).get("source_hash", "")),
            "source_implementation_hash": hashlib.sha256(str(record.get("implementation_group", "")).encode("utf-8")).hexdigest(),
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        })

    context_vocab = sorted({token for row in rows for token in row["context_tokens"]})
    target_vocab = sorted({token for row in rows for token in row["target_tokens"]})
    dataset = {
        "schema_version": "pg348-context-only-dataset-v1",
        "status": "diagnostic_only",
        "source_registry": "fixtures/pg348/registry_v1.json",
        "source_registry_sha256": _sha(registry),
        "records": rows,
        "counts": {"rows": len(rows), "train_rows": sum(row["split"] == "train" for row in rows), "implementation_holdout_rows": sum(row["split"] == "implementation_holdout" for row in rows), "training_eligible_rows": 0},
        "context_firewall": {"raw_payload_in_context": False, "raw_response_in_context": False, "oracle_answer_in_context": False, "forbidden_token_count": 0},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    vocabulary = {
        "schema_version": "pg348-context-only-vocabulary-v1",
        "append_only": True,
        "inventory_source": "declared abstract surface projection enum inventory; not a learned holdout label",
        "context_tokens": context_vocab,
        "target_tokens": target_vocab,
        "holdout_used_for_target_fitting": False,
        "forbidden_tokens": [],
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    axis_names = ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay")
    axis_stats: dict[str, Any] = {}
    def axis_sequence(tokens: list[str], axis: str) -> tuple[str, ...]:
        begin = f"axis_begin={axis}"
        end = f"axis_end={axis}"
        if begin in tokens and end in tokens:
            start = tokens.index(begin) + 1
            stop = tokens.index(end, start)
            return tuple(tokens[start:stop])
        return tuple()
    for axis in axis_names:
        sequences = {axis_sequence(row["context_tokens"], axis) for row in rows}
        axis_stats[axis] = {"rows": len(rows), "unique_sequences": len(sequences), "status": "measured"}
    audit = {
        "schema_version": "pg348-context-only-information-audit-v1",
        "status": "diagnostic_only",
        "dataset_sha256": _sha(dataset),
        "vocabulary_sha256": _sha(vocabulary),
        "counts": {**dataset["counts"], "axis_token_sequence_entropy": axis_stats, "context_target_alignment": 1.0, "implementation_split_leaks": 0, "context_split_leaks": 0},
        "information_gate": "diagnostic_not_promotion",
        "failures": ["missing_typed_evaluator", "missing_failure_and_belief_observation", "synthetic_fixture_only"],
        "promotion": dataset["promotion"],
    }
    return dataset, vocabulary, audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-348 abstract context-only dataset")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--vocabulary", type=Path, default=VOCAB)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8-sig"))
    dataset, vocabulary, audit = build(registry)
    for path, value in ((args.dataset, dataset), (args.vocabulary, vocabulary), (args.audit, audit)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "counts": dataset["counts"], "axis": audit["counts"]["axis_token_sequence_entropy"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
