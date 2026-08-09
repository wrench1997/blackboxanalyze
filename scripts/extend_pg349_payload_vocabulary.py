"""Append source-grounded payload-shape tokens to a PG-349 vocabulary.

This is a vocabulary reservation step, not a payload importer.  It reads only
the abstract dimensions in ``pg349_payload_probe_vocabulary_v1.json`` and
adds namespaced ``probe_shape_*`` tokens.  No source examples or executable
strings are copied into the model vocabulary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "research" / "pg349_dynamic_typed_vocabulary_v7.json"
DEFAULT_PAYLOAD = ROOT / "research" / "pg349_payload_probe_vocabulary_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg349_dynamic_typed_vocabulary_v8.json"
_SAFE = re.compile(r"^[a-z0-9_=-]+$")
_FORBIDDEN = ("javascript:", "<script", "alert(", "document.cookie", "http://", "https://", "raw_payload", "response_body")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"not an object: {path}")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(payload: dict[str, Any]) -> list[str]:
    dimensions = dict(payload.get("token_dimensions") or {})
    result: list[str] = ["probe_shape_vocab_version=pg349_v1"]
    for dimension, values in dimensions.items():
        name = str(dimension).strip().lower()
        if not name or not isinstance(values, list):
            raise ValueError("payload vocabulary dimensions must be non-empty lists")
        for value in values:
            token = f"probe_shape_{name}={str(value).strip().lower()}"
            if not _SAFE.fullmatch(token) or any(marker in token for marker in _FORBIDDEN):
                raise ValueError("payload vocabulary contains unsafe token material")
            result.append(token)
    return list(dict.fromkeys(result))


def extend(*, base_path: Path = DEFAULT_BASE, payload_path: Path = DEFAULT_PAYLOAD) -> dict[str, Any]:
    base = _load(base_path)
    payload = _load(payload_path)
    if not (base.get("vocabulary_policy") or {}).get("append_only", False):
        raise ValueError("base vocabulary is not append-only")
    if (payload.get("storage_policy") or {}).get("raw_strings_stored") is not False:
        raise ValueError("payload vocabulary must declare raw_strings_stored=false")
    added = _tokens(payload)
    context = list(dict.fromkeys([*(str(token) for token in base.get("context_tokens") or []), *added]))
    target = list(dict.fromkeys(str(token) for token in base.get("target_tokens") or []))
    shared = sorted((set(context) | set(target) | set(str(token) for token in base.get("shared_tokens") or [])) - {"[PAD]", "[UNK]"})
    result = dict(base)
    result["schema_version"] = "pg349-dynamic-typed-web-token-v8-payload-shape-reserved"
    result["base_vocabulary"] = str(base_path.resolve().relative_to(ROOT.resolve()))
    result["base_vocabulary_sha256"] = _file_sha(base_path)
    result["payload_probe_vocabulary"] = str(payload_path.resolve().relative_to(ROOT.resolve()))
    result["payload_probe_vocabulary_sha256"] = _file_sha(payload_path)
    result["payload_probe_vocabulary_status"] = payload.get("status")
    result["context_tokens"] = sorted(context)
    result["target_tokens"] = sorted(target)
    result["shared_tokens"] = shared
    counts = dict(result.get("counts") or {})
    counts.update(
        {
            "context_observed": int(counts.get("context_observed", 0)),
            "target_observed": int(counts.get("target_observed", 0)),
            "payload_probe_reserved": len(added),
            "context_total": len(context),
            "target_total": len(target),
            "shared_total": len(shared),
        }
    )
    result["counts"] = counts
    policy = dict(result.get("vocabulary_policy") or {})
    policy.update(
        {
            "append_only": True,
            "raw_literal_tokens_allowed": False,
            "payload_probe_shapes_abstract_only": True,
            "payload_probe_raw_binding": "evaluator_only_runtime_probe_ref",
        }
    )
    result["vocabulary_policy"] = policy
    result["promotion"] = {"training": False, "memory": False, "payload": False, "vulnerability": False}
    result["vocabulary_sha256"] = ""
    result["vocabulary_sha256"] = _digest(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-vocabulary", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--payload-vocabulary", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = extend(base_path=args.base_vocabulary, payload_path=args.payload_vocabulary)
    args.output.resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.get("status"), "counts": result.get("counts"), "vocabulary_sha256": result.get("vocabulary_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
