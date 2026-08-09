"""Collect PG-36's independent maze with the shared bounded surface channels.

PG-36 is a separately implemented HTTP maze with two route layouts and four
phases (screen/confirm/error/timeout).  The older PG-36 collector is preserved
as a historical baseline.  This wrapper replays it into versioned PG-94
artifacts while adding only the shared label-free ``surface_observation`` and
``generic_effect_geometry`` channels.  It never stores a request or response
body and remains evaluation-only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg53_cross_source_oracle import generic_effect_geometry, surface_observation  # noqa: E402


BASE_SCRIPT = ROOT / "scripts" / "run_pg36_independent_maze_catalog.py"
CATALOG_OUTPUT = ROOT / "research" / "pg94_pg36_surface_catalog_v1.json"
TRACE_OUTPUT = ROOT / "research" / "pg94_pg36_surface_trace_v1.json"
TARGET_PORT = 31994
PROJECTION_SCHEMA = "canonical_effect_projection_v3_surface_signal"
PROTOCOL_ID = "pg-pk-94-pg36-surface-collector-v1"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response_projection_from_original(base: Any, original_projection: Any, response: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    projection, parsed = original_projection(response)
    if not isinstance(parsed, dict):
        raise RuntimeError("PG-36 response must be a JSON object for semantic projection")
    # This signal is computed before the body leaves local memory.  The
    # excluded evaluator keys, bounded counts and field-name hash buckets are
    # all label-free; the raw body is never persisted.
    projection["effect_surface"] = surface_observation(parsed)
    projection["effect_geometry"] = generic_effect_geometry(parsed)
    projection["projection_schema"] = PROJECTION_SCHEMA
    # The base collector already declared a hash over its pre-extension
    # projection.  Remove it before computing the new canonical hash; keeping
    # the old digest would make the digest self-referential and fail validation.
    projection.pop("projection_sha256", None)
    projection["projection_sha256"] = base.sha256_json(projection)
    return projection, parsed


class _FreshTarget:
    """Use PG-36's server lifecycle with a versioned PG-94 loopback port."""

    def __init__(self, base: Any, implementation: str, port: int = TARGET_PORT) -> None:
        self.base = base
        self.implementation = str(implementation)
        self.port = int(port)
        self.server = base.make_pg36_server(self.port, self.implementation)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.client: Any = None

    def __enter__(self) -> Any:
        self.thread.start()
        deadline = self.base.time.monotonic() + 5.0
        while self.base.time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                self.base.time.sleep(0.01)
        else:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2.0)
            raise RuntimeError("PG-94 PG-36 fixture did not start")
        self.client = self.base.httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=3.0, follow_redirects=False, headers={"accept": "application/json"})
        return self.client

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.client is not None:
            self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def main() -> int:
    base = _load(BASE_SCRIPT, "pg94_pg36_base_runtime")
    base.CATALOG_OUTPUT = CATALOG_OUTPUT
    base.TRACE_OUTPUT = TRACE_OUTPUT
    base.TARGET_PORT = TARGET_PORT
    wrapper_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_hash = hashlib.sha256((ROOT / "app" / "pg36_independent_maze_fixture.py").read_bytes()).hexdigest()
    original_source = base._source
    # Keep the original source contract but bind provenance to this wrapper,
    # so the new projection channel cannot masquerade as the old collector.
    base._source = lambda implementation, _collector_hash, _fixture_hash: original_source(implementation, wrapper_hash, fixture_hash)
    original_projection = base._response_projection
    base._response_projection = lambda response: _response_projection_from_original(base, original_projection, response)
    base._FreshTarget = lambda implementation, port=TARGET_PORT: _FreshTarget(base, implementation, port)
    result = base.main()
    catalog = json.loads(CATALOG_OUTPUT.read_text(encoding="utf-8"))
    trace = json.loads(TRACE_OUTPUT.read_text(encoding="utf-8"))
    catalog["schema_version"] = "pg-pk-94-pg36-surface-catalog-v1"
    catalog["catalog_id"] = "pg94-pg36-surface-v1"
    catalog["protocol_id"] = PROTOCOL_ID
    catalog["collector_profile"] = "pg36_original_maze_runner_plus_shared_bounded_surface_channels"
    catalog["projection_schema"] = PROJECTION_SCHEMA
    catalog["projection_repair_post_hoc"] = False
    catalog["sample_count"] = len(catalog.get("samples", []))
    catalog["model_evaluation_completed"] = False
    trace["schema_version"] = "pg-pk-94-pg36-surface-trace-v1"
    trace["protocol_id"] = PROTOCOL_ID
    trace["projection_schema"] = PROJECTION_SCHEMA
    trace["projection_repair_post_hoc"] = False
    CATALOG_OUTPUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_OUTPUT.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "catalog": str(CATALOG_OUTPUT.relative_to(ROOT)),
        "trace": str(TRACE_OUTPUT.relative_to(ROOT)),
        "sample_count": len(catalog.get("samples", [])),
        "typed_positive_count": catalog.get("typed_positive_count", 0),
        "negative_control_count": catalog.get("negative_control_count", 0),
        "target_instance_count": catalog.get("target_instance_count", 0),
        "source_count": catalog.get("source_count", 0),
        "projection_schema": PROJECTION_SCHEMA,
        "training_eligible": False,
        "result": result,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
