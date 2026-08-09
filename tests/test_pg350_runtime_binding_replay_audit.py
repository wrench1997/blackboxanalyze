from __future__ import annotations

import json
from pathlib import Path

from app.pg348_dynamic_runtime import load_registry
from scripts.audit_pg350_runtime_binding_replay import _audit
from scripts.run_pg350_runtime_binding_replay import replay


ROOT = Path(__file__).resolve().parents[1]


def test_pg350_audit_passes_and_is_fail_closed() -> None:
    registry = load_registry(ROOT / "fixtures" / "pg348" / "registry_v1.json")
    report, sidecars, _ = replay(registry, seeds=(35003,), max_routes=2)
    audit = _audit(report, sidecars)
    assert audit["status"] == "passed_evaluator_only"
    assert audit["counts"] == {"episodes": 2, "confirmed_positive": 2, "raw_firewall_violations": 0}
    tampered = json.loads(json.dumps(sidecars))
    tampered["sidecars"][0]["roles"]["candidate"]["binding"]["raw_wire_stored"] = True
    blocked = _audit(report, tampered)
    assert blocked["status"] == "blocked"
    assert any("binding_firewall" in failure for failure in blocked["failures"])

