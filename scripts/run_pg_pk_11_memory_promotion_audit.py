"""Audit PG-PK-10 promotion rows with the shared promotion runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.experiment_ledger import ExperimentLedger  # noqa: E402
from app.promotion_runner import run_promotion_audit  # noqa: E402


SOURCE_REPORT = ROOT / "research" / "pg_pk_10_logic_access_v1.json"
REPORT_PATH = ROOT / "research" / "pg_pk_11_memory_promotion_audit_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_11_memory_promotion_audit_v1.md"
LEDGER_PATH = ROOT / "artifacts" / "experiment-ledger-pg-pk-11.jsonl"


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    rows = list(source.get("promotion_ledger") or [])
    # The runner consumes only bounded ledger rows.  Source hash is carried as
    # provenance, not as a payload or response body.
    source_hash = str((source.get("provenance") or {}).get("source_hashes", [""])[0] if (source.get("provenance") or {}).get("source_hashes") else "")
    for row in rows:
        row.setdefault("source_hash", source_hash)
    audit = run_promotion_audit(rows, rule_keys=["access_control::typed_boundary", "logic::typed_boundary"])
    ledger = ExperimentLedger(LEDGER_PATH, ROOT)
    record = ledger.append({
        "protocol_id": "pg-pk-11-memory-promotion-audit-v1",
        "run_id": "pg-pk-11-promotion-audit",
        "dataset_id": "logic-access-pg10",
        "target_instance_id": "+".join(audit["provenance"]["target_instance_ids"]),
        "sampling_seed": 20260812,
        "status": "promote" if audit["all_promoted"] else "quarantine",
        "memory_write_allowed": audit["memory_write_allowed"],
        "provenance": audit["provenance"],
        "replay_queue_count": len(audit["replay_queue"]),
        "local_only": True,
    })
    report = {
        "schema_version": "sift-pg-pk-11-memory-promotion-audit-report-v1",
        "protocol_id": "pg-pk-11-memory-promotion-audit-v1",
        "status": "pass" if audit["all_promoted"] else "quarantine",
        "source_report": str(SOURCE_REPORT.relative_to(ROOT)),
        "audit": audit,
        "ledger": {
            "path": str(LEDGER_PATH.relative_to(ROOT)),
            "record_hash": record["record_hash"],
            "verification": ledger.verify(),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-11 统一 memory promotion audit\n\n"
        f"状态：`{report['status']}`；memory write：`{audit['memory_write_allowed']}`；replay queue：{len(audit['replay_queue'])}。\n\n"
        f"datasets：{len(audit['provenance']['dataset_ids'])}；targets：{len(audit['provenance']['target_instance_ids'])}；seeds：{len(audit['provenance']['sampling_seeds'])}；evidence hashes：{audit['provenance']['evidence_hash_count']}。\n\n"
        "失败时只生成 fresh-reset replay queue，不直接写长期记忆。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "status": report["status"],
        "memory_write_allowed": audit["memory_write_allowed"],
        "replay_queue_count": len(audit["replay_queue"]),
        "provenance": audit["provenance"],
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
