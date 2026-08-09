#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.juice_shop_adapter import DockerJuiceShopManager, EvidenceLedger, JuiceShopAdapter, JuiceShopEpisode  # noqa: E402


OUTPUT = ROOT / "research/juice_shop_loop_12_nonmutating_method_audit.json"
EVIDENCE = ROOT / "artifacts/juice-shop-loop-12/nonmutating-method-audit.jsonl"
PATHS = ["/", "/robots.txt", "/.well-known/security.txt", "/sitemap.xml", "/metrics", "/ftp/"]
SEED = 20262071


def main() -> None:
    adapter = JuiceShopAdapter()
    manager = DockerJuiceShopManager(adapter)
    environment = manager.reset(SEED)
    before = adapter.evaluator_solved_state()
    rows = []
    with JuiceShopEpisode(adapter, ledger=EvidenceLedger(EVIDENCE, ROOT)) as episode:
        for path in PATHS:
            before_step = adapter.evaluator_solved_state()
            observation = episode.act({"method": "OPTIONS", "path": path})
            after_step = adapter.evaluator_solved_state()
            changed = [key for key, value in after_step.items() if value and not before_step.get(key, False)]
            rows.append({
                "path": path,
                "status_code": observation["observation"]["status_code"],
                "headers": observation["observation"]["headers"],
                "body_length": observation["observation"]["summary"]["body_length"],
                "challenge_transitions": changed,
            })
    after = adapter.evaluator_solved_state()
    report = {
        "schema_version": "sift-juice-shop-loop-12-nonmutating-method-audit-v1",
        "status": "pass" if before == after else "failed_nonmutation",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "method": "OPTIONS",
        "paths": PATHS,
        "initial_solved_count": sum(before.values()),
        "final_solved_count": sum(after.values()),
        "rows": rows,
        "evidence": str(EVIDENCE.relative_to(ROOT)),
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
