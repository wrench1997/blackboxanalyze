"""Generate the offline PG-31 Rule IR evaluation manifest.

The command only serializes synthetic projections.  It never reaches a
network, Docker, browser, database, or model checkpoint, and it does not
create a training artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ood_rule_ir_dataset import DEFAULT_SEEDS, generate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research" / "pg_pk_31_ood_rule_ir_evaluation_manifest_v1.json",
    )
    parser.add_argument("--samples-per-role", type=int, default=12)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    manifest = generate_manifest(seeds=tuple(args.seeds), samples_per_role=args.samples_per_role)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(args.output),
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": len(manifest["samples"]),
        "dataset_test_count": len(manifest["dataset_tests"]),
        "training_eligible": manifest["training_eligible"],
        "model_evaluation_completed": manifest["model_evaluation_completed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
