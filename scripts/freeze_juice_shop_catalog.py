from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.juice_shop_adapter import JuiceShopAdapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the evaluator-only Juice Shop Loop 12 catalog and family split.")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "juice_shop_loop_12_catalog_v3.json")
    parser.add_argument("--image-digest", required=True)
    args = parser.parse_args()

    adapter = JuiceShopAdapter()
    manifest = {
        "schema_version": "sift-juice-shop-catalog-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "base_url": adapter.base_url,
            "image_digest": args.image_digest,
            "target_container": "sift-loop12-juice-v20",
            "fixed_ingress_proxy": "sift-loop12-proxy",
            "target_network_internal": True,
        },
        "information_classification": "evaluator-only; never serialize challenge metadata into agent prompts",
        **adapter.safe_split_manifest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "catalog_count": manifest["catalog_count"],
        "selection_count": manifest["selection_count"],
        "splits": manifest["splits"],
        "catalog_sha256": manifest["catalog_sha256"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
