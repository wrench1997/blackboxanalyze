"""PG-362 full Rule-IR next-token candidate over the PG-361 slot dataset.

PG-361 queried each slot independently.  This wrapper tests the complementary
hypothesis that a causal decoder needs the whole ordered Rule-IR target to
learn cross-slot consistency (ASK/repair/negative/safe-to-send).  It reuses the
reviewed PG-351 evaluator only as an abstract offline trainer; it never reads
raw payloads, responses, routes, or evaluator sidecars and never enables
promotion.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "run_pg351_a800_ask_oracle_composition_candidate.py"
spec = importlib.util.spec_from_file_location("pg351_full_candidate_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load abstract full-target candidate base")
BASE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = BASE
spec.loader.exec_module(BASE)

# Keep this order append-only and identical to the PG-361 dataset contract.
BASE.TARGET_KEY_ORDER = (
    "question",
    "ask_reason",
    "next_action",
    "repair_action",
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "syntax_category_ref",
    "probe_variant_ref",
    "safe_to_send",
    "payload_shape_ref",
    "oracle_ref",
    "negative_control_presence_ref",
)
BASE.TARGET_PREFIXES = tuple(BASE.TARGET_PREFIXES) + ("syntax_category_ref=",)
BASE.SCHEMA_VERSION = "pg362-a800-full-rule-ir-candidate-v1"
BASE.SEEDS = (36201, 36202, 36203)


if __name__ == "__main__":
    raise SystemExit(BASE.main())
