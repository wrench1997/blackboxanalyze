"""Read-only typed oracle for local business/authorization boundary replay.

The endpoint accepts only abstract probe classes.  It does not access users,
tokens, cookies, a database, or mutable state; the returned boundary effect is
the evaluator contract for a local maze fixture.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


PROBE_CLASSES = frozenset({"normal", "boundary_candidate", "invariant_break"})
SURFACES = frozenset({"authorization_boundary", "business_invariant", "ordinary_response"})


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_logic_replay_oracle(probe_class: str, *, surface: str = "authorization_boundary") -> dict[str, Any]:
    probe_class = str(probe_class)
    surface = str(surface)
    if probe_class not in PROBE_CLASSES:
        raise ValueError("unknown abstract logic probe class")
    if surface not in SURFACES:
        raise ValueError("unknown logic replay surface")
    positive = probe_class != "normal"
    effect = surface
    projection = {
        "oracle": "controlled_logic_boundary_v1",
        "surface": surface,
        "probe_class": probe_class,
        "candidate_signal": positive,
        "typed_boundary_observed": positive,
        "confirmed_effect": effect if positive else "none",
        "state_mutated": False,
        "credentials_accessed": False,
        "database_touched": False,
        "network_access": False,
        "external_network": False,
    }
    projection["evidence_hash"] = _digest(projection)
    return projection


__all__ = ["PROBE_CLASSES", "SURFACES", "run_logic_replay_oracle"]
