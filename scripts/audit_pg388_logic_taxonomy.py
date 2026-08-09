"""Read-only coverage audit for the PG-388 business-logic taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg388_logic_invariant_projection import ALL_LOGIC_CASES, LOGIC_CASES, SUPPLEMENTAL_LOGIC_CASES


SCHEMA_VERSION = "pg388-logic-taxonomy-audit-v1"

# These anchors are deliberately abstract case references, not route literals or
# business values.  A category can be represented by multiple contracts.
TAXONOMY: dict[str, dict[str, tuple[str, ...]]] = {
    "installation": {
        "covered": ("install_reentry_gate", "install_artifact_exposure", "update_authorization_gate"),
    },
    "transaction": {
        "covered": ("purchase_price_binding", "purchase_status_transition", "purchase_quantity_floor", "purchase_amount_floor", "purchase_concurrency_lock", "nonce_replay"),
    },
    "risk_control": {
        "covered": ("coupon_reuse_boundary", "cashout_balance_binding", "rate_limit_scope"),
    },
    "account_registration_password": {
        "covered": ("registration_overwrite", "registration_enumeration", "password_storage_contract", "password_strength_contract"),
    },
    "identity_normalization_cookie_login": {
        "covered": ("identity_canonicalization", "phone_canonicalization", "cookie_integrity", "login_rate_limit", "lock_unlock_contract"),
    },
    "password_recovery_change_appeal_update": {
        "covered": ("password_reset_subject_binding", "password_reset_expiry", "password_reset_response_secrecy", "password_reset_host_binding", "change_password_old_secret", "appeal_identity_binding", "update_field_allowlist", "update_authorization_gate"),
    },
    "information_query_authorization": {
        "covered": ("query_object_scope", "query_identifier_entropy", "unauthorized_static_resource", "vertical_role_scope", "cross_role_scope"),
    },
    "two_factor": {
        "covered": ("two_factor_reset_binding", "two_factor_attempt_budget"),
        "gaps": ("oauth_second_factor", "activation_link_second_factor", "csrf_disable_second_factor"),
    },
    "captcha": {
        "covered": ("captcha_reuse", "captcha_expiry", "captcha_attempt_limit"),
        "gaps": ("captcha_predictability", "captcha_response_exposure", "captcha_client_validation", "captcha_delivery_abuse"),
    },
    "session_randomness_other": {
        "covered": ("session_fixation_boundary", "session_scope_boundary", "session_storage_boundary", "random_seed_quality", "crypto_mode_binding", "execution_order", "sensitive_projection"),
        "gaps": ("session_guessing", "session_forgery", "session_leakage"),
    },
}


def _sha256(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_audit() -> dict[str, Any]:
    refs = {str(item["case_ref"]) for item in ALL_LOGIC_CASES}
    categories: dict[str, Any] = {}
    missing_anchors: list[str] = []
    declared_gaps: list[dict[str, str]] = []
    candidate_only: list[dict[str, str]] = []
    for category, spec in TAXONOMY.items():
        covered = [ref for ref in spec.get("covered", ()) if ref in refs]
        missing = [ref for ref in spec.get("covered", ()) if ref not in refs]
        if missing:
            missing_anchors.extend(f"{category}:{ref}" for ref in missing)
        unresolved = [gap for gap in spec.get("gaps", ()) if gap not in refs]
        candidate = [gap for gap in spec.get("gaps", ()) if gap in refs]
        categories[category] = {
            "covered_case_refs": covered,
            "covered_count": len(covered),
            "missing_anchor_case_refs": missing,
            "candidate_only_case_refs": candidate,
            "unresolved_next_cases": unresolved,
        }
        declared_gaps.extend({"category": category, "gap": gap} for gap in unresolved)
        candidate_only.extend({"category": category, "case_ref": gap} for gap in candidate)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_candidate_coverage_all_anchors" if not missing_anchors and not declared_gaps else "blocked_missing_taxonomy_anchors",
        "case_count": len(ALL_LOGIC_CASES),
        "core_case_count": len(LOGIC_CASES),
        "supplemental_case_count": len(SUPPLEMENTAL_LOGIC_CASES),
        "case_ref_sha256": _sha256(sorted(refs)),
        "categories": categories,
        "missing_anchor_count": len(missing_anchors),
        "diagnostic_gap_count": len(declared_gaps),
        "diagnostic_gaps": declared_gaps,
        "candidate_only_count": len(candidate_only),
        "candidate_only_contracts": candidate_only,
        "training_eligible": 0,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "interpretation": "coverage inventory only; supplemental contracts are candidate-only and still require fresh typed evidence before training or promotion",
    }
    payload["audit_sha256"] = _sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("research/pg388_logic_taxonomy_audit_v1.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{report['status']}: cases={report['case_count']} diagnostic_gaps={report['diagnostic_gap_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main